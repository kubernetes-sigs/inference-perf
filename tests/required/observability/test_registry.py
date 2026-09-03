# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from typing import Optional

import pytest
from prometheus_client import Counter, Gauge, Histogram
from prometheus_client.exposition import generate_latest

from inference_perf.apis.base import ErrorResponseInfo, InferenceInfo, RequestLifecycleMetric
from inference_perf.config import APIConfig, Config
from inference_perf.observability.metrics import (
    MetricSpec,
    MetricStability,
    MetricsHub,
    PrometheusMetricsServer,
    build_metrics,
)
from inference_perf.payloads import RequestMetrics, Text


# Builds one finished request for the hooks to observe: scheduled and started at
# 0s, ended at 1s, 1 prompt token. failed=True attaches a "Timeout" error so the
# same request counts as a failure instead of a success.
def _metric(stage_id: Optional[int], failed: bool = False) -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        stage_id=stage_id,
        scheduled_time=0.0,
        start_time=0.0,
        end_time=1.0,
        request_data="r",
        info=InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=1))),
        error=ErrorResponseInfo(error_type="Timeout", error_msg="boom") if failed else None,
    )


# The hub's registry rendered as the text a scrape would return, so a test can
# assert on names, HELP lines and series exactly as a consumer sees them.
def _exposition(hub: MetricsHub) -> str:
    return generate_latest(hub.registry).decode()


# Builds the real ALL_SPECS registry from a default Config. Expects the always-on
# metrics to be there, checked through run_elapsed_seconds and requests_total.
def test_core_metrics_exported_on_default_config() -> None:
    hub = build_metrics(Config())
    body = _exposition(hub)
    assert "inference_perf_run_elapsed_seconds" in body
    assert "inference_perf_requests_total" in body


# Two specs, one enabled and one whose predicate returns False. Expects test_on in
# the scrape and test_off missing entirely, not present at 0.
def test_disabled_spec_is_absent_not_zero() -> None:
    specs = (
        MetricSpec(name="test_on", documentation="enabled", metric_type=Gauge),
        MetricSpec(name="test_off", documentation="disabled", metric_type=Gauge, enabled=lambda config: False),
    )
    hub = build_metrics(Config(), specs=specs)
    body = _exposition(hub)
    assert "test_on" in body
    assert "test_off" not in body


# One spec gated on api.streaming, built twice: streaming off, then on. Expects the
# metric absent from the first scrape and present in the second.
def test_enabled_predicate_reads_config() -> None:
    specs = (
        MetricSpec(
            name="test_streaming_only",
            documentation="only exported for streaming runs",
            metric_type=Histogram,
            buckets=(0.1, 1.0),
            enabled=lambda config: config.api.streaming,
        ),
    )
    off = build_metrics(Config(api=APIConfig(streaming=False)), specs=specs)
    on = build_metrics(Config(api=APIConfig(streaming=True)), specs=specs)
    assert "test_streaming_only" not in _exposition(off)
    assert "test_streaming_only" in _exposition(on)


# Four observed requests: two successes and one failure in stage 0, plus one
# success with no stage. Expects requests_total to read 2 success and 1 failure at
# stage="0", and 1 success at stage="" for the stageless one.
def test_requests_counted_at_completion_by_stage_and_status() -> None:
    hub = build_metrics(Config())
    hub.observe_request(_metric(stage_id=0))
    hub.observe_request(_metric(stage_id=0))
    hub.observe_request(_metric(stage_id=0, failed=True))
    hub.observe_request(_metric(stage_id=None))

    counter = "inference_perf_requests_total"
    assert hub.registry.get_sample_value(counter, {"stage": "0", "status": "success"}) == 2.0
    assert hub.registry.get_sample_value(counter, {"stage": "0", "status": "failure"}) == 1.0
    assert hub.registry.get_sample_value(counter, {"stage": "", "status": "success"}) == 1.0


# Reads run_elapsed_seconds before on_run_start, then twice 10ms apart after it.
# Expects exactly 0.0 before the run starts, then a value that keeps rising.
def test_run_elapsed_zero_until_run_start_then_advances() -> None:
    hub = build_metrics(Config())
    gauge = "inference_perf_run_elapsed_seconds"
    assert hub.registry.get_sample_value(gauge) == 0.0

    hub.on_run_start()
    time.sleep(0.01)
    first = hub.registry.get_sample_value(gauge)
    assert first is not None and first > 0.0
    time.sleep(0.01)
    second = hub.registry.get_sample_value(gauge)
    assert second is not None and second > first


# Two request hooks, the first raising RuntimeError, driven with two requests.
# Expects the second hook to still count both (test_ok_total == 2) and the raising
# one to be logged once, not once per request.
def test_failing_hook_is_isolated_and_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    def _boom(counter: Counter, metric: RequestLifecycleMetric) -> None:
        raise RuntimeError("boom")

    def _count(counter: Counter, metric: RequestLifecycleMetric) -> None:
        counter.inc()

    # The failing spec comes first to prove the fan-out continues past it.
    specs = (
        MetricSpec(name="test_boom", documentation="always fails", metric_type=Counter, on_request=_boom),
        MetricSpec(name="test_ok", documentation="counts requests", metric_type=Counter, on_request=_count),
    )
    hub = build_metrics(Config(), specs=specs)
    with caplog.at_level(logging.ERROR):
        hub.observe_request(_metric(stage_id=0))
        hub.observe_request(_metric(stage_id=0))

    assert hub.registry.get_sample_value("test_ok_total") == 2.0
    boom_logs = [r for r in caplog.records if "test_boom" in r.getMessage()]
    assert len(boom_logs) == 1


# Passes the same spec twice in one set. Expects build_metrics to raise ValueError
# rather than let prometheus_client fail on a duplicate registration later.
def test_duplicate_metric_names_rejected() -> None:
    spec = MetricSpec(name="test_dupe", documentation="duplicated", metric_type=Gauge)
    with pytest.raises(ValueError, match="duplicate metric names"):
        build_metrics(Config(), specs=(spec, spec))


# Declares buckets on a Counter. Expects MetricSpec construction itself to raise
# ValueError, since buckets mean nothing outside a Histogram.
def test_buckets_rejected_for_non_histograms() -> None:
    with pytest.raises(ValueError, match="buckets"):
        MetricSpec(name="test_bad", documentation="bad", metric_type=Counter, buckets=(1.0,))


# Observes one stage-1 success, serves the hub's registry on an ephemeral port and
# fetches /metrics. Expects the scraped body to carry
# inference_perf_requests_total{stage="1",status="success"} 1.0.
def test_hub_registry_served_over_http() -> None:
    hub = build_metrics(Config())
    hub.observe_request(_metric(stage_id=1))
    server = PrometheusMetricsServer(hub.registry, port=0)
    server.start()
    try:
        import urllib.request

        assert server.bound_port is not None
        with urllib.request.urlopen(f"http://127.0.0.1:{server.bound_port}/metrics", timeout=2) as resp:
            body = resp.read().decode()
        assert 'inference_perf_requests_total{stage="1",status="success"} 1.0' in body
    finally:
        server.stop()


# --- stability levels -----------------------------------------------------------


# A spec that says nothing about stability, plus one declared STABLE. Expects the
# level to lead the scraped HELP line of each: "[ALPHA] not promised yet" and
# "[STABLE] promised for this major version", with the metric names untouched.
def test_stability_level_leads_the_help_text() -> None:
    specs = (
        MetricSpec(name="test_new", documentation="not promised yet", metric_type=Gauge),
        MetricSpec(
            name="test_settled",
            documentation="promised for this major version",
            metric_type=Gauge,
            stability=MetricStability.STABLE,
        ),
    )
    body = _exposition(build_metrics(Config(), specs=specs))
    assert "# HELP test_new [ALPHA] not promised yet" in body
    assert "# HELP test_settled [STABLE] promised for this major version" in body


# Same check for a bucketed Histogram, which build_metrics instantiates through a
# separate branch. Expects "[BETA] latency of something" on the HELP line.
def test_stability_level_reaches_histogram_help_text() -> None:
    specs = (
        MetricSpec(
            name="test_hist_seconds",
            documentation="latency of something",
            metric_type=Histogram,
            buckets=(0.1, 1.0),
            stability=MetricStability.BETA,
        ),
    )
    body = _exposition(build_metrics(Config(), specs=specs))
    assert "# HELP test_hist_seconds [BETA] latency of something" in body


# Reads the default off a bare spec. Expects ALPHA, so a metric added without a
# thought about compatibility promises nothing.
def test_specs_are_alpha_by_default() -> None:
    spec = MetricSpec(name="test_default", documentation="d", metric_type=Gauge)
    assert spec.stability is MetricStability.ALPHA


# Every level's promise wording, read through the property the generated doc uses.
# Expects a non-empty sentence for each of ALPHA, BETA and STABLE.
def test_every_level_states_its_promise() -> None:
    for level in MetricStability:
        assert level.promise.strip().endswith(".")
