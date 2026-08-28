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
"""Progress bars are a view of the exported metrics, and only of those."""

import io
from typing import Any, Tuple

import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge
from rich.console import Console

from inference_perf.config import APIConfig, APIType, Config, LoadConfig, StandardLoadStage
from inference_perf.observability.context import StageContext
from inference_perf.observability.metrics.registry import MetricSpec, build_metrics, exposition_name
from inference_perf.observability.metrics.sets import ALL_SPECS
from inference_perf.observability.metrics.sets.progress import STAGE_REQUESTS_FINISHED
from inference_perf.observability.progress import (
    BAR_SPECS,
    STAGE_REQUESTS_BAR,
    BarSpec,
    ProgressBars,
    snapshot,
    validate_bar_specs,
)


def _console() -> Console:
    # force_terminal makes rich render under pytest; record buffers the frames
    # so a test can read back what was painted.
    return Console(file=io.StringIO(), force_terminal=True, width=120, record=True)


def _registry_with(planned: float, finished: float, stage: str = "0") -> Tuple[CollectorRegistry, Gauge]:
    # A registry holding just the two series a stage-requests bar reads, so a
    # test can set them to any pair without running a stage. Returns the
    # finished gauge too, for tests that move it mid-run.
    registry = CollectorRegistry()
    Gauge("inference_perf_stage_requests_planned", "planned", ["stage"], registry=registry).labels(stage).set(planned)
    finished_gauge = Gauge("inference_perf_stage_requests_finished", "finished", ["stage"], registry=registry)
    finished_gauge.labels(stage).set(finished)
    return registry, finished_gauge


def _paint(bars: ProgressBars) -> None:
    # Re-read the metrics, then force rich to paint. rich coalesces repaints
    # on a timer of its own, so without this a test can only ever see the
    # final frame and not the states the bar passed through.
    bars.refresh(force=True)
    bars._progress.refresh()


# --- rendering ------------------------------------------------------------


def test_bar_shows_the_metric_values() -> None:
    # planned=10, finished=3 in the registry, one refresh.
    # Expects the rendered frame to read "3/10", straight off the metrics.
    registry, _ = _registry_with(planned=10, finished=3)
    console = _console()
    with ProgressBars(registry, console=console, refresh_hz=0) as bars:
        bars.open(STAGE_REQUESTS_BAR, stage="0")
        _paint(bars)
    rendered = console.export_text()
    assert "3/10" in rendered
    assert "Stage 0 Requests" in rendered


def test_bar_follows_the_metric_when_it_moves() -> None:
    # finished goes 1 -> 7 between refreshes with planned=10 throughout.
    # Expects both "1/10" and "7/10" to have been painted: the bar tracks the
    # metric rather than holding its opening value.
    registry, finished_gauge = _registry_with(planned=10, finished=1)
    console = _console()
    with ProgressBars(registry, console=console, refresh_hz=0) as bars:
        bars.open(STAGE_REQUESTS_BAR, stage="0")
        _paint(bars)
        finished_gauge.labels("0").set(7)
        _paint(bars)
    rendered = console.export_text()
    assert "1/10" in rendered
    assert "7/10" in rendered


def test_missing_series_renders_as_empty_rather_than_crashing() -> None:
    # An empty registry, so neither of the bar's metrics has a series yet.
    # Expects completed to read 0 and the total to render as unknown ("?"),
    # which is what a bar opened before its stage records anything must show.
    console = _console()
    with ProgressBars(CollectorRegistry(), console=console, refresh_hz=0) as bars:
        bars.open(STAGE_REQUESTS_BAR, stage="0")
        bars.refresh(force=True)
    assert "0/?" in console.export_text()


def test_counter_backed_bar_resolves_the_total_suffix() -> None:
    # A bar whose numerator is a Counter, which prometheus_client exposes as
    # <name>_total. Expects "4/10": the bar must resolve the sample name the
    # same way the exposition format does, not read spec.name literally.
    registry = CollectorRegistry()
    counter_spec = MetricSpec[Counter](name="test_done", documentation="done", metric_type=Counter, labelnames=("stage",))
    total_spec = MetricSpec[Gauge](name="test_planned", documentation="planned", metric_type=Gauge, labelnames=("stage",))
    Counter("test_done", "done", ["stage"], registry=registry).labels("0").inc(4)
    Gauge("test_planned", "planned", ["stage"], registry=registry).labels("0").set(10)

    assert exposition_name(counter_spec) == "test_done_total"
    console = _console()
    with ProgressBars(registry, console=console, refresh_hz=0) as bars:
        bars._open.append(
            (
                bars._progress.add_task(description="Counter Bar", total=None),
                BarSpec(name="t", description="Counter Bar", completed=counter_spec, total=total_spec),
                {"stage": "0"},
            )
        )
        bars.refresh(force=True)
    assert "4/10" in console.export_text()


def test_no_registry_means_no_bars() -> None:
    # ProgressBars built without a registry, which is what a LoadGenerator
    # constructed outside main.py has. Expects opening and refreshing to be
    # safe no-ops and nothing to be painted: a bar with no metric behind it is
    # the thing this module exists to prevent, so it is not drawn at all.
    console = _console()
    with ProgressBars(None, console=console, refresh_hz=0) as bars:
        handle = bars.open(STAGE_REQUESTS_BAR, stage="0")
        bars.refresh(force=True)
        handle.close()
    assert console.export_text().strip() == ""


def test_refresh_is_throttled_between_ticks() -> None:
    # refresh_hz=1 with two back-to-back refreshes.
    # Expects only the first to collect: driving refresh from a hot loop must
    # cost a clock read, not a registry collect, per iteration.
    registry, _ = _registry_with(planned=10, finished=1)
    collects = 0
    real_collect = registry.collect

    def counting_collect() -> Any:
        nonlocal collects
        collects += 1
        return real_collect()

    registry.collect = counting_collect  # type: ignore[method-assign]
    with ProgressBars(registry, console=_console(), refresh_hz=1) as bars:
        bars.open(STAGE_REQUESTS_BAR, stage="0")  # opening collects once
        before = collects
        bars.refresh()
        bars.refresh()
    assert collects == before + 1


# --- the declared bars are enforceable ------------------------------------


def test_shipped_bars_all_read_exported_unconditional_metrics() -> None:
    # The bars this repo ships, checked against the metrics it exports.
    # Expects no error: every bar's two metrics are in ALL_SPECS and are
    # exported on every run, so no config can leave a bar without a source.
    validate_bar_specs(BAR_SPECS, ALL_SPECS)


def test_bar_reading_an_unexported_metric_is_rejected() -> None:
    # A bar naming a metric that is not in the exported set at all.
    # Expects a ValueError at validation, not a bar that silently sits empty.
    stray = MetricSpec[Gauge](name="not_exported", documentation="stray", metric_type=Gauge)
    bad = BarSpec(name="bad", description="Bad", completed=stray, total=stray)
    with pytest.raises(ValueError, match="not in ALL_SPECS"):
        validate_bar_specs([bad], ALL_SPECS)


def test_bar_reading_a_config_conditional_metric_is_rejected() -> None:
    # A bar reading a metric that is only exported for streaming runs.
    # Expects a ValueError: the bar would have no source on a non-streaming
    # run, which is a broken display rather than a missing metric.
    def only_streaming(config: Config) -> bool:
        """Only when streaming is on."""
        return bool(config.api and config.api.streaming)

    conditional = MetricSpec[Gauge](name="sometimes", documentation="sometimes", metric_type=Gauge, enabled=only_streaming)
    bad = BarSpec(name="bad", description="Bad", completed=conditional, total=conditional)
    with pytest.raises(ValueError, match="config-conditional"):
        validate_bar_specs([bad], [conditional])


# --- the freeze hazard ----------------------------------------------------


def test_finished_series_is_pinned_when_its_stage_ends() -> None:
    # Stage 0 finishes 5 requests, then stage 1 starts and its live counter
    # reads 2. Expects stage 0 to stay at 5. The probes read counters the load
    # generator resets per stage, so a series left bound to a live probe would
    # report stage 1's numbers under stage 0's label. Pinning has to rebind
    # set_function, because set() on a child that already has one is a no-op.
    config = Config(
        api=APIConfig(type=APIType.Completion),
        load=LoadConfig(stages=[StandardLoadStage(rate=1, duration=1)] * 2),
    )
    hub = build_metrics(config)
    hub.on_run_start()

    live = {"finished": 0}
    stage_0 = StageContext(stage_id=0, planned_requests=5, requests_finished=lambda: live["finished"])
    hub.on_stage_start(stage_0)
    live["finished"] = 5
    hub.on_stage_end(stage_0)

    live["finished"] = 0  # the load generator resets the counter for stage 1
    stage_1 = StageContext(stage_id=1, planned_requests=5, requests_finished=lambda: live["finished"])
    hub.on_stage_start(stage_1)
    live["finished"] = 2

    values = snapshot(hub.registry)
    finished = exposition_name(STAGE_REQUESTS_FINISHED)
    assert values[(finished, (("stage", "0"),))] == 5.0
    assert values[(finished, (("stage", "1"),))] == 2.0
