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
"""Tests for inference-perf's BR0.2 partial-report emission.

Inference-perf emits its slice of a BR0.2 report (``version``, ``run`` with
the generated uid and stage time window, ``results`` from request metrics).
A downstream composer merges in stack/scenario/observability/etc to produce
a full BR0.2 document. These tests pin down the shape and mergeability of
the emitted partial.
"""

import time
from typing import Any, Dict

from inference_perf.apis import (
    InferenceInfo,
    RequestLifecycleMetric,
    StreamedResponseMetrics,
)
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.br.v0_2 import build_partial_report, generate_run_uid
from inference_perf.reportgen.br.v0_2.schema_v0_2 import VERSION

import yaml


def _metric(start: float, end: float) -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=start - 0.001,
        start_time=start,
        end_time=end,
        request_data="{}",
        response_data="ok",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=128)),
            response_metrics=StreamedResponseMetrics(
                response_chunks=[],
                chunk_times=[start + 0.05, end - 0.005],
                output_tokens=2,
                output_token_times=[start + 0.05, end - 0.005],
            ),
        ),
        error=None,
    )


def test_generate_run_uid_is_unique_and_stage_tagged() -> None:
    a = generate_run_uid(0)
    b = generate_run_uid(0)
    assert a != b
    assert a.startswith("inference-perf-stage-0-")
    assert generate_run_uid(7).startswith("inference-perf-stage-7-")


def test_partial_report_top_level_shape() -> None:
    now = time.time()
    metrics = [_metric(now + i * 0.1, now + i * 0.1 + 0.08) for i in range(5)]

    partial = build_partial_report(metrics, tokenizer=None, run_uid="test-uid-abc")

    assert set(partial.keys()) == {"version", "run", "results"}
    assert partial["version"] == VERSION
    assert partial["run"]["uid"] == "test-uid-abc"
    assert "time" in partial["run"]
    assert "results" in partial
    assert "request_performance" in partial["results"]


def test_partial_report_run_time_is_iso8601() -> None:
    start = 1_700_000_000.0
    end = start + 4.5
    metrics = [_metric(start, end)]

    partial = build_partial_report(metrics, tokenizer=None, run_uid="uid")

    run_time = partial["run"]["time"]
    assert run_time["start"].startswith("2023-")  # 1.7e9 → 2023
    assert run_time["start"].endswith("+00:00") or run_time["start"].endswith("Z")
    assert run_time["duration"].startswith("PT") and run_time["duration"].endswith("S")
    assert run_time["duration"] == "PT4.500S"


def test_partial_report_strips_nulls_recursively() -> None:
    """None-valued fields must be absent so yq-merge does not clobber with null."""
    now = time.time()
    metrics = [_metric(now, now + 0.1)]

    partial = build_partial_report(metrics, tokenizer=None, run_uid="uid")

    def assert_no_none(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert v is not None, f"null value found at {path}.{k}"
                assert_no_none(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                assert_no_none(item, f"{path}[{i}]")

    assert_no_none(partial)


def test_partial_report_round_trips_via_yaml_safe() -> None:
    """The emitted dict must serialize/deserialize via yaml.safe_* unchanged."""
    now = time.time()
    metrics = [_metric(now, now + 0.1)]

    partial = build_partial_report(metrics, tokenizer=None, run_uid="uid")
    rendered = yaml.safe_dump(partial, sort_keys=False, default_flow_style=False)
    reloaded = yaml.safe_load(rendered)

    assert reloaded == partial


def test_partial_report_deep_merges_cleanly_with_other_producer() -> None:
    """A composer doing a deep-merge of (inference-perf partial, other partial)
    must end up with both inference-perf's fields and the other producer's
    fields, with no overwrite on either side.
    """
    now = time.time()
    metrics = [_metric(now, now + 0.5)]
    ip_partial = build_partial_report(metrics, tokenizer=None, run_uid="ip-uid")

    other_partial: Dict[str, Any] = {
        "run": {
            "eid": "exp-42",
            "user": "brendan",
            "description": "smoke run",
        },
        "scenario": {
            "stack": [{"name": "vllm", "version": "0.10.0"}],
        },
    }

    merged = _deep_merge(ip_partial, other_partial)

    # Both partials' fields coexist; neither overwrites the other.
    assert merged["version"] == VERSION
    assert merged["run"]["uid"] == "ip-uid"
    assert merged["run"]["eid"] == "exp-42"
    assert merged["run"]["user"] == "brendan"
    assert "start" in merged["run"]["time"]
    assert merged["scenario"]["stack"][0]["name"] == "vllm"
    assert "request_performance" in merged["results"]


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal deep-merge mirroring ``yq '. * load(...)'`` semantics on maps."""
    out: Dict[str, Any] = {}
    for k in set(a) | set(b):
        if k in a and k in b and isinstance(a[k], dict) and isinstance(b[k], dict):
            out[k] = _deep_merge(a[k], b[k])
        elif k in b:
            out[k] = b[k]
        else:
            out[k] = a[k]
    return out
