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
    SessionLifecycleMetric,
    StreamedResponseMetrics,
)
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.br.v0_2 import build_partial_report, generate_experiment_eid, generate_run_uid
from inference_perf.reportgen.br.v0_2.schema import VERSION

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


def test_generate_experiment_eid_is_unique_and_prefixed() -> None:
    a = generate_experiment_eid()
    b = generate_experiment_eid()
    assert a != b
    assert a.startswith("inference-perf-experiment-")


def test_partial_report_stamps_shared_run_eid_across_stages() -> None:
    """run.eid is the invocation-wide experiment id: two stage partials built
    with the same run_eid must both emit it verbatim, so a composer can group
    a sweep's files as one experiment without parsing uids or filenames."""
    now = time.time()
    eid = "inference-perf-experiment-abcd1234"

    stage_0 = build_partial_report(
        [_metric(now, now + 0.1)], tokenizer=None, run_uid="stage-0-uid", run_eid=eid, stage_start=now, stage_end=now + 1.0
    )
    stage_1 = build_partial_report(
        [_metric(now + 1.0, now + 1.1)],
        tokenizer=None,
        run_uid="stage-1-uid",
        run_eid=eid,
        stage_start=now + 1.0,
        stage_end=now + 2.0,
    )

    assert stage_0["run"]["eid"] == eid
    assert stage_1["run"]["eid"] == eid
    assert stage_0["run"]["uid"] != stage_1["run"]["uid"]


def test_partial_report_omits_run_eid_when_not_supplied() -> None:
    """Without a run_eid the field must be absent, not null: a fabricated or
    null eid would clobber a composer's grouping on merge."""
    now = time.time()

    partial = build_partial_report([_metric(now, now + 0.1)], tokenizer=None, run_uid="uid")

    assert "eid" not in partial["run"]


def test_partial_report_top_level_shape() -> None:
    now = time.time()
    metrics = [_metric(now + i * 0.1, now + i * 0.1 + 0.08) for i in range(5)]

    partial = build_partial_report(metrics, tokenizer=None, run_uid="test-uid-abc", stage_start=now, stage_end=now + 1.0)

    assert set(partial.keys()) == {"version", "run", "results"}
    assert partial["version"] == VERSION
    assert partial["run"]["uid"] == "test-uid-abc"
    assert "time" in partial["run"]
    assert "results" in partial
    assert "request_performance" in partial["results"]


def test_partial_report_run_time_is_iso8601() -> None:
    start = 1_700_000_000.0
    metrics = [_metric(start, start + 4.0)]

    partial = build_partial_report(metrics, tokenizer=None, run_uid="uid", stage_start=start, stage_end=start + 4.5)

    run_time = partial["run"]["time"]
    assert run_time["start"].startswith("2023-")  # 1.7e9 → 2023
    assert run_time["start"].endswith("+00:00") or run_time["start"].endswith("Z")
    assert run_time["duration"].startswith("PT") and run_time["duration"].endswith("S")
    assert run_time["duration"] == "PT4.500S"


def test_partial_report_run_time_from_stage_wall_clock_not_request_clocks() -> None:
    """Request lifecycle timestamps come from a monotonic clock whose origin
    is arbitrary (roughly seconds since boot), so they must never be
    interpreted as epoch: deriving run.time from them put every run in
    January 1970. run.time comes from the stage's wall-clock window."""
    metrics = [_metric(345_678.0, 345_682.5)]  # perf_counter-style values
    stage_start = 1_770_000_000.0

    partial = build_partial_report(
        metrics, tokenizer=None, run_uid="uid", stage_start=stage_start, stage_end=stage_start + 60.0
    )

    run_time = partial["run"]["time"]
    assert run_time["start"].startswith("2026-")  # 1.77e9 → 2026, not 1970
    assert run_time["end"].startswith("2026-")
    assert run_time["duration"] == "PT60.000S"


def test_partial_report_omits_run_time_without_stage_window() -> None:
    """When the stage's wall-clock window is unknown, run.time is omitted
    (absence over fabrication) so a composer's own value survives the
    merge; the request timestamps are no substitute, being monotonic."""
    now = time.time()
    metrics = [_metric(now, now + 0.1)]

    partial = build_partial_report(metrics, tokenizer=None, run_uid="uid")

    assert "time" not in partial["run"]


def test_partial_report_strips_nulls_recursively() -> None:
    """None-valued fields must be absent so yq-merge does not clobber with null."""
    now = time.time()
    metrics = [_metric(now, now + 0.1)]

    partial = build_partial_report(metrics, tokenizer=None, run_uid="uid", stage_start=now, stage_end=now + 0.2)

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

    partial = build_partial_report(metrics, tokenizer=None, run_uid="uid", stage_start=now, stage_end=now + 0.2)
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
    ip_partial = build_partial_report(metrics, tokenizer=None, run_uid="ip-uid", stage_start=now, stage_end=now + 0.6)

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


def _session_metric(session_id: str, start: float, duration: float = 1.0) -> SessionLifecycleMetric:
    return SessionLifecycleMetric(
        session_id=session_id,
        stage_id=0,
        file_path=f"/traces/{session_id}.json",
        start_time=start,
        end_time=start + duration,
        duration_sec=duration,
        num_events=3,
        num_events_completed=3,
        success=True,
        total_input_tokens=128,
        total_output_tokens=64,
    )


# The block is only emitted for stages that actually ran sessions, so a partial
# from a non-session workload yq-merges exactly as it did before the field was
# populated.
def test_partial_omits_session_performance_without_sessions() -> None:
    partial = build_partial_report([_metric(100.0, 101.0)], None, run_uid="ip-uid")

    assert "session_performance" not in partial["results"]


def test_partial_includes_session_performance_for_session_stages() -> None:
    partial = build_partial_report(
        [_metric(100.0, 101.0)],
        None,
        run_uid="ip-uid",
        session_metrics=[_session_metric("s1", 100.0), _session_metric("s2", 102.0)],
    )

    sessions = partial["results"]["session_performance"]["sessions"]
    assert sessions["total"] == 2
    assert sessions["succeeded"] == 2
    assert sessions["total_events"] == 6


# Sessions the stage stranded have no metric to be passed alongside, so their
# count travels as its own argument and lands in `total`.
def test_partial_counts_stranded_sessions_in_the_session_total() -> None:
    partial = build_partial_report(
        [_metric(100.0, 101.0)],
        None,
        run_uid="ip-uid",
        session_metrics=[_session_metric("s1", 100.0)],
        sessions_not_completed=3,
    )

    sessions = partial["results"]["session_performance"]["sessions"]
    assert sessions["total"] == 4
    assert sessions["succeeded"] == 1
    assert sessions["failed"] == 0


# A stage whose sessions all stranded has no request metrics either: the
# partial still carries the session block, and the results map simply has no
# request side to report.
def test_partial_reports_a_fully_stranded_stage() -> None:
    partial = build_partial_report([], None, run_uid="ip-uid", sessions_not_completed=2)

    sessions = partial["results"]["session_performance"]["sessions"]
    assert sessions["total"] == 2
    assert sessions["succeeded"] == 0
    assert sessions["failed"] == 0
    assert "request_performance" not in partial["results"]


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
