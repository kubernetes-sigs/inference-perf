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
"""Tests for the BR0.2 fillability table and the schema walk behind it."""

from pathlib import Path
from typing import List, Set

import pytest

from inference_perf.apis import (
    ErrorResponseInfo,
    InferenceInfo,
    RequestLifecycleMetric,
    SessionLifecycleMetric,
    StreamedResponseMetrics,
)
from inference_perf.apis.base import STAGE_TEARDOWN_CANCELLED_ERROR_TYPE
from inference_perf.client.modelserver.metrics import CounterResult, GaugeResult
from inference_perf.client.server_metrics import ModelServerMetrics
from inference_perf.payloads import Audio, Audios, Image, Images, RequestMetrics, Text, Video, Videos
from inference_perf.reportgen.br.v0_2 import build_partial_report
from inference_perf.reportgen.br.v0_2.fillability import (
    FILLABILITY,
    Fillability,
    document_leaves,
    fillability,
    schema_leaves,
)

RATCHET_FILE = Path(__file__).parent / "fillability_ratchet.txt"


# The leaf paths listed in fillability_ratchet.txt, comments and blank lines skipped.
def _ratchet() -> Set[str]:
    lines = RATCHET_FILE.read_text().splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


# Every leaf the table marks WRITTEN or RESERVED.
def _owned() -> Set[str]:
    return {path for path, state in FILLABILITY.items() if state in (Fillability.WRITTEN, Fillability.RESERVED)}


# A successful streamed request at start..start+1s with 4 tokens and one image,
# one 2-frame video and one audio clip, a failed one with an HTTP 500, or one
# cancelled at stage teardown.
def _request(start: float, *, failed: bool = False, cancelled: bool = False) -> RequestLifecycleMetric:
    token_times = [start + 0.2, start + 0.4, start + 0.6, start + 0.8]
    return RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=start,
        start_time=start,
        end_time=start + 1.0,
        request_data='{"prompt": "hi"}',
        response_data=None if failed else "ok",
        info=InferenceInfo(
            request_metrics=RequestMetrics(
                text=Text(input_tokens=16),
                image=Images(count=1, instances=[Image(pixels=100, bytes=10, aspect_ratio=1)]),
                video=Videos(count=1, instances=[Video(pixels=200, bytes=20, aspect_ratio=1, frames=2)]),
                audio=Audios(count=1, instances=[Audio(bytes=30, seconds=1)]),
            ),
            response_metrics=None
            if failed or cancelled
            else StreamedResponseMetrics(
                response_chunks=[], chunk_times=token_times, output_tokens=4, output_token_times=token_times
            ),
        ),
        error=ErrorResponseInfo(error_type="HTTP Error 500", error_msg="boom")
        if failed
        else ErrorResponseInfo(error_type=STAGE_TEARDOWN_CANCELLED_ERROR_TYPE, error_msg="cancelled")
        if cancelled
        else None,
    )


# A session at start..start+2s with 3 events, 1 of them cancelled, and token totals set.
def _session(session_id: str, start: float, *, success: bool) -> SessionLifecycleMetric:
    return SessionLifecycleMetric(
        session_id=session_id,
        stage_id=0,
        file_path=f"/traces/{session_id}.json",
        start_time=start,
        end_time=start + 2.0,
        duration_sec=2.0,
        num_events=3,
        num_events_completed=2,
        num_events_cancelled=1,
        success=success,
        total_input_tokens=48,
        total_output_tokens=12,
    )


# Each of the 211 schema leaves has exactly one entry, so a schema bump that adds
# a leaf fails here until someone classifies it.
def test_table_covers_every_schema_leaf() -> None:
    leaves = schema_leaves()
    unclassified = sorted(leaves - set(FILLABILITY))
    stale = sorted(set(FILLABILITY) - leaves)
    assert unclassified == [], f"classify these new BR0.2 leaves in fillability.py: {unclassified}"
    assert stale == [], f"these table entries are not in the installed schema: {stale}"


# A leaf newly marked WRITTEN or RESERVED has to be appended to the ratchet,
# so the next test can hold it there.
def test_every_owned_leaf_is_in_the_ratchet() -> None:
    missing = sorted(_owned() - _ratchet())
    assert missing == [], f"append these to {RATCHET_FILE.name}: {missing}"


# A ratchet leaf the table now marks FILLABLE or DEFAULTED fails: consumers have
# read it as measured by inference-perf, so a base report may never set it.
def test_no_ratchet_leaf_becomes_fillable() -> None:
    released = sorted(_ratchet() - _owned())
    assert released == [], f"these were WRITTEN or RESERVED and must stay so (see {RATCHET_FILE.name}): {released}"


# A vLLM stage summary with every gauge and counter the partial reads: KV cache
# usage, running and waiting requests, preemptions, prefix cache hits and queries.
def _server_summary() -> ModelServerMetrics:
    return ModelServerMetrics(
        kv_cache_usage=GaugeResult(avg=0.42, median=0.4, p90=0.7, p99=0.9),
        num_requests_running=GaugeResult(avg=6.5, median=6.0, p90=9.0, p99=12.0),
        queue_length=GaugeResult(avg=1.5, median=1.0, p90=4.0, p99=7.0),
        num_preemptions_total=CounterResult(total=3.0, avg=0.05, per_second=0.05),
        prefix_cache_hits=CounterResult(total=250.0),
        prefix_cache_queries=CounterResult(total=1000.0),
    )


# A partial from streamed multimodal requests (one failed, one cancelled at
# teardown), sessions with cancellations and a vLLM stage summary sets every
# WRITTEN and DEFAULTED leaf and nothing else.
def test_written_and_defaulted_leaves_are_what_a_full_partial_emits() -> None:
    partial = build_partial_report(
        [_request(100.0), _request(100.5), _request(101.0, failed=True), _request(101.5, cancelled=True)],
        None,
        run_uid="uid",
        run_eid="eid",
        stage_start=1750000000.0,
        stage_end=1750000002.0,
        session_metrics=[_session("s1", 100.0, success=True), _session("s2", 101.0, success=False)],
        server_metrics={"vllm": _server_summary()},
    )
    emitted = set(document_leaves(partial))
    declared = {path for path, state in FILLABILITY.items() if state in (Fillability.WRITTEN, Fillability.DEFAULTED)}
    assert sorted(emitted - declared) == [], "emitted but not marked WRITTEN or DEFAULTED"
    assert sorted(declared - emitted) == [], "marked WRITTEN or DEFAULTED but never emitted"


# A leaf from a newer 0.2.x schema than the table is one inference-perf cannot
# write, so it is FILLABLE. A known leaf keeps its own class.
def test_unknown_leaf_is_fillable() -> None:
    assert fillability("results.a_field_from_0_2_9") == Fillability.FILLABLE
    assert fillability("run.uid") == Fillability.WRITTEN


# Only FILLABLE and DEFAULTED leaves may come from a base report.
@pytest.mark.parametrize(
    ("state", "allowed"),
    [
        (Fillability.FILLABLE, True),
        (Fillability.DEFAULTED, True),
        (Fillability.WRITTEN, False),
        (Fillability.RESERVED, False),
    ],
)
def test_base_may_set(state: Fillability, allowed: bool) -> None:
    assert state.base_may_set is allowed


# scenario.stack.standardized is Generic | InferenceEngine. The walk takes both
# arms: kind is in both, accelerator.count only in InferenceEngine.
def test_schema_leaves_follow_every_union_arm() -> None:
    leaves = schema_leaves()
    assert "scenario.stack.standardized.kind" in leaves
    assert "scenario.stack.standardized.accelerator.count" in leaves


# A Statistics block is one leaf: the walk stops at request_latency and does not
# list request_latency.mean.
def test_schema_leaves_stop_at_statistics() -> None:
    leaves = schema_leaves()
    assert "results.request_performance.aggregate.latency.request_latency" in leaves
    assert "results.request_performance.aggregate.latency.request_latency.mean" not in leaves


# {"run": {"descripton": ...}} has a key Run does not, so it is an error naming run.descripton.
def test_document_leaves_rejects_an_unknown_key() -> None:
    with pytest.raises(ValueError, match=r"run\.descripton"):
        document_leaves({"run": {"descripton": "typo"}})


# results.observability takes extra keys, so an ad-hoc vllm_kv block is a leaf, not an error.
def test_document_leaves_counts_extra_keys_where_the_model_allows_them() -> None:
    leaves = document_leaves({"results": {"observability": {"vllm_kv": {"aggregated": 1}}}})
    assert leaves == ["results.observability.vllm_kv"]


# Two stack components give each leaf path once, with no list index in it.
def test_document_leaves_walks_list_items_without_indices() -> None:
    component = {"metadata": {"label": "a", "cfg_id": "c"}}
    leaves: List[str] = document_leaves({"scenario": {"stack": [component, component]}})
    assert leaves == ["scenario.stack.metadata.label", "scenario.stack.metadata.cfg_id"]


# A Statistics block in a document is one leaf, whatever keys it holds.
def test_document_leaves_stops_at_statistics() -> None:
    latency = {"request_latency": {"units": "s", "mean": 1.0, "p50": 1.0}}
    leaves = document_leaves({"results": {"request_performance": {"aggregate": {"latency": latency}}}})
    assert leaves == ["results.request_performance.aggregate.latency.request_latency"]


# run: "x" is a scalar where a model belongs. It counts as the leaf "run" and
# schema validation is left to reject it.
def test_document_leaves_counts_a_scalar_in_a_model_field_as_a_leaf() -> None:
    assert document_leaves({"run": "x"}) == ["run"]
