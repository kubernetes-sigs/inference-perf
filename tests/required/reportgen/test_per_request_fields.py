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

from typing import Optional, cast

import pytest

from inference_perf.apis import (
    ErrorResponseInfo,
    InferenceInfo,
    RequestLifecycleMetric,
    StreamedResponseMetrics,
    UnaryResponseMetrics,
)
from inference_perf.config import PerRequestFieldsConfig
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.base import build_per_request_lifecycle_entry, summarize_requests
from inference_perf.utils.custom_tokenizer import CustomTokenizer


class _FakeTokenizer:
    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        return len(text.split())


def _metric() -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=0.0,
        start_time=1.0,
        end_time=2.0,
        request_data="raw request",
        response_data="raw response",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=5)),
            response_metrics=StreamedResponseMetrics(
                output_tokens=2,
                response_chunks=['{"choices": [{"text": "hello"}]}'],
                chunk_times=[1.5],
                output_token_times=[1.5],
            ),
        ),
        error=None,
    )


def test_per_request_fields_default_preserves_full_report_shape() -> None:
    entry = build_per_request_lifecycle_entry(_metric(), PerRequestFieldsConfig())

    assert entry["start_time"] == 1.0
    assert entry["end_time"] == 2.0
    assert entry["request"] == "raw request"
    assert entry["response"] == "raw response"
    assert entry["info"]["response_metrics"]["response_chunks"] == ['{"choices": [{"text": "hello"}]}']
    assert entry["error"] is None


def test_per_request_fields_can_omit_top_level_raw_fields() -> None:
    entry = build_per_request_lifecycle_entry(
        _metric(),
        PerRequestFieldsConfig(request=False, response=False, info=False),
    )

    assert "request" not in entry
    assert "response" not in entry
    assert "info" not in entry
    assert entry["start_time"] == 1.0
    assert entry["end_time"] == 2.0
    assert entry["error"] is None


def test_per_request_fields_can_omit_response_chunks_without_mutating_metric() -> None:
    metric = _metric()
    entry = build_per_request_lifecycle_entry(metric, PerRequestFieldsConfig(response_chunks=False))

    response_metrics = entry["info"]["response_metrics"]
    assert "response_chunks" not in response_metrics
    assert response_metrics["chunk_times"] == [1.5]
    assert response_metrics["output_token_times"] == [1.5]
    assert isinstance(metric.info.response_metrics, StreamedResponseMetrics)
    assert metric.info.response_metrics.response_chunks == ['{"choices": [{"text": "hello"}]}']


_CONTENT_CHUNKS = ['{"choices": [{"delta": {"content": "hi"}}]}']
_TOOL_CHUNKS = ['{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}}]}']
_REASONING_CHUNKS = [
    '{"choices": [{"delta": {"reasoning_content": "let"}}]}',
    '{"choices": [{"delta": {"reasoning_content": " me think"}}]}',
]


# Tool arguments at 1.1s, reasoning at 1.2s and 1.4s, then answer text at 1.6s.
def _reasoning_metric() -> RequestLifecycleMetric:
    metric = _metric()
    metric.info.response_metrics = StreamedResponseMetrics(
        output_tokens=6,
        tool_chunks=list(_TOOL_CHUNKS),
        tool_chunk_times=[1.1],
        has_tool_calls=True,
        response_chunks=list(_CONTENT_CHUNKS),
        chunk_times=[1.6],
        reasoning_chunks=list(_REASONING_CHUNKS),
        reasoning_chunk_times=[1.2, 1.4],
        output_token_times=[1.1, 1.1, 1.2, 1.4, 1.4, 1.6],
    )
    return metric


# response_chunks=False on that stream writes no chunk list from any channel.
# All channels' arrival times stay, and the in-memory metric keeps its chunks.
def test_per_request_fields_omit_response_chunks_drops_all_channels() -> None:
    metric = _reasoning_metric()
    entry = build_per_request_lifecycle_entry(metric, PerRequestFieldsConfig(response_chunks=False))

    response_metrics = entry["info"]["response_metrics"]
    assert [key for key in response_metrics if key.endswith("_chunks")] == []
    assert response_metrics["chunk_times"] == [1.6]
    assert response_metrics["reasoning_chunk_times"] == [1.2, 1.4]
    assert response_metrics["tool_chunk_times"] == [1.1]
    assert isinstance(metric.info.response_metrics, StreamedResponseMetrics)
    assert metric.info.response_metrics.response_chunks == _CONTENT_CHUNKS
    assert metric.info.response_metrics.reasoning_chunks == _REASONING_CHUNKS
    assert metric.info.response_metrics.tool_chunks == _TOOL_CHUNKS


# response_chunks=True on the same stream writes all chunk lists exactly as received.
def test_per_request_fields_include_response_chunks_keeps_all_channels() -> None:
    entry = build_per_request_lifecycle_entry(_reasoning_metric(), PerRequestFieldsConfig(response_chunks=True))

    response_metrics = entry["info"]["response_metrics"]
    assert response_metrics["response_chunks"] == _CONTENT_CHUNKS
    assert response_metrics["reasoning_chunks"] == _REASONING_CHUNKS
    assert response_metrics["tool_chunks"] == _TOOL_CHUNKS


def test_per_request_fields_computed_metrics_disabled_by_default() -> None:
    entry = build_per_request_lifecycle_entry(_metric(), PerRequestFieldsConfig())

    assert "computed_metrics" not in entry


# One streamed chunk at 1.5 on a request from 1.0 to 2.0, with computed_metrics on.
# The entry has latency 1.0, NTPOT 0.5 (1.0 over 2 tokens) and TTFT 0.5. TPOT and ITL need two
# timestamps. There is no server usage, so output_tokens falls back to the client's 2, which is
# also output_len and the count TPOT/NTPOT divide by.
def test_per_request_fields_computed_metrics_can_be_enabled() -> None:
    entry = build_per_request_lifecycle_entry(_metric(), PerRequestFieldsConfig(computed_metrics=True))

    assert entry["computed_metrics"] == {
        "request_latency": 1.0,
        "normalized_time_per_output_token": 0.5,
        "time_to_first_token": 0.5,
        "time_to_first_output_token": 0.5,
        "time_per_output_token": None,
        "inter_token_latency": None,
        "inter_token_latencies": [],
        "input_tokens": 5,
        "output_tokens": 2,
        "output_len": 2,
        "tpot_output_tokens": 2,
        "ttft_slo_sec": None,
        "tpot_slo_sec": None,
    }


# A streamed request whose client re-tokenization counted client_tokens output tokens, carrying
# server_usage as the raw usage dict the server sent (None when it sent none).
def _metric_with_usage(client_tokens: int, server_usage: Optional[dict[str, object]]) -> RequestLifecycleMetric:
    metric = _metric()
    metric.info.response_metrics = StreamedResponseMetrics(
        output_tokens=client_tokens,
        server_usage=server_usage,
        output_token_times=[1.2] + [1.5] * (client_tokens - 1),
    )
    return metric


# Server says 12 output tokens, the client re-tokenized 10. computed_metrics.output_tokens must be
# the server's 12, the number the summary aggregates carry, while output_len keeps the client's 10
# and tpot_output_tokens shows TPOT divided by 10 because use_server_output_tokens is off (#1105).
def test_per_request_computed_metrics_output_tokens_is_the_server_count() -> None:
    entry = build_per_request_lifecycle_entry(
        _metric_with_usage(10, {"prompt_tokens": 5, "completion_tokens": 12}),
        PerRequestFieldsConfig(computed_metrics=True),
    )

    computed = entry["computed_metrics"]
    assert computed["output_tokens"] == 12
    assert computed["output_len"] == 10
    assert computed["tpot_output_tokens"] == 10


# The same request with use_server_output_tokens on: TPOT and NTPOT divide by the server's 12, so
# tpot_output_tokens reports 12, while output_tokens and output_len keep their meanings.
def test_per_request_computed_metrics_tpot_output_tokens_follows_the_server_flag() -> None:
    entry = build_per_request_lifecycle_entry(
        _metric_with_usage(10, {"completion_tokens": 12}),
        PerRequestFieldsConfig(computed_metrics=True),
        use_server_output_tokens=True,
    )

    computed = entry["computed_metrics"]
    assert computed["output_tokens"] == 12
    assert computed["output_len"] == 10
    assert computed["tpot_output_tokens"] == 12


# A server that reported no usage leaves no server count to prefer, so output_tokens falls back to
# the client's 10 under either flag setting, matching the summary's fallback, and TPOT divides by 10.
def test_per_request_computed_metrics_falls_back_to_the_client_count() -> None:
    for use_server in (False, True):
        entry = build_per_request_lifecycle_entry(
            _metric_with_usage(10, None),
            PerRequestFieldsConfig(computed_metrics=True),
            use_server_output_tokens=use_server,
        )

        computed = entry["computed_metrics"]
        assert computed["output_tokens"] == 10
        assert computed["output_len"] == 10
        assert computed["tpot_output_tokens"] == 10


# Usage that carries only prompt tokens has not counted the output, so the entry keeps the client's
# 10 instead of reading the absent output key as 0.
def test_per_request_computed_metrics_falls_back_when_usage_has_no_output_count() -> None:
    entry = build_per_request_lifecycle_entry(
        _metric_with_usage(10, {"prompt_tokens": 5}),
        PerRequestFieldsConfig(computed_metrics=True),
    )

    assert entry["computed_metrics"]["output_tokens"] == 10
    assert entry["computed_metrics"]["output_len"] == 10


# The Anthropic Messages API reports output_tokens inside usage; that spelling is the server count
# and wins over the client's 10 exactly like completion_tokens does.
def test_per_request_computed_metrics_reads_the_anthropic_output_spelling() -> None:
    entry = build_per_request_lifecycle_entry(
        _metric_with_usage(10, {"input_tokens": 5, "output_tokens": 99}),
        PerRequestFieldsConfig(computed_metrics=True),
    )

    assert entry["computed_metrics"]["output_tokens"] == 99
    assert entry["computed_metrics"]["output_len"] == 10


# A server-reported 0 is a count, not a missing count: output_tokens stays 0 and, with the flag on,
# TPOT and NTPOT divide by 0 so both report null rather than borrowing the client's 7.
def test_per_request_computed_metrics_keeps_a_server_reported_zero() -> None:
    entry = build_per_request_lifecycle_entry(
        _metric_with_usage(7, {"completion_tokens": 0}),
        PerRequestFieldsConfig(computed_metrics=True),
        use_server_output_tokens=True,
    )

    computed = entry["computed_metrics"]
    assert computed["output_tokens"] == 0
    assert computed["output_len"] == 7
    assert computed["tpot_output_tokens"] == 0
    assert computed["normalized_time_per_output_token"] is None
    assert computed["time_per_output_token"] is None


# docs/config.md says the per-request values are the ones the summary aggregates: for a single
# request, the summary's output_tokens mean equals computed_metrics.output_tokens, flag on or off.
def test_per_request_output_tokens_matches_the_summary_aggregate() -> None:
    for use_server in (False, True):
        metric = _metric_with_usage(10, {"completion_tokens": 12})
        entry = build_per_request_lifecycle_entry(
            metric,
            PerRequestFieldsConfig(computed_metrics=True),
            use_server_output_tokens=use_server,
        )

        summary = summarize_requests([metric], [50])
        assert entry["computed_metrics"]["output_tokens"] == summary.successes["output_tokens"]["mean"]


# A non-streamed response carries the same three counts; TTFT and TPOT stay null because there are
# no token arrival times to measure them from.
def test_per_request_computed_metrics_reports_counts_for_unary_responses() -> None:
    metric = _metric()
    metric.info.response_metrics = UnaryResponseMetrics(output_tokens=10, server_usage={"completion_tokens": 12})

    entry = build_per_request_lifecycle_entry(metric, PerRequestFieldsConfig(computed_metrics=True))

    computed = entry["computed_metrics"]
    assert computed["output_tokens"] == 12
    assert computed["output_len"] == 10
    assert computed["tpot_output_tokens"] == 10
    assert computed["time_to_first_token"] is None


# A successful request with no response metrics has counted nothing on either side, so all three
# count fields are null rather than 0.
def test_per_request_computed_metrics_reports_null_counts_without_response_metrics() -> None:
    metric = _metric()
    metric.info.response_metrics = None

    entry = build_per_request_lifecycle_entry(metric, PerRequestFieldsConfig(computed_metrics=True))

    computed = entry["computed_metrics"]
    assert computed["output_tokens"] is None
    assert computed["output_len"] is None
    assert computed["tpot_output_tokens"] is None


def test_per_request_fields_disabled_computed_metrics_does_not_mutate_token_metrics() -> None:
    metric = _metric()

    build_per_request_lifecycle_entry(
        metric, PerRequestFieldsConfig(computed_metrics=False), tokenizer=cast(CustomTokenizer, _FakeTokenizer())
    )

    assert isinstance(metric.info.response_metrics, StreamedResponseMetrics)
    assert metric.info.response_metrics.output_tokens == 2
    assert metric.info.response_metrics.output_token_times == [1.5]


def test_per_request_fields_computed_metrics_omitted_for_failed_requests() -> None:
    metric = _metric()
    metric.error = ErrorResponseInfo(error_type="TimeoutError", error_msg="boom")

    entry = build_per_request_lifecycle_entry(metric, PerRequestFieldsConfig(computed_metrics=True))

    assert "computed_metrics" not in entry


def test_per_request_fields_computed_metrics_uses_tokenizer_corrected_chunks() -> None:
    metric = RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=0.0,
        start_time=1.0,
        end_time=2.0,
        request_data="raw request",
        response_data="raw response",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=5)),
            response_metrics=StreamedResponseMetrics(
                output_tokens=4,
                response_chunks=[
                    '{"choices": [{"text": "hello"}]}',
                    '{"choices": [{"text": " world foo"}]}',
                ],
                chunk_times=[1.2, 1.6],
                output_token_times=[],
            ),
        ),
        error=None,
    )

    entry = build_per_request_lifecycle_entry(
        metric, PerRequestFieldsConfig(computed_metrics=True), tokenizer=cast(CustomTokenizer, _FakeTokenizer())
    )

    assert entry["info"]["response_metrics"]["output_token_times"] == pytest.approx([1.2, 1.6, 1.6])
    assert entry["computed_metrics"]["time_to_first_token"] == pytest.approx(0.2)
    assert entry["computed_metrics"]["time_per_output_token"] == pytest.approx(0.1333333333333334)
    assert entry["computed_metrics"]["inter_token_latency"] == pytest.approx(0.2)
    assert entry["computed_metrics"]["inter_token_latencies"] == pytest.approx([0.4, 0.0])
    assert entry["computed_metrics"]["input_tokens"] == 5
    # The correction sets output_token_times only; output_tokens keeps the API layer's
    # whole-message count (see #564).
    assert entry["computed_metrics"]["output_tokens"] == 4


def test_per_request_fields_include_session_ids_when_set() -> None:
    metric = _metric()
    metric.session_id = "wekatrace5_05658118df5d398eec4703ce4fafe679ab30"
    metric.request_session_id = "wekatrace5_05658118df5d398eec4703ce4fafe679ab30::sa:agent-1:s0"

    entry = build_per_request_lifecycle_entry(metric, PerRequestFieldsConfig(request=False, response=False, info=False))

    assert entry["session_id"] == "wekatrace5_05658118df5d398eec4703ce4fafe679ab30"
    assert entry["request_session_id"] == "wekatrace5_05658118df5d398eec4703ce4fafe679ab30::sa:agent-1:s0"


def test_per_request_fields_keep_parent_session_id_without_request_session_id() -> None:
    metric = _metric()
    metric.session_id = "wekatrace5_05658118df5d398eec4703ce4fafe679ab30"

    entry = build_per_request_lifecycle_entry(metric, PerRequestFieldsConfig())

    assert entry["session_id"] == "wekatrace5_05658118df5d398eec4703ce4fafe679ab30"
    assert "request_session_id" not in entry


def test_per_request_fields_omit_session_ids_when_absent() -> None:
    entry = build_per_request_lifecycle_entry(_metric(), PerRequestFieldsConfig())

    assert "session_id" not in entry
    assert "request_session_id" not in entry


@pytest.mark.parametrize("include_raw", [True, False])
@pytest.mark.parametrize("failed", [True, False])
def test_per_request_entries_preserve_stage_and_schedule(include_raw: bool, failed: bool) -> None:
    fields = PerRequestFieldsConfig(request=include_raw, response=include_raw, info=include_raw)
    entries = []
    for stage_id, scheduled_time in [(0, 0.0), (2, 0.125)]:
        metric = _metric()
        metric.stage_id = stage_id
        metric.scheduled_time = scheduled_time
        if failed:
            metric.error = ErrorResponseInfo(error_type="test", error_msg="request failed")
        entries.append(build_per_request_lifecycle_entry(metric, fields))

    # Identical send times cannot identify the stage or recover its scheduled time.
    assert [entry["start_time"] for entry in entries] == [1.0, 1.0]
    assert [entry["stage_id"] for entry in entries] == [0, 2]
    assert [entry["scheduled_time"] for entry in entries] == [0.0, 0.125]


def test_per_request_fields_omit_stage_id_when_absent() -> None:
    metric = _metric()
    metric.stage_id = None

    entry = build_per_request_lifecycle_entry(metric, PerRequestFieldsConfig())

    assert "stage_id" not in entry
    assert entry["scheduled_time"] == metric.scheduled_time
