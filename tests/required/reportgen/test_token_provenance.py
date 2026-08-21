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

from typing import Any, Optional

from inference_perf.apis.base import InferenceInfo, RequestLifecycleMetric, StreamedResponseMetrics
from inference_perf.config.reportgen.config import RequestLifecycleMetricsReportConfig
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.base import (
    SERVER_OUTPUT_TOKEN_KEYS,
    SERVER_PROMPT_TOKEN_KEYS,
    count_client_fallbacks,
    summarize_output_token_usage,
    summarize_prompt_token_usage,
    summarize_requests,
)
from inference_perf.utils.cli_summary import token_source_caption

DEFAULT_PERCENTILES = RequestLifecycleMetricsReportConfig().percentiles


# One finished request. input_tokens is what the report already resolved for the prompt side,
# client_output_tokens is the client's re-tokenization of the response, and server_usage is the
# usage dict the server returned (None when it returned none).
def _request(input_tokens: int, client_output_tokens: int, server_usage: Optional[dict[str, Any]]) -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        scheduled_time=0.0,
        start_time=0.0,
        end_time=1.0,
        request_data="r",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=input_tokens)),
            response_metrics=StreamedResponseMetrics(output_tokens=client_output_tokens, server_usage=server_usage),
        ),
        error=None,
    )


# Two requests, both with full OpenAI-style usage. Every count is server-sourced, so neither
# side counts a fallback and output_tokens totals 100 + 200 = 300.
def test_no_fallback_when_every_request_has_server_usage() -> None:
    requests = [
        _request(10, 99, {"prompt_tokens": 10, "completion_tokens": 100}),
        _request(20, 201, {"prompt_tokens": 20, "completion_tokens": 200}),
    ]

    assert count_client_fallbacks(requests, SERVER_PROMPT_TOKEN_KEYS) == 0
    assert count_client_fallbacks(requests, SERVER_OUTPUT_TOKEN_KEYS) == 0
    assert summarize_output_token_usage(requests, DEFAULT_PERCENTILES)["total"] == 300.0


# Three requests, one of which came back with no usage at all. That request contributes its
# client count (7) to output_tokens, and each side counts exactly one fallback: totals are
# 10 + 20 + 30 = 60 prompt and 100 + 200 + 7 = 307 output.
def test_missing_usage_counts_one_fallback_on_both_sides() -> None:
    requests = [
        _request(10, 99, {"prompt_tokens": 10, "completion_tokens": 100}),
        _request(20, 201, {"prompt_tokens": 20, "completion_tokens": 200}),
        _request(30, 7, None),
    ]

    assert count_client_fallbacks(requests, SERVER_PROMPT_TOKEN_KEYS) == 1
    assert count_client_fallbacks(requests, SERVER_OUTPUT_TOKEN_KEYS) == 1
    assert summarize_prompt_token_usage(requests, DEFAULT_PERCENTILES)["total"] == 60.0
    assert summarize_output_token_usage(requests, DEFAULT_PERCENTILES)["total"] == 307.0


# One request whose usage uses the Anthropic Messages key names (input_tokens/output_tokens)
# instead of the OpenAI ones. Those counts are server-reported, so output_tokens must total the
# server's 500 with no fallback, not the client's 480 and not 0.
def test_anthropic_usage_keys_count_as_server_reported() -> None:
    requests = [_request(40, 480, {"input_tokens": 40, "output_tokens": 500})]

    assert count_client_fallbacks(requests, SERVER_PROMPT_TOKEN_KEYS) == 0
    assert count_client_fallbacks(requests, SERVER_OUTPUT_TOKEN_KEYS) == 0
    assert summarize_output_token_usage(requests, DEFAULT_PERCENTILES)["total"] == 500.0


# A partial usage dict: the server reported the prompt side but no output count. The output side
# falls back to the client's 480 and is counted as a fallback; the prompt side is not.
def test_partial_usage_falls_back_only_on_the_missing_side() -> None:
    requests = [_request(40, 480, {"prompt_tokens": 40})]

    assert count_client_fallbacks(requests, SERVER_PROMPT_TOKEN_KEYS) == 0
    assert count_client_fallbacks(requests, SERVER_OUTPUT_TOKEN_KEYS) == 1
    assert summarize_output_token_usage(requests, DEFAULT_PERCENTILES)["total"] == 480.0


# Two requests through the full report path, one with usage and one without. The report carries
# the counts as successes.client_fallback_counts = {prompt_tokens: 1, output_tokens: 1}, and the
# token summaries stay a total plus a pure distribution, with no count mixed in.
def test_report_carries_fallback_counts_outside_the_distributions() -> None:
    requests = [
        _request(10, 100, {"prompt_tokens": 10, "completion_tokens": 100}),
        _request(10, 100, None),
    ]

    successes = summarize_requests(requests, [50]).successes

    assert successes["client_fallback_counts"] == {"prompt_tokens": 1, "output_tokens": 1}
    assert set(successes["output_tokens"]) == {"total", "mean", "min", "max", "median"}
    assert set(successes["output_tokens"].values()) - {200.0} == {100.0}


# One stage report with 2 prompt-side and 3 output-side fallbacks. The CLI caption names the
# source of each column and reports the 5 fallbacks in one line.
def test_caption_reports_sources_and_fallback_total() -> None:
    stage = {"successes": {"client_fallback_counts": {"prompt_tokens": 2, "output_tokens": 3}}}

    caption = token_source_caption([stage], has_server_output=True)

    assert "Prompt: server-reported usage where available" in caption
    assert "Out (client)" in caption
    assert "Out (server)" in caption
    assert "5 counts fell back to client tokenization" in caption


# A run against a server that reported usage for everything: no fallback sentence at all, and no
# server column described, since the caller found no server-sourced distribution to show.
def test_caption_omits_fallback_line_when_every_count_is_server_sourced() -> None:
    stage = {"successes": {"client_fallback_counts": {"prompt_tokens": 0, "output_tokens": 0}}}

    caption = token_source_caption([stage], has_server_output=False)

    assert "fell back" not in caption
    assert "Out (server)" not in caption
