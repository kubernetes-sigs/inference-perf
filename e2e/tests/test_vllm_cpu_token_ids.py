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
"""Live check of the token timeline built from token ids a real vLLM returns (#917).

With ``api.return_token_ids`` the client asks vLLM for the ids of the tokens each
streamed chunk generated and builds the timeline from them. Two server-side
ledgers must agree with the id count exactly:

- in-band: every request's timeline has as many entries as the server's
  ``usage.completion_tokens``, and the summary flags no mismatch
- out-of-band: the server's cumulative ``vllm:generation_tokens_total`` grows by
  the total number of ids the client received

The /metrics delta assumes no other traffic reaches the server during the run.
The ``xdist_group`` mark pins every vLLM CPU test to one worker for that reason.
"""

import pytest

from utils.accuracy import (
    SUMMARY_REPORT,
    assert_successful_run,
    request_body,
    response_metrics,
    server_completion_tokens,
)
from utils.benchmark import run_benchmark_minimal
from utils.net import get_free_port
from utils.vllm_server import VLLMServerRunner

pytestmark = pytest.mark.xdist_group(name="vllm-cpu-server")

RATE = 2
DURATION = 5
EXPECTED_REQUESTS = RATE * DURATION


# Sum of every sample of one counter family in a /metrics page, with or without
# the OpenMetrics _total suffix. Two samples of 3 and 4 give 7.0.
def _counter_total(metrics_text: str, base_name: str) -> float:
    total = 0.0
    found = False
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        sample = line.split("{")[0].split(" ")[0]
        if sample in (base_name, f"{base_name}_total"):
            total += float(line.rsplit(" ", 1)[1])
            found = True
    assert found, f"counter {base_name} not found in /metrics exposition"
    return total


# 10 streamed requests with return_token_ids on, once per API. Every request sent
# the field and got ids back, each timeline is exactly completion_tokens long, the
# summary counts 10 id requests and 0 mismatches, and the server's generation
# counter grew by the number of ids received.
@pytest.mark.asyncio
@pytest.mark.parametrize("api_type", ["completion", "chat"])
@pytest.mark.skipif(not VLLMServerRunner.is_available(), reason="no vLLM server or executable available")
async def test_token_id_count_matches_completion_tokens(api_type):
    async with VLLMServerRunner(port=get_free_port()) as server:
        before = await server.fetch_metrics()

        result = await run_benchmark_minimal(
            {
                "data": {"type": "mock"},
                "load": {
                    "type": "constant",
                    "stages": [{"rate": RATE, "duration": DURATION}],
                    "num_workers": 2,
                },
                "api": {"type": api_type, "streaming": True, "return_token_ids": True},
                "server": {
                    "type": "vllm",
                    "model_name": server.model,
                    "base_url": server.base_url,
                    "ignore_eos": True,
                },
                "tokenizer": {"pretrained_model_name_or_path": server.model},
                "report": {
                    "request_lifecycle": {
                        "summary": True,
                        "per_stage": True,
                        "per_request": True,
                    },
                },
            },
            timeout_sec=300,
        )

        after = await server.fetch_metrics()

    entries = assert_successful_run(result, EXPECTED_REQUESTS)

    ids_received = 0
    for entry in entries:
        assert request_body(entry)["return_token_ids"] is True
        metrics = response_metrics(entry)
        assert metrics["token_ids_returned"] is True, "server returned no token ids"
        server_tokens = server_completion_tokens(entry)
        assert server_tokens is not None, "server reported no completion_tokens"
        token_times = metrics["output_token_times"]
        assert len(token_times) == server_tokens, f"{len(token_times)} token ids vs completion_tokens {server_tokens}"
        assert token_times == sorted(token_times)
        ids_received += len(token_times)

    summary = result.reports[SUMMARY_REPORT]["successes"]
    assert summary["requests_with_token_ids"] == EXPECTED_REQUESTS
    assert summary["token_count_mismatches"] == 0
    assert summary["tool_token_count_mismatches"] == 0

    # Exact because ignore_eos pins generation and nothing else talked to the
    # server between the two scrapes.
    gen_delta = _counter_total(after, "vllm:generation_tokens") - _counter_total(before, "vllm:generation_tokens")
    assert gen_delta == ids_received, f"server generated {gen_delta} tokens, client received {ids_received} ids"
