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

The tool-call case needs a model that calls tools and a server started with a
tool-call parser, which the default opt-125m server is not. It skips unless
``E2E_VLLM_TOOL_CALLS=1``. Local run:

    VLLM_MODEL=Qwen/Qwen3-0.6B VLLM_CHAT_TEMPLATE= VLLM_MAX_MODEL_LEN=4096 \
      VLLM_EXTRA_ARGS='--enable-auto-tool-choice --tool-call-parser hermes --default-chat-template-kwargs {"enable_thinking":false}' \
      e2e/vllm_cpu_server.sh start v0.33.0
    E2E_VLLM_BASE_URL=http://127.0.0.1:8000 E2E_VLLM_TOOL_CALLS=1 \
      pdm run pytest e2e/tests/test_vllm_cpu_token_ids.py -k tool
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
from utils.vllm_server import VLLMServerRunner, serves_tool_calls

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


# 2 agent sessions, each one tool round then an answer, 4 streamed chat requests
# with a 2-tool catalog and tool_choice left to the model. Every timeline is exactly
# completion_tokens long. On a turn that called a tool the ids outnumber the tokens
# of the visible text, and the first id lands before the first visible tool text,
# because the parser swallows the wrapper token. The summary flags no mismatch.
@pytest.mark.asyncio
@pytest.mark.skipif(not VLLMServerRunner.is_available(), reason="no vLLM server or executable available")
@pytest.mark.skipif(not serves_tool_calls(), reason="server not declared to parse tool calls (E2E_VLLM_TOOL_CALLS=1)")
async def test_token_ids_cover_tokens_the_tool_parser_swallows():
    async with VLLMServerRunner(port=get_free_port()) as server:
        result = await run_benchmark_minimal(
            {
                "data": {
                    "type": "synthetic_agentic",
                    "synthetic_agentic": {
                        "num_sessions": 2,
                        "seed": 1,
                        "theme_mix": {"generic": {"weight": 1.0}},
                        "turns_per_session": {"type": "fixed", "mean": 1},
                        "fanout_probability": 0.0,
                        "tool_loop_depth": {"type": "fixed", "mean": 1},
                        "tool_catalog_size_per_agent": {"type": "fixed", "mean": 2},
                        "input_tokens_per_turn": {"type": "fixed", "mean": 50},
                        "output_tokens_per_turn": {"type": "fixed", "mean": 20},
                        "tool_call_latency_sec": {"type": "fixed", "mean": 0},
                        # the model decides, so the call goes through the tool parser
                        # and not through a schema-constrained tool_choice
                        "tool_choice_mode": "as_recorded",
                    },
                },
                "load": {
                    "type": "trace_session_replay",
                    "stages": [{"concurrent_sessions": 1}],
                    "num_workers": 1,
                    "worker_max_concurrency": 2,
                },
                "api": {"type": "chat", "streaming": True, "return_token_ids": True},
                "server": {"type": "vllm", "model_name": server.model, "base_url": server.base_url},
                "tokenizer": {"pretrained_model_name_or_path": server.model},
                "report": {
                    "request_lifecycle": {
                        "summary": True,
                        "per_stage": True,
                        "per_request": True,
                    },
                },
            },
            timeout_sec=900,
        )

    entries = assert_successful_run(result, 4)

    tool_turns = 0
    for entry in entries:
        metrics = response_metrics(entry)
        assert metrics["token_ids_returned"] is True, "server returned no token ids"
        token_times = metrics["output_token_times"]
        server_tokens = server_completion_tokens(entry)
        assert len(token_times) == server_tokens, f"{len(token_times)} token ids vs completion_tokens {server_tokens}"
        if not metrics["has_tool_calls"]:
            continue
        tool_turns += 1
        assert len(token_times) > metrics["output_tokens"], "no swallowed tokens on a tool turn"
        assert token_times[0] < metrics["tool_chunk_times"][0], "first id did not precede the first visible tool text"

    assert tool_turns > 0, "the model called no tool, the swallowed-token path was not exercised"

    summary = result.reports[SUMMARY_REPORT]["successes"]
    assert summary["requests_with_token_ids"] == 4
    assert summary["token_count_mismatches"] == 0
    assert summary["tool_token_count_mismatches"] == 0
