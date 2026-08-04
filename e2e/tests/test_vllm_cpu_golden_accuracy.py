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
"""Live golden accuracy against a real vLLM server in CPU mode (#627).

The sim goldens (#631) control ground truth but share none of a real
server's tokenization; this test uses the real server itself as the oracle.
Every request is sent with ``ignore_eos`` and the client-default
``max_tokens``, so the server must generate exactly that many tokens, and:

- server-reported ``usage.completion_tokens`` == the configured budget,
  per request, zero tolerance (proves the run is deterministic in length)
- client-derived ``output_len`` == server ``completion_tokens``, per
  request, zero tolerance (the #564-class check: our re-tokenization of
  the real model's real output must agree with the real server's count)
- for completions, client ``input_tokens`` == server ``prompt_tokens``,
  zero tolerance (both sides prepend special tokens; opt-125m's tokenizer
  adds a BOS, keeping #564-lineage special-token handling in play)

CPU mode changes generation speed, not token accounting, so this runs on a
plain CI runner. See utils.vllm_server for provisioning; without a server
or a ``vllm`` executable the test skips.
"""

import pytest

from utils.accuracy import (
    SUMMARY_REPORT,
    assert_output_token_accounting,
    assert_successful_run,
    chunk_times,
    response_metrics,
    server_completion_tokens,
    ttft,
)
from utils.benchmark import run_benchmark_minimal
from utils.net import get_free_port
from utils.vllm_server import VLLMServerRunner

# The client-level default max_completion_tokens (openai_client.py); mock
# datagen sets no per-request max_tokens, so with ignore_eos every request
# must produce exactly this many output tokens.
EXPECTED_OUTPUT_TOKENS = 30

RATE = 2
DURATION = 5
EXPECTED_REQUESTS = RATE * DURATION


@pytest.mark.asyncio
@pytest.mark.skipif(not VLLMServerRunner.is_available(), reason="no vLLM server or executable available")
@pytest.mark.parametrize(
    ("api_type", "streaming"),
    [
        pytest.param("completion", True, id="completion-stream"),
        pytest.param("completion", False, id="completion-unary"),
        pytest.param("chat", True, id="chat-stream"),
        pytest.param("chat", False, id="chat-unary"),
    ],
)
async def test_golden_accuracy_vllm_cpu(api_type: str, streaming: bool):
    async with VLLMServerRunner(port=get_free_port()) as server:
        result = await run_benchmark_minimal(
            {
                "data": {"type": "mock"},
                "load": {
                    "type": "constant",
                    "stages": [{"rate": RATE, "duration": DURATION}],
                    "num_workers": 2,
                },
                "api": {"type": api_type, "streaming": streaming},
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

    entries = assert_successful_run(result, EXPECTED_REQUESTS)

    for entry in entries:
        # The live-oracle core: client output_len == server completion_tokens
        # == the request's token budget, no tolerance.
        assert_output_token_accounting(entry, expected=EXPECTED_OUTPUT_TOKENS, tolerance=0)

        # Prompt side, against the server's own count. Completion prompts are
        # tokenized by both sides as sequence starts (special tokens
        # included), so the counts must agree exactly. Chat prompts go
        # through the server-side template, whose special-token convention
        # may differ from a raw encode by the BOS, so allow exactly that.
        server_prompt = response_metrics(entry)["server_usage"]["prompt_tokens"]
        client_prompt = entry["info"]["request_metrics"]["text"]["input_tokens"]
        if api_type == "completion":
            assert client_prompt == server_prompt, f"client prompt_len {client_prompt} != server prompt_tokens {server_prompt}"
        else:
            assert abs(client_prompt - server_prompt) <= 1, (
                f"client prompt_len {client_prompt} vs server prompt_tokens {server_prompt} differs by more than a BOS"
            )

        if streaming:
            # Real vLLM streams roughly one token per chunk, but may coalesce
            # under load, so chunk structure is asserted, not chunk count:
            # a streamed response must arrive in more than one chunk and
            # never in more chunks than tokens.
            times = chunk_times(entry)
            assert 2 <= len(times) <= EXPECTED_OUTPUT_TOKENS, f"{len(times)} chunks for {EXPECTED_OUTPUT_TOKENS} tokens"
            assert times == sorted(times), "chunk_times are not monotonically nondecreasing"
            assert ttft(entry) > 0, "nonpositive TTFT"

    # Summary-level exactness follows from per-request exactness.
    summary = result.reports[SUMMARY_REPORT]["successes"]
    assert summary["output_tokens"]["total"] == float(EXPECTED_OUTPUT_TOKENS * EXPECTED_REQUESTS)
    assert all(server_completion_tokens(e) == EXPECTED_OUTPUT_TOKENS for e in entries)
