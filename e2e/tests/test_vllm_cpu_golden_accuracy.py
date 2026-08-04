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
Every request is sent with ``ignore_eos``, so the server must generate
exactly the request's token budget, and:

- server-reported ``usage.completion_tokens`` == the budget, per request,
  zero tolerance (proves the run is deterministic in length)
- client-derived ``output_len`` == server ``completion_tokens``, per
  request, zero tolerance (the #564-class check: our re-tokenization of
  the real model's real output must agree with the real server's count)
- for completions, client ``input_tokens`` == server ``prompt_tokens``,
  zero tolerance (both sides prepend special tokens; opt-125m's tokenizer
  adds a BOS, keeping #564-lineage special-token handling in play)

The token budget reaches the server by two different paths, and the table
covers both:

- mock rows: mock datagen sets no per-request ``max_tokens``, so the
  client-level default (``openai_client.py``) applies; these rows cover
  the completion/chat x stream/unary matrix.
- random rows: a fixed output distribution stamps an explicit
  ``max_tokens`` on every request, proving the configured budget survives
  datagen -> request body -> server, a path the mock rows never exercise.
  The 2-token row sits just above the single-token case that masked the
  pre-#410 accounting flaw, and runs unary only: real vLLM may coalesce a
  tiny response into one chunk, so the >=2-chunk structure invariant is
  only asserted for budgets comfortably above the coalescing scale.
  Random datagen is completion-only today; chat rows with explicit
  budgets become possible once random chat-template support (#693) lands.

CPU mode changes generation speed, not token accounting, so this runs on a
plain CI runner. See utils.vllm_server for provisioning; without a server
or a ``vllm`` executable the tests skip. The server is module-scoped, so
spawned mode pays one model load for the whole table rather than one per
row (external mode only re-runs the health check).
"""

from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict

import pytest

from utils.accuracy import (
    SUMMARY_REPORT,
    assert_output_token_accounting,
    assert_successful_run,
    chunk_times,
    request_body,
    response_metrics,
    server_completion_tokens,
    ttft,
)
from utils.benchmark import run_benchmark_minimal
from utils.net import get_free_port
from utils.vllm_server import VLLMServerRunner

# One worker for all vLLM CPU tests: the token-mismatch module's /metrics
# counter deltas need exclusive access to the shared server.
pytestmark = pytest.mark.xdist_group(name="vllm-cpu-server")

# The client-level default max_completion_tokens (openai_client.py); with
# no per-request max_tokens and ignore_eos, every request must produce
# exactly this many output tokens.
CLIENT_DEFAULT_OUTPUT_TOKENS = 30

RATE = 2
DURATION = 5
EXPECTED_REQUESTS = RATE * DURATION

# Deterministic prompts for the random rows: greedy decoding means fixed
# prompts produce fixed outputs, so a row whose text round-trips exactly
# once round-trips exactly on every run instead of flaking.
BASE_SEED = 42


@dataclass(frozen=True)
class Case:
    api_type: str
    streaming: bool
    data: Dict[str, Any]
    # Ground truth for every request in the row: server completion_tokens,
    # client output_len, and the summary total must all equal it exactly.
    expected_output_tokens: int
    # True when datagen must stamp max_tokens == the budget on the wire
    # (random rows); False when the client default is the budget (mock).
    explicit_budget: bool


def fixed_distribution(value: int) -> Dict[str, Any]:
    # "fixed" emits total_count copies of mean; the headroom over
    # EXPECTED_REQUESTS keeps lazy data indexing in range no matter how
    # workers split the stream.
    return {
        "type": "fixed",
        "min": value,
        "max": value,
        "mean": value,
        "total_count": 4 * EXPECTED_REQUESTS,
    }


def random_data(budget: int) -> Dict[str, Any]:
    """Data config whose every request carries an explicit max_tokens."""
    return {
        "type": "random",
        "input_distribution": fixed_distribution(16),
        "output_distribution": fixed_distribution(budget),
    }


MOCK_DATA: Dict[str, Any] = {"type": "mock"}

CASES = [
    # Client-default budget path across the full API matrix.
    pytest.param(Case("completion", True, MOCK_DATA, CLIENT_DEFAULT_OUTPUT_TOKENS, False), id="completion-stream"),
    pytest.param(Case("completion", False, MOCK_DATA, CLIENT_DEFAULT_OUTPUT_TOKENS, False), id="completion-unary"),
    pytest.param(Case("chat", True, MOCK_DATA, CLIENT_DEFAULT_OUTPUT_TOKENS, False), id="chat-stream"),
    pytest.param(Case("chat", False, MOCK_DATA, CLIENT_DEFAULT_OUTPUT_TOKENS, False), id="chat-unary"),
    # Explicit-budget path: the configured output length must reach the wire.
    pytest.param(Case("completion", True, random_data(16), 16, True), id="completion-stream-budget16"),
    # Smallest multi-token budget: the boundary just above the 1-token case
    # that masked the pre-#410 flaw. Unary only, see the module docstring.
    pytest.param(Case("completion", False, random_data(2), 2, True), id="completion-unary-budget2"),
]


@pytest.fixture(scope="module")
async def vllm_server() -> AsyncIterator[VLLMServerRunner]:
    async with VLLMServerRunner(port=get_free_port()) as server:
        yield server


@pytest.mark.asyncio
@pytest.mark.skipif(not VLLMServerRunner.is_available(), reason="no vLLM server or executable available")
@pytest.mark.parametrize("case", CASES)
async def test_golden_accuracy_vllm_cpu(vllm_server: VLLMServerRunner, case: Case) -> None:
    result = await run_benchmark_minimal(
        {
            "data": case.data,
            "load": {
                "type": "constant",
                "stages": [{"rate": RATE, "duration": DURATION}],
                "num_workers": 2,
                "base_seed": BASE_SEED,
            },
            "api": {"type": case.api_type, "streaming": case.streaming},
            "server": {
                "type": "vllm",
                "model_name": vllm_server.model,
                "base_url": vllm_server.base_url,
                "ignore_eos": True,
            },
            "tokenizer": {"pretrained_model_name_or_path": vllm_server.model},
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
        assert_output_token_accounting(entry, expected=case.expected_output_tokens, tolerance=0)

        if case.explicit_budget:
            # The budget must arrive via the request body, not the client
            # default: datagen -> request -> server is the path under test.
            sent = request_body(entry).get("max_tokens")
            assert sent == case.expected_output_tokens, (
                f"request carries max_tokens {sent}, expected {case.expected_output_tokens}"
            )

        # Prompt side, against the server's own count. Completion prompts are
        # tokenized by both sides as sequence starts (special tokens
        # included), so the counts must agree exactly. Chat prompts go
        # through the server-side template, whose special-token convention
        # may differ from a raw encode by the BOS, so allow exactly that.
        server_prompt = response_metrics(entry)["server_usage"]["prompt_tokens"]
        client_prompt = entry["info"]["request_metrics"]["text"]["input_tokens"]
        if case.api_type == "completion":
            assert client_prompt == server_prompt, f"client prompt_len {client_prompt} != server prompt_tokens {server_prompt}"
        else:
            assert abs(client_prompt - server_prompt) <= 1, (
                f"client prompt_len {client_prompt} vs server prompt_tokens {server_prompt} differs by more than a BOS"
            )

        if case.streaming:
            # Real vLLM streams roughly one token per chunk, but may coalesce
            # under load, so chunk structure is asserted, not chunk count:
            # a streamed response must arrive in more than one chunk and
            # never in more chunks than tokens.
            times = chunk_times(entry)
            assert 2 <= len(times) <= case.expected_output_tokens, (
                f"{len(times)} chunks for {case.expected_output_tokens} tokens"
            )
            assert times == sorted(times), "chunk_times are not monotonically nondecreasing"
            assert ttft(entry) > 0, "nonpositive TTFT"

    # Summary-level exactness follows from per-request exactness.
    summary = result.reports[SUMMARY_REPORT]["successes"]
    assert summary["output_tokens"]["total"] == float(case.expected_output_tokens * EXPECTED_REQUESTS)
    assert all(server_completion_tokens(e) == case.expected_output_tokens for e in entries)
