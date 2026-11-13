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
"""api.type sglang_generate from config to report, in both of SGLang's streaming modes (#986).

llm-d-inference-sim does not serve /generate, so inference-perf runs against
``FakeSGLangServer`` (``tests/required/integration/fake_sglang_server.py``),
which streams the way SGLang's tokenizer_manager does: cumulative chunks by
default, delta chunks under --incremental-streaming-output. Every request gets
the same N one-token chunks, and the server stamps when each request arrived
and when each chunk was sent.

A correct run reports N output tokens per request, matching the server's
completion_tokens, no token count mismatch, and TTFT and TPOT that agree with
the server's stamps. Read without per-request stream state, the cumulative
stream counts N(N+1)/2 tokens; stored as raw chunks, it reports no TTFT, TPOT
or ITL and a mismatch on every request.
"""

import statistics
import sys
from pathlib import Path
from typing import Any, List

import pytest
from transformers import AutoTokenizer

from utils.accuracy import SUMMARY_REPORT, assert_successful_run, client_output_tokens, server_completion_tokens
from utils.benchmark import run_benchmark_minimal
from utils.testdata import extract_tarball

# The fake server lives with the integration tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "required" / "integration"))
from fake_sglang_server import FakeSGLangServer  # noqa: E402

TOKENIZER_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"
CORPUS = "The quick brown fox jumps over the lazy dog while the farmer counts his sheep before dawn."
N_TOKENS = 16
TOKEN_DELAY = 0.02
RATE = 2
DURATION = 3
N_REQUESTS = RATE * DURATION

# How far a reported mean may sit from the server's, in seconds (as in
# test_reasoning_metric_parity.py).
FIRST_TOKEN_TOLERANCE = 0.06
PER_TOKEN_TOLERANCE = 0.008

pytestmark = pytest.mark.xdist_group(name="sglang-generate")


# The first n tokens of `corpus` as n strings of one token each, cut at the
# tokenizer's own offsets. Fails if a piece does not re-encode to one token.
def _one_token_chunks(tokenizer: Any, corpus: str, n: int) -> List[str]:
    offsets = tokenizer(corpus, add_special_tokens=False, return_offsets_mapping=True).offset_mapping
    assert len(offsets) >= n, f"corpus has {len(offsets)} tokens, need {n}"
    starts = [offset[0] for offset in offsets[:n]] + [offsets[n - 1][1]]
    chunks = [corpus[a:b] for a, b in zip(starts[:-1], starts[1:], strict=True)]

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    assert count("".join(chunks)) == n and all(count(chunk) == 1 for chunk in chunks), chunks
    return chunks


# 6 streamed requests of 16 one-token chunks, 20 ms apart, in each mode ->
# 16 output tokens per request (client and server), no mismatch, TTFT, TPOT
# and ITL reported, TTFT and TPOT within tolerance of the server's stamps.
@pytest.mark.asyncio
@pytest.mark.parametrize("incremental", [False, True], ids=["cumulative", "incremental"])
async def test_sglang_generate_stream_from_config_to_report(incremental: bool) -> None:
    tokenizer_path = str(extract_tarball(TOKENIZER_TARBALL))
    tokens = _one_token_chunks(AutoTokenizer.from_pretrained(tokenizer_path), CORPUS, N_TOKENS)

    async with FakeSGLangServer(tokens, incremental=incremental, delay=TOKEN_DELAY) as server:
        result = await run_benchmark_minimal(
            {
                "data": {"type": "mock"},
                "load": {"type": "constant", "stages": [{"rate": RATE, "duration": DURATION}], "num_workers": 1},
                "api": {"type": "sglang_generate", "streaming": True},
                "server": {"type": "sglang", "model_name": "fake-sglang", "base_url": server.base_url, "ignore_eos": True},
                "tokenizer": {"pretrained_model_name_or_path": tokenizer_path},
                "report": {"request_lifecycle": {"summary": True, "per_stage": True, "per_request": True}},
            },
            timeout_sec=120,
        )

    entries = assert_successful_run(result, N_REQUESTS)
    assert [client_output_tokens(entry) for entry in entries] == [N_TOKENS] * N_REQUESTS
    assert [server_completion_tokens(entry) for entry in entries] == [N_TOKENS] * N_REQUESTS

    assert len(server.served) == N_REQUESTS
    assert all(r.body.get("stream") is True and "model" not in r.body for r in server.served)

    assert result.reports is not None
    summary = result.reports[SUMMARY_REPORT]["successes"]
    assert summary["token_count_mismatches"] == 0
    latency = summary["latency"]
    assert latency["inter_token_latency"] is not None

    expected_ttft = statistics.mean(r.send_times[0] - r.arrival_time for r in server.served)
    expected_per_token = statistics.mean((r.send_times[-1] - r.send_times[0]) / (N_TOKENS - 1) for r in server.served)
    assert latency["time_to_first_token"]["mean"] == pytest.approx(expected_ttft, abs=FIRST_TOKEN_TOLERANCE)
    assert latency["time_per_output_token"]["mean"] == pytest.approx(expected_per_token, abs=PER_TOKEN_TOLERANCE)
