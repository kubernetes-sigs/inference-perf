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
"""Sim-backed golden accuracy e2e (#631).

Ground truth is fully controlled: the golden sim constructs each response
from a real tokenizer's token ids (known count N, known chunk count K,
controlled chunk pacing), so token accounting is asserted EXACTLY, zero
tolerance:

- client-derived output_len == server-reported completion_tokens == N
- content chunk count == K; reportgen's expanded output_token_times has one
  entry per token with exactly K-1 nonzero inter-token gaps

The matrix includes a per-chunk BOS-prepending tokenizer (the #564 failure
mode: Llama-3.1, plus the vendored Gemma-3 so the merge-blocking path needs
no network) and a mid-token chunk split, so per-chunk re-tokenization bugs
cannot hide behind polite chunk boundaries.

Timing: sample counts are exact; gap VALUES are asserted against the sim's
recorded actual send gaps (not the configured interval), because wall-clock
zero tolerance over a real socket does not exist. A systematic ITL
distortion (#566 was ~K-fold deflation) fails these bounds; scheduler jitter
does not.

Expected values are computed with a raw transformers AutoTokenizer, NOT
inference_perf's CustomTokenizer, so a counting bug in the product cannot
cancel itself out in the test.
"""

import statistics
from collections import Counter
from typing import List

import pytest

from utils.accuracy import (
    SUMMARY_REPORT,
    assert_output_token_accounting,
    assert_streaming_bookkeeping,
    assert_successful_run,
    chunk_gaps,
    request_body,
    server_completion_tokens,
    ttft,
)
from utils.benchmark import run_benchmark_minimal
from utils.golden_sim import GoldenCase, GoldenSimServer
from utils.net import get_free_port
from utils.testdata import extract_tarball

GEMMA_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"
LLAMA_31 = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"  # the #564 tokenizer (public mirror)

TTFT_DELAY = 0.15
INTERVAL = 0.08
RATE = 2
DURATION = 5
EXPECTED_REQUESTS = RATE * DURATION

# n_tokens doubles as the case id (reported as usage.completion_tokens), so it
# must be unique. n_chunks == n_tokens is the one-token-per-chunk regime where
# #564's per-chunk BOS inflation was worst (~2x); the codepoint split places
# chunk boundaries mid-token so tokenization is deliberately non-compositional
# across chunks.
CASES = [
    GoldenCase(n_tokens=24, n_chunks=1, split="token"),
    GoldenCase(n_tokens=36, n_chunks=6, split="token"),
    GoldenCase(n_tokens=48, n_chunks=12, split="codepoint"),
    GoldenCase(n_tokens=16, n_chunks=16, split="token"),
]


def _load_oracle(pretrained: str):
    """Independent ground-truth tokenizer (raw transformers, not CustomTokenizer)."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(pretrained)
    except Exception as e:  # gated/offline hub tokenizers skip instead of erroring
        pytest.skip(f"tokenizer {pretrained} unavailable: {e}")


def _resolve_tokenizer(key: str) -> str:
    if key == "gemma-3-270m":
        return str(extract_tarball(GEMMA_TARBALL))
    if key == "llama-3.1":
        return LLAMA_31
    raise ValueError(key)


def _prompt_text(body: dict) -> str:
    if "prompt" in body:
        return body["prompt"]
    return "".join(m.get("content", "") for m in body.get("messages", []) if isinstance(m.get("content"), str))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tokenizer_key", "api_type", "streaming", "use_server_output_tokens"),
    [
        pytest.param("gemma-3-270m", "completion", True, False, id="gemma-completion-stream"),
        pytest.param("gemma-3-270m", "chat", True, False, id="gemma-chat-stream"),
        pytest.param("gemma-3-270m", "completion", False, False, id="gemma-completion-unary"),
        pytest.param("gemma-3-270m", "completion", True, True, id="gemma-completion-stream-serveroracle"),
        pytest.param("llama-3.1", "completion", True, False, id="llama31-completion-stream"),
        pytest.param("llama-3.1", "chat", True, False, id="llama31-chat-stream"),
    ],
)
async def test_golden_accuracy(tokenizer_key: str, api_type: str, streaming: bool, use_server_output_tokens: bool):
    tokenizer_path = _resolve_tokenizer(tokenizer_key)
    oracle = _load_oracle(tokenizer_path)

    port = get_free_port()
    async with GoldenSimServer(
        oracle,
        CASES,
        model="golden-model",
        ttft_delay=TTFT_DELAY,
        inter_chunk_interval=INTERVAL,
        port=port,
    ) as sim:
        # Fixture cross-check with the oracle: the served text of each case
        # must re-encode to exactly n_tokens, and the per-chunk token sums
        # (what reportgen's timestamp expansion sees) are precomputed here.
        case_by_n = {c.n_tokens: c for c in CASES}
        chunk_token_sums = {}
        for case in CASES:
            chunks = sim.get_chunks(case.n_tokens)
            assert len(chunks) == case.n_chunks
            n = len(oracle("".join(chunks), add_special_tokens=False).input_ids)
            assert n == case.n_tokens, f"fixture broken: case {case} text re-encodes to {n}"
            chunk_token_sums[case.n_tokens] = sum(len(oracle(c, add_special_tokens=False).input_ids) for c in chunks)

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
                    "model_name": "golden-model",
                    "base_url": f"http://{sim.host}:{sim.port}",
                    "ignore_eos": True,
                },
                "tokenizer": {"pretrained_model_name_or_path": tokenizer_path},
                "report": {
                    "request_lifecycle": {
                        "summary": True,
                        "per_stage": True,
                        "per_request": True,
                        "use_server_output_tokens": use_server_output_tokens,
                    },
                },
            },
            timeout_sec=180,
        )

    entries = assert_successful_run(result, EXPECTED_REQUESTS)
    assert len(sim.requests) == EXPECTED_REQUESTS, (
        f"sim served {len(sim.requests)} requests, expected {EXPECTED_REQUESTS} (client retried or dropped?)"
    )

    # --- Exact token accounting, per request, zero tolerance ---
    observed_ns = []
    client_gap_samples: List[float] = []
    for entry in entries:
        n_server = server_completion_tokens(entry)
        assert n_server in case_by_n, f"server reported unknown case n={n_server}"
        case = case_by_n[n_server]
        observed_ns.append(n_server)

        # output_len == completion_tokens == N, no tolerance. This holds for
        # ALL splits, including mid-token: the client counts the concatenated
        # text once, so chunking must not affect it.
        assert_output_token_accounting(entry, expected=case.n_tokens, tolerance=0)

        # Prompt side: recount the exact prompt the client sent. Prompts start
        # a sequence, so special tokens are included (matches server
        # prompt_tokens semantics).
        expected_prompt = len(oracle(_prompt_text(request_body(entry)), add_special_tokens=True).input_ids)
        got_prompt = entry["info"]["request_metrics"]["text"]["input_tokens"]
        assert got_prompt == expected_prompt, f"prompt_len {got_prompt} != oracle count {expected_prompt}"

        if streaming:
            # Chunk/token bookkeeping is exact. output_token_times is chunk
            # arrival times expanded via per-chunk counts, so its expected
            # length is the per-chunk sum: equal to N for token-aligned splits,
            # and whatever the oracle says for mid-token splits.
            assert_streaming_bookkeeping(
                entry,
                expected_chunks=case.n_chunks,
                expected_token_times=chunk_token_sums[case.n_tokens],
            )
            # Timing: TTFT cannot be smaller than the sim's enforced delay, and
            # no client-observed gap may fall below half the configured
            # interval (#566-class deflation is a >= 2x effect).
            assert ttft(entry) >= TTFT_DELAY, f"TTFT {ttft(entry):.4f}s below configured delay {TTFT_DELAY}s"
            gaps = chunk_gaps(entry)
            assert len(gaps) == case.n_chunks - 1
            for g in gaps:
                assert g >= INTERVAL * 0.5, f"chunk gap {g * 1000:.1f}ms below half the configured {INTERVAL * 1000:.0f}ms"
            client_gap_samples.extend(gaps)

    # Case cycling is by arrival order, so exactly this multiset was served.
    expected_ns = [CASES[i % len(CASES)].n_tokens for i in range(EXPECTED_REQUESTS)]
    assert Counter(observed_ns) == Counter(expected_ns), (
        f"served case multiset {Counter(observed_ns)} != expected {Counter(expected_ns)}"
    )

    # --- Timing against what the sim actually did, not what was configured ---
    if streaming:
        sim_gap_samples = [
            t2 - t1 for served in sim.requests for t1, t2 in zip(served.send_times, served.send_times[1:], strict=False)
        ]
        assert len(sim_gap_samples) == len(client_gap_samples)
        client_mean = statistics.mean(client_gap_samples)
        sim_mean = statistics.mean(sim_gap_samples)
        assert abs(client_mean - sim_mean) <= 0.02, (
            f"client mean chunk gap {client_mean * 1000:.1f}ms deviates from sim's actual "
            f"mean send gap {sim_mean * 1000:.1f}ms by more than 20ms"
        )

    # --- Summary-level exactness ---
    summary = result.reports[SUMMARY_REPORT]["successes"]
    assert summary["output_tokens"]["total"] == float(sum(expected_ns))
    assert summary["output_len"]["min"] == float(min(expected_ns))
    assert summary["output_len"]["max"] == float(max(expected_ns))
    if streaming:
        # reportgen flags requests whose per-chunk token sums disagree with the
        # server count; only mid-token splits may legitimately do that.
        expected_mismatches = sum(1 for n in observed_ns if chunk_token_sums[n] != n)
        assert summary["token_count_mismatches"] == expected_mismatches


# --- Helper self-tests: prove the assertions can actually fail. -------------
# Guards the helpers against refactors that would make them vacuous (the
# meta-lesson of #564: green tests that could not have gone red).


def _fake_entry(client_n: int, server_n: int, times: List[float], token_times: List[float]) -> dict:
    return {
        "start_time": 0.0,
        "end_time": times[-1] if times else 1.0,
        "request": '{"prompt": "x"}',
        "error": None,
        "info": {
            "request_metrics": {"text": {"input_tokens": 1}},
            "response_metrics": {
                "output_tokens": client_n,
                "server_usage": {"completion_tokens": server_n},
                "chunk_times": times,
                "output_token_times": token_times,
            },
        },
    }


def test_token_accounting_helper_rejects_mismatch():
    entry = _fake_entry(client_n=25, server_n=24, times=[0.1], token_times=[0.1])
    with pytest.raises(AssertionError, match="exceeds tolerance"):
        assert_output_token_accounting(entry, expected=24, tolerance=0)
    # ...but a one-token tolerance (the #630 weak form) accepts it.
    assert_output_token_accounting(entry, tolerance=1)


def test_streaming_bookkeeping_helper_rejects_wrong_counts():
    entry = _fake_entry(client_n=4, server_n=4, times=[0.1, 0.2], token_times=[0.1, 0.1, 0.2, 0.2])
    assert_streaming_bookkeeping(entry, expected_chunks=2, expected_token_times=4)
    with pytest.raises(AssertionError, match="content chunks"):
        assert_streaming_bookkeeping(entry, expected_chunks=3, expected_token_times=4)
    with pytest.raises(AssertionError, match="output_token_times"):
        assert_streaming_bookkeeping(entry, expected_chunks=2, expected_token_times=5)
    # Inflated timestamp expansion (per-chunk BOS, the #564/#566 signature)
    # shows up as extra nonzero gaps.
    entry = _fake_entry(client_n=4, server_n=4, times=[0.1, 0.2], token_times=[0.1, 0.15, 0.2, 0.25])
    with pytest.raises(AssertionError, match="nonzero inter-token gaps"):
        assert_streaming_bookkeeping(entry, expected_chunks=2, expected_token_times=4)
