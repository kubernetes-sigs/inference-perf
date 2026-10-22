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
"""Mooncake trace replay against llm-d-inference-sim.

Pins what a replayed Mooncake trace has to reproduce: one request per
recorded line, each prompt tokenizing to the recorded input_length, each
response the recorded output_length, requests leaving at the recorded
offsets, and the prefix structure the hash ids describe: requests that share
their leading hash ids are sent with the same leading text, requests that
share none are not.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from inference_perf.config import CustomTokenizerConfig
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from utils.accuracy import assert_successful_run, request_body, server_completion_tokens
from utils.benchmark import run_benchmark_minimal
from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.net import get_free_port
from utils.testdata import extract_tarball

TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"

BLOCK_SIZE = 16

# Ten requests over 3.2 seconds in three bursts. Requests 0, 1 and 4 share
# their first 8 hash ids (128 tokens), requests 2 and 3 share their first 3
# (48 tokens), request 5 has no hash ids at all, and the rest share nothing
# with anyone. Every hash_ids list fits ceil(input_length / 16) blocks.
OFFSETS_MS = [0, 200, 400, 1500, 1600, 1700, 1800, 3000, 3100, 3200]
INPUT_TOKENS = [160, 144, 64, 80, 200, 40, 300, 96, 256, 120]
OUTPUT_TOKENS = [8, 16, 12, 24, 10, 40, 20, 14, 32, 18]
HASH_IDS: List[List[int]] = [
    list(range(0, 8)) + [100, 101],
    list(range(0, 8)) + [102],
    [200, 201, 202, 203],
    [200, 201, 202, 300, 301],
    list(range(0, 8)) + list(range(400, 405)),
    [],
    list(range(500, 519)),
    list(range(600, 606)),
    list(range(700, 716)),
    list(range(800, 808)),
]
# (request, request, shared leading blocks)
SHARED = [(0, 1, 8), (0, 4, 8), (1, 4, 8), (2, 3, 3)]
# Pairs that share no leading block at all.
DISJOINT = [(5, 0), (5, 2), (6, 7), (0, 2)]

# How far the client's send times may drift from the recorded offsets.
# The stage runner hands requests to worker processes, so a couple of
# hundred milliseconds of skew per request is normal; a whole second is not.
SEND_TOLERANCE_SEC = 0.3


def write_mooncake_trace(path: Path) -> Path:
    lines = []
    for offset, n_in, n_out, hash_ids in zip(OFFSETS_MS, INPUT_TOKENS, OUTPUT_TOKENS, HASH_IDS, strict=True):
        lines.append(json.dumps({"timestamp": offset, "input_length": n_in, "output_length": n_out, "hash_ids": hash_ids}))
    path.write_text("\n".join(lines) + "\n")
    return path


def mooncake_workload_config(trace: Path) -> Dict[str, Any]:
    """The record-layer spelling: the format is named once, the block size
    is the trace's, and the send times come from the arrangement, so load
    has no trace block."""
    return {
        "data": {
            "type": "workload_replay",
            "workload": {"format": "Mooncake", "file": str(trace), "block_size": BLOCK_SIZE},
        },
        "load": {"type": "trace_replay", "num_workers": 2, "stages": [{"rate": 1, "duration": 1}]},
    }


def assert_replays_the_trace(entries: List[Dict[str, Any]], tokenizer: CustomTokenizer) -> List[str]:
    """The report, ordered by send time, must line up with the trace line by
    line. Returns the prompts in that order for the prefix check.

    Prompt lengths are measured with the benchmark's own tokenizer: the sim
    estimates prompt_tokens rather than tokenizing, so its usage numbers say
    nothing about what was sent. Output lengths are what the server produced,
    which with ignore_eos is exactly the max_tokens the client asked for.
    """
    entries = sorted(entries, key=lambda e: e["start_time"])
    prompts: List[str] = []
    for entry, n_in, n_out in zip(entries, INPUT_TOKENS, OUTPUT_TOKENS, strict=True):
        body = request_body(entry)
        got_in = tokenizer.count_tokens(body["prompt"])
        assert got_in == n_in, f"prompt of {got_in} tokens where the trace recorded {n_in}"
        assert body["max_tokens"] == n_out, f"asked for {body['max_tokens']} tokens where the trace recorded {n_out}"
        assert server_completion_tokens(entry) == n_out, f"server produced {server_completion_tokens(entry)}, expected {n_out}"
        prompts.append(body["prompt"])

    first = entries[0]["start_time"]
    sent_at = [e["start_time"] - first for e in entries]
    for got, want_ms in zip(sent_at, OFFSETS_MS, strict=True):
        want = want_ms / 1000.0
        assert abs(got - want) <= SEND_TOLERANCE_SEC, f"sent at {got:.3f}s where the trace says {want:.3f}s (all: {sent_at})"
    return prompts


def assert_prefixes_follow_the_hash_ids(prompts: List[str], tokenizer: CustomTokenizer) -> None:
    """Requests that share their first k hash ids were sent with the same
    first k * 16 tokens and differ inside the block after; requests that
    share none differ inside their first 16. The prompts are re-tokenized
    without special tokens, which is what the server sees. The last shared
    token is excluded from the comparison: the text diverges right after
    it, and the tokenizer sometimes merges that boundary token with what
    follows (seen on one run out of two for a 3-block prefix), so k * 16 - 1
    leading ids is the claim that holds on every run.
    """
    hf = tokenizer.get_tokenizer()
    ids = [hf.encode(prompt, add_special_tokens=False) for prompt in prompts]
    for a, b, k in SHARED:
        shared = k * BLOCK_SIZE
        assert ids[a][: shared - 1] == ids[b][: shared - 1], (
            f"requests {a} and {b} share {k} hash ids but their first {shared - 1} tokens differ"
        )
        assert ids[a][shared:][:BLOCK_SIZE] != ids[b][shared:][:BLOCK_SIZE], (
            f"requests {a} and {b} share only {k} hash ids but agree past block {k}"
        )
    for a, b in DISJOINT:
        assert ids[a][:BLOCK_SIZE] != ids[b][:BLOCK_SIZE], f"requests {a} and {b} share no hash ids but start alike"


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
async def test_mooncake_trace_replay(tmp_path: Path) -> None:
    model_path = extract_tarball(TEST_MODEL_TARBALL)
    trace = write_mooncake_trace(tmp_path / "conversation_trace.jsonl")

    async with LLMDInferenceSimRunner(TEST_MODEL_NAME, port=get_free_port()) as sim:
        result = await run_benchmark_minimal(
            {
                **mooncake_workload_config(trace),
                "api": {"type": "completion", "streaming": True},
                "server": {
                    "type": "vllm",
                    "model_name": TEST_MODEL_NAME,
                    "base_url": f"http://{sim.host}:{sim.port}",
                    "ignore_eos": True,
                },
                "tokenizer": {"pretrained_model_name_or_path": str(model_path)},
                "report": {"request_lifecycle": {"summary": True, "per_stage": True, "per_request": True}},
            }
        )

    entries = assert_successful_run(result, expected_requests=len(OFFSETS_MS))
    tokenizer = CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path=str(model_path)))
    prompts = assert_replays_the_trace(entries, tokenizer)
    assert_prefixes_follow_the_hash_ids(prompts, tokenizer)
