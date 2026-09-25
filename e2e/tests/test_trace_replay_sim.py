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
"""Trace replay against llm-d-inference-sim.

Pins what a replayed trace has to reproduce, whichever code path replays it:
one request per recorded line, each prompt tokenizing to the recorded input
length, each response the recorded output length, and requests leaving at
the recorded offsets from each other.
"""

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

# Ten requests over 3.2 seconds in three bursts, with input and output
# lengths that differ enough per request to be told apart in the report.
OFFSETS_SEC = [0.0, 0.2, 0.4, 1.5, 1.6, 1.7, 1.8, 3.0, 3.1, 3.2]
INPUT_TOKENS = [64, 128, 96, 200, 50, 300, 150, 80, 256, 120]
OUTPUT_TOKENS = [8, 16, 12, 24, 10, 40, 20, 14, 32, 18]

# How far the client's send times may drift from the recorded offsets.
# The stage runner hands requests to worker processes, so a couple of
# hundred milliseconds of skew per request is normal; a whole second is not.
SEND_TOLERANCE_SEC = 0.3


def write_azure_trace(path: Path) -> Path:
    lines = ["TIMESTAMP,ContextTokens,GeneratedTokens"]
    for offset, n_in, n_out in zip(OFFSETS_SEC, INPUT_TOKENS, OUTPUT_TOKENS, strict=True):
        seconds = int(offset)
        micros = int(round((offset - seconds) * 1_000_000))
        lines.append(f"2026-01-01 00:00:{seconds:02d}.{micros:06d},{n_in},{n_out}")
    path.write_text("\n".join(lines) + "\n")
    return path


def azure_trace_config(trace: Path) -> Dict[str, Any]:
    """The pre-record-layer spelling: data.type random plus a trace block on
    both data and load."""
    trace_block = {"file": str(trace), "format": "AzurePublicDataset"}
    return {
        "data": {"type": "random", "trace": trace_block},
        "load": {"type": "trace_replay", "trace": trace_block, "num_workers": 2, "stages": [{"rate": 1, "duration": 1}]},
    }


def azure_workload_config(trace: Path) -> Dict[str, Any]:
    """The record-layer spelling: the format is named once, and the send
    times come from the arrangement, so load has no trace block."""
    return {
        "data": {"type": "workload_replay", "workload": {"format": "AzurePublicDataset", "file": str(trace)}},
        "load": {"type": "trace_replay", "num_workers": 2, "stages": [{"rate": 1, "duration": 1}]},
    }


def assert_replays_the_trace(entries: List[Dict[str, Any]], tokenizer: CustomTokenizer) -> None:
    """The report, ordered by send time, must line up with the trace line by line.

    Prompt lengths are measured with the benchmark's own tokenizer: the sim
    estimates prompt_tokens rather than tokenizing, so its usage numbers say
    nothing about what was sent. Output lengths are what the server produced,
    which with ignore_eos is exactly the max_tokens the client asked for.
    """
    entries = sorted(entries, key=lambda e: e["start_time"])
    for entry, n_in, n_out in zip(entries, INPUT_TOKENS, OUTPUT_TOKENS, strict=True):
        body = request_body(entry)
        got_in = tokenizer.count_tokens(body["prompt"])
        assert got_in == n_in, f"prompt of {got_in} tokens where the trace recorded {n_in}"
        assert body["max_tokens"] == n_out, f"asked for {body['max_tokens']} tokens where the trace recorded {n_out}"
        assert server_completion_tokens(entry) == n_out, f"server produced {server_completion_tokens(entry)}, expected {n_out}"

    first = entries[0]["start_time"]
    sent_at = [e["start_time"] - first for e in entries]
    for got, want in zip(sent_at, OFFSETS_SEC, strict=True):
        assert abs(got - want) <= SEND_TOLERANCE_SEC, f"sent at {got:.3f}s where the trace says {want:.3f}s (all: {sent_at})"


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
@pytest.mark.parametrize(
    "make_config",
    [
        pytest.param(azure_trace_config, id="azure_trace_block"),
        pytest.param(azure_workload_config, id="azure_workload_format"),
    ],
)
async def test_azure_trace_replay(tmp_path: Path, make_config: Any) -> None:
    model_path = extract_tarball(TEST_MODEL_TARBALL)
    trace = write_azure_trace(tmp_path / "trace.csv")

    async with LLMDInferenceSimRunner(TEST_MODEL_NAME, port=get_free_port()) as sim:
        result = await run_benchmark_minimal(
            {
                **make_config(trace),
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

    entries = assert_successful_run(result, expected_requests=len(OFFSETS_SEC))
    tokenizer = CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path=str(model_path)))
    assert_replays_the_trace(entries, tokenizer)
