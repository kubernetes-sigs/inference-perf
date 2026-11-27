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
"""Compare benchmark reports with the real SGLang /generate stream (#1092)."""

import json
import os

import pytest

from utils.accuracy import assert_successful_run, chunk_times, request_body, response_metrics, server_completion_tokens
from utils.benchmark import run_benchmark_minimal


@pytest.mark.parametrize("budget", [1, 16], ids=["single-token", "multi-token"])
# A live /generate stream must preserve the server's token count and normalize
# its text without dropping deltas or repeating cumulative prefixes.
async def test_sglang_generate_live(budget: int) -> None:
    base_url = os.environ.get("E2E_SGLANG_BASE_URL")
    if base_url is None:
        pytest.skip("E2E_SGLANG_BASE_URL is not set; no live SGLang server was requested")
    model = os.environ["E2E_SGLANG_MODEL"]
    mode = os.environ["E2E_SGLANG_STREAM_MODE"]
    assert mode in ("cumulative", "incremental"), f"unknown SGLang stream mode: {mode}"

    result = await run_benchmark_minimal(
        {
            "data": {
                "type": "synthetic",
                "input_distribution": {"type": "fixed", "min": 8, "max": 8, "mean": 8, "total_count": 8},
                "output_distribution": {"type": "fixed", "min": budget, "max": budget, "mean": budget, "total_count": 8},
            },
            "load": {"type": "constant", "stages": [{"rate": 1, "duration": 2}], "num_workers": 0, "base_seed": 42},
            "api": {"type": "sglang_generate", "streaming": True},
            "server": {"type": "sglang", "model_name": model, "base_url": base_url, "ignore_eos": True},
            "tokenizer": {"pretrained_model_name_or_path": model},
            "report": {"request_lifecycle": {"per_request": True}},
        },
        timeout_sec=120,
    )
    assert result.success, result.stdout
    for entry in assert_successful_run(result, expected_requests=2):
        sent = request_body(entry)
        assert sent["stream"] is True
        assert sent["sampling_params"]["max_new_tokens"] == budget
        assert sent["sampling_params"]["ignore_eos"] is True

        # SGLang sends a single JSON object per SSE data line. Read the raw
        # report body independently of inference-perf's stream parser.
        payloads = [line[5:].strip() for line in entry["response"].splitlines() if line.startswith("data:")]
        assert payloads and payloads[-1] == "[DONE]", "missing SGLang stream terminator"
        frames = [json.loads(payload) for payload in payloads[:-1]]
        assert frames, "SGLang returned no generation frames"

        counts = [frame["meta_info"]["completion_tokens"] for frame in frames]
        sizes = [len(frame["output_ids"]) for frame in frames]
        assert counts[-1] == server_completion_tokens(entry) == budget
        if budget > 1:
            # One frame looks the same in both modes, so it cannot prove that
            # the server was launched with the intended streaming format.
            assert 0 < counts[0] < counts[-1], f"did not exercise multiple generation frames: {counts}"
        if mode == "incremental":
            assert sizes == [count - prev for count, prev in zip(counts, [0, *counts[:-1]], strict=True)]
            expected_text = "".join(frame["text"] for frame in frames)
        else:
            assert sizes == counts
            expected_text = frames[-1]["text"]

        if budget > 1:
            assert expected_text, "multi-token stream did not return text"
        chunks = response_metrics(entry)["response_chunks"]
        assert len(chunk_times(entry)) == len(chunks)
        # Compare text, not exact re-encoded token counts: detokenized text
        # can re-encode to different IDs without any loss of streamed text.
        assert "".join(json.loads(chunk)["text"] for chunk in chunks) == expected_text
        if expected_text:
            assert chunk_times(entry), "text arrived without a content timestamp"
