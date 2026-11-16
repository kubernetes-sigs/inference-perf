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
"""A trace_session_replay stage ended by a measured stop_condition, against llm-d-inference-sim.

Runs the real CLI with worker processes, so the condition is fed by the request
metric collector in the main process, the path unit tests stub out. Skipped when
`llm-d-inference-sim` is not on PATH.
"""

import re

import pytest

from utils.benchmark import run_benchmark_minimal
from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.net import get_free_port
from utils.testdata import extract_tarball

TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"

STOP_CONDITION = "(t >= 120) | (completed >= 20)"
_HELD_RE = re.compile(r"stop condition '\(t >= 120\) \| \(completed >= 20\)' held at t=([\d.]+)s")


# One single-request synthetic_agentic session per slot, 4 slots, over a 5-session corpus that cycles: the stage
# ends COMPLETED once 20 requests have finished, well before its 120s deadline, with at least 20 successes reported.
@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
async def test_session_stage_ends_on_measured_stop_condition() -> None:
    model_path = extract_tarball(TEST_MODEL_TARBALL)
    port = get_free_port()
    config = {
        "api": {"type": "chat", "streaming": True},
        "data": {
            "type": "synthetic_agentic",
            "synthetic_agentic": {
                "num_sessions": 5,
                "seed": 1,
                "theme_mix": {"generic": {"weight": 1.0}},
                "turns_per_session": {"type": "fixed", "mean": 1},
                "fanout_probability": 0.0,
                "tool_loop_depth": {"type": "fixed", "mean": 0},
                "tool_catalog_size_per_agent": {"type": "fixed", "mean": 0},
                "input_tokens_per_turn": {"type": "fixed", "mean": 50},
                "output_tokens_per_turn": {"type": "fixed", "mean": 10},
                "tool_call_latency_sec": {"type": "fixed", "mean": 1},
            },
        },
        "load": {
            "type": "trace_session_replay",
            "stages": [{"concurrent_sessions": 4, "stop_condition": STOP_CONDITION}],
            "num_workers": 2,
        },
        "server": {
            "type": "vllm",
            "model_name": TEST_MODEL_NAME,
            "base_url": f"http://127.0.0.1:{port}",
            "ignore_eos": True,
        },
        "tokenizer": {"pretrained_model_name_or_path": str(model_path)},
        "report": {"request_lifecycle": {"summary": True, "per_stage": True}},
    }

    async with LLMDInferenceSimRunner(TEST_MODEL_NAME, port=port):
        result = await run_benchmark_minimal(config, timeout_sec=180)

    assert result.success, f"Benchmark did not complete cleanly:\n{result.stdout}"
    held = _HELD_RE.search(result.stdout)
    assert held, f"stop condition never logged as held:\n{result.stdout}"
    assert float(held.group(1)) < 60, "the stage ran toward its 120s deadline instead of ending on completed >= 20"
    assert "Stage 0 - session-based run completed" in result.stdout

    summary = result.reports["summary_lifecycle_metrics.json"]
    assert summary["successes"]["count"] >= 20
    assert summary["failures"]["count"] == 0
