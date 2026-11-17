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
"""
A concurrent stage whose `concurrency_level` is an expression delivers that level on the wire.

The sim answers every request in the same 375ms, so how many requests are in
flight at any moment is set by the load generator alone. The delivered level
is rebuilt from each request's `start_time` and `end_time`, not read from the
report's `concurrency` field, which only echoes the config.

Requires `llm-d-inference-sim` in PATH (see test_llm_d_inference_sim.py). If it
is missing, the tests are skipped automatically.
"""

from typing import Any, Dict, List, Sequence, Tuple

import pytest

from utils.accuracy import assert_successful_run
from utils.benchmark import run_benchmark_minimal
from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.load_shape import Segment, inflight_segments, mean_inflight
from utils.net import get_free_port
from utils.testdata import extract_tarball

TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"

# every request takes 150ms + 15 * 15ms = 375ms
SIM_TTFT_MS = 150
SIM_ITL_MS = 15
OUTPUT_TOKENS = 16

# the level changes this many seconds into the stage
STEP_AT_SEC = 4.0
# kept clear of the step on both sides: a lowered level is only reached once the
# requests already in flight finish, which is one request latency
SETTLE_SEC = 1.0
# the same half slot assert_delivered_concurrency allows a fixed level
SLACK = 0.5


# The most requests in flight at any moment inside `window`.
# E.g. segments [(0, 1, 2), (1, 2, 8)] over window (0, 0.9) -> 2.
def _max_inflight_within(segments: Sequence[Segment], window: Tuple[float, float]) -> int:
    lo, hi = window
    return max((n for seg_lo, seg_hi, n in segments if min(seg_hi, hi) > max(seg_lo, lo)), default=0)


# One concurrent stage of `num_requests` requests at `level` over 2 workers, with per-request records on.
# E.g. level "Piecewise((2, t < 4), (8, True))" -> 2 in flight for 4s, then 8.
def _config(sim: LLMDInferenceSimRunner, level: str, num_requests: int) -> Dict[str, Any]:
    return {
        "data": {
            "type": "synthetic",
            "input_distribution": {"type": "fixed", "min": 32, "max": 32, "mean": 32},
            "output_distribution": {"type": "fixed", "min": OUTPUT_TOKENS, "max": OUTPUT_TOKENS, "mean": OUTPUT_TOKENS},
        },
        "load": {
            "type": "concurrent",
            "stages": [{"num_requests": num_requests, "concurrency_level": level}],
            "num_workers": 2,
        },
        "api": {"type": "completion", "streaming": True},
        "server": {
            "type": "vllm",
            "model_name": TEST_MODEL_NAME,
            "base_url": f"http://{sim.host}:{sim.port}",
            "ignore_eos": True,
        },
        "tokenizer": {"pretrained_model_name_or_path": str(extract_tarball(TEST_MODEL_TARBALL))},
        "report": {"request_lifecycle": {"summary": True, "per_stage": True, "per_request": True}},
    }


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
@pytest.mark.parametrize(
    ("level", "before", "after", "num_requests"),
    [
        # about 21 requests in the first 4s at 2 in flight, the other 100 at 8 in flight take about 5s more
        pytest.param("Piecewise((2, t < 4), (8, True))", 2, 8, 120, id="step_up"),
        # about 85 requests in the first 4s at 8 in flight, the other 25 at 2 in flight take about 5s more
        pytest.param("Piecewise((8, t < 4), (2, True))", 8, 2, 110, id="step_down"),
    ],
)
# A stage whose level steps at t=4s against a sim that answers in 375ms.
# step_up: 2 in flight before the step and never more, 8 after. step_down: 8 before, 2 after and never more.
# Either way the report gives the peak, 8, as `concurrency` and the expression as `concurrency_expression`.
async def test_delivered_concurrency_follows_a_step_expression(level: str, before: int, after: int, num_requests: int) -> None:
    async with LLMDInferenceSimRunner(
        TEST_MODEL_NAME,
        # past the peak, so the sim never queues a request and stretches its latency
        *("--max-num-seqs", "64"),
        *("--time-to-first-token", f"{SIM_TTFT_MS}ms"),
        *("--inter-token-latency", f"{SIM_ITL_MS}ms"),
        *("--time-to-first-token-std-dev", "0"),
        *("--inter-token-latency-std-dev", "0"),
        *("--seed", "42"),
        port=get_free_port(),
    ) as sim:
        result = await run_benchmark_minimal(_config(sim, level, num_requests), timeout_sec=120)

    entries: List[Dict[str, Any]] = assert_successful_run(result, num_requests)

    assert result.reports
    load_summary = result.reports["stage_0_lifecycle_metrics.json"]["load_summary"]
    assert load_summary["concurrency"] == max(before, after), load_summary
    assert load_summary["concurrency_expression"] == level, load_summary

    starts = sorted(float(entry["start_time"]) for entry in entries)
    stage_start, last_start = starts[0], starts[-1]
    step = stage_start + STEP_AT_SEC
    # from the moment the first level is full until just short of the step
    before_window = (starts[before - 1], step - SETTLE_SEC)
    # from just past the step until the last request goes out, after which the stage only drains
    after_window = (step + SETTLE_SEC, last_start)
    assert after_window[1] - after_window[0] >= 1.0, f"the stage ended too soon after the step to measure: {after_window}"

    segments = inflight_segments(entries)
    peaks = (_max_inflight_within(segments, before_window), _max_inflight_within(segments, after_window))
    assert peaks[0] <= before and peaks[1] <= after, f"in flight peaked at {peaks}, configured ({before}, {after})"

    means = (mean_inflight(segments, before_window), mean_inflight(segments, after_window))
    assert abs(means[0] - before) <= SLACK and abs(means[1] - after) <= SLACK, (
        f"mean in flight {means[0]:.2f} then {means[1]:.2f}, configured {before} then {after}"
    )
