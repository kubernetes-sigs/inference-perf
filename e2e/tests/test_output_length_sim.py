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
End-to-end check of the requested-versus-delivered output length fields (#655)
against llm-d-inference-sim, extending the ground-truth pattern of #614.

The sim honours ``ignore_eos``: with it on, every response runs to exactly
``max_tokens`` and reports ``finish_reason: length``; with it off, it stops at
a random earlier point and reports ``finish_reason: stop``. That gives two
known outcomes for the same config:

- ``ignore_eos: true``: no request is short, so ``finish_reasons`` is all
  ``length``, ``output_shortfalls`` is 0 and nothing is reclassified as failed.
- ``ignore_eos: false``: short responses are legitimate. They stay successes,
  ``finish_reasons`` shows the ``stop`` bucket, and ``output_shortfalls`` equals
  the number of requests whose server-reported ``completion_tokens`` fell
  below the ``max_tokens`` recorded on their per-request entry, which is the
  cross-check between the two report files.

The truncation *failure* (short under ``ignore_eos``) is a condition the sim
does not produce on its own, so it lives in the Integration tier against a
fake (tests/required/integration/test_truncated_response.py) per the #606
"fake the conditions, never the oracle" rule.
"""

import pytest
from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.benchmark import run_benchmark_minimal
from utils.net import get_free_port
from utils.testdata import extract_tarball

TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"
MAX_TOKENS = 40


# Pins every request's max_tokens to `tokens` through the random datagen's
# output distribution (a degenerate distribution: min == max == mean).
def _exact_length_distribution(tokens: int) -> dict:
    return {"min": tokens, "max": tokens, "mean": tokens, "std_dev": 0, "total_count": 100}


# Runs 10 streaming completion requests (rate 2 for 5s) against the sim with the
# given ignore_eos and max_tokens 40, and returns the summary and per-request
# reports. Fails if the run itself failed or produced no reports.
async def _run(ignore_eos: bool) -> tuple[dict, list]:
    model_path = extract_tarball(TEST_MODEL_TARBALL)
    async with LLMDInferenceSimRunner(
        TEST_MODEL_NAME,
        *("--time-to-first-token", "20"),
        *("--inter-token-latency", "5"),
        *("--max-num-seqs", "64"),
        *("--seed", "42"),
        port=get_free_port(),
    ) as sim:
        result = await run_benchmark_minimal(
            {
                "api": {"type": "completion", "streaming": True},
                "data": {
                    "type": "random",
                    "input_distribution": _exact_length_distribution(8),
                    "output_distribution": _exact_length_distribution(MAX_TOKENS),
                },
                "load": {"type": "constant", "stages": [{"rate": 2, "duration": 5}], "num_workers": 2},
                "server": {
                    "type": "vllm",
                    "model_name": TEST_MODEL_NAME,
                    "base_url": f"http://{sim.host}:{sim.port}",
                    "ignore_eos": ignore_eos,
                },
                "tokenizer": {"pretrained_model_name_or_path": str(model_path)},
                "report": {"request_lifecycle": {"summary": True, "per_stage": False, "per_request": True}},
            }
        )
    assert result.success, "Benchmark failed"
    assert result.reports, "No reports generated from benchmark"
    return result.reports["summary_lifecycle_metrics.json"], result.reports["per_request_lifecycle_metrics.json"]


# ignore_eos on, max_tokens 40 on all 10 requests: the sim delivers exactly 40
# with finish_reason "length" every time, so failures == 0, successes == 10,
# finish_reasons == {"length": 10}, output_shortfalls == 0, and every per-request
# entry records max_tokens 40 with completion_tokens 40.
@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
async def test_full_budget_under_ignore_eos_reports_length_and_no_shortfall():
    summary, per_request = await _run(ignore_eos=True)

    assert summary["failures"]["count"] == 0
    assert summary["successes"]["count"] == len(per_request) == 10
    assert summary["successes"]["finish_reasons"] == {"length": 10}
    assert summary["successes"]["output_shortfalls"] == 0
    for entry in per_request:
        assert entry["max_tokens"] == MAX_TOKENS
        assert entry["info"]["response_metrics"]["server_usage"]["completion_tokens"] == MAX_TOKENS
        assert entry["info"]["response_metrics"]["finish_reason"] == "length"


# ignore_eos off, same config: the sim stops early at random, so short responses
# are legitimate and stay successes (failures == 0). Every request carries a
# finish_reason (buckets sum to 10, keys within {"stop", "length"}), and
# output_shortfalls equals the number of per-request entries whose server
# completion_tokens is below their recorded max_tokens (40), which with 10
# random-length responses is at least 1.
@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
async def test_early_stops_without_ignore_eos_stay_successes_and_are_counted():
    summary, per_request = await _run(ignore_eos=False)

    assert summary["failures"]["count"] == 0
    assert summary["successes"]["count"] == len(per_request) == 10
    finish_reasons = summary["successes"]["finish_reasons"]
    assert set(finish_reasons) <= {"stop", "length"}, finish_reasons
    assert sum(finish_reasons.values()) == 10

    short = [e for e in per_request if e["info"]["response_metrics"]["server_usage"]["completion_tokens"] < e["max_tokens"]]
    assert all(e["max_tokens"] == MAX_TOKENS for e in per_request)
    assert len(short) >= 1, "the sim produced no early stop in 10 requests; the ignore_eos=false path is untested"
    assert summary["successes"]["output_shortfalls"] == len(short)
