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
End-to-end integration testing of inference-perf using llm-d-inference-sim[1].

In order for these tests to run, you must have `llm-d-inference-sim` in your
PATH. The GitHub Actions runner will have this, but you may also install it
locally by following llm-d-inference-sim's README or by entering the Nix shell
of this repository (i.e. `nix develop`).

If your local environment is missing `llm-d-inference-sim`, tests here will
automatically be skipped.

[1]: https://github.com/llm-d/llm-d-inference-sim
"""

import os
import pytest

from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.benchmark import run_benchmark_minimal

TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"
TEST_SHAREGPT_PATH = os.path.abspath("e2e/testdata/sharegpt.json")


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
@pytest.mark.parametrize(
    "data",
    [
        pytest.param(
            {
                "type": "mock",
            },
            id="data_mock",
        ),
        pytest.param(
            {
                "type": "shareGPT",
                "path": TEST_SHAREGPT_PATH,
            },
            id="data_sharegpt",
        ),
        pytest.param(
            {
                "type": "shared_prefix",
                "shared_prefix": {
                    "num_groups": 256,
                    "num_prompts_per_group": 16,
                    "system_prompt_len": 512,
                    "question_len": 256,
                    "output_len": 256,
                },
            },
            id="data_shared_prefix",
        ),
    ],
)
@pytest.mark.parametrize(
    "load",
    [
        pytest.param(
            {
                "type": "constant",
                "stages": [{"rate": 1, "duration": 5}],
                "num_workers": 2,
            },
            id="load_constant_slow",
        ),
        pytest.param(
            {
                "type": "constant",
                "interval": 2,
                "stages": [{"rate": 1, "duration": 5}, {"rate": 2, "duration": 5}],
                "num_workers": 2,
            },
            id="load_constant_slow_two_stages",
        ),
        pytest.param(
            {
                "type": "constant",
                "stages": [{"rate": 100, "duration": 5}],
                "num_workers": 2,
            },
            id="load_constant_fast",
        ),
    ],
)
async def test_completion_successful_run(data: dict, load: dict):
    """
    Very simple inference-perf integration test that ensures a wide range of
    vLLM benchmarking configurations can run successfully.
    """
    model_name = TEST_MODEL_NAME

    async with LLMDInferenceSimRunner(model_name, port=18000) as sim:
        result = await run_benchmark_minimal(
            {
                "data": data,
                "load": load,
            },
            url=f"http://{sim.host}:{sim.port}",
        )

    assert result.success, "Benchmark failed"
    assert result.reports, "No reports generated from benchmark"

    summary_report = result.reports["summary_lifecycle_metrics.json"]
    assert summary_report, "Missing summary report"
    assert summary_report["successes"]["count"] > 1


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
async def test_completion_auto_config():
    """Test that inference-perf can run with no config file, relying on auto-detection."""
    model_name = TEST_MODEL_NAME

    async with LLMDInferenceSimRunner(model_name, port=18001) as sim:
        result = await run_benchmark_minimal(
            {},  # Empty config
            url=f"http://{sim.host}:{sim.port}",
        )

    assert result.success, "Benchmark failed"
    assert result.reports, "No reports generated from benchmark"
    summary_report = result.reports["summary_lifecycle_metrics.json"]
    assert summary_report, "Missing summary report"
