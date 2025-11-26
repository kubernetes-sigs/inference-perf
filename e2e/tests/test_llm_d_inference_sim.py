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

import pytest
import logging

from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.benchmark import run_benchmark_minimal
from utils.testdata import extract_tarball


TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"


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
    model_path = extract_tarball(TEST_MODEL_TARBALL)

    async with LLMDInferenceSimRunner(model_name, port=18000) as sim:
        result = await run_benchmark_minimal(
            {
                "data": data,
                "load": load,
                "api": {
                    "type": "completion",
                    "streaming": True,
                },
                "server": {
                    "type": "vllm",
                    "model_name": model_name,
                    "base_url": f"http://{sim.host}:{sim.port}",
                    "ignore_eos": True,
                },
                "tokenizer": {
                    "pretrained_model_name_or_path": str(model_path),
                },
                "report": {
                    "request_lifecycle": {
                        "summary": True,
                        "per_stage": True,
                        "per_request": True,
                    },
                },
            }
        )

    assert result.success, "Benchmark failed"
    assert result.reports, "No reports generated from benchmark"

    requests_report = result.reports["per_request_lifecycle_metrics.json"]
    expect_requests_report_num = sum([stage["duration"] * stage["rate"] for stage in load["stages"]])
    assert requests_report and len(requests_report) == expect_requests_report_num, "Unexpected number of requests in report"

    summary_report = result.reports["summary_lifecycle_metrics.json"]
    assert summary_report, "Missing summary report"
    assert summary_report["successes"]["count"] > 1
