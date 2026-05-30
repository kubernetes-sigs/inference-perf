import pytest
from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.benchmark import run_benchmark_minimal
from utils.testdata import extract_tarball

TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
async def test_preprocess_sweep_successful():
    """
    Test that the load generator can run preprocessing (sweep) successfully
    against llm-d-inference-sim.
    """
    model_name = TEST_MODEL_NAME
    model_path = extract_tarball(TEST_MODEL_TARBALL)

    async with LLMDInferenceSimRunner(model_name, port=18001) as sim:
        result = await run_benchmark_minimal(
            {
                "data": {
                    "type": "mock",
                },
                "load": {
                    "type": "constant",
                    "sweep": {
                        "type": "linear",
                        "num_stages": 2,
                        "stage_duration": 5,
                        "timeout": 10,
                    },
                    "num_workers": 1,
                },
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

    assert result.success, f"Benchmark failed with output:\n{result.stdout}"
    assert result.reports, "No reports generated from benchmark"


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
async def test_preprocess_sweep_saturation():
    """
    Test that the load generator detects saturation during preprocessing.
    """
    model_name = TEST_MODEL_NAME
    model_path = extract_tarball(TEST_MODEL_TARBALL)

    async with LLMDInferenceSimRunner(
        model_name, "--inter-token-latency", "50", "--max-num-seqs", "10", port=18005, stdout=None, stderr=None
    ) as sim:
        result = await run_benchmark_minimal(
            {
                "data": {
                    "type": "mock",
                },
                "load": {
                    "type": "constant",
                    "sweep": {
                        "type": "linear",
                        "num_stages": 3,
                        "stage_duration": 5,
                        "timeout": 10,
                    },
                    "num_workers": 1,
                },
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
                    },
                },
            }
        )

    assert result.success, f"Benchmark failed with output:\n{result.stdout}"
    assert "Saturation detected!" in result.stdout, "Expected saturation to be detected"
    assert "Step 1.0:" in result.stdout, "Expected saturation detection at Step 1.0"

    assert "Saturation point estimated at" in result.stdout, "Expected saturation point estimation to be reported"
    import re

    match = re.search(r"Saturation point estimated at ([\d.]+) QPS", result.stdout)
    assert match, "Failed to parse estimated saturation point from logs"
    saturation_val = float(match.group(1))
    assert saturation_val > 0.0, f"Expected non-zero saturation point, got {saturation_val}"
