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

import json
import pathlib
import re
import shutil
import sys
import logging
from typing import Any
import pytest
import requests
import textwrap

from utils.benchmark import run_benchmark_minimal
from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.testdata import extract_tarball


logger = logging.getLogger(__name__)


PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
MAIN_PY_PATH = PROJECT_ROOT / "inference_perf" / "main.py"

TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"


def is_prometheus_available() -> bool:
    return shutil.which("prometheus") is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
@pytest.mark.skipif(not is_prometheus_available(), reason="local environment missing prometheus")
async def test_prometheus_metrics_collection(prometheus_server):
    """Verifies that inference-perf can collect metrics from Prometheus."""
    model_name = TEST_MODEL_NAME
    model_path = extract_tarball(TEST_MODEL_TARBALL)

    prometheus_url = prometheus_server

    async with LLMDInferenceSimRunner(model_name, port=18000) as sim:
        # Run a short benchmark
        result = await run_benchmark_minimal(
            {
                "data": {
                    "type": "shared_prefix",
                    "shared_prefix": {
                        "num_groups": 1,
                        "num_prompts_per_group": 25,
                        "system_prompt_len": 512,
                        "question_len": 256,
                        "output_len": 256,
                    },
                },
                "load": {
                    "type": "constant",
                    "stages": [{"rate": 5, "duration": 30}],
                    "num_workers": 1,
                },
                "api": {
                    "type": "completion",
                    "streaming": True,
                },
                "server": {
                    "type": "vllm",
                    "model_name": model_name,
                    "base_url": f"http://127.0.0.1:{sim.port}",
                    "ignore_eos": True,
                },
                "tokenizer": {
                    "pretrained_model_name_or_path": str(model_path),
                },
                "metrics": {
                    "type": "prometheus",
                    "prometheus": {
                        "url": prometheus_url,
                        "scrape_interval": 5,
                    },
                },
                "report": {
                    "request_lifecycle": {
                        "summary": True,
                    },
                    "prometheus": {
                        "summary": True,
                    },
                },
            },
            executable=[sys.executable, str(MAIN_PY_PATH)],
            extra_env={"PYTHONPATH": str(PROJECT_ROOT)},
        )

        # Verify benchmark succeeded before proceeding
        assert result.success, f"Benchmark failed with output: {format_content(result.stdout)}"

        # Debug and verify metrics exposed by simulator
        try:
            resp = requests.get("http://127.0.0.1:18000/metrics", timeout=1)
            metrics_content = resp.text
            logger.debug(f"Simulator Metrics Content: {format_content(metrics_content)}")

            # Verify simulator recorded 150 successes (rate 5 * duration 30)
            match = re.search(r"vllm:request_success(?:_total)?\{.*?\} (\d+)", metrics_content)
            assert match, "vllm:request_success(_total) not found in simulator metrics"

            sim_success_count = int(match.group(1))
            assert sim_success_count == 150, f"Expected 150 successes in simulator metrics, got {sim_success_count}"

            logger.debug("Verified 150 successes in simulator metrics")
        except Exception as e:
            logger.debug(f"Failed to get or verify simulator metrics: {e}")
            raise

    # Check if Prometheus metrics report was generated
    assert result.reports, "No reports generated"

    report_names = list(result.reports.keys())
    logger.debug(f"Generated reports: {report_names}")

    assert "summary_prometheus_metrics.json" in result.reports, f"Missing prometheus report in {report_names}"

    prom_report = result.reports["summary_prometheus_metrics.json"]
    assert prom_report, "Prometheus report is empty"
    assert isinstance(prom_report, dict), "Report should be a dictionary"

    logger.debug(f"Prometheus Report Content: {format_content(prom_report)}")

    # Assertions on content
    assert "successes" in prom_report, "Report missing 'successes' section"
    successes_obj = prom_report["successes"]

    assert "request_success_count" in successes_obj, "Missing 'request_success_count'"
    success_count = successes_obj["request_success_count"]
    logger.debug(f"Asserting request_success_count ({success_count}) is greater than 100")
    assert success_count > 100.0, f"Expected > 100 successes in report, got {success_count}"

    assert "rate" in successes_obj, "Missing 'rate' (requests_per_second)"
    rps = successes_obj["rate"]
    logger.debug(f"Asserting rate/requests_per_second ({rps}) is reasonable")
    assert 3.0 < rps < 7.0, f"Expected rate around 5, got {rps}"

    lifecycle_report = result.reports.get("summary_lifecycle_metrics.json")
    if lifecycle_report:
        logger.debug(f"Lifecycle Report Content: {format_content(lifecycle_report)}")
        if lifecycle_report.get("failures", {}).get("count", 0) > 0:
            logger.debug(f"Benchmark Stdout:\n{result.stdout}")


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
@pytest.mark.skipif(not is_prometheus_available(), reason="local environment missing prometheus")
async def test_prometheus_metrics_collection_chat(prometheus_server):
    """Verifies that inference-perf can collect metrics from Prometheus for Chat API."""
    model_name = TEST_MODEL_NAME
    model_path = extract_tarball(TEST_MODEL_TARBALL)

    prometheus_url = prometheus_server

    async with LLMDInferenceSimRunner(model_name, port=18001) as sim:
        # Run a short benchmark with chat API
        result = await run_benchmark_minimal(
            {
                "data": {
                    "type": "mock",
                },
                "load": {
                    "type": "constant",
                    "stages": [{"rate": 5, "duration": 30}],
                    "num_workers": 1,
                },
                "api": {
                    "type": "chat",
                    "streaming": True,
                },
                "server": {
                    "type": "vllm",
                    "model_name": model_name,
                    "base_url": f"http://127.0.0.1:{sim.port}",
                    "ignore_eos": True,
                },
                "tokenizer": {
                    "pretrained_model_name_or_path": str(model_path),
                },
                "metrics": {
                    "type": "prometheus",
                    "prometheus": {
                        "url": prometheus_url,
                        "scrape_interval": 5,
                    },
                },
                "report": {
                    "request_lifecycle": {
                        "summary": True,
                    },
                    "prometheus": {
                        "summary": True,
                    },
                },
            },
            executable=[sys.executable, str(MAIN_PY_PATH)],
            extra_env={"PYTHONPATH": str(PROJECT_ROOT)},
        )

        # Verify benchmark succeeded before proceeding
        assert result.success, f"Benchmark failed with output:\n{result.stdout}"

        # Debug and verify metrics exposed by simulator
        try:
            resp = requests.get("http://127.0.0.1:18001/metrics", timeout=1)
            metrics_content = resp.text
            logger.debug(f"Simulator Metrics Content: {format_content(metrics_content)}")

            # Verify simulator recorded 150 successes
            match = re.search(r"vllm:request_success(?:_total)?\{.*?\} (\d+)", metrics_content)
            assert match, "vllm:request_success(_total) not found in simulator metrics"
            sim_success_count = int(match.group(1))
            assert sim_success_count == 150, f"Expected 150 successes in simulator metrics, got {sim_success_count}"
            logger.debug("Verified 150 successes in simulator metrics")
        except Exception as e:
            logger.debug(f"Failed to get or verify simulator metrics: {e}")
            raise

    # Check if Prometheus metrics report was generated
    assert result.reports, "No reports generated"

    report_names = list(result.reports.keys())
    logger.debug(f"Generated reports: {report_names}")

    assert "summary_prometheus_metrics.json" in result.reports, f"Missing prometheus report in {report_names}"

    prom_report = result.reports["summary_prometheus_metrics.json"]
    assert prom_report, "Prometheus report is empty"
    assert isinstance(prom_report, dict), "Report should be a dictionary"

    logger.debug(f"Prometheus Report Content: {format_content(prom_report)}")

    # Assertions on content
    assert "successes" in prom_report, "Report missing 'successes' section"
    successes_obj = prom_report["successes"]

    assert "request_success_count" in successes_obj, "Missing 'request_success_count'"
    success_count = successes_obj["request_success_count"]
    logger.debug(f"Asserting request_success_count ({success_count}) is greater than 100")
    assert success_count > 100.0, f"Expected > 100 successes in report, got {success_count}"

    assert "rate" in successes_obj, "Missing 'rate' (requests_per_second)"
    rps = successes_obj["rate"]
    logger.debug(f"Asserting rate/requests_per_second ({rps}) is reasonable")
    assert 3.0 < rps < 7.0, f"Expected rate around 5, got {rps}"


def format_content(content: str | Any) -> str:
    content_str = content if isinstance(content, str) else json.dumps(content, indent=2)
    return "\n" + textwrap.indent(content_str, "  | ")
