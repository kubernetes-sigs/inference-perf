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
import os
import sys
import shutil
import subprocess
import time
from pathlib import Path
import pytest
import requests

from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.benchmark import run_benchmark_minimal
from utils.testdata import extract_tarball

TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"


def is_prometheus_available() -> bool:
    return shutil.which("prometheus") is not None


@pytest.fixture(scope="module")
def prometheus_server():
    """Starts a lightweight ephemeral Prometheus instance."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        config_path = tmp_path / "prometheus.yml"

        # Write minimal config pointing to the simulator
        config_path.write_text(
            """
global:
  scrape_interval: 1s
scrape_configs:
  - job_name: 'llm-d-inference-sim'
    static_configs:
      - targets: ['127.0.0.1:18000']
""",
            encoding="utf-8",
        )

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Start prometheus
        proc = subprocess.Popen(
            [
                "prometheus",
                f"--config.file={config_path}",
                f"--storage.tsdb.path={data_dir}",
                "--web.listen-address=127.0.0.1:9090",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Wait for ready
        ready = False
        for _ in range(30):
            try:
                resp = requests.get("http://127.0.0.1:9090/-/ready", timeout=1)
                if resp.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(1)

        if not ready:
            proc.terminate()
            stdout, _ = proc.communicate()
            raise Exception(f"Prometheus failed to become ready. Output:\n{stdout.decode()}")

        yield "http://127.0.0.1:9090"

        # Teardown
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


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
                    "stages": [{"rate": 5, "duration": 5}],
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
                        "scrape_interval": 1,
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
            executable=[sys.executable, "/workspace/inference_perf/main.py"],
            extra_env={"PYTHONPATH": "/workspace"}
        )

    if not result.success:
        print(f"Simulator Stdout:\n{sim.stdout}")
    assert result.success, f"Benchmark failed with output:\n{result.stdout}"

    # Check if Prometheus metrics report was generated
    assert result.reports, "No reports generated"
    
    report_names = list(result.reports.keys())
    print(f"Generated reports: {report_names}")
    
    assert "summary_prometheus_metrics.json" in result.reports, f"Missing prometheus report in {report_names}"
    
    prom_report = result.reports["summary_prometheus_metrics.json"]
    assert prom_report, "Prometheus report is empty"
    assert isinstance(prom_report, dict), "Report should be a dictionary"
    
    print(f"Prometheus Report Content:\n{json.dumps(prom_report, indent=2)}")
    
    lifecycle_report = result.reports.get("summary_lifecycle_metrics.json")
    if lifecycle_report:
        print(f"Lifecycle Report Content:\n{json.dumps(lifecycle_report, indent=2)}")
        if lifecycle_report.get("failures", {}).get("count", 0) > 0:
            print(f"Benchmark Stdout:\n{result.stdout}")
