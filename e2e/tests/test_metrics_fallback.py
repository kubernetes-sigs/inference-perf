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

import http.server
import json
import pathlib
import sys
import threading
import pytest

from utils.benchmark import run_benchmark_minimal
from test_prometheus import is_prometheus_available

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
MAIN_PY_PATH = PROJECT_ROOT / "inference_perf" / "main.py"


class MockHandler(http.server.BaseHTTPRequestHandler):
    metric_name = "vllm:request_success"
    success_count = 0

    @classmethod
    def reset(cls, metric_name: str) -> None:
        cls.metric_name = metric_name
        cls.success_count = 0

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
        elif self.path == "/metrics":
            body = (
                f"# HELP {self.metric_name} Count of successfully processed requests.\n"
                f"# TYPE {self.metric_name} counter\n"
                f'{self.metric_name}{{model_name="facebook/opt-125m"}} {float(MockHandler.success_count)}\n'
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/v1/completions":
            MockHandler.success_count += 1
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "id": "cmpl-mock",
                "object": "text_completion",
                "created": 12345,
                "model": "facebook/opt-125m",
                "choices": [{"text": " mock response", "finish_reason": "length"}],
                "usage": {"prompt_tokens": 1, "total_tokens": 5, "completion_tokens": 4},
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return


def start_mock_server(port: int, metric_name: str) -> http.server.HTTPServer:
    MockHandler.reset(metric_name)
    server = http.server.HTTPServer(("127.0.0.1", port), MockHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    return server


def _benchmark_config(prometheus_url: str, port: int) -> dict:
    return {
        "data": {
            "type": "shared_prefix",
            "shared_prefix": {
                "num_groups": 1,
                "num_prompts_per_group": 5,
                "system_prompt_len": 10,
                "question_len": 10,
                "output_len": 10,
            },
        },
        "load": {
            "type": "constant",
            "stages": [{"rate": 1, "duration": 15}],
            "num_workers": 1,
        },
        "api": {
            "type": "completion",
            "streaming": True,
        },
        "server": {
            "type": "vllm",
            "model_name": "facebook/opt-125m",
            "base_url": f"http://127.0.0.1:{port}",
            "ignore_eos": True,
        },
        "tokenizer": {
            "pretrained_model_name_or_path": "facebook/opt-125m",
        },
        "metrics": {
            "type": "prometheus",
            "prometheus": {
                "url": prometheus_url,
                "scrape_interval": 5,
            },
        },
        "report": {
            "prometheus": {
                "summary": True,
            },
        },
    }


@pytest.mark.asyncio
@pytest.mark.skipif(not is_prometheus_available(), reason="local environment missing prometheus")
async def test_legacy_metric_name(prometheus_server):
    """Verifies that inference-perf can collect metrics using the legacy name 'vllm:request_success'."""
    server = start_mock_server(18000, "vllm:request_success")

    try:
        result = await run_benchmark_minimal(
            _benchmark_config(prometheus_server, 18000),
            executable=[sys.executable, str(MAIN_PY_PATH)],
            extra_env={"PYTHONPATH": str(PROJECT_ROOT)},
        )

        assert result.success, f"Benchmark failed: {result.stdout}"
        assert result.reports and "summary_prometheus_metrics.json" in result.reports
        report = result.reports["summary_prometheus_metrics.json"]
        assert "successes" in report
        success_count = report["successes"]["request_success_count"]
        assert success_count > 0, f"Expected non-zero success count from mock, got {success_count}"

    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.asyncio
@pytest.mark.skipif(not is_prometheus_available(), reason="local environment missing prometheus")
async def test_new_metric_name(prometheus_server):
    """Verifies that inference-perf can collect metrics using the new name 'vllm:request_success_total'."""
    server = start_mock_server(18001, "vllm:request_success_total")

    try:
        result = await run_benchmark_minimal(
            _benchmark_config(prometheus_server, 18001),
            executable=[sys.executable, str(MAIN_PY_PATH)],
            extra_env={"PYTHONPATH": str(PROJECT_ROOT)},
        )

        assert result.success, f"Benchmark failed: {result.stdout}"
        assert result.reports and "summary_prometheus_metrics.json" in result.reports
        report = result.reports["summary_prometheus_metrics.json"]
        assert "successes" in report
        success_count = report["successes"]["request_success_count"]
        assert success_count > 0, f"Expected non-zero success count from mock, got {success_count}"

    finally:
        server.shutdown()
        server.server_close()
