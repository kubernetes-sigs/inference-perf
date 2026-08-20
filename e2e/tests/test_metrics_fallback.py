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
    # "" mimics a legacy exposition (bare counter names), "_total" a modern
    # prometheus_client one; both must resolve through the same declared names.
    counter_suffix = ""
    success_count = 0
    prompt_tokens = 0

    @classmethod
    def reset(cls, counter_suffix: str) -> None:
        cls.counter_suffix = counter_suffix
        cls.success_count = 0
        cls.prompt_tokens = 0

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
        elif self.path == "/metrics":
            body = ""
            for base, help_text, value in (
                ("vllm:request_success", "Count of successfully processed requests.", MockHandler.success_count),
                ("vllm:prompt_tokens", "Number of prefill tokens processed.", MockHandler.prompt_tokens),
            ):
                name = f"{base}{MockHandler.counter_suffix}"
                body += (
                    f"# HELP {name} {help_text}\n"
                    f"# TYPE {name} counter\n"
                    f'{name}{{model_name="facebook/opt-125m"}} {float(value)}\n'
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
            MockHandler.prompt_tokens += 10
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


def start_mock_server(port: int, counter_suffix: str) -> http.server.HTTPServer:
    MockHandler.reset(counter_suffix)
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
    """Verifies that inference-perf can collect metrics from a legacy exposition using bare
    counter names ('vllm:request_success', 'vllm:prompt_tokens')."""
    server = start_mock_server(prometheus_server.sim_port, "")

    try:
        result = await run_benchmark_minimal(
            _benchmark_config(prometheus_server.url, prometheus_server.sim_port),
            executable=[sys.executable, str(MAIN_PY_PATH)],
            extra_env={"PYTHONPATH": str(PROJECT_ROOT)},
        )

        assert result.success, f"Benchmark failed: {result.stdout}"
        assert result.reports and "summary_prometheus_metrics.json" in result.reports
        report = result.reports["summary_prometheus_metrics.json"]
        assert "successes" in report
        success_count = report["successes"]["request_success_count"]
        assert success_count > 0, f"Expected non-zero success count from mock, got {success_count}"
        prompt_rate = report["successes"]["prompt_len"]["rate"]
        assert prompt_rate > 0, f"Expected non-zero prompt token rate from mock, got {prompt_rate}"

    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.asyncio
@pytest.mark.skipif(not is_prometheus_available(), reason="local environment missing prometheus")
async def test_new_metric_name(prometheus_server):
    """Verifies that inference-perf can collect metrics from a modern prometheus_client
    exposition using '_total'-suffixed counter names ('vllm:request_success_total',
    'vllm:prompt_tokens_total'), which is what a stock vLLM stores (#567)."""
    server = start_mock_server(prometheus_server.sim_port, "_total")

    try:
        result = await run_benchmark_minimal(
            _benchmark_config(prometheus_server.url, prometheus_server.sim_port),
            executable=[sys.executable, str(MAIN_PY_PATH)],
            extra_env={"PYTHONPATH": str(PROJECT_ROOT)},
        )

        assert result.success, f"Benchmark failed: {result.stdout}"
        assert result.reports and "summary_prometheus_metrics.json" in result.reports
        report = result.reports["summary_prometheus_metrics.json"]
        assert "successes" in report
        success_count = report["successes"]["request_success_count"]
        assert success_count > 0, f"Expected non-zero success count from mock, got {success_count}"
        prompt_rate = report["successes"]["prompt_len"]["rate"]
        assert prompt_rate > 0, f"Expected non-zero prompt token rate from mock, got {prompt_rate}"

    finally:
        server.shutdown()
        server.server_close()
