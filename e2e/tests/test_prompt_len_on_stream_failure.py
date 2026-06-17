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
"""prompt_len must survive a streaming request that fails mid-stream.

Reproduces a real consumer report: a streaming-completion run against the
`random` dataset whose request-lifecycle report carried no prompt_len metrics.
The cause is the same HTTP-200-then-broken-stream condition as issue #531: when
an SSE body breaks partway (here, a Content-Length larger than the bytes the
server actually sends, which makes aiohttp raise ClientPayloadError), the
request is recorded as a failure. The failed request's prompt is fully known at
send time, so its prompt_len (input_tokens) belongs in the report regardless of
whether the response completed.

The mock server below mirrors the consumer's client_flags (streaming completion,
`random` dataset, ignore_eos). The single mock-server pattern follows
test_metrics_fallback.py; here it deliberately truncates the stream so every
request lands in the failures bucket.
"""

import http.server
import json
import pathlib
import sys
import threading

import pytest

from utils.benchmark import run_benchmark_minimal

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
MAIN_PY_PATH = PROJECT_ROOT / "inference_perf" / "main.py"

# Token count the mock claims to have processed. The point of the test is the
# client-side prompt_len (from the tokenizer over the generated prompt), not this
# number, but it keeps the mock honest about being a completion server.
MOCK_PROMPT_TOKENS = 7


class TruncatedStreamHandler(http.server.BaseHTTPRequestHandler):
    """Answers /health, then 200s every completion but breaks the SSE stream.

    Promises more bytes via Content-Length than it writes, then closes the
    socket. aiohttp raises ClientPayloadError mid-stream, so inference-perf
    records the request as a failure on an HTTP 200 (the issue-#531 shape).
    """

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/v1/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        # Promise far more than we will send, then bail out -> ClientPayloadError.
        self.send_header("Content-Length", "100000")
        self.end_headers()
        for text in ("Hello", " there"):
            chunk = {"choices": [{"text": text, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()
        # Close with fewer bytes than the advertised Content-Length.
        self.connection.close()

    def log_message(self, format, *args):
        return


def start_mock_server(port: int) -> http.server.HTTPServer:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), TruncatedStreamHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    return server


def _benchmark_config(port: int) -> dict:
    # Mirrors the consumer's client_flags: streaming completion (skip_chat_template
    # -> completion API), the `random` dataset, ignore_eos. Lengths are small so the
    # tokenizer generates prompts quickly; the bug is independent of magnitude.
    return {
        "api": {"type": "completion", "streaming": True},
        "data": {
            "type": "random",
            "input_distribution": {"min": 20, "max": 40, "mean": 30, "std_dev": 5, "total_count": 8},
            "output_distribution": {"min": 20, "max": 40, "mean": 30, "std_dev": 5, "total_count": 8},
        },
        "load": {
            "type": "concurrent",
            "stages": [{"concurrency_level": 4, "num_requests": 8}],
            "num_workers": 1,
        },
        "server": {
            "type": "vllm",
            "model_name": "facebook/opt-125m",
            "base_url": f"http://127.0.0.1:{port}",
            "ignore_eos": True,
        },
        "tokenizer": {"pretrained_model_name_or_path": "facebook/opt-125m"},
        "report": {"request_lifecycle": {"summary": True, "per_stage": True, "per_request": True}},
    }


@pytest.mark.asyncio
async def test_prompt_len_preserved_when_stream_fails():
    """A run whose streaming requests all fail mid-stream must still report prompt_len."""
    server = start_mock_server(18030)
    try:
        result = await run_benchmark_minimal(
            _benchmark_config(18030),
            executable=[sys.executable, str(MAIN_PY_PATH)],
            extra_env={"PYTHONPATH": str(PROJECT_ROOT)},
            timeout_sec=180,
        )

        assert result.success, f"Benchmark process failed: {result.stdout}"
        assert result.reports and "summary_lifecycle_metrics.json" in result.reports, "missing summary report"
        summary = result.reports["summary_lifecycle_metrics.json"]

        # The truncated stream must have driven every request into the failures bucket;
        # otherwise the test is not exercising the failed-request path at all.
        failures = summary["failures"]
        assert failures["count"] == 8, f"expected all 8 requests to fail mid-stream, got {failures}"
        assert summary["successes"]["count"] == 0, "no request should have completed its stream"

        # The regression: a failed request still has a fully-known prompt, so its
        # prompt_len (input_tokens) must be recorded. Before the fix this summarized
        # to all-zeros (or None), which is what the consumer saw as "no prompt_len".
        prompt_len = failures["prompt_len"]
        assert prompt_len is not None, "failures.prompt_len missing entirely"
        assert prompt_len["mean"] > 0, f"failed-request prompt_len not recorded: {prompt_len}"
        # Prompts were generated in [20, 40] tokens, so the mean must land in range.
        assert 20 <= prompt_len["mean"] <= 40, f"prompt_len out of expected range: {prompt_len}"
    finally:
        server.shutdown()
        server.server_close()
