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

import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any
import pytest
from inference_perf.detector.detector import detect_server_type
from inference_perf.config import ModelServerType


class MockHandler(BaseHTTPRequestHandler):
    mock_responses: dict[str, tuple[int, str]] = {}

    def do_GET(self) -> None:
        if self.path in self.mock_responses:
            code, content = self.mock_responses[self.path]
            self.send_response(code)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress logging to keep test output clean
        return


@pytest.fixture
def mock_server() -> Any:
    server = HTTPServer(("127.0.0.1", 0), MockHandler)
    ip = str(server.server_address[0])
    port = server.server_address[1]
    url = f"http://{ip}:{port}"

    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    yield url, MockHandler.mock_responses

    server.shutdown()
    thread.join()
    MockHandler.mock_responses.clear()


def test_integration_detect_vllm_via_metrics(mock_server: Any) -> None:
    url, mock_responses = mock_server
    mock_responses["/metrics"] = (200, "vllm:num_requests_waiting 0")

    server_type = detect_server_type(url)
    assert server_type == ModelServerType.VLLM


def test_integration_detect_sglang_via_metrics(mock_server: Any) -> None:
    url, mock_responses = mock_server
    mock_responses["/metrics"] = (200, "sglang:num_queue_reqs 0")

    server_type = detect_server_type(url)
    assert server_type == ModelServerType.SGLANG


def test_integration_detect_tgi_via_metrics(mock_server: Any) -> None:
    url, mock_responses = mock_server
    mock_responses["/metrics"] = (200, "tgi_queue_size 0")

    server_type = detect_server_type(url)
    assert server_type == ModelServerType.TGI


def test_integration_detect_vllm_via_v1_models(mock_server: Any) -> None:
    url, mock_responses = mock_server
    mock_responses["/metrics"] = (404, "")
    mock_responses["/v1/models"] = (200, "{}")

    server_type = detect_server_type(url)
    assert server_type == ModelServerType.VLLM


def test_integration_detect_fallback_to_vllm(mock_server: Any) -> None:
    url, mock_responses = mock_server
    # All endpoints return 404
    mock_responses["/metrics"] = (404, "")
    mock_responses["/v1/models"] = (404, "")
    mock_responses["/generate"] = (404, "")

    server_type = detect_server_type(url)
    # Fallback is VLLM
    assert server_type == ModelServerType.VLLM


def test_integration_detect_api_type_chat(mock_server: Any) -> None:
    from inference_perf.config import APIType
    from inference_perf.detector.detector import detect_api_type

    url, mock_responses = mock_server
    mock_responses["/openapi.json"] = (200, '{"paths": {"/v1/chat/completions": {}}}')

    api_type = detect_api_type(url)
    assert api_type == APIType.Chat


def test_integration_detect_api_type_completion(mock_server: Any) -> None:
    from inference_perf.config import APIType
    from inference_perf.detector.detector import detect_api_type

    url, mock_responses = mock_server
    mock_responses["/openapi.json"] = (200, '{"paths": {"/v1/completions": {}}}')

    api_type = detect_api_type(url)
    assert api_type == APIType.Completion


def test_integration_detect_api_type_fallback(mock_server: Any) -> None:
    from inference_perf.config import APIType
    from inference_perf.detector.detector import detect_api_type

    url, mock_responses = mock_server
    mock_responses["/openapi.json"] = (404, "")

    api_type = detect_api_type(url)
    assert api_type == APIType.Completion


def test_integration_detect_model_name_success(mock_server: Any) -> None:
    from inference_perf.detector.detector import detect_model_name

    url, mock_responses = mock_server
    mock_responses["/v1/models"] = (
        200,
        '{"object": "list", "data": [{"id": "test-model", "object": "model"}]}',
    )

    model_name = detect_model_name(url)
    assert model_name == "test-model"


def test_integration_detect_model_name_fallback(mock_server: Any) -> None:
    from inference_perf.detector.detector import detect_model_name

    url, mock_responses = mock_server
    mock_responses["/v1/models"] = (404, "")

    model_name = detect_model_name(url)
    assert model_name is None
