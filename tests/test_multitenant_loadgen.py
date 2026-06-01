# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import http.server
import threading
import socket
import sys
import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Generator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference_perf.apis import (
    LazyLoadInferenceAPIData,
    RequestLifecycleMetric,
    InferenceInfo,
    StreamedResponseMetrics,
)
from inference_perf.config import APIConfig, APIType, DataConfig, OTelTraceReplayConfig
from inference_perf.datagen.otel_trace_replay_datagen import OTelTraceReplayDataGenerator
from inference_perf.client.modelserver.openai_client import openAIModelServerClient, openAIModelServerClientSession
from inference_perf.payloads import RequestMetrics, Text

# Bypassed Hub Connection globally
import inference_perf.client.modelserver.openai_client

inference_perf.client.modelserver.openai_client.CustomTokenizer = MagicMock()  # type: ignore[attr-defined]


class ConcreteOpenAIClient(openAIModelServerClient):
    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Chat, APIType.Completion]

    def get_prometheus_metric_metadata(self) -> Any:
        return MagicMock()


# ---------------------------------------------------------------------------
# Mock HTTP Server Fixture
# ---------------------------------------------------------------------------


class MockHandler(http.server.BaseHTTPRequestHandler):
    received_headers: ClassVar[Dict[str, str]] = {}
    received_body: ClassVar[bytes | None] = None

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        MockHandler.received_body = body

        # Capture headers case-sensitively
        MockHandler.received_headers = {k: v for k, v in self.headers.items()}

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        # Return a mock OpenAI-like response
        response = {
            "choices": [{"message": {"role": "assistant", "content": "Mock response content"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        }
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Suppress logging


@pytest.fixture
def mock_server() -> Generator[str, None, None]:
    # Find a free port
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()

    server = http.server.HTTPServer(("127.0.0.1", port), MockHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    yield f"http://127.0.0.1:{port}"

    server.shutdown()
    server.server_close()
    thread.join()
    MockHandler.received_headers = {}
    MockHandler.received_body = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMultiTenantLoadGen:
    def test_otel_attributes_extraction_and_mapping(self, tmp_path: Path) -> None:
        """Verify OTel span attributes extract and map to headers/labels correctly."""
        # 1. Create mock spans
        spans = [
            {
                "span_id": "span_1",
                "trace_id": "trace_1",
                "start_time": "2026-06-01T00:00:00Z",
                "end_time": "2026-06-01T00:00:01Z",
                "name": "chat completions",
                "attributes": {
                    "gen_ai.input.messages": json.dumps([{"role": "user", "content": "hello"}]),
                    "gen_ai.output.text": "hi",
                    "gen_ai.request.model": "test-model",
                    "otel.genai.tenant_id": "tenant-123",
                    "otel.genai.token": "secret-token",
                    # This one is NOT mapped, should stay in extra_attributes but not map to header/label
                    "custom.attribute": "keep-me",
                    "otel.genai.nil_attr": None,  # Test None value fallback
                },
            }
        ]

        # Write mock trace to a temp file
        trace_file = tmp_path / "mock_trace.json"
        trace_file.write_text(json.dumps({"spans": spans}), encoding="utf-8")

        # 2. Configure generator with mapping
        api_config = APIConfig(type=APIType.Chat, streaming=False)
        otel_config = OTelTraceReplayConfig(
            trace_files=[str(trace_file)],
            attribute_to_header_map={
                "otel.genai.token": "x-tenant-token",
                "otel.genai.nil_attr": "x-nil-header",  # Should map to 'default'
                "missing.attr": "x-missing-header",  # Test missing attribute fallback
            },
            attribute_to_label_map={
                "otel.genai.tenant_id": "tenant_id",
                "otel.genai.nil_attr": "nil_label",  # Should map to 'default'
                "missing.attr": "missing_label",  # Test missing attribute fallback
            },
        )

        data_config = DataConfig(type="otel_trace_replay", otel_trace_replay=otel_config)

        # Initialize generator (this will load the trace file and build graph)
        gen = OTelTraceReplayDataGenerator(api_config=api_config, config=data_config, tokenizer=None, num_workers=1)

        # 3. Load lazy data and assert mappings
        lazy_data = LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0)
        api_data = gen.load_lazy_data(lazy_data)

        # Assert headers mapped correctly
        assert api_data.headers is not None
        assert api_data.headers.get("x-tenant-token") == "secret-token"
        assert api_data.headers.get("x-nil-header") == "default"  # None value fallback
        assert api_data.headers.get("x-missing-header") == "default"  # Missing key fallback

        # Assert labels mapped correctly
        assert api_data.labels is not None
        assert api_data.labels.get("tenant_id") == "tenant-123"
        assert api_data.labels.get("nil_label") == "default"  # None value fallback
        assert api_data.labels.get("missing_label") == "default"  # Missing key fallback

    def test_otel_attributes_extraction_and_mapping_missing_attributes(self, tmp_path: Path) -> None:
        """Verify OTel span attributes mapping still injects defaults when span has no extra attributes."""
        # 1. Create mock spans with ONLY excluded standard attributes
        spans = [
            {
                "span_id": "span_1",
                "trace_id": "trace_1",
                "start_time": "2026-06-01T00:00:00Z",
                "end_time": "2026-06-01T00:00:01Z",
                "name": "chat completions",
                "attributes": {
                    "gen_ai.input.messages": json.dumps([{"role": "user", "content": "hello"}]),
                    "gen_ai.output.text": "hi",
                    "gen_ai.request.model": "test-model",
                },
            }
        ]

        # Write mock trace to a temp file
        trace_file = tmp_path / "mock_trace_missing_attrs.json"
        trace_file.write_text(json.dumps({"spans": spans}), encoding="utf-8")

        # 2. Configure generator with mapping
        api_config = APIConfig(type=APIType.Chat, streaming=False)
        otel_config = OTelTraceReplayConfig(
            trace_files=[str(trace_file)],
            attribute_to_header_map={
                "missing.attr": "x-missing-header",
            },
            attribute_to_label_map={
                "missing.attr": "missing_label",
            },
        )

        data_config = DataConfig(type="otel_trace_replay", otel_trace_replay=otel_config)

        # Initialize generator
        gen = OTelTraceReplayDataGenerator(api_config=api_config, config=data_config, tokenizer=None, num_workers=1)

        # 3. Load lazy data and assert mappings still receive defaults
        lazy_data = LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0)
        api_data = gen.load_lazy_data(lazy_data)

        # Assert headers mapped to default
        assert api_data.headers is not None
        assert api_data.headers.get("x-missing-header") == "default"

        # Assert labels mapped to default
        assert api_data.labels is not None
        assert api_data.labels.get("missing_label") == "default"

    @pytest.mark.asyncio
    async def test_client_slo_resolution(self) -> None:
        """Verify that client combined SLO resolution logic resolves thresholds correctly."""
        # Setup mock client and session
        metrics_collector = MagicMock()
        api_config = APIConfig(
            type=APIType.Chat,
            streaming=True,
            headers={"x-slo-ttft-ms": "100", "x-slo-tpot-ms": "20"},  # Global SLOs
        )
        client = ConcreteOpenAIClient(
            metrics_collector=metrics_collector,
            api_config=api_config,
            uri="http://mock",
            model_name="test-model",
            tokenizer_config=None,
            max_tcp_connections=1,
            additional_filters=[],
        )
        session = openAIModelServerClientSession(client)

        # Mock response and info to trigger SLO parsing block
        response_metrics = StreamedResponseMetrics(
            output_tokens=2, output_token_times=[1.0, 1.1], response_chunks=["a", "b"], chunk_times=[1.0, 1.1]
        )
        info = InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=5)), response_metrics=response_metrics)

        # Mock data.process_response to return our info
        data = MagicMock()
        data.headers = {"x-slo-ttft-ms": "50"}  # Request-specific override (TTFT only)
        data.labels = {}
        data.session_id = "sess-1"
        data.get_route.return_value = "/v1/chat/completions"
        data.to_request_body = AsyncMock(return_value={})
        data.process_response = AsyncMock(return_value=info)
        data.process_failure = AsyncMock(return_value=info)

        # Mock aiohttp post
        mock_resp = MagicMock()
        mock_resp.status = 200

        mock_post = MagicMock()
        mock_post.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_post.__aexit__ = AsyncMock(return_value=None)
        # Run process_request with mocked post
        with patch.object(session.session, "post", return_value=mock_post):
            await session.process_request(data, stage_id=1, scheduled_time=0.0)

        # Verify recorded metric
        assert metrics_collector.record_metric.called
        metric = metrics_collector.record_metric.call_args[0][0]
        assert isinstance(metric, RequestLifecycleMetric)

        # SLOs should be resolved:
        # Global TTFT (100ms) overridden by Request-specific TTFT (50ms) -> 0.05s
        # Global TPOT (20ms) kept because request didn't override it -> 0.02s
        assert metric.ttft_slo_sec == 0.05
        assert metric.tpot_slo_sec == 0.02

        await session.close()

    @pytest.mark.asyncio
    async def test_mock_server_headers_transmission(self, mock_server: str) -> None:
        """Verify that mapped headers are physically transmitted on the wire."""
        metrics_collector = MagicMock()
        api_config = APIConfig(type=APIType.Chat, streaming=False)
        client = ConcreteOpenAIClient(
            metrics_collector=metrics_collector,
            api_config=api_config,
            uri=mock_server,  # Point to our mock server
            model_name="test-model",
            tokenizer_config=None,
            max_tcp_connections=1,
            additional_filters=[],
        )
        session = openAIModelServerClientSession(client)

        # Create request data with custom headers
        data = MagicMock()
        data.headers = {"x-llm-d-inference-fairness-id": "tenant-fairness-456", "x-custom-auth": "token-789"}
        data.labels = {}
        data.session_id = "sess-1"
        data.get_route.return_value = "/v1/chat/completions"
        # We need to return a valid body dict for json.dumps
        data.to_request_body = AsyncMock(return_value={"model": "test-model", "messages": []})

        # Mock process_response to avoid tokenizer dependencies in this wire test
        info = InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=5)), response_metrics=None)
        data.process_response = AsyncMock(return_value=info)

        # Run process_request
        await session.process_request(data, stage_id=1, scheduled_time=0.0)

        # Assert headers were received by the server
        headers = MockHandler.received_headers
        assert headers is not None

        # Case-insensitive check
        headers_lower = {k.lower(): v for k, v in headers.items()}
        assert headers_lower.get("x-llm-d-inference-fairness-id") == "tenant-fairness-456"
        assert headers_lower.get("x-custom-auth") == "token-789"
        assert headers_lower.get("content-type") == "application/json"

        await session.close()

    @pytest.mark.asyncio
    async def test_client_slo_resolution_malformed_values(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify that client handles malformed SLO thresholds gracefully by logging warning and keeping None."""
        # Setup mock client and session
        metrics_collector = MagicMock()
        api_config = APIConfig(
            type=APIType.Chat,
            streaming=True,
            headers={"x-slo-ttft-ms": "not-a-float", "x-slo-tpot-ms": "also-not-a-float"},  # Global malformed SLOs
        )
        client = ConcreteOpenAIClient(
            metrics_collector=metrics_collector,
            api_config=api_config,
            uri="http://mock",
            model_name="test-model",
            tokenizer_config=None,
            max_tcp_connections=1,
            additional_filters=[],
        )
        session = openAIModelServerClientSession(client)

        # Mock response and info to trigger SLO parsing block
        response_metrics = StreamedResponseMetrics(
            output_tokens=2, output_token_times=[1.0, 1.1], response_chunks=["a", "b"], chunk_times=[1.0, 1.1]
        )
        info = InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=5)), response_metrics=response_metrics)

        # Mock data
        data = MagicMock()
        data.headers = {"x-slo-ttft-ms": "bad-request-value"}  # Request-specific override (also malformed)
        data.labels = {}
        data.session_id = "sess-1"
        data.get_route.return_value = "/v1/chat/completions"
        data.to_request_body = AsyncMock(return_value={})
        data.process_response = AsyncMock(return_value=info)
        data.process_failure = AsyncMock(return_value=info)

        # Mock aiohttp post
        mock_resp = MagicMock()
        mock_resp.status = 200

        mock_post = MagicMock()
        mock_post.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_post.__aexit__ = AsyncMock(return_value=None)
        # Run process_request while capturing warnings and mocking post
        with caplog.at_level(logging.WARNING), patch.object(session.session, "post", return_value=mock_post):
            await session.process_request(data, stage_id=1, scheduled_time=0.0)

        # Verify recorded metric
        assert metrics_collector.record_metric.called
        metric = metrics_collector.record_metric.call_args[0][0]
        assert isinstance(metric, RequestLifecycleMetric)

        # SLOs should be None
        assert metric.ttft_slo_sec is None
        assert metric.tpot_slo_sec is None

        # Assert warnings were logged
        warnings = [record.message for record in caplog.records if record.levelname == "WARNING"]
        assert any("Invalid TTFT SLO value: bad-request-value" in w for w in warnings)
        assert any("Invalid TPOT SLO value: also-not-a-float" in w for w in warnings)

        await session.close()

    @pytest.mark.asyncio
    async def test_mock_server_headers_transmission_casing_overwrite(self, mock_server: str) -> None:
        """Verify that request headers overwrite global headers case-insensitively on the wire without duplicates."""
        metrics_collector = MagicMock()
        # Configure global headers with mixed casing
        api_config = APIConfig(type=APIType.Chat, streaming=False, headers={"X-LLM-D-Inference-Fairness-ID": "global-val"})
        client = ConcreteOpenAIClient(
            metrics_collector=metrics_collector,
            api_config=api_config,
            uri=mock_server,
            model_name="test-model",
            tokenizer_config=None,
            max_tcp_connections=1,
            additional_filters=[],
        )
        session = openAIModelServerClientSession(client)

        # Configure request headers with different casing
        data = MagicMock()
        data.headers = {"x-llm-d-inference-fairness-id": "request-override"}
        data.labels = {}
        data.session_id = "sess-1"
        data.get_route.return_value = "/v1/chat/completions"
        data.to_request_body = AsyncMock(return_value={"model": "test-model", "messages": []})

        info = InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=5)), response_metrics=None)
        data.process_response = AsyncMock(return_value=info)

        # Run process_request
        await session.process_request(data, stage_id=1, scheduled_time=0.0)

        # Assert headers received
        headers = MockHandler.received_headers
        assert headers is not None

        # Count how many times the fairness-id header appears (case-insensitively)
        fairness_keys = [k for k in headers.keys() if k.lower() == "x-llm-d-inference-fairness-id"]

        # We expect exactly one matching key
        assert len(fairness_keys) == 1, f"Expected exactly one fairness-id header, but got: {fairness_keys}"

        # We expect the value to be the request-specific override
        target_key = fairness_keys[0]
        assert headers[target_key] == "request-override"

        # The key itself should have the casing of the request-specific header
        assert target_key == "x-llm-d-inference-fairness-id"

        await session.close()
