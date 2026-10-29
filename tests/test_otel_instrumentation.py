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

"""Tests for OpenTelemetry instrumentation."""

import base64
from multiprocessing.queues import SimpleQueue
from typing import Any, Optional

import pytest

from inference_perf.client.modelserver.otel_instrumentation import (
    OTelInstrumentation,
    get_otel_instrumentation,
    summarize_media,
)
from inference_perf.utils.mp_context import MP_CONTEXT


def test_otel_instrumentation_init() -> None:
    """Test OTEL instrumentation initialization."""
    otel = OTelInstrumentation(service_name="test-service", enabled=True)
    assert otel.service_name == "test-service"
    # Should be enabled if packages are available, otherwise disabled
    assert isinstance(otel.enabled, bool)


def test_otel_instrumentation_disabled() -> None:
    """Test OTEL instrumentation when disabled."""
    otel = OTelInstrumentation(service_name="test-service", enabled=False)
    assert otel.service_name == "test-service"
    assert otel.enabled is False
    assert otel.tracer is None


def test_trace_llm_request_disabled() -> None:
    """Test tracing when OTEL is disabled."""
    otel = OTelInstrumentation(service_name="test-service", enabled=False)

    with otel.trace_llm_request(
        operation_name="chat.completions",
        model_name="test-model",
        request_data={"max_tokens": 100},
    ) as span:
        assert span is None


def test_record_response_metrics_disabled() -> None:
    """Test recording metrics when OTEL is disabled."""
    otel = OTelInstrumentation(service_name="test-service", enabled=False)

    # Should not raise an error
    otel.record_response_metrics(
        span=None,
        response_info={"prompt_tokens": 10, "completion_tokens": 20},
        error=None,
    )


def test_get_otel_instrumentation() -> None:
    """Test getting global OTEL instrumentation instance."""
    otel1 = get_otel_instrumentation(service_name="test-service", enabled=True)
    otel2 = get_otel_instrumentation(service_name="test-service", enabled=True)

    # Should return the same instance
    assert otel1 is otel2


def test_trace_llm_request_with_data() -> None:
    """Test tracing with request data."""
    otel = OTelInstrumentation(service_name="test-service", enabled=True)

    request_data = {
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": True,
    }

    with otel.trace_llm_request(
        operation_name="chat.completions",
        model_name="test-model",
        request_data=request_data,
    ) as span:
        # Span may be None if OTEL packages are not installed
        if span is not None:
            # If span exists, it should have the operation name set
            pass


def test_record_response_metrics_with_data() -> None:
    """Test recording response metrics with data."""
    otel = OTelInstrumentation(service_name="test-service", enabled=True)

    response_info = {
        "prompt_tokens": 50,
        "completion_tokens": 100,
        "time_to_first_token": 0.5,
        "time_per_output_token": 0.01,
        "total_latency": 1.5,
        "finish_reason": "stop",
        "response_id": "test-123",
    }

    with otel.trace_llm_request(
        operation_name="chat.completions",
        model_name="test-model",
    ) as span:
        # Should not raise an error
        otel.record_response_metrics(
            span=span,
            response_info=response_info,
            error=None,
        )


def test_record_response_metrics_with_error() -> None:
    """Test recording response metrics with error."""
    otel = OTelInstrumentation(service_name="test-service", enabled=True)

    with otel.trace_llm_request(
        operation_name="chat.completions",
        model_name="test-model",
    ) as span:
        # Should not raise an error
        otel.record_response_metrics(
            span=span,
            response_info=None,
            error="Connection timeout",
        )


def test_summarize_media_replaces_inline_payloads_and_keeps_everything_else() -> None:
    """Spans carry a size and hash summary of each inline image or audio clip, never its bytes."""
    png = base64.b64encode(b"\x89PNG fake image bytes").decode()
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}},
            ],
        },
        {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}]},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]},
    ]
    summary = summarize_media(messages)

    assert summary[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,[21 bytes, sha256:")
    assert png not in str(summary)
    assert summary[0]["content"][0] == {"type": "text", "text": "hi"}
    assert summary[1]["content"][0]["input_audio"]["format"] == "wav"
    assert summary[1]["content"][0]["input_audio"]["data"].startswith("[3 bytes, sha256:")
    assert summary[2] == messages[2]
    assert messages[0]["content"][1]["image_url"]["url"].endswith(png)  # the request itself is untouched


# Runs in the worker: opens one span and reports whether it is recording.
def _span_records_in_worker(otel: OTelInstrumentation, result: "SimpleQueue[bool]") -> None:
    with otel.trace_llm_request(operation_name="completions", model_name="test-model") as span:
        result.put(span is not None and span.is_recording())


# An enabled instance goes to a forkserver worker, once with the console exporter and once with OTLP.
# The worker reports True: the instance pickled and the worker built a live tracer of its own.
@pytest.mark.parametrize("otlp_endpoint", [None, "http://localhost:4317"])
def test_enabled_instrumentation_traces_in_forkserver_worker(
    monkeypatch: pytest.MonkeyPatch, otlp_endpoint: Optional[str]
) -> None:
    pytest.importorskip("opentelemetry.sdk.trace")
    if otlp_endpoint is None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    else:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", otlp_endpoint)
    otel = OTelInstrumentation(service_name="test-service", enabled=True)

    result: "SimpleQueue[bool]" = MP_CONTEXT.SimpleQueue()
    worker = MP_CONTEXT.Process(target=_span_records_in_worker, args=(otel, result))  # type: ignore[attr-defined]
    worker.start()
    worker.join(timeout=60)

    assert worker.exitcode == 0
    assert result.get() is True
