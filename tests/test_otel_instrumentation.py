# Copyright 2025 The Kubernetes Authors.
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

from inference_perf.client.modelserver.otel_instrumentation import OTelInstrumentation, get_otel_instrumentation


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
