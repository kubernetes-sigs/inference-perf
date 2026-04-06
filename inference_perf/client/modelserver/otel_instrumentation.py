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

"""
OpenTelemetry instrumentation for LLM API calls.

This module provides standard GenAI OTEL instrumentation following the
OpenTelemetry Semantic Conventions for GenAI operations.

Environment Variables:
    OTEL_TRACES_ENABLED: Set to "true" to enable OTEL tracing (default: false)
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint for exporting traces (e.g., "http://localhost:4317")
    OTEL_SERVICE_NAME: Service name for tracing (default: "inference-perf")
    OTEL_TRACE_PER_STAGE: Set to "true" to create one trace per stage instead of per session (default: false)
"""

import logging
import os
from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING, cast
from contextlib import contextmanager

if TYPE_CHECKING:
    # Local stub for type checking - avoids "Any" issues with --ignore-missing-imports
    class IdGeneratorBase:
        """Stub base class for type checking."""

        def generate_span_id(self) -> int: ...
        def generate_trace_id(self) -> int: ...

    OTEL_AVAILABLE: bool
    # Import other types for type checking
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter
    from opentelemetry.sdk.trace.id_generator import IdGenerator
    from opentelemetry.semconv_ai import SpanAttributes
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
else:
    try:
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        from opentelemetry.sdk.trace.id_generator import IdGenerator as IdGeneratorBase
        from opentelemetry.semconv_ai import SpanAttributes
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

        OTEL_AVAILABLE = True
    except ImportError:
        IdGeneratorBase = object
        OTEL_AVAILABLE = False
        trace = None  # type: ignore[assignment]
        SpanAttributes = None  # type: ignore[assignment]
        TraceContextTextMapPropagator = None  # type: ignore[assignment]
        Status = None  # type: ignore[assignment]
        StatusCode = None  # type: ignore[assignment]
        TracerProvider = None  # type: ignore[assignment]
        ConsoleSpanExporter = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class CryptographicIdGenerator(IdGeneratorBase):
    """
    Custom ID generator that uses os.urandom() for cryptographically secure random IDs.

    This bypasses Python's random module completely, ensuring unique Trace and Span IDs
    even when the random seed is fixed for reproducibility in other parts of the application.

    OpenTelemetry spec requires that IDs cannot be 0, so we regenerate if we get 0.
    """

    def generate_span_id(self) -> int:
        """Generate a random 64-bit span ID using os.urandom()."""
        # Generate 8 random bytes and convert to a 64-bit integer
        # Ensure the ID is never 0 (OpenTelemetry spec requirement)
        span_id = 0
        while span_id == 0:
            span_id = int.from_bytes(os.urandom(8), byteorder="big")
        return span_id

    def generate_trace_id(self) -> int:
        """Generate a random 128-bit trace ID using os.urandom()."""
        # Generate 16 random bytes and convert to a 128-bit integer
        # Ensure the ID is never 0 (OpenTelemetry spec requirement)
        trace_id = 0
        while trace_id == 0:
            trace_id = int.from_bytes(os.urandom(16), byteorder="big")
        return trace_id


class OTelInstrumentation:
    """
    OpenTelemetry instrumentation for LLM API calls.

    Provides tracing capabilities following GenAI semantic conventions.
    """

    def __init__(self, service_name: Optional[str] = None, enabled: Optional[bool] = None):
        """
        Initialize OTEL instrumentation.

        Args:
            service_name: Service name for tracing (overrides OTEL_SERVICE_NAME env var)
            enabled: Whether to enable tracing (overrides OTEL_TRACES_ENABLED env var)

        Reads configuration from environment variables if not provided:
        - OTEL_TRACES_ENABLED: Set to "true" to enable tracing (default: false)
        - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (e.g., "http://localhost:4317")
        - OTEL_SERVICE_NAME: Service name (default: "inference-perf")
        - OTEL_TRACE_PER_STAGE: Set to "true" to create one trace per stage instead of per session (default: false)
        """
        # Read configuration from environment variables or use provided values
        if enabled is None:
            enabled = os.getenv("OTEL_TRACES_ENABLED", "false").lower() == "true"
        if service_name is None:
            service_name = os.getenv("OTEL_SERVICE_NAME", "inference-perf")

        self.service_name = service_name
        self.otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        self.trace_per_stage = os.getenv("OTEL_TRACE_PER_STAGE", "false").lower() == "true"

        self.enabled = enabled and OTEL_AVAILABLE
        self.tracer: Optional[Any] = None

        if not OTEL_AVAILABLE and enabled:
            logger.warning(
                "OpenTelemetry packages not installed. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk "
                "opentelemetry-exporter-otlp-proto-grpc opentelemetry-semantic-conventions-ai"
            )
            self.enabled = False

        if self.enabled:
            self._setup_tracer()
            logger.info(f"OTEL tracing enabled for service: {self.service_name}")

    def _setup_tracer(self) -> None:
        """Set up the OpenTelemetry tracer."""
        if not OTEL_AVAILABLE:
            return

        # Get or create tracer provider
        provider = trace.get_tracer_provider()

        # If no provider is set, create a default one
        if not hasattr(provider, "add_span_processor"):
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor
            from opentelemetry.sdk.resources import Resource, SERVICE_NAME

            # Create resource with service name
            resource = Resource(attributes={SERVICE_NAME: self.service_name})

            # Use custom ID generator to ensure unique IDs even with fixed random seed
            id_generator = CryptographicIdGenerator()
            # Cast needed because our stub doesn't match the real IdGenerator type
            provider = TracerProvider(resource=resource, id_generator=cast("IdGenerator", id_generator))
            trace.set_tracer_provider(provider)

            logger.info("Using CryptographicIdGenerator for unique trace/span IDs")

            # Configure exporter based on otlp_endpoint
            if self.otlp_endpoint:
                # Use OTLP exporter for Jaeger/other backends
                try:
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

                    otlp_exporter = OTLPSpanExporter(
                        endpoint=self.otlp_endpoint,
                        insecure=True,  # Use insecure connection by default
                    )
                    # Use BatchSpanProcessor with shorter intervals for better reliability
                    batch_processor = BatchSpanProcessor(
                        otlp_exporter,
                        max_queue_size=2048,
                        schedule_delay_millis=1000,  # Export every 1 second
                        max_export_batch_size=512,
                    )
                    provider.add_span_processor(batch_processor)
                    logger.info(f"Created OTEL tracer provider with OTLP exporter to {self.otlp_endpoint}")
                except ImportError:
                    logger.warning(
                        "OTLP exporter not available. Install with: pip install opentelemetry-exporter-otlp-proto-grpc"
                    )
                    logger.info("Falling back to console exporter")
                    console_exporter = ConsoleSpanExporter()
                    provider.add_span_processor(SimpleSpanProcessor(console_exporter))
            else:
                # Use console exporter for debugging
                console_exporter = ConsoleSpanExporter()
                provider.add_span_processor(SimpleSpanProcessor(console_exporter))
                logger.info("Created OTEL tracer provider with console exporter")

        self.tracer = trace.get_tracer(self.service_name)
        self._provider = provider  # Store provider for shutdown
        logger.info(f"OTEL instrumentation enabled for service: {self.service_name}")

    def shutdown(self) -> None:
        """Shutdown the tracer provider and flush all pending spans."""
        if hasattr(self, "_provider") and self._provider:
            try:
                self._provider.force_flush(timeout_millis=5000)
                self._provider.shutdown()
                logger.info("OTEL tracer provider shutdown successfully")
            except Exception as e:
                logger.warning(f"Error during OTEL shutdown: {e}")

    @contextmanager
    def trace_llm_request(
        self,
        operation_name: str,
        model_name: str,
        request_data: Optional[Dict[str, Any]] = None,
        parent_context: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Context manager for tracing LLM requests.

        Args:
            operation_name: Name of the operation (e.g., "chat.completions", "completions")
            model_name: Name of the model being used
            request_data: Optional request data for additional context
            parent_context: Optional serialized parent context for linking spans

        Yields:
            Span object if OTEL is enabled, None otherwise
        """
        if not self.enabled or self.tracer is None:
            yield None
            return

        # Extract parent context if provided
        ctx = None
        if parent_context and OTEL_AVAILABLE and TraceContextTextMapPropagator is not None:
            try:
                propagator = TraceContextTextMapPropagator()
                ctx = propagator.extract(parent_context)
            except Exception as e:
                logger.warning(f"Failed to extract parent context: {e}")
                ctx = None

        with self.tracer.start_as_current_span(f"llm.{operation_name}", kind=trace.SpanKind.CLIENT, context=ctx) as span:
            try:
                # Set standard GenAI attributes using semantic conventions
                if OTEL_AVAILABLE and SpanAttributes is not None:
                    # Core GenAI attributes
                    span.set_attribute(SpanAttributes.LLM_SYSTEM, "openai_compatible")
                    span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model_name)
                    # Map operation name to GenAI semantic convention operation types
                    operation_type = "chat" if operation_name == "chat.completions" else "text_completion"
                    span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, operation_type)

                    # Add request-specific attributes if available
                    if request_data:
                        if "max_tokens" in request_data:
                            span.set_attribute(SpanAttributes.LLM_REQUEST_MAX_TOKENS, request_data["max_tokens"])
                        if "temperature" in request_data:
                            span.set_attribute(SpanAttributes.LLM_REQUEST_TEMPERATURE, request_data["temperature"])
                        if "top_p" in request_data:
                            span.set_attribute(SpanAttributes.LLM_REQUEST_TOP_P, request_data["top_p"])
                        if "stream" in request_data:
                            span.set_attribute(SpanAttributes.LLM_IS_STREAMING, request_data["stream"])

                yield span

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def record_response_metrics(
        self,
        span: Optional[Any],
        response_info: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Record response metrics on the span.

        Args:
            span: The OTEL span to record metrics on
            response_info: Response information including token counts, latency, etc.
            error: Error message if the request failed
        """
        if not self.enabled or span is None:
            return

        try:
            if error:
                span.set_status(Status(StatusCode.ERROR, error))
                # Note: There's no standard attribute for error in the semantic conventions yet
                span.set_attribute("error.message", error)
            else:
                span.set_status(Status(StatusCode.OK))

            if response_info and OTEL_AVAILABLE and SpanAttributes is not None:
                # Token usage
                if "prompt_tokens" in response_info:
                    span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, response_info["prompt_tokens"])
                if "completion_tokens" in response_info:
                    span.set_attribute(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, response_info["completion_tokens"])

                # Calculate total tokens if both are available
                if "prompt_tokens" in response_info and "completion_tokens" in response_info:
                    total_tokens = response_info["prompt_tokens"] + response_info["completion_tokens"]
                    span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)

                # Latency metrics (custom attributes as they're not in standard semantic conventions)
                if "time_to_first_token" in response_info:
                    span.set_attribute("gen_ai.response.time_to_first_token", response_info["time_to_first_token"])
                if "time_per_output_token" in response_info:
                    span.set_attribute("gen_ai.response.time_per_output_token", response_info["time_per_output_token"])
                if "total_latency" in response_info:
                    span.set_attribute("gen_ai.response.total_latency", response_info["total_latency"])

                # Finish reason
                if "finish_reason" in response_info:
                    span.set_attribute(SpanAttributes.LLM_RESPONSE_FINISH_REASON, response_info["finish_reason"])

                # Input - either messages (for chat) or prompt (for completions)
                if "input_messages" in response_info:
                    # Chat completion - gen_ai.input.messages as JSON string
                    span.set_attribute("gen_ai.input.messages", response_info["input_messages"])
                elif "input_prompt" in response_info:
                    # Text completion - gen_ai.prompt as string
                    span.set_attribute("gen_ai.prompt", response_info["input_prompt"])

                # Output text (gen_ai.output.text)
                if "output_text" in response_info:
                    span.set_attribute("gen_ai.output.text", response_info["output_text"])

                # Response ID (custom attribute)
                if "response_id" in response_info:
                    span.set_attribute("gen_ai.response.id", response_info["response_id"])

        except Exception as e:
            logger.warning(f"Failed to record response metrics: {e}")

    def start_session_span(
        self,
        session_id: str,
        session_info: Optional[Dict[str, Any]] = None,
        parent_context: Optional[Dict[str, str]] = None,
    ) -> Tuple[Optional[Any], Optional[Dict[str, str]]]:
        """
        Start a session-level span and return both the span and its serialized context.

        Args:
            session_id: Unique identifier for the session
            session_info: Optional metadata about the session (num_graph_events, file_path, etc.)
            parent_context: Optional serialized parent context for linking spans

        Returns:
            Tuple of (span, context_dict) where:
            - span: The OTEL span object (or None if OTEL disabled)
            - context_dict: Serialized context for cross-process propagation (or None)
        """
        if not self.enabled or self.tracer is None:
            return None, None

        # Extract parent context if provided
        ctx = None
        if parent_context and OTEL_AVAILABLE and TraceContextTextMapPropagator is not None:
            try:
                propagator = TraceContextTextMapPropagator()
                ctx = propagator.extract(parent_context)
            except Exception as e:
                logger.warning(f"Failed to extract parent context: {e}")
                ctx = None

        # Start the session span with optional parent context
        if ctx:
            span = self.tracer.start_span(f"session.{session_id}", kind=trace.SpanKind.INTERNAL, context=ctx)
        else:
            span = self.tracer.start_span(f"session.{session_id}", kind=trace.SpanKind.INTERNAL)

        # Set session attributes
        if session_info:
            if "num_graph_events" in session_info:
                span.set_attribute("session.num_events", session_info["num_graph_events"])
            if "file_path" in session_info:
                span.set_attribute("session.file_path", session_info["file_path"])

        span.set_attribute("session.id", session_id)

        # Serialize the span context for cross-process propagation
        context_dict: Dict[str, str] = {}
        if OTEL_AVAILABLE and TraceContextTextMapPropagator is not None:
            propagator = TraceContextTextMapPropagator()
            # Create a context with this span as current
            ctx = trace.set_span_in_context(span)
            propagator.inject(context_dict, context=ctx)

        return span, context_dict

    def start_stage_span(
        self,
        stage_id: int,
        stage_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Any], Optional[Dict[str, str]]]:
        """
        Start a stage-level span and return both the span and its serialized context.

        Args:
            stage_id: Unique identifier for the stage
            stage_info: Optional metadata about the stage (num_sessions, duration, etc.)

        Returns:
            Tuple of (span, context_dict) where:
            - span: The OTEL span object (or None if OTEL disabled)
            - context_dict: Serialized context for cross-process propagation (or None)
        """
        if not self.enabled or self.tracer is None:
            return None, None

        # Start the stage span
        span = self.tracer.start_span(f"stage.{stage_id}", kind=trace.SpanKind.INTERNAL)

        # Set stage attributes
        span.set_attribute("stage.id", stage_id)
        if stage_info:
            if "num_sessions" in stage_info:
                span.set_attribute("stage.num_sessions", stage_info["num_sessions"])
            if "duration" in stage_info:
                span.set_attribute("stage.duration", stage_info["duration"])
            if "rate" in stage_info:
                span.set_attribute("stage.rate", stage_info["rate"])
            if "concurrent_sessions" in stage_info:
                span.set_attribute("stage.concurrent_sessions", stage_info["concurrent_sessions"])

        # Serialize the span context for cross-process propagation
        context_dict: Dict[str, str] = {}
        if OTEL_AVAILABLE and TraceContextTextMapPropagator is not None:
            propagator = TraceContextTextMapPropagator()
            # Create a context with this span as current
            ctx = trace.set_span_in_context(span)
            propagator.inject(context_dict, context=ctx)

        return span, context_dict

    def end_session_span(self, span: Optional[Any], error: Optional[str] = None) -> None:
        """
        End a session span.

        Args:
            span: The OTEL span to end
            error: Optional error message if session failed
        """
        if not self.enabled or span is None:
            return

        try:
            if error:
                span.set_status(Status(StatusCode.ERROR, error))
            else:
                span.set_status(Status(StatusCode.OK))
            span.end()
        except Exception as e:
            logger.warning(f"Failed to end session span: {e}")

    def end_stage_span(self, span: Optional[Any], error: Optional[str] = None) -> None:
        """
        End a stage span.

        Args:
            span: The OTEL span to end
            error: Optional error message if stage failed
        """
        if not self.enabled or span is None:
            return

        try:
            if error:
                span.set_status(Status(StatusCode.ERROR, error))
            else:
                span.set_status(Status(StatusCode.OK))
            span.end()
        except Exception as e:
            logger.warning(f"Failed to end stage span: {e}")


# Global instance
_global_instrumentation: Optional[OTelInstrumentation] = None


def get_otel_instrumentation(service_name: Optional[str] = None, enabled: Optional[bool] = None) -> OTelInstrumentation:
    """
    Get or create the global OTEL instrumentation instance.

    Args:
        service_name: Service name for tracing (overrides OTEL_SERVICE_NAME env var)
        enabled: Whether to enable tracing (overrides OTEL_TRACES_ENABLED env var)

    Configuration is read from environment variables if not provided:
    - OTEL_TRACES_ENABLED: Set to "true" to enable tracing
    - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (e.g., "http://localhost:4317")
    - OTEL_SERVICE_NAME: Service name (default: "inference-perf")

    Returns:
        OTelInstrumentation instance
    """
    global _global_instrumentation

    if _global_instrumentation is None:
        _global_instrumentation = OTelInstrumentation(service_name=service_name, enabled=enabled)

    return _global_instrumentation
