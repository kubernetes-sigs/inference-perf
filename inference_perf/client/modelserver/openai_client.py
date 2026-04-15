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

from abc import abstractmethod
from inference_perf.client.requestdatacollector import RequestDataCollector
from inference_perf.config import APIConfig, APIType, CustomTokenizerConfig, MultiLoRAConfig
from inference_perf.apis import InferenceAPIData, InferenceInfo, RequestLifecycleMetric, ErrorResponseInfo
from inference_perf.utils import CustomTokenizer
from .base import ModelServerClient, ModelServerClientSession, PrometheusMetricMetadata
from .otel_instrumentation import get_otel_instrumentation
from typing import List, Optional, Any, Dict
import aiohttp
import asyncio
import json
import time
import logging
import requests
import ssl


logger = logging.getLogger(__name__)


class openAIModelServerClient(ModelServerClient):
    _session: "Optional[openAIModelServerClientSession]" = None
    _session_lock = asyncio.Lock()

    def __init__(
        self,
        metrics_collector: RequestDataCollector,
        api_config: APIConfig,
        uri: str,
        model_name: Optional[str],
        tokenizer_config: Optional[CustomTokenizerConfig],
        max_tcp_connections: int,
        additional_filters: List[str],
        ignore_eos: bool = True,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        cert_path: Optional[str] = None,
        key_path: Optional[str] = None,
        lora_config: Optional[List[MultiLoRAConfig]] = None,
    ) -> None:
        super().__init__(api_config, timeout)
        self.uri = uri
        self.max_completion_tokens = 30  # default to use when not set at the request level
        self.ignore_eos = ignore_eos
        self.metrics_collector = metrics_collector
        self.max_tcp_connections = max_tcp_connections
        self.additional_filters = additional_filters
        self.api_key = api_key
        self.cert_path = cert_path
        self.key_path = key_path
        self.lora_config = lora_config

        # Initialize OTEL instrumentation (configured via environment variables)
        self.otel = get_otel_instrumentation()

        if model_name is None:
            supported_models = self.get_supported_models()
            if not supported_models:
                logger.error("No supported models found")
                raise Exception("openAI client init failed, no model_name could be found")
            self.model_name = supported_models[0].get("id")
            logger.info(f"Inferred model {self.model_name}")
            if len(supported_models) > 1:
                logger.warning(f"More than one supported model found {supported_models}, selecting {self.model_name}")
        else:
            self.model_name = model_name

        if self.lora_config is not None:
            supported_models = self.get_supported_models()
            supported_model_names = set()
            for model in supported_models:
                supported_model_names.add(model.get("id"))
            lora_adapters = [config.name for config in self.lora_config]
            for adapter in lora_adapters:
                if adapter not in supported_model_names:
                    raise ValueError(f"LoRA adapter {adapter} not found in model server's available models")

        if tokenizer_config and not tokenizer_config.pretrained_model_name_or_path:
            tokenizer_config.pretrained_model_name_or_path = self.model_name
        elif not tokenizer_config:
            tokenizer_config = CustomTokenizerConfig(pretrained_model_name_or_path=self.model_name)
        self.tokenizer = CustomTokenizer(tokenizer_config)

    def new_session(self) -> "ModelServerClientSession":
        return openAIModelServerClientSession(self)

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        """
        Create an internal client session if not already, then use that to
        process the request.
        """
        session: openAIModelServerClientSession
        # ensure session is only created once.
        async with self._session_lock:
            if self._session is None:
                self._session = openAIModelServerClientSession(self)
            session = self._session
        await session.process_request(data, stage_id, scheduled_time, lora_adapter)

    async def close(self) -> None:
        """Close the internal session created by process_request, if any."""
        if self._session is not None:
            await self._session.close()
            self._session = None

        # Shutdown OTEL instrumentation to flush pending spans
        if self.otel:
            self.otel.shutdown()

    def get_supported_apis(self) -> List[APIType]:
        return []

    @abstractmethod
    def get_prometheus_metric_metadata(self) -> PrometheusMetricMetadata:
        raise NotImplementedError

    def get_supported_models(self) -> List[dict[str, Any]]:
        try:
            response = requests.get(f"{self.uri}/v1/models")
            response.raise_for_status()
            data = response.json()
            if "data" in data and isinstance(data["data"], list):
                return data["data"]
            else:
                return []
        except Exception as e:
            logger.error(f"Got exception retrieving supported models {e}")
            return []


class openAIModelServerClientSession(ModelServerClientSession):
    def __init__(self, client: openAIModelServerClient):
        timeout = aiohttp.ClientTimeout(total=client.timeout) if client.timeout else aiohttp.helpers.sentinel
        connector = None
        if client.cert_path and client.key_path:
            ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)  # Use system trust store
            ssl_context.load_cert_chain(certfile=client.cert_path, keyfile=client.key_path)
            connector = aiohttp.TCPConnector(limit=client.max_tcp_connections, ssl=ssl_context)
        else:
            connector = aiohttp.TCPConnector(limit=client.max_tcp_connections)

        self.client = client
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)

    def _get_session_otel_context(self, data: InferenceAPIData) -> Optional[Dict[str, str]]:
        """Get session OTEL context if available (for OTel trace replay)."""

        if hasattr(data, "otel_context") and data.otel_context is not None:
            return data.otel_context

        return None

    def _record_otel_metrics(
        self,
        span: Any,
        data: InferenceAPIData,
        response: Optional[aiohttp.ClientResponse],
        response_info: Optional[InferenceInfo],
        response_content: str,
        error: Optional[ErrorResponseInfo],
        start_time: float,
        end_time: float,
    ) -> None:
        """Record OTEL metrics for the request."""
        if response_info:
            otel_response_info = {
                "prompt_tokens": response_info.input_tokens,
                "completion_tokens": response_info.output_tokens,
                "total_latency": end_time - start_time,
            }

            # Calculate TTFT if token times are available
            if response_info.output_token_times and len(response_info.output_token_times) > 0:
                ttft = response_info.output_token_times[0] - start_time
                otel_response_info["time_to_first_token"] = ttft

                # Calculate average TPOT if we have multiple tokens
                if len(response_info.output_token_times) > 1:
                    total_decode_time = response_info.output_token_times[-1] - response_info.output_token_times[0]
                    num_decode_tokens = len(response_info.output_token_times) - 1
                    tpot = total_decode_time / num_decode_tokens if num_decode_tokens > 0 else 0
                    otel_response_info["time_per_output_token"] = tpot

            # Add finish reason from extra_info if available
            if "finish_reason" in response_info.extra_info:
                otel_response_info["finish_reason"] = response_info.extra_info["finish_reason"]

            # Extract input and output following GenAI semantic conventions
            try:
                # Extract input based on request type
                if hasattr(data, "messages"):
                    # Chat completion - serialize messages as JSON string (gen_ai.input.messages)
                    input_messages = [{"role": msg.role, "content": msg.content} for msg in data.messages]
                    otel_response_info["input_messages"] = json.dumps(input_messages)  # type: ignore[assignment]
                elif hasattr(data, "prompt"):
                    # Text completion - store as prompt string (gen_ai.prompt)
                    otel_response_info["input_prompt"] = data.prompt

                # Extract output text (gen_ai.output.text)
                if response and response.status == 200 and response_content:
                    response_json = json.loads(response_content)
                    choices = response_json.get("choices", [])
                    if choices:
                        if "message" in choices[0]:
                            # Chat completion response
                            output_text = choices[0].get("message", {}).get("content", "")
                            otel_response_info["output_text"] = output_text
                        elif "text" in choices[0]:
                            # Text completion response
                            output_text = choices[0].get("text", "")
                            otel_response_info["output_text"] = output_text
            except Exception as e:
                logger.debug(f"Failed to extract messages for OTEL: {e}")

            self.client.otel.record_response_metrics(
                span=span,
                response_info=otel_response_info,
                error=error.error_msg if error else None,
            )
        elif error:
            self.client.otel.record_response_metrics(
                span=span,
                error=error.error_msg,
            )

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        # Compute effective model name: use LoRA adapter if provided, otherwise use client's model name
        effective_model_name = lora_adapter if lora_adapter else self.client.model_name
        payload = await data.to_payload(
            effective_model_name=effective_model_name,
            max_tokens=self.client.max_completion_tokens,
            ignore_eos=self.client.ignore_eos,
            streaming=self.client.api_config.streaming,
        )

        # Add response_format for structured output if configured
        if self.client.api_config.response_format:
            payload["response_format"] = self.client.api_config.response_format.to_api_format()

        headers = {"Content-Type": "application/json"}

        if self.client.api_key:
            headers["Authorization"] = f"Bearer {self.client.api_key}"

        if self.client.api_config.headers:
            headers.update(self.client.api_config.headers)

        request_data = json.dumps(payload)

        # Determine operation name based on API type
        operation_name = "chat.completions" if self.client.api_config.type == APIType.Chat else "completions"

        start = time.perf_counter()
        response: Optional[aiohttp.ClientResponse] = None
        response_info = None
        error = None
        response_content = ""
        caught_exception: Optional[Exception] = None

        # Get session OTEL context if available (for OTel trace replay)
        parent_context = self._get_session_otel_context(data)

        # Start OTEL tracing
        with self.client.otel.trace_llm_request(
            operation_name=operation_name,
            model_name=effective_model_name,
            request_data=payload,
            parent_context=parent_context,
        ) as span:
            try:
                async with self.session.post(self.client.uri + data.get_route(), headers=headers, data=request_data) as resp:
                    response = resp
                    try:
                        if self.client.api_config.streaming and response.status == 200:
                            response_info = await data.process_response(
                                response=response,
                                config=self.client.api_config,
                                tokenizer=self.client.tokenizer,
                                lora_adapter=lora_adapter,
                            )
                            response_content = response_info.extra_info.get("raw_response", "") if response_info else ""
                        else:
                            # Read response body once to avoid double-read issue
                            response_content = await response.text()

                            if response.status == 200:
                                response_info = await data.process_response(
                                    response=response,
                                    config=self.client.api_config,
                                    tokenizer=self.client.tokenizer,
                                    lora_adapter=lora_adapter,
                                )

                        if response.status != 200:
                            # Handle HTTP error responses (status != 200).
                            #
                            # For OTel trace replay, process_failure() is called to:
                            # 1. Mark the session as failed in WorkerSessionTracker
                            # 2. Call registry.record_failure() to unblock dependent events via EventFailedError
                            # 3. Immediately notify the main process via session_completion_queue
                            #
                            # This ensures that if request X fails and request Y depends on X's output,
                            # Y raises EventFailedError and skips rather than hanging indefinitely.
                            #
                            # Note: We call process_failure() for all data types on non-200 responses
                            # to ensure proper state cleanup (e.g. releasing locks in multi-turn chat)
                            # and failure propagation.
                            if response is not None:
                                error = ErrorResponseInfo(
                                    error_msg=response_content,
                                    error_type=f"HTTP Error {response.status}",
                                )
                                exception = Exception(f"{error.error_type}: {error.error_msg}")
                                response_info = await data.process_failure(
                                    response=response,
                                    config=self.client.api_config,
                                    tokenizer=self.client.tokenizer,
                                    exception=exception,
                                    lora_adapter=lora_adapter,
                                )
                    except Exception as read_error:
                        # Handle errors reading response body
                        if not response_content:
                            response_content = f"Failed to read response text: {read_error}"
                        raise

            except aiohttp.ClientError as e:
                caught_exception = e
                logger.error("Client error during request:", exc_info=True)
                error = ErrorResponseInfo(error_msg=str(e), error_type=type(e).__name__)
            except asyncio.TimeoutError as e:
                caught_exception = e
                logger.error("Request timed out:", exc_info=True)
                error = ErrorResponseInfo(error_msg="Request timed out", error_type="TimeoutError")
            except Exception as e:
                caught_exception = e
                logger.error("Unexpected error during request processing:", exc_info=True)
                error = ErrorResponseInfo(error_msg=str(e), error_type=type(e).__name__)

            end_time = time.perf_counter()

            # Record OTEL metrics
            self._record_otel_metrics(
                span=span,
                data=data,
                response=response,
                response_info=response_info,
                response_content=response_content,
                error=error,
                start_time=start,
                end_time=end_time,
            )

        if caught_exception is not None and not response_info:
            response_info = await data.process_failure(
                response=response,
                config=self.client.api_config,
                tokenizer=self.client.tokenizer,
                exception=caught_exception,
                lora_adapter=lora_adapter,
            )

        metric = RequestLifecycleMetric(
            stage_id=stage_id,
            session_id=data.session_id if isinstance(data.session_id, str) else None,
            request_data=request_data,
            response_data=response_content,
            info=response_info if response_info else InferenceInfo(),
            error=error,
            start_time=start,
            end_time=end_time,
            scheduled_time=scheduled_time,
        )

        # Grab TTFT and TPOT thresholds from request headers if available for streaming requests with token-level timestamps
        if metric.info and metric.info.output_token_times:
            ttft_threshold = None
            tpot_threshold = None
            slo_unit = getattr(self.client.api_config, "slo_unit", None) or "ms"

            default_ttft_header = f"x-slo-ttft-{slo_unit}"
            default_tpot_header = f"x-slo-tpot-{slo_unit}"
            ttft_header = getattr(self.client.api_config, "slo_ttft_header", None) or default_ttft_header
            tpot_header = getattr(self.client.api_config, "slo_tpot_header", None) or default_tpot_header
            if self.client.api_config.headers:
                ttft_threshold = self.client.api_config.headers.get(ttft_header)
                tpot_threshold = self.client.api_config.headers.get(tpot_header)

                unit = slo_unit.lower()
                unit_to_s = {"s": 1.0, "ms": 0.001, "us": 0.000001}
                factor = unit_to_s.get(unit, 1.0)

                if ttft_threshold is not None:
                    metric.ttft_slo_sec = float(ttft_threshold) * factor

                if tpot_threshold is not None:
                    metric.tpot_slo_sec = float(tpot_threshold) * factor

        # Record the metric
        self.client.metrics_collector.record_metric(metric)

    async def close(self) -> None:
        await self.session.close()
