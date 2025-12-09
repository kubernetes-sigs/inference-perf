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

from abc import abstractmethod
from inference_perf.client.requestdatacollector import RequestDataCollector
from inference_perf.config import APIConfig, APIType, CustomTokenizerConfig
from inference_perf.apis import InferenceAPIData, InferenceInfo, RequestLifecycleMetric, ErrorResponseInfo
from inference_perf.utils import CustomTokenizer
from .base import ModelServerClient, ModelServerClientSession, PrometheusMetricMetadata
from typing import List, Optional
import aiohttp
import asyncio
import json
import time
import logging
import requests

logger = logging.getLogger(__name__)


class openAIModelServerClient(ModelServerClient):
    _session: "openAIModelServerClientSession | None" = None
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
    ) -> None:
        super().__init__(api_config, timeout)
        self.uri = uri
        self.max_completion_tokens = 30  # default to use when not set at the request level
        self.ignore_eos = ignore_eos
        self.metrics_collector = metrics_collector
        self.max_tcp_connections = max_tcp_connections
        self.additional_filters = additional_filters
        self.api_key = api_key

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

        if tokenizer_config and not tokenizer_config.pretrained_model_name_or_path:
            tokenizer_config.pretrained_model_name_or_path = self.model_name
        elif not tokenizer_config:
            tokenizer_config = CustomTokenizerConfig(pretrained_model_name_or_path=self.model_name)
        self.tokenizer = CustomTokenizer(tokenizer_config)

    def new_session(self) -> "ModelServerClientSession":
        return openAIModelServerClientSession(self)

    async def process_request(self, data: InferenceAPIData, stage_id: int, scheduled_time: float) -> None:
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
        await session.process_request(data, stage_id, scheduled_time)

    async def close(self) -> None:
        """Close the internal session created by process_request, if any."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    def get_supported_apis(self) -> List[APIType]:
        return []

    @abstractmethod
    def get_prometheus_metric_metadata(self) -> PrometheusMetricMetadata:
        raise NotImplementedError

    def get_supported_models(self) -> List[str]:
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
        self.client = client
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=client.timeout) if client.timeout else aiohttp.helpers.sentinel,
            connector=aiohttp.TCPConnector(limit=client.max_tcp_connections),
        )

    async def process_request(self, data: InferenceAPIData, stage_id: int, scheduled_time: float) -> None:
        payload = await data.to_payload(
            model_name=self.client.model_name,
            max_tokens=self.client.max_completion_tokens,
            ignore_eos=self.client.ignore_eos,
            streaming=self.client.api_config.streaming,
        )
        headers = {"Content-Type": "application/json"}

        if self.client.api_key:
            headers["Authorization"] = f"Bearer {self.client.api_key}"

        if self.client.api_config.headers:
            headers.update(self.client.api_config.headers)

        request_data = json.dumps(payload)

        start = time.perf_counter()
        try:
            async with self.session.post(self.client.uri + data.get_route(), headers=headers, data=request_data) as response:
                response_info = await data.process_response(
                    response=response, config=self.client.api_config, tokenizer=self.client.tokenizer
                )
                response_content = await response.text()

                end_time = time.perf_counter()
                error = None
                if response.status != 200:
                    error = ErrorResponseInfo(
                        error_msg=response_content,
                        error_type=f"{response.status} {response.reason}",
                    )

                metric = RequestLifecycleMetric(
                        stage_id=stage_id,
                        request_data=request_data,
                        response_data=response_content,
                        info=response_info,
                        error=error,
                        start_time=start,
                        end_time=end_time,
                        scheduled_time=scheduled_time,
                    )
                    
                    # Calculate TTFT and TPOT if we have timing data
                if response_info.output_token_times:
                        # TTFT = time to first token
                        metric.ttft = response_info.output_token_times[0] - start
    
                        # TPOT = average time between tokens
                        if len(response_info.output_token_times) > 1:
                            token_gaps = []
                            for i in range(1, len(response_info.output_token_times)):
                                gap = response_info.output_token_times[i] - response_info.output_token_times[i-1]
                                token_gaps.append(gap)
                            metric.tpot = sum(token_gaps) / len(token_gaps)

                # Evaluate SLO - check custom_headers first, then api_config.headers
                ttft_threshold = None
                tpot_threshold = None

                if self.client.api_config.headers:
                    ttft_threshold = self.client.api_config.headers.get('x-slo-ttft-ms')
                if self.client.api_config.headers:
                    tpot_threshold = self.client.api_config.headers.get('x-slo-tpot-ms')

                if ttft_threshold and metric.ttft is not None:
                    metric.ttft_slo = float(ttft_threshold)
                    metric.ttft_slo_met = metric.ttft <= float(ttft_threshold) / 1000  # convert ms to s

                if tpot_threshold and metric.tpot is not None:
                    metric.tpot_slo = float(tpot_threshold)
                    metric.tpot_slo_met = metric.tpot <= float(tpot_threshold) / 1000  # convert ms to s

                 # Record the metric
                self.client.metrics_collector.record_metric(metric)

        except Exception as e:
            if isinstance(e, asyncio.exceptions.TimeoutError):
                logger.error("request timed out:", exc_info=True)
            else:
                logger.error("error occured during request processing:", exc_info=True)
            failure_info = await data.process_failure(
                response=response if "response" in locals() else None,
                config=self.client.api_config,
                tokenizer=self.client.tokenizer,
                exception=e,
            )
            self.client.metrics_collector.record_metric(
                RequestLifecycleMetric(
                    stage_id=stage_id,
                    request_data=request_data,
                    response_data=response_content if "response_content" in locals() else "",
                    info=failure_info if failure_info else InferenceInfo(),
                    error=ErrorResponseInfo(
                        error_msg=str(e),
                        error_type=type(e).__name__,
                    ),
                    start_time=start,
                    end_time=time.perf_counter(),
                    scheduled_time=scheduled_time,
                )
            )

    async def close(self) -> None:
        await self.session.close()
