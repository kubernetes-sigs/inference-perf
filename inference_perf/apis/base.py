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
from typing import Any, List, Optional
from aiohttp import ClientResponse
from pydantic import BaseModel, Field, SerializeAsAny, computed_field, model_serializer
from inference_perf.payloads import RequestBody, RequestMetrics
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.config import APIConfig, APIType


class ResponseMetrics(BaseModel):
    output_tokens: int = 0
    # Last-seen `usage` dict reported by the server (streaming: the trailing
    # usage chunk; non-streaming: the response body's `usage`).
    server_usage: Optional[dict[str, Any]] = None


class UnaryResponseMetrics(ResponseMetrics):
    pass


class StreamedResponseMetrics(ResponseMetrics):
    response_chunks: List[str] = []
    chunk_times: List[float] = []
    output_token_times: List[float] = []


class InferenceInfo(BaseModel):
    request_metrics: RequestMetrics
    response_metrics: Optional[SerializeAsAny[ResponseMetrics]] = None
    extra_info: dict[str, Any] = {}
    lora_adapter: Optional[str] = None
    graph_event_id: Optional[str] = None
    labels: dict[str, str] = {}
    # Extra attempts spent on connection faults raised before a response was
    # established, and whether one of them went on to succeed. 0/False on the
    # overwhelming majority of requests; a request that exhausted its retries reports
    # the count with retries_recovered False. See LoadConfig.request_retries.
    retries_attempted: int = 0
    retries_recovered: bool = False
    # Time this request spent on attempts that failed, plus the backoff waited between
    # them: the share of its end-to-end latency that bought nothing. Counted on
    # exhausted requests too, where every attempt was wasted. None unless a retry
    # happened, so RequestLifecycleMetric.start_time stays the single definition of
    # when the request was dispatched.
    retry_wasted_sec: Optional[float] = None

    @model_serializer(mode="wrap")
    def _omit_absent_retry_wasted_sec(self, handler: Any) -> dict[str, Any]:
        """Drop `retry_wasted_sec` entirely when it is None.

        A request that never retried must serialize exactly as it did before retries
        existed, so the field appears only on the requests it describes rather than as
        a `null` on every entry of per_request_lifecycle_metrics.json.
        """
        dumped: dict[str, Any] = handler(self)
        if dumped.get("retry_wasted_sec") is None:
            dumped.pop("retry_wasted_sec", None)
        return dumped

    # DEPRECATED: mirror of request_metrics.text.input_tokens kept at the top
    # level for back-compat with parsers of pre-multimodal
    # per_request_lifecycle_metrics.json. Slated for removal alongside the
    # lifecycle-report rewrite — new consumers should read
    # request_metrics.text.input_tokens.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def input_tokens(self) -> int:
        return self.request_metrics.text.input_tokens


class ErrorResponseInfo(BaseModel):
    error_type: str
    error_msg: str


class RequestLifecycleMetric(BaseModel):
    stage_id: Optional[int] = None
    session_id: Optional[str] = None
    scheduled_time: float
    start_time: float
    end_time: float
    request_data: str
    response_data: Optional[str] = None
    info: InferenceInfo
    error: Optional[ErrorResponseInfo]

    ttft_slo_sec: Optional[float] = None
    tpot_slo_sec: Optional[float] = None


class SessionLifecycleMetric(BaseModel):
    session_id: str
    stage_id: int
    file_path: str
    start_time: float
    end_time: float
    duration_sec: float
    num_events: int
    num_events_completed: int
    num_events_cancelled: Optional[int] = None
    # Per-session count of events whose live tool_call response was
    # detected malformed and replaced with the recorded assistant
    # message at substitution time. None when bad_tool_call_handling
    # is `none` (the upstream-default). Zero when handling is enabled
    # but no malformed tool_calls were observed.
    n_recorded_substitutions: Optional[int] = None
    recorded_substitution_event_ids: Optional[List[str]] = None
    # Per-session sum of extra attempts spent on connection faults raised before
    # response headers were obtained, and how many of those requests went on to
    # succeed. 0/0 when the session hit no transport faults; see
    # LoadConfig.request_retries.
    retries_attempted: int = 0
    retries_recovered: int = 0
    success: Optional[bool] = None
    error: Optional[ErrorResponseInfo] = None
    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
    # Per-session sum of prompt tokens served from the server-side KV/prefix
    # cache (usage.prompt_tokens_details.cached_tokens). None when no request in
    # the session reported server usage (non-streaming, or a vLLM build that
    # omits or nulls prompt_tokens_details) — distinct from 0, which means the
    # field was present but no tokens were cached.
    total_cached_tokens: Optional[int] = None
    # Denominator for the per-session cache hit rate: sum of the server-reported
    # usage.prompt_tokens over exactly the requests that contributed to
    # total_cached_tokens. This is deliberately NOT total_input_tokens, which is
    # the client-side re-tokenized count — mixing a server numerator with a
    # client denominator skews the ratio by the tokenizer's disagreement with the
    # server (chat template and tool-schema overhead the client does not model)
    # and can push it above 1.0. None whenever total_cached_tokens is None.
    total_cacheable_input_tokens: Optional[int] = None
    # TFUT (Time to First User Token) — per-session user-perceived latency.
    # Anchored on session dispatch time (includes client-side queue delay),
    # not the moment the first request reached the server.
    user_facing_event_ids: Optional[List[str]] = None
    num_structured_output_excluded: Optional[int] = None
    tfut_sec: Optional[float] = None
    tfut_none_reason: Optional[str] = None
    dispatch_perf_counter: Optional[float] = Field(default=None, exclude=True)


class InferenceAPIData(BaseModel):
    # loadgen should assign this request to preferred worker if possible
    preferred_worker_id: int = -1  # no preferred worker by default
    session_id: Optional[str] = None  # set by loadgen for session-based workloads
    graph_event_id: Optional[str] = None
    otel_context: Optional[dict[str, str]] = None  # OTEL trace context for distributed tracing
    stage_id: int = 0  # stage id
    headers: Optional[dict[str, str]] = None
    labels: dict[str, str] = {}

    @abstractmethod
    def get_api_type(self) -> APIType:
        raise NotImplementedError

    @abstractmethod
    def get_route(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def to_request_body(
        self, effective_model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool
    ) -> RequestBody:
        raise NotImplementedError

    @abstractmethod
    async def process_response(
        self, response: ClientResponse, config: APIConfig, tokenizer: CustomTokenizer, lora_adapter: Optional[str] = None
    ) -> InferenceInfo:
        raise NotImplementedError

    def on_completion(self, info: InferenceInfo) -> None:
        """Called after every request completes (real or mock). Override to add post-completion side effects."""
        pass

    async def process_failure(
        self,
        response: Optional[ClientResponse],
        config: APIConfig,
        tokenizer: CustomTokenizer,
        exception: Exception,
        lora_adapter: Optional[str] = None,
    ) -> Optional[InferenceInfo]:
        pass  # no-op by default


class LazyLoadInferenceAPIData(InferenceAPIData):
    """
    InferenceAPIData that loads data lazily.
    This is useful for multiprocessing where the data cannot be pickled or need to be initialized in worker space.
    this class shouldn't go with any data but payload for data generator to return API data later.
    in most cases, generator should depends on data_index as reference. If more payload needed, try to extend this class.
    """

    data_index: int

    def get_api_type(self) -> APIType:
        raise NotImplementedError("LazyLoadInferenceAPIData doesn't support this operation")

    def get_route(self) -> str:
        raise NotImplementedError("LazyLoadInferenceAPIData doesn't support this operation")

    async def to_request_body(
        self, effective_model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool
    ) -> RequestBody:
        raise NotImplementedError("LazyLoadInferenceAPIData doesn't support this operation")

    async def process_response(
        self, response: ClientResponse, config: APIConfig, tokenizer: CustomTokenizer, lora_adapter: Optional[str] = None
    ) -> InferenceInfo:
        raise NotImplementedError("LazyLoadInferenceAPIData doesn't support this operation")

    async def process_failure(
        self,
        response: Optional[ClientResponse],
        config: APIConfig,
        tokenizer: CustomTokenizer,
        exception: Exception,
        lora_adapter: Optional[str] = None,
    ) -> Optional[InferenceInfo]:
        raise NotImplementedError("LazyLoadInferenceAPIData doesn't support this operation")
