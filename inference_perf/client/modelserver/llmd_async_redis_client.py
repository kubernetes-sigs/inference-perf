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
"""Client that sends requests through llm-d-async's Redis sorted-set queue.

Requests are enqueued the way llm-d-async's ``RedisSortedSetProducer`` enqueues
them on a standalone Redis (producer/redis_sortedset_producer.go), so the
dispatcher orders them earliest deadline first and sends them to the inference
gateway. Each worker process reads the results back from its own result list.
"""

import asyncio
import importlib
import json
import logging
import math
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Sequence

from inference_perf.apis import (
    ErrorResponseInfo,
    InferenceAPIData,
    InferenceInfo,
    RequestLifecycleMetric,
    UnaryResponseMetrics,
)
from inference_perf.apis.base import is_stage_teardown_cancelled, stage_teardown_cancelled_error
from inference_perf.apis.response_errors import EmptyResponseError, InBandError, in_band_error
from inference_perf.config import APIConfig, APIType, LlmdAsyncRedisConfig
from inference_perf.metrics.request_collector import RequestMetricCollector
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.utils.custom_tokenizer import CustomTokenizer

from .base import ModelServerClient
from .metrics import BaseMetrics

logger = logging.getLogger(__name__)

# submitRequestScript from llm-d-async producer/redis_sortedset_producer.go, unchanged.
# KEYS: enqueue sequence, cancellation marker, active-token marker, request queue.
# ARGV: request token, active TTL (ms), sequence expiry (unix s), envelope before and
# after the sequence number, deadline (unix s).
SUBMIT_REQUEST_SCRIPT = """
local seq = redis.call("INCR", KEYS[1])
redis.call("EXPIREAT", KEYS[1], ARGV[3])
redis.call("DEL", KEYS[2])
redis.call("SET", KEYS[3], ARGV[1], "PX", ARGV[2])
local score = tonumber(ARGV[6]) + math.min(seq, 2097151) / 2097152
redis.call("ZADD", KEYS[4], string.format("%.17g", score), ARGV[4] .. seq .. ARGV[5])
if KEYS[5] then
  redis.call("SET", KEYS[5], ARGV[7], "PX", ARGV[8])
end
return seq
"""

# The producer keeps a deadline's sequence counter an hour past the deadline.
ENQUEUE_SEQ_GRACE_SEC = 3600
# How long a cancellation marker lives, as in the producer's CancelRequests.
CANCELLATION_MARKER_TTL_SEC = 7 * 24 * 3600
# BRPOP wait per call, so the collector notices cancellation.
RESULT_POLL_SEC = 1.0


class RedisCommands(Protocol):
    """The redis.asyncio calls this client makes."""

    async def eval(self, script: str, numkeys: int, *keys_and_args: str | int) -> object: ...

    async def set(self, name: str, value: str, ex: int) -> object: ...

    async def brpop(self, keys: List[str], timeout: float) -> Optional[Sequence[str]]: ...


def connect_redis(url: str, password: Optional[str]) -> RedisCommands:
    # redis is an optional dependency, imported only when this server type is selected.
    try:
        redis_asyncio = importlib.import_module("redis.asyncio")
    except ImportError as e:
        raise ImportError("server.type 'llmd_async_redis' needs the redis package: pip install 'redis>=5'") from e
    client: RedisCommands = redis_asyncio.from_url(url, password=password, decode_responses=True)
    return client


def active_ttl_ms(deadline: int, now: float) -> int:
    """Milliseconds until the deadline, at least 1, as the producer sets request-active:<id>."""
    return max(int((deadline - now) * 1000), 1)


def result_wait_sec(deadline: int, grace_sec: float, now: float) -> float:
    """How much longer to wait for a result: until the deadline plus the grace, never negative."""
    return max(deadline + grace_sec - now, 0.0)


def request_queue_keys(queue: str, request_id: str, deadline: int) -> List[str]:
    return [f"request-seq:{queue}:{deadline}", f"request-cancel:{request_id}", f"request-active:{request_id}", queue]


def split_envelope(internal: dict[str, str], data: dict[str, object]) -> tuple[str, str]:
    """The queued message as the text before and after its enqueue sequence number.

    The script inserts the sequence it assigns between the two halves, as the Go
    producer does, giving llm-d-async's tagged envelope:
    {"internal": {..., "enqueue_seq": <seq>}, "request_kind": "plain", "data": {...}}.
    """
    internal_json = json.dumps(internal, separators=(",", ":"))
    before = '{"internal":' + internal_json[:-1] + ',"enqueue_seq":'
    after = '},"request_kind":"plain","data":' + json.dumps(data, separators=(",", ":")) + "}"
    return before, after


@dataclass
class QueueResult:
    status_code: int
    payload: str
    error_code: str
    error_message: str


def parse_result(raw: str) -> Optional[tuple[str, QueueResult]]:
    """The request ID and outcome of one result-list entry, or None if it is not a result."""
    try:
        record = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict) or not isinstance(record.get("id"), str):
        return None
    status = record.get("status_code")
    payload = record.get("payload")
    error_code = record.get("error_code")
    error_message = record.get("error_message")
    return record["id"], QueueResult(
        status_code=status if isinstance(status, int) else 0,
        payload=payload if isinstance(payload, str) else "",
        error_code=error_code if isinstance(error_code, str) else "",
        error_message=error_message if isinstance(error_message, str) else "",
    )


@dataclass
class _ProcessState:
    loop: asyncio.AbstractEventLoop
    redis: RedisCommands
    result_list: str
    pending: dict[str, "asyncio.Future[QueueResult]"] = field(default_factory=dict)
    collector: Optional["asyncio.Task[None]"] = None


class LlmdAsyncRedisModelServerClient(ModelServerClient):
    def __init__(
        self,
        metrics_collector: RequestMetricCollector,
        api_config: APIConfig,
        redis_url: str,
        queue_config: Optional[LlmdAsyncRedisConfig],
        model_name: Optional[str],
        tokenizer: Optional[CustomTokenizer] = None,
        ignore_eos: bool = True,
        connect: Callable[[str, Optional[str]], RedisCommands] = connect_redis,
    ) -> None:
        # No request timeout or transport retries: the deadline bounds the wait, and llm-d-async retries dispatches itself.
        super().__init__(api_config)
        if api_config.streaming:
            raise ValueError("server.type 'llmd_async_redis' returns whole response bodies: set api.streaming to false.")
        if model_name is None:
            raise ValueError("server.type 'llmd_async_redis' needs server.model_name.")
        self.metrics_collector = metrics_collector
        self.redis_url = redis_url
        self.queue_config = queue_config if queue_config is not None else LlmdAsyncRedisConfig()
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.ignore_eos = ignore_eos
        self.max_completion_tokens = 30  # default to use when not set at the request level, as the HTTP clients do
        self.connect = connect
        self._state: Optional[_ProcessState] = None

    def __getstate__(self) -> dict[str, object]:
        # Workers get a copy of this client; each opens its own connection and result list.
        state = self.__dict__.copy()
        state["_state"] = None
        return state

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def get_prometheus_metric_metadata(self) -> BaseMetrics:
        return BaseMetrics()

    def _process_state(self) -> _ProcessState:
        loop = asyncio.get_running_loop()
        if self._state is None or self._state.loop is not loop:
            password = self.queue_config.password.get_secret_value() if self.queue_config.password else None
            state = _ProcessState(
                loop=loop,
                redis=self.connect(self.redis_url, password),
                result_list=f"{self.queue_config.result_queue_name}:{uuid.uuid4().hex}",
            )
            state.collector = loop.create_task(self._collect(state))
            self._state = state
        return self._state

    async def _collect(self, state: _ProcessState) -> None:
        while True:
            try:
                popped = await state.redis.brpop([state.result_list], timeout=RESULT_POLL_SEC)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # The requests waiting on this list time out at their deadline if Redis stays down.
                logger.warning("Reading results from %s failed: %s", state.result_list, e)
                await asyncio.sleep(RESULT_POLL_SEC)
                continue
            if popped is None:
                continue
            parsed = parse_result(popped[1])
            if parsed is None:
                logger.warning("Ignoring a result-list entry that is not an llm-d-async result")
                continue
            request_id, result = parsed
            waiter = state.pending.get(request_id)
            if waiter is not None and not waiter.done():
                waiter.set_result(result)

    def _response_info(self, api_type: APIType, body: str, lora_adapter: Optional[str]) -> InferenceInfo:
        """Token counts of a 200 response body, as the HTTP clients count a unary response."""
        data = json.loads(body)
        if (error_payload := in_band_error(data)) is not None:
            raise InBandError(error_payload)
        if not isinstance(data, dict):
            raise EmptyResponseError()
        usage = data.get("usage")
        server_usage = usage if isinstance(usage, dict) else None
        choices = data.get("choices")
        first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        if api_type == APIType.Chat:
            message = first.get("message")
            content = message.get("content") if isinstance(message, dict) else None
        else:
            content = first.get("text")
        output_text = content if isinstance(content, str) else ""
        if not output_text and server_usage is None:
            raise EmptyResponseError()
        prompt_tokens = server_usage.get("prompt_tokens") if server_usage else None
        completion_tokens = server_usage.get("completion_tokens") if server_usage else None
        if isinstance(completion_tokens, int):
            output_tokens = completion_tokens
        elif self.tokenizer is not None:
            output_tokens = self.tokenizer.count_tokens(output_text, add_special_tokens=False)
        else:
            output_tokens = 0
        return InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=prompt_tokens if isinstance(prompt_tokens, int) else 0)),
            response_metrics=UnaryResponseMetrics(output_tokens=output_tokens, server_usage=server_usage),
            lora_adapter=lora_adapter,
        )

    async def _cancel_queued(self, state: _ProcessState, request_id: str, token: str) -> None:
        """Stop llm-d-async from dispatching a request nobody waits for any more, as CancelRequests does."""
        try:
            await state.redis.set(f"request-cancel:{request_id}", token, ex=CANCELLATION_MARKER_TTL_SEC)
        except Exception as e:
            logger.debug("Could not mark request %s cancelled: %s", request_id, e)

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        effective_model_name = lora_adapter if lora_adapter else self.model_name
        payload = await data.to_request_body(
            effective_model_name=effective_model_name,
            max_tokens=self.max_completion_tokens,
            ignore_eos=self.ignore_eos,
            streaming=False,
        )
        request_data = json.dumps(payload)
        queue = self.queue_config.request_queue_name
        request_id = uuid.uuid4().hex
        # The producer's request generation token: 16 random bytes, hex encoded.
        token = secrets.token_hex(16)

        start = time.perf_counter()
        submitted_at = time.time()
        deadline = math.ceil(submitted_at + self.queue_config.default_deadline_seconds)

        info: Optional[InferenceInfo] = None
        error: Optional[ErrorResponseInfo] = None
        response_content = ""
        try:
            state = self._process_state()
            message: dict[str, object] = {
                "id": request_id,
                "created": int(submitted_at),
                "deadline": deadline,
                "payload": payload,
                "endpoint": data.get_route(),
                "model": effective_model_name,
            }
            headers = {**(self.api_config.headers or {}), **(data.headers or {})}
            if headers:
                message["headers"] = headers
            before, after = split_envelope(
                {"request_token": token, "request_queue_name": queue, "result_queue_name": state.result_list}, message
            )
            waiter: asyncio.Future[QueueResult] = state.loop.create_future()
            state.pending[request_id] = waiter
            try:
                await state.redis.eval(
                    SUBMIT_REQUEST_SCRIPT,
                    4,
                    *request_queue_keys(queue, request_id, deadline),
                    token,
                    active_ttl_ms(deadline, submitted_at),
                    deadline + ENQUEUE_SEQ_GRACE_SEC,
                    before,
                    after,
                    deadline,
                )
                wait_sec = result_wait_sec(deadline, self.queue_config.result_grace_seconds, time.time())
                result = await asyncio.wait_for(waiter, timeout=wait_sec)
            finally:
                del state.pending[request_id]
            response_content = result.payload
            if result.error_code:
                error = ErrorResponseInfo(error_msg=result.error_message or result.payload, error_type=result.error_code)
            elif result.status_code != 200:
                error = ErrorResponseInfo(error_msg=result.payload, error_type=f"HTTP Error {result.status_code}")
            else:
                info = self._response_info(self.api_config.type, result.payload, lora_adapter)
        except TimeoutError:
            error = ErrorResponseInfo(
                error_msg="No result by the deadline plus result_grace_seconds", error_type="TimeoutError"
            )
            await self._cancel_queued(state, request_id, token)
        except asyncio.CancelledError:
            error = stage_teardown_cancelled_error()
            await self._cancel_queued(state, request_id, token)
        except Exception as e:
            error = ErrorResponseInfo(error_msg=str(e), error_type=type(e).__name__)
        end_time = time.perf_counter()
        cancelled = is_stage_teardown_cancelled(error)

        if info is not None:
            data.on_completion(info)
        else:
            if error is not None and not cancelled and self.tokenizer is not None:
                # As the HTTP clients do, so a session waiting on this request learns it failed.
                info = await data.process_failure(
                    response=None,
                    config=self.api_config,
                    tokenizer=self.tokenizer,
                    exception=Exception(f"{error.error_type}: {error.error_msg}"),
                    lora_adapter=lora_adapter,
                )
            if info is None:
                info = InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=0)), lora_adapter=lora_adapter)
        if data.labels:
            info.labels = data.labels
        if data.graph_event_id:
            info.graph_event_id = data.graph_event_id

        self.metrics_collector.record_metric(
            RequestLifecycleMetric(
                stage_id=stage_id,
                session_id=data.session_id,
                request_session_id=data.request_session_id,
                request_data=request_data,
                response_data=response_content,
                info=info,
                error=error,
                start_time=start,
                end_time=end_time,
                scheduled_time=scheduled_time,
            )
        )

        if cancelled:
            raise asyncio.CancelledError()
