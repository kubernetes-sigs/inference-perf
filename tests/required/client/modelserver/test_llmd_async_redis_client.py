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
"""The llm-d-async Redis client: what it enqueues, how it reads results back, and how it records each outcome."""

import asyncio
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import pytest
from pydantic import SecretStr

import inference_perf.main as inference_perf_main
from inference_perf.apis import ChatCompletionAPIData, ChatMessage, CompletionAPIData, InferenceAPIData
from inference_perf.apis.base import STAGE_TEARDOWN_CANCELLED_ERROR_TYPE
from inference_perf.client.modelserver.llmd_async_redis_client import (
    SUBMIT_REQUEST_SCRIPT,
    LlmdAsyncRedisModelServerClient,
    RedisCommands,
)
from inference_perf.config import APIConfig, APIType, LlmdAsyncRedisConfig
from inference_perf.metrics.request_collector import LocalRequestMetricCollector


@dataclass
class Queued:
    """One message the client enqueued, in llm-d-async's tagged envelope."""

    internal: dict[str, object]
    request_kind: str
    data: dict[str, object]


class FakeRedis:
    """The three commands the client sends, with the submit script's effect done in Python."""

    def __init__(self, url: str, password: Optional[str]) -> None:
        self.connected_with = (url, password)
        self.values: dict[str, str] = {}
        self.expiries: dict[str, int] = {}
        self.seqs: dict[str, int] = {}
        self.queue: list[str] = []
        self.results: dict[str, "asyncio.Queue[str]"] = {}
        self.eval_calls: list[tuple[str, int, tuple[str | int, ...]]] = []
        self.enqueued = asyncio.Event()

    async def eval(self, script: str, numkeys: int, *keys_and_args: str | int) -> object:
        self.eval_calls.append((script, numkeys, keys_and_args))
        seq_key, cancel_key, active_key, _ = (str(key) for key in keys_and_args[:numkeys])
        token, _, _, before, after, _ = keys_and_args[numkeys:]
        seq = self.seqs[seq_key] = self.seqs.get(seq_key, 0) + 1
        self.values.pop(cancel_key, None)
        self.values[active_key] = str(token)
        self.queue.append(f"{before}{seq}{after}")
        self.enqueued.set()
        return seq

    async def set(self, name: str, value: str, ex: int) -> object:
        self.values[name], self.expiries[name] = value, ex
        return True

    async def brpop(self, keys: List[str], timeout: float) -> Optional[Sequence[str]]:
        results = self.results.setdefault(keys[0], asyncio.Queue())
        try:
            return keys[0], await asyncio.wait_for(results.get(), timeout)
        except TimeoutError:
            return None

    # Waits for the next enqueued message and takes it off the queue.
    async def next_message(self) -> Queued:
        while not self.queue:
            self.enqueued.clear()
            await asyncio.wait_for(self.enqueued.wait(), 5)
        message = json.loads(self.queue.pop(0))
        assert set(message) == {"internal", "request_kind", "data"}
        return Queued(message["internal"], message["request_kind"], message["data"])

    # Puts one entry on a result list, as llm-d-async does with a result.
    async def push(self, result_list: str, entry: str) -> None:
        await self.results.setdefault(result_list, asyncio.Queue()).put(entry)

    # Takes the next enqueued message and answers it with `result` on the result list it names.
    async def answer(self, result: dict[str, object]) -> Queued:
        message = await self.next_message()
        await self.push(str(message.internal["result_queue_name"]), json.dumps({"id": message.data["id"], **result}))
        return message


FAKES: list[FakeRedis] = []


# Hands the client a fresh FakeRedis and keeps it in FAKES for the test to drive.
def fake_connect(url: str, password: Optional[str]) -> RedisCommands:
    FAKES.append(FakeRedis(url, password))
    return FAKES[-1]


# A client for model "test-model" on redis://r:6379/0, connected to a FakeRedis.
def make_client(
    api_type: APIType = APIType.Completion,
    queue_config: Optional[LlmdAsyncRedisConfig] = None,
    headers: Optional[dict[str, str]] = None,
) -> tuple[LlmdAsyncRedisModelServerClient, LocalRequestMetricCollector]:
    collector = LocalRequestMetricCollector()
    api_config = APIConfig(type=api_type, headers=headers)
    client = LlmdAsyncRedisModelServerClient(
        collector, api_config, "redis://r:6379/0", queue_config, "test-model", connect=fake_connect
    )
    return client, collector


# Sends one request, answers it with `result` and waits until the client has recorded it.
async def send(client: LlmdAsyncRedisModelServerClient, data: InferenceAPIData, result: dict[str, object]) -> Queued:
    task = asyncio.create_task(client.process_request(data, 0, 0.0))
    await asyncio.sleep(0)
    message = await FAKES[-1].answer(result)
    await task
    return message


# A completion request for prompt "hi" with no max_tokens of its own.
def hi() -> CompletionAPIData:
    return CompletionAPIData(prompt="hi", max_tokens=0)


COMPLETION_BODY = json.dumps({"choices": [{"text": "hello"}], "usage": {"prompt_tokens": 5, "completion_tokens": 7}})
CHAT_BODY = json.dumps({"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}})
OK: dict[str, object] = {"status_code": 200, "payload": COMPLETION_BODY}
SHORT = LlmdAsyncRedisConfig(default_deadline_seconds=0.01, result_grace_seconds=0)


# A completion request with api.headers {x-llm-d-tenant: t1, x-a: 1} and request headers {x-a: 2}. Expects llm-d-async's
# producer script with its keys in order (sequence counter for the deadline, cancel marker, active marker, queue), a
# deadline 300 s out that the active marker lasts until, the counter an hour past it, and the envelope with enqueue_seq 1,
# a "result-list:" list, the headers merged (the request's value wins) and the HTTP clients' payload defaults.
async def test_submit_matches_the_llm_d_async_producer() -> None:
    client, _ = make_client(headers={"x-llm-d-tenant": "t1", "x-a": "1"})
    before = time.time()
    message = await send(client, CompletionAPIData(prompt="hi", max_tokens=0, headers={"x-a": "2"}), OK)
    script, numkeys, keys_and_args = FAKES[-1].eval_calls[0]
    request_id, deadline = message.data["id"], message.data["deadline"]
    assert isinstance(deadline, int) and math.ceil(before + 300) <= deadline <= math.ceil(time.time() + 300)
    assert script == SUBMIT_REQUEST_SCRIPT and numkeys == 4
    assert list(keys_and_args[:4]) == [
        f"request-seq:request-sortedset:{deadline}",
        f"request-cancel:{request_id}",
        f"request-active:{request_id}",
        "request-sortedset",
    ]
    token, ttl_ms, seq_expiry, _, _, score_base = keys_and_args[4:]
    assert token == message.internal["request_token"] and isinstance(ttl_ms, int) and 299_000 <= ttl_ms <= 301_000
    assert seq_expiry == deadline + 3600 and score_base == deadline
    assert message.request_kind == "plain" and message.internal["enqueue_seq"] == 1
    assert message.internal["request_queue_name"] == "request-sortedset"
    assert str(message.internal["result_queue_name"]).startswith("result-list:")
    assert message.data["endpoint"] == "/v1/completions" and message.data["model"] == "test-model"
    assert message.data["headers"] == {"x-llm-d-tenant": "t1", "x-a": "2"}
    payload = {"model": "test-model", "prompt": "hi", "max_tokens": 30, "ignore_eos": True, "stream": False}
    assert message.data["payload"] == payload


# 200 bodies that make a completion: text with usage {5, 7}, chat content with usage {3, 2}, text alone, usage alone.
# Expects a success with the server's (input, output) counts, 0 where it sends none, both bodies kept, and no cancel marker.
@pytest.mark.parametrize(
    "api_type, body, tokens",
    [
        pytest.param(APIType.Completion, COMPLETION_BODY, (5, 7), id="completion"),
        pytest.param(APIType.Chat, CHAT_BODY, (3, 2), id="chat"),
        pytest.param(APIType.Completion, json.dumps({"choices": [{"text": "hello"}]}), (0, 0), id="text-only"),
        pytest.param(APIType.Chat, json.dumps({"choices": [], "usage": {"prompt_tokens": 1}}), (1, 0), id="usage-only"),
    ],
)
async def test_completed_result_is_a_success(api_type: APIType, body: str, tokens: tuple[int, int]) -> None:
    client, collector = make_client(api_type=api_type)
    data: InferenceAPIData = hi()
    if api_type == APIType.Chat:
        data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="hi")], max_tokens=0)
    message = await send(client, data, {"status_code": 200, "payload": body})
    metric = collector.metrics[0]
    assert metric.error is None and metric.info.response_metrics is not None
    assert (metric.info.input_tokens, metric.info.response_metrics.output_tokens) == tokens
    assert metric.request_data == json.dumps(message.data["payload"]) and metric.response_data == body
    assert f"request-cancel:{message.data['id']}" not in FAKES[-1].values


# Failed results: an HTTP error status, an llm-d-async error code with its message and without one (the payload is the
# message), a 200 body carrying an error, and one with neither content nor usage. Expects the error type and message
# the HTTP clients would record, and the body kept.
@pytest.mark.parametrize(
    "result, error_type, error_msg",
    [
        pytest.param({"status_code": 503, "payload": "busy"}, "HTTP Error 503", "busy", id="http-status"),
        pytest.param(
            {"payload": "{}", "error_code": "EXPIRED", "error_message": "too late"},
            "EXPIRED",
            "too late",
            id="dispatcher-error",
        ),
        pytest.param(
            {"payload": "dropped", "error_code": "GATE_DROPPED"}, "GATE_DROPPED", "dropped", id="dispatcher-error-no-message"
        ),
        pytest.param({"status_code": 200, "payload": '{"error": "overloaded"}'}, "InBandError", "overloaded", id="in-band"),
        pytest.param({"status_code": 200, "payload": '{"choices": []}'}, "EmptyResponseError", "no usage", id="empty-body"),
    ],
)
async def test_failed_result_is_recorded_as_an_error(result: dict[str, object], error_type: str, error_msg: str) -> None:
    client, collector = make_client()
    await send(client, hi(), result)
    error = collector.metrics[0].error
    assert error is not None and error.error_type == error_type and error_msg in error.error_msg
    assert collector.metrics[0].response_data == result["payload"]


# No result arrives, with a deadline 0.01 s out and no grace. Expects a TimeoutError once the deadline passes, and
# request-cancel:<id> set to the request's token for 7 days, as llm-d-async's CancelRequests sets it.
async def test_no_result_by_the_deadline_times_out_and_cancels() -> None:
    client, collector = make_client(queue_config=SHORT)
    await asyncio.wait_for(client.process_request(hi(), 0, 0.0), 5)
    error = collector.metrics[0].error
    assert error is not None and error.error_type == "TimeoutError"
    message = await FAKES[-1].next_message()
    marker = f"request-cancel:{message.data['id']}"
    assert FAKES[-1].values[marker] == message.internal["request_token"] and FAKES[-1].expiries[marker] == 7 * 24 * 3600


# A deadline at most ~1 s out, 3 s of grace, and the result 1.5 s after sending. Expects a success.
async def test_result_within_grace_is_a_success() -> None:
    client, collector = make_client(queue_config=SHORT.model_copy(update={"result_grace_seconds": 3}))
    task = asyncio.create_task(client.process_request(hi(), 0, 0.0))
    await asyncio.sleep(1.5)
    await FAKES[-1].answer(OK)
    await task
    assert collector.metrics[0].error is None


# The load generator cancels a queued request for stage 3, as at stage teardown. Expects a StageTeardownCancelled
# metric for stage 3, the cancellation re-raised, and the request's cancel marker set.
async def test_stage_teardown_cancels_the_queued_request() -> None:
    client, collector = make_client()
    task = asyncio.create_task(client.process_request(hi(), 3, 0.0))
    await asyncio.sleep(0)
    message = await FAKES[-1].next_message()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    metric = collector.metrics[0]
    assert metric.stage_id == 3
    assert metric.error is not None and metric.error.error_type == STAGE_TEARDOWN_CANCELLED_ERROR_TYPE
    assert FAKES[-1].values[f"request-cancel:{message.data['id']}"] == message.internal["request_token"]


# Right after a request's result, its list gets a second copy of it (llm-d-async delivers at least once), another
# request's result and text that is not JSON. Expects all three ignored and the next request still answered.
async def test_result_list_entries_for_nothing_waiting_are_ignored() -> None:
    client, collector = make_client(queue_config=SHORT.model_copy(update={"default_deadline_seconds": 1}))
    first = asyncio.create_task(client.process_request(hi(), 0, 0.0))
    await asyncio.sleep(0)
    answered = await FAKES[-1].answer(OK)
    result_list = str(answered.internal["result_queue_name"])
    await FAKES[-1].push(result_list, json.dumps({"id": answered.data["id"], **OK}))
    await FAKES[-1].push(result_list, json.dumps({"id": "another-request", "status_code": 500, "payload": "boom"}))
    await FAKES[-1].push(result_list, "not json")
    await first
    await send(client, hi(), OK)
    assert [m.error for m in collector.metrics] == [None, None]


# A client with Redis password "pw" used from two event loops in turn, as a run without workers does across stages,
# then copied as the load generator copies it to a worker. Expects every request answered, and each loop and the
# copy to connect on its own, with the URL and the password, and to read its own result list.
def test_each_event_loop_and_worker_copy_reads_its_own_result_list() -> None:
    client, collector = make_client(queue_config=LlmdAsyncRedisConfig(password=SecretStr("pw")))

    async def result_list_of_one_request(c: LlmdAsyncRedisModelServerClient) -> str:
        return str((await send(c, hi(), OK)).internal["result_queue_name"])

    lists = [asyncio.run(result_list_of_one_request(client)) for _ in range(2)]
    lists.append(asyncio.run(result_list_of_one_request(pickle.loads(pickle.dumps(client)))))
    assert len(set(lists)) == 3 and [m.error for m in collector.metrics] == [None, None]
    assert [fake.connected_with for fake in FAKES[-3:]] == [("redis://r:6379/0", "pw")] * 3


# api.streaming true, then no model name. Expects construction to fail for each: llm-d-async returns whole bodies,
# and there is no /v1/models behind a queue to ask for the model.
def test_streaming_and_missing_model_name_are_rejected() -> None:
    collector = LocalRequestMetricCollector()
    with pytest.raises(ValueError, match="streaming"):
        LlmdAsyncRedisModelServerClient(collector, APIConfig(streaming=True), "redis://r:6379/0", None, "m")
    with pytest.raises(ValueError, match="model_name"):
        LlmdAsyncRedisModelServerClient(collector, APIConfig(), "redis://r:6379/0", None, None)


class ClientBuilt(Exception):
    """Raised in place of building the client, to stop main_cli there."""


# The flags --server.type llmd_async_redis, a Redis base_url, a model, ignore_eos false and a request queue name.
# Expects main_cli to build this client from them: base_url as the Redis URL, the queue block, the model and ignore_eos.
def test_main_builds_this_client_for_its_server_type(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[dict[str, object]] = []

    def build(*args: object, **kwargs: object) -> None:
        built.append(kwargs)
        raise ClientBuilt()

    monkeypatch.setattr(inference_perf_main, "LlmdAsyncRedisModelServerClient", build)
    monkeypatch.setattr(inference_perf_main, "setup_logging", lambda level: None)
    argv = ["inference-perf", "--server.type", "llmd_async_redis", "--server.base_url", "redis://redis:6379/0"]
    argv += ["--server.model_name", "m", "--server.ignore_eos", "false", "--server.llmd_async_redis.request_queue_name", "q"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ClientBuilt):
        inference_perf_main.main_cli()
    assert built[0]["redis_url"] == "redis://redis:6379/0" and built[0]["model_name"] == "m"
    assert built[0]["queue_config"] == LlmdAsyncRedisConfig(request_queue_name="q") and built[0]["ignore_eos"] is False
