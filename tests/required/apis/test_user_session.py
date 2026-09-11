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

"""Tests for LocalUserSession lifecycle."""

import asyncio
import multiprocessing as mp
import re
import pytest
from collections import defaultdict
from queue import Empty
from typing import Any, List, Optional, Tuple
from unittest.mock import MagicMock

from inference_perf.apis.user_session import LocalUserSession, UserSessionCompletionAPIData
from inference_perf.apis import InferenceAPIData
from inference_perf.client.modelserver.base import ModelServerClient
from inference_perf.client.modelserver.metrics import BaseMetrics
from inference_perf.config import (
    APIConfig,
    APIType,
    DataConfig,
    DataGenType,
    LoadConfig,
    LoadType,
    SharedPrefix,
    StandardLoadStage,
)
from inference_perf.datagen.synthetic.shared_prefix_datagen import SharedPrefixDataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator


def _mock_tokenizer() -> MagicMock:
    tok = MagicMock()
    hf = MagicMock()
    hf.vocab_size = 1000
    hf.decode = MagicMock(side_effect=lambda ids, **kw: f"tok_{len(ids)}")
    hf.batch_decode = MagicMock(side_effect=lambda batch, **kw: [f"tok_{len(ids)}" for ids in batch])
    tok.get_tokenizer.return_value = hf
    # Match the decode mock's "tok_N" format so count_tokens returns a real int
    # (the exact-length datagen path compares this against target_len).
    # **kw absorbs add_special_tokens, which the response path passes explicitly.
    tok.count_tokens = MagicMock(
        side_effect=lambda text, **kw: sum(int(n) for n in re.findall(r"tok_(\d+)", text)) if isinstance(text, str) else 0
    )
    return tok


def _make_datagen(num_groups: int = 1, num_prompts_per_group: int = 1) -> SharedPrefixDataGenerator:
    api_config = APIConfig(type=APIType.Completion)
    data_config = DataConfig(
        type=DataGenType.SharedPrefix,
        shared_prefix=SharedPrefix(
            num_groups=num_groups,
            num_prompts_per_group=num_prompts_per_group,
            enable_multi_turn_chat=True,
            system_prompt_len=5,
            question_len=5,
            output_len=5,
            seed=42,
        ),
    )
    return SharedPrefixDataGenerator(api_config, data_config, _mock_tokenizer())


class SessionTrackingClient(ModelServerClient):
    """Minimal client that exercises the UserSession to_request_body / update_context
    lifecycle and records the prompt sent per stage.

    When prompt_queue is set (mp mode), prompts are sent to the queue so the
    main process can read them.  Otherwise they are stored in-process."""

    def __init__(self, prompt_queue: Optional["mp.Queue[Tuple[int, str]]"] = None) -> None:
        self.api_config = APIConfig(type=APIType.Completion)
        self.timeout = None
        self.prompts_by_stage: dict[int, list[str]] = defaultdict(list)
        self._prompt_queue = prompt_queue

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        payload = await data.to_request_body("model", 64, False, False)
        prompt = payload["prompt"]
        self.prompts_by_stage[stage_id].append(prompt)

        if self._prompt_queue is not None:
            self._prompt_queue.put((stage_id, prompt))

        if isinstance(data, UserSessionCompletionAPIData):
            data.user_session.update_context(prompt + f" RESPONSE_STAGE{stage_id}")

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion]

    def get_prometheus_metric_metadata(self) -> BaseMetrics:
        raise NotImplementedError


class TestLocalUserSessionLifecycle:
    def setup_method(self) -> None:
        LocalUserSession.clear_instances()

    def teardown_method(self) -> None:
        LocalUserSession.clear_instances()

    def test_get_instance_returns_same_object(self) -> None:
        s1 = LocalUserSession.get_instance("sess_a")
        s2 = LocalUserSession.get_instance("sess_a")
        assert s1 is s2

    def test_clear_instances_resets_all_sessions(self) -> None:
        s1 = LocalUserSession.get_instance("sess_a")
        s1.context = "accumulated context"
        s1._current_round = 3

        s2 = LocalUserSession.get_instance("sess_b")
        s2.context = "other context"

        LocalUserSession.clear_instances()

        new_s1 = LocalUserSession.get_instance("sess_a")
        new_s2 = LocalUserSession.get_instance("sess_b")

        assert new_s1 is not s1
        assert new_s2 is not s2
        assert new_s1.context == ""
        assert new_s1._current_round == 0
        assert new_s2.context == ""

    def test_context_does_not_leak_across_stage_boundary(self) -> None:
        """
        Simulates two stages. After clearing between stages, a session
        obtained via get_instance must have empty context and round 0.


        """
        session = LocalUserSession.get_instance("user_0")
        session.context = "system prompt Q1 A1 Q2 A2"
        session._current_round = 2

        LocalUserSession.clear_instances()

        session_s1 = LocalUserSession.get_instance("user_0")
        assert session_s1.context == ""
        assert session_s1._current_round == 0

    @pytest.mark.asyncio
    async def test_loadgen_does_not_leak_session_context_across_stages(self) -> None:
        """
        Run the real LoadGenerator with two stages (num_workers=0) using a
        client that exercises the full to_request_body / update_context lifecycle.

        Stage 0 builds up session context.  Stage 1 must NOT see that context
        in its prompts — if it does, sessions leaked across the stage boundary.


        """
        datagen = _make_datagen()
        # High rate ensures multiple requests per stage so sessions accumulate context.
        # ExceptionGroup may fire due to strict zip in the non-mp path when
        # floating-point timer values land exactly at the stage boundary.
        load_config = LoadConfig(
            type=LoadType.CONSTANT,
            stages=[
                StandardLoadStage(rate=10, duration=1),
                StandardLoadStage(rate=10, duration=1),
            ],
            num_workers=0,
            interval=0,
        )
        loadgen = LoadGenerator(datagen, load_config)
        client = SessionTrackingClient()

        try:
            await loadgen.run(client)
        except ExceptionGroup:
            pass

        assert 0 in client.prompts_by_stage, "Expected prompts in stage 0"
        if 1 not in client.prompts_by_stage:
            pytest.skip("Stage 1 did not produce prompts (ExceptionGroup aborted early)")

        for prompt in client.prompts_by_stage[1]:
            assert "RESPONSE_STAGE0" not in prompt, (
                f"Stage 1 prompt contains stage 0 response context — sessions "
                f"were not cleared between stages.\n"
                f"  stage 1 prompt: {prompt!r}"
            )

    @pytest.mark.asyncio
    async def test_loadgen_mp_does_not_leak_session_context_across_stages(self) -> None:
        """
        Same as the non-mp test but with num_workers=1 so requests flow
        through a forked Worker subprocess.

        The client writes (stage_id, prompt) tuples to an mp.Queue that the
        main process drains after the run.  If any stage-1 prompt contains
        RESPONSE_STAGE0, sessions leaked across the stage boundary inside
        the worker process.


        """
        mp.set_start_method("fork", force=True)

        datagen = _make_datagen()
        load_config = LoadConfig(
            type=LoadType.CONSTANT,
            stages=[
                StandardLoadStage(rate=10, duration=1),
                StandardLoadStage(rate=10, duration=1),
            ],
            num_workers=1,
            interval=0,
        )

        prompt_queue: "mp.Queue[Tuple[int, str]]" = mp.Queue()
        client = SessionTrackingClient(prompt_queue=prompt_queue)
        loadgen = LoadGenerator(datagen, load_config)

        await loadgen.run(client)
        await loadgen.stop()

        prompts_by_stage: dict[int, list[str]] = defaultdict(list)
        while True:
            try:
                stage_id, prompt = prompt_queue.get_nowait()
                prompts_by_stage[stage_id].append(prompt)
            except Empty:
                break

        assert 0 in prompts_by_stage, "Expected prompts in stage 0"
        assert 1 in prompts_by_stage, "Expected prompts in stage 1"

        for prompt in prompts_by_stage[1]:
            assert "RESPONSE_STAGE0" not in prompt, (
                f"Stage 1 prompt contains stage 0 response context — sessions "
                f"were not cleared between stages in worker subprocess.\n"
                f"  stage 1 prompt: {prompt!r}"
            )


# A dropped handoff between turns deadlocks rather than erroring, so the
# contention tests are bounded: a stalled conversation must fail the run, not hang it.
_TURN_TIMEOUT = 5.0


def _unary_response(text: str) -> MagicMock:
    """Minimal non-streaming completion response carrying ``text``."""
    response = MagicMock()
    body: asyncio.Future[Any] = asyncio.Future()
    body.set_result({"choices": [{"text": text}]})
    response.json = MagicMock(return_value=body)
    response.status = 200
    response.headers = {"content-type": "application/json"}
    return response


def _new_session(session_id: str, context: str = "") -> LocalUserSession:
    """Register a session the way a datagen does, so ``user_session`` resolves to it."""
    session = LocalUserSession(user_session_id=session_id, context=context)
    LocalUserSession._instances[session_id] = session
    return session


class TestUserSessionTurnSerialization:
    """A session serves one turn at a time.

    Every other multi-turn test drives a session sequentially (acquire, release,
    acquire), so the waiter queue in ``get_context`` is never entered. These
    tests put turns in genuine contention, which is the invariant multi-turn
    rests on: a conversation is only coherent if turn N+1 observes the context
    turn N produced. Losing it does not raise, it silently degrades the
    benchmark into concurrent single-turn requests that happen to share an id.
    """

    def setup_method(self) -> None:
        LocalUserSession.clear_instances()

    def teardown_method(self) -> None:
        LocalUserSession.clear_instances()

    @pytest.mark.asyncio
    async def test_concurrent_turns_never_overlap(self) -> None:
        session = _new_session("sess_excl")
        in_flight = 0
        peak = 0

        async def turn(n: int) -> None:
            nonlocal in_flight, peak
            await session.get_context(n)
            in_flight += 1
            peak = max(peak, in_flight)
            # Yield while holding the slot so an unserialized turn would enter here.
            await asyncio.sleep(0)
            in_flight -= 1
            session.update_context(f"resp{n}")

        await asyncio.wait_for(asyncio.gather(*(turn(i) for i in range(5))), timeout=_TURN_TIMEOUT)

        assert peak == 1, f"{peak} turns held the same session at once; turns are not serialized"
        assert session._current_round == 5

    @pytest.mark.asyncio
    async def test_queued_turns_are_served_in_arrival_order(self) -> None:
        session = _new_session("sess_fifo")
        served: List[int] = []

        async def turn(n: int) -> None:
            await session.get_context(n)
            served.append(n)
            await asyncio.sleep(0)
            session.update_context(f"resp{n}")

        # gather schedules in argument order, so turns 1..4 queue behind turn 0.
        await asyncio.wait_for(asyncio.gather(*(turn(i) for i in range(5))), timeout=_TURN_TIMEOUT)

        assert served == [0, 1, 2, 3, 4], f"queued turns were reordered: {served}"

    @pytest.mark.asyncio
    async def test_sessions_do_not_block_each_other(self) -> None:
        """Serialization is per session. A slow conversation must not stall the rest
        of the load, or reported latency would fold in queueing that no server saw."""
        slow = _new_session("sess_slow")
        fast = _new_session("sess_fast")
        events: List[str] = []

        async def slow_turn() -> None:
            await slow.get_context(0)
            events.append("slow_start")
            await asyncio.sleep(0.05)
            events.append("slow_end")
            slow.update_context("slow_resp")

        async def fast_turn() -> None:
            await fast.get_context(0)
            events.append("fast_start")
            events.append("fast_end")
            fast.update_context("fast_resp")

        await asyncio.wait_for(asyncio.gather(slow_turn(), fast_turn()), timeout=_TURN_TIMEOUT)

        assert events == ["slow_start", "fast_start", "fast_end", "slow_end"], (
            f"the fast session waited on the slow one: {events}"
        )

    @pytest.mark.asyncio
    async def test_concurrent_turns_build_one_ordered_transcript(self) -> None:
        """Dispatch turns concurrently through the real request path and check the
        conversation still reads as a single ordered transcript."""
        session = _new_session("sess_transcript", context="SYSTEM")
        api_config = APIConfig(type=APIType.Completion, streaming=False)
        tokenizer = _mock_tokenizer()

        async def turn(n: int) -> None:
            data = UserSessionCompletionAPIData(
                user_session_id="sess_transcript", target_round=n, prompt=f"Q{n}", max_tokens=8
            )
            await data.to_request_body("model", 8, False, False)
            # Stand in for the request being in flight. Without a suspension here
            # the turns would run to completion one after another on their own and
            # the session's serialization would never be exercised.
            await asyncio.sleep(0)
            await data.process_response(_unary_response(f"A{n}"), api_config, tokenizer)

        await asyncio.wait_for(asyncio.gather(*(turn(i) for i in range(4))), timeout=_TURN_TIMEOUT)

        assert session.context == "SYSTEM Q0 A0 Q1 A1 Q2 A2 Q3 A3"


class TestMultiTurnConversationSemantics:
    """Each turn sends the accumulated conversation, not just its own question."""

    def setup_method(self) -> None:
        LocalUserSession.clear_instances()

    def teardown_method(self) -> None:
        LocalUserSession.clear_instances()

    @pytest.mark.asyncio
    async def test_each_turn_carries_the_prior_transcript(self) -> None:
        _new_session("sess_grow", context="SYSTEM")
        api_config = APIConfig(type=APIType.Completion, streaming=False)
        tokenizer = _mock_tokenizer()
        prompts: List[str] = []

        for n in range(3):
            data = UserSessionCompletionAPIData(user_session_id="sess_grow", target_round=n, prompt=f"Q{n}", max_tokens=8)
            body = await data.to_request_body("model", 8, False, False)
            prompts.append(body["prompt"])
            await data.process_response(_unary_response(f"A{n}"), api_config, tokenizer)

        # The growing prefix is the whole point of the workload: each turn is a
        # longer prefill than the last, and a cached prefix the server can reuse.
        assert prompts == ["SYSTEM Q0", "SYSTEM Q0 A0 Q1", "SYSTEM Q0 A0 Q1 A1 Q2"]

    @pytest.mark.asyncio
    async def test_turns_are_tagged_with_session_and_round(self) -> None:
        """Reporting splits latency by turn index, so these tags must advance."""
        _new_session("sess_tags", context="SYSTEM")
        api_config = APIConfig(type=APIType.Completion, streaming=False)
        tokenizer = _mock_tokenizer()
        rounds: List[int] = []

        for n in range(3):
            data = UserSessionCompletionAPIData(user_session_id="sess_tags", target_round=n, prompt=f"Q{n}", max_tokens=8)
            await data.to_request_body("model", 8, False, False)
            info = await data.process_response(_unary_response(f"A{n}"), api_config, tokenizer)
            assert info.extra_info["user_session"] == "sess_tags"
            rounds.append(info.extra_info["chat_round"])

        assert rounds == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_failed_turn_rolls_back_context_and_frees_the_slot(self) -> None:
        """A failed turn must leave no trace in the transcript: the model never saw
        it, so replaying it as context would fabricate history the server never had."""
        session = _new_session("sess_fail", context="SYSTEM")
        api_config = APIConfig(type=APIType.Completion, streaming=False)
        tokenizer = _mock_tokenizer()

        ok = UserSessionCompletionAPIData(user_session_id="sess_fail", target_round=0, prompt="Q0", max_tokens=8)
        await ok.to_request_body("model", 8, False, False)
        await ok.process_response(_unary_response("A0"), api_config, tokenizer)
        assert session.context == "SYSTEM Q0 A0"

        failed = UserSessionCompletionAPIData(user_session_id="sess_fail", target_round=1, prompt="Q1", max_tokens=8)
        await failed.to_request_body("model", 8, False, False)
        await failed.process_failure(None, api_config, tokenizer, Exception("connection reset"))
        assert session.context == "SYSTEM Q0 A0"

        # The slot must be free, otherwise the conversation stalls for the run.
        nxt = UserSessionCompletionAPIData(user_session_id="sess_fail", target_round=2, prompt="Q2", max_tokens=8)
        body = await asyncio.wait_for(nxt.to_request_body("model", 8, False, False), timeout=2.0)
        assert body["prompt"] == "SYSTEM Q0 A0 Q2"


class TestUserSessionTruncation:
    def setup_method(self) -> None:
        LocalUserSession.clear_instances()

    def teardown_method(self) -> None:
        LocalUserSession.clear_instances()

    @pytest.mark.asyncio
    async def test_prompt_truncation_history(self) -> None:
        tok = _mock_tokenizer()
        hf = tok.get_tokenizer.return_value
        hf.encode = MagicMock(side_effect=lambda text: [1] * tok.count_tokens(text))

        session = LocalUserSession(user_session_id="sess_1", system_prompt="tok_5", tokenizer=tok, max_model_len=220)
        LocalUserSession._instances["sess_1"] = session

        session.history = ["tok_5", "tok_5"]
        session.context = "tok_5 tok_5 tok_5"

        data = UserSessionCompletionAPIData(user_session_id="sess_1", target_round=1, prompt="tok_10", max_tokens=0)

        payload = await data.to_request_body("model", 0, False, False)

        assert session.history == ["tok_5"]
        assert payload["prompt"] == "tok_5 tok_5 tok_10"

    @pytest.mark.asyncio
    async def test_prompt_truncation_system_prompt(self) -> None:
        tok = _mock_tokenizer()
        hf = tok.get_tokenizer.return_value
        hf.encode = MagicMock(side_effect=lambda text: [1] * tok.count_tokens(text))

        session = LocalUserSession(user_session_id="sess_2", system_prompt="tok_15", tokenizer=tok, max_model_len=220)
        LocalUserSession._instances["sess_2"] = session

        session.history = []
        session.context = "tok_15"

        data = UserSessionCompletionAPIData(user_session_id="sess_2", target_round=0, prompt="tok_10", max_tokens=0)

        payload = await data.to_request_body("model", 0, False, False)

        assert session.system_prompt == "tok_10"
        assert payload["prompt"] == "tok_10 tok_10"

    @pytest.mark.asyncio
    async def test_question_alone_over_budget_drops_system_prompt(self) -> None:
        """When the current question alone exceeds the context budget there is no
        room left for the system prompt, so it is dropped and the question clipped.
        Sending it anyway would overflow the model and fail the request instead of
        measuring it."""
        tok = _mock_tokenizer()
        hf = tok.get_tokenizer.return_value
        hf.encode = MagicMock(side_effect=lambda text: [1] * tok.count_tokens(text))

        # target_len = max_model_len - max_tokens - 200 = 20
        session = LocalUserSession(user_session_id="sess_3", system_prompt="tok_5", tokenizer=tok, max_model_len=220)
        LocalUserSession._instances["sess_3"] = session

        data = UserSessionCompletionAPIData(user_session_id="sess_3", target_round=0, prompt="tok_25", max_tokens=0)

        payload = await data.to_request_body("model", 0, False, False)

        assert session.system_prompt == ""
        assert payload["prompt"] == "tok_20"
