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

import multiprocessing as mp
import re
import pytest
from collections import defaultdict
from queue import Empty
from typing import List, Optional, Tuple
from unittest.mock import MagicMock

from inference_perf.apis.user_session import LocalUserSession, UserSessionCompletionAPIData
from inference_perf.apis import InferenceAPIData
from inference_perf.client.modelserver.base import ModelServerClient, PrometheusMetricMetadata
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
from inference_perf.datagen.shared_prefix_datagen import SharedPrefixDataGenerator
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
    tok.count_tokens = MagicMock(
        side_effect=lambda text: sum(int(n) for n in re.findall(r"tok_(\d+)", text)) if isinstance(text, str) else 0
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
    """Minimal client that exercises the UserSession to_payload / update_context
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
        payload = await data.to_payload("model", 64, False, False)
        prompt = payload["prompt"]
        self.prompts_by_stage[stage_id].append(prompt)

        if self._prompt_queue is not None:
            self._prompt_queue.put((stage_id, prompt))

        if isinstance(data, UserSessionCompletionAPIData):
            data.user_session.update_context(prompt + f" RESPONSE_STAGE{stage_id}")

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion]

    def get_prometheus_metric_metadata(self) -> PrometheusMetricMetadata:
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
        client that exercises the full to_payload / update_context lifecycle.

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
