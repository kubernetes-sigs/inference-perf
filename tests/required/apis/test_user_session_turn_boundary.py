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

"""A session's next turn starts with the previous prompt and the response text as the server returned it.

The server keeps a turn's KV blocks under the token ids of its prompt followed by the ids it
generated. The next turn can reuse those blocks only if its prompt starts with that same text.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Callable, List
from unittest.mock import MagicMock

import pytest

from inference_perf.apis.user_session import LocalUserSession, UserSessionCompletionAPIData
from inference_perf.config import APIConfig, APIType, CustomTokenizerConfig
from inference_perf.datagen.replay.conversation_replay_datagen import _ConversationReplayAPIData
from inference_perf.utils.custom_tokenizer import CustomTokenizer

API_CONFIG = APIConfig(type=APIType.Completion, streaming=False)

QUESTIONS = [" Q0", "Q1", " Q2", "Q3"]
# Server text for each turn: a leading space, none, and trailing newlines, as a completion can return them.
REPLIES = [" A0.\n", "A1\n\n", " A2", " A3"]


# _unary_response(" A0.") is a non-streaming /v1/completions reply whose choices[0].text is " A0.".
def _unary_response(text: str) -> MagicMock:
    response = MagicMock()
    body: asyncio.Future[Any] = asyncio.Future()
    body.set_result({"choices": [{"text": text}]})
    response.json = MagicMock(return_value=body)
    response.status = 200
    response.headers = {"content-type": "application/json"}
    return response


# Counts whitespace-separated words: "SYSTEM  Q0 A0." is 3 tokens, far below any limit used here.
def _word_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.count_tokens = MagicMock(side_effect=lambda text, **kw: len(text.split()))
    tokenizer.get_tokenizer.return_value.encode = MagicMock(side_effect=lambda text, **kw: text.split())
    return tokenizer


# "plain" has context "SYSTEM" and no tokenizer; turn 0 with " Q0" sends "SYSTEM  Q0".
def _plain_session(session_id: str) -> LocalUserSession:
    return LocalUserSession(user_session_id=session_id, context="SYSTEM")


# "history" keeps turns after system prompt "SYSTEM", as conversation_replay does; turn 0 with " Q0" sends "SYSTEM  Q0".
def _history_session(session_id: str) -> LocalUserSession:
    return LocalUserSession(
        user_session_id=session_id,
        context="SYSTEM",
        system_prompt="SYSTEM",
        tokenizer=_word_tokenizer(),
        max_model_len=10_000,
    )


# "history_no_system" keeps turns with an empty system prompt; turn 0 with " Q0" sends " Q0", leading space kept.
def _history_no_system_session(session_id: str) -> LocalUserSession:
    return LocalUserSession(
        user_session_id=session_id,
        tokenizer=_word_tokenizer(),
        max_model_len=10_000,
        accumulate_history=True,
    )


# "shared_prefix" keeps turns after prefix "PREFIX" with no added space, as shared_prefix does; " Q0" sends "PREFIX Q0".
def _shared_prefix_session(session_id: str) -> LocalUserSession:
    return LocalUserSession(
        user_session_id=session_id,
        context="PREFIX",
        system_prompt="PREFIX",
        tokenizer=_word_tokenizer(),
        max_model_len=10_000,
        accumulate_history=True,
        preserve_initial_prompt_boundary=True,
    )


# Sends QUESTIONS through to_request_body and process_response with REPLIES; returns the four prompts sent.
async def _send_turns(api_data: type[UserSessionCompletionAPIData], session_id: str) -> List[str]:
    sent: List[str] = []
    for n, (question, reply) in enumerate(zip(QUESTIONS, REPLIES, strict=True)):
        data = api_data(user_session_id=session_id, target_round=n, prompt=question, max_tokens=8)
        body = await data.to_request_body("model", 8, False, False)
        sent.append(body["prompt"])
        await data.process_response(_unary_response(reply), API_CONFIG, _word_tokenizer())
    return sent


class TestNextTurnExtendsPreviousTurn:
    def setup_method(self) -> None:
        LocalUserSession.clear_instances()

    def teardown_method(self) -> None:
        LocalUserSession.clear_instances()

    # Turn 0 sends "SYSTEM  Q0" and gets " A0.\n"; turn 1 must start with "SYSTEM  Q0 A0.\n", with no space added
    # before the reply and the trailing newline kept. Same for turns 2 and 3, for each session kind and for both
    # the session request and conversation_replay's request.
    @pytest.mark.parametrize(
        "make_session",
        [_plain_session, _history_session, _history_no_system_session, _shared_prefix_session],
        ids=["plain", "history", "history_no_system", "shared_prefix"],
    )
    @pytest.mark.parametrize(
        "api_data",
        [UserSessionCompletionAPIData, _ConversationReplayAPIData],
        ids=["user_session", "conversation_replay"],
    )
    @pytest.mark.asyncio
    async def test_next_prompt_starts_with_previous_prompt_and_reply(
        self,
        make_session: Callable[[str], LocalUserSession],
        api_data: type[UserSessionCompletionAPIData],
    ) -> None:
        session = make_session("sess_boundary")
        LocalUserSession._instances["sess_boundary"] = session

        sent = await _send_turns(api_data, "sess_boundary")

        for n in range(len(sent) - 1):
            assert sent[n + 1].startswith(sent[n] + REPLIES[n]), (
                f"turn {n + 1} does not continue turn {n} and its reply:\n"
                f"  expected prefix: {sent[n] + REPLIES[n]!r}\n"
                f"  sent:            {sent[n + 1]!r}"
            )


# A Llama-style tokenizer loaded from a local directory: every space becomes "▁" and starts a new piece,
# so "Q0  The" is ▁Q0, ▁, ▁The.
def _metaspace_tokenizer(tmp_path: Path) -> CustomTokenizer:
    pieces = ["<unk>", "▁", "▁PREFIX", "▁Q0", "▁Q1", "▁The", "▁answer", "▁is", "▁Paris."]
    spec: dict[str, Any] = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": True},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": {p: i for i, p in enumerate(pieces)}, "unk_token": "<unk>"},
    }
    (tmp_path / "tokenizer.json").write_text(json.dumps(spec), encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}), encoding="utf-8"
    )
    return CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path=str(tmp_path)))


# Turn 0 sends "PREFIX Q0" (▁PREFIX ▁Q0); the server generates ▁The ▁answer ▁is ▁Paris. and returns
# " The answer is Paris.". Turn 1's ids must start with those six ids; an added space puts a lone ▁ third.
@pytest.mark.asyncio
async def test_next_turn_token_ids_start_with_cached_prompt_and_generated_ids(tmp_path: Path) -> None:
    LocalUserSession.clear_instances()
    try:
        tokenizer = _metaspace_tokenizer(tmp_path)
        hf = tokenizer.get_tokenizer()
        LocalUserSession._instances["sess_ids"] = LocalUserSession(
            user_session_id="sess_ids",
            context="PREFIX",
            system_prompt="PREFIX",
            tokenizer=tokenizer,
            max_model_len=10_000,
            accumulate_history=True,
            preserve_initial_prompt_boundary=True,
        )
        generated = hf.encode(" The answer is Paris.", add_special_tokens=False)
        assert hf.convert_ids_to_tokens(generated) == ["▁The", "▁answer", "▁is", "▁Paris."]

        first = UserSessionCompletionAPIData(user_session_id="sess_ids", target_round=0, prompt=" Q0", max_tokens=8)
        first_prompt = (await first.to_request_body("model", 8, False, False))["prompt"]
        await first.process_response(_unary_response(" The answer is Paris."), API_CONFIG, tokenizer)

        second = UserSessionCompletionAPIData(user_session_id="sess_ids", target_round=1, prompt=" Q1", max_tokens=8)
        second_prompt = (await second.to_request_body("model", 8, False, False))["prompt"]

        cached = hf.encode(first_prompt, add_special_tokens=False) + generated
        second_ids = hf.encode(second_prompt, add_special_tokens=False)
        assert second_ids[: len(cached)] == cached, (
            f"turn 1 {hf.convert_ids_to_tokens(second_ids)} does not start with "
            f"turn 0's prompt and generated tokens {hf.convert_ids_to_tokens(cached)}"
        )
    finally:
        LocalUserSession.clear_instances()
