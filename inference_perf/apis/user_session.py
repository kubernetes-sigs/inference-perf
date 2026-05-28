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
import logging
import asyncio
from typing import Optional
from pydantic import ConfigDict, Field

from aiohttp import ClientResponse
from inference_perf.apis import CompletionAPIData, InferenceInfo
from inference_perf.payloads import RequestBody, RequestMetrics, Text
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.config import APIConfig

logger = logging.getLogger(__name__)


def join_conversation_context(system_prompt: str, history: list[str], current_prompt: str = "") -> str:
    """Assembles a full context string from system prompt, conversation history, and an optional current prompt."""
    parts = []
    if system_prompt:
        parts.append(system_prompt)
    if history:
        parts.append(" ".join(history))
    if current_prompt:
        parts.append(current_prompt)
    return " ".join(parts)


class LocalUserSession:
    user_session_id: str
    context: str
    system_prompt: str
    max_model_len: Optional[int]
    history: list[str]
    history_tokens: list[int]

    _instances: dict[str, "LocalUserSession"] = {}

    def __init__(
        self,
        user_session_id: str,
        context: str = "",
        system_prompt: str = "",
        tokenizer: Optional[CustomTokenizer] = None,
        max_model_len: Optional[int] = None,
    ):
        self.user_session_id = user_session_id
        self.context = context if context else ""
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer
        self.max_model_len = max_model_len
        self.history = []
        self.history_tokens = []
        self._current_round = 0
        self._in_flight: Optional[asyncio.Lock] = None
        self._waiting_rounds: Optional[asyncio.Queue[asyncio.Future[bool]]] = None

    @classmethod
    def get_instance(cls, user_session_id: str) -> "LocalUserSession":
        if user_session_id not in cls._instances:
            cls._instances[user_session_id] = LocalUserSession(user_session_id)
        return cls._instances[user_session_id]

    @classmethod
    def clear_instances(cls) -> None:
        cls._instances.clear()

    def _ensure_initialized(self) -> None:
        if self._in_flight is None:
            self._in_flight = asyncio.Lock()
        if self._waiting_rounds is None:
            self._waiting_rounds = asyncio.Queue()

    async def get_context(self, round: int) -> str:
        self._ensure_initialized()
        assert self._waiting_rounds is not None
        assert self._in_flight is not None

        if not self._waiting_rounds.empty() or self._in_flight.locked():
            future: asyncio.Future[bool] = asyncio.Future()
            self._waiting_rounds.put_nowait(future)
            await future
        await self._in_flight.acquire()
        self._current_round += 1
        return self.context

    def update_context(self, response: str, response_len: Optional[int] = None) -> None:
        """
        Updates the session's conversation history and context with the new turn's content.

        This function extracts the newly added prompt/response from the full response string,
        appends it to history, updates the cached history token lengths, and applies a sliding
        window truncation if the total tokens exceed max_model_len. Finally, it releases
        the session's concurrency lock to allow the next turn of this user to begin.
        """
        if self.system_prompt and self.tokenizer and self.max_model_len:
            history_context = " ".join(self.history) if self.history else ""
            base_len = len(self.system_prompt)
            if history_context:
                base_len += len(history_context) + 1
            # Extract the content added in this turn by discarding previous system prompt & history
            turn_content = response[base_len:].strip()
            if turn_content:
                self.history.append(turn_content)
                if response_len is not None:
                    self.history_tokens.append(response_len)
                else:
                    self.history_tokens.append(self.tokenizer.count_tokens(turn_content))

            system_tokens = self.tokenizer.count_tokens(self.system_prompt)
            total_tokens = system_tokens + sum(self.history_tokens)

            # Drop older turns from history if the aggregated context length exceeds limits
            if total_tokens > self.max_model_len:
                while self.history and total_tokens > self.max_model_len:
                    self.history.pop(0)
                    popped_len = self.history_tokens.pop(0)
                    total_tokens -= popped_len

            self.context = join_conversation_context(self.system_prompt, self.history)
        else:
            self.context = response

        self._ensure_initialized()
        assert self._waiting_rounds is not None
        assert self._in_flight is not None

        if not self._waiting_rounds.empty():
            future = self._waiting_rounds.get_nowait()
            future.set_result(True)

        # Release defensively: failure paths can call update_context after the
        # success path already released (e.g. process_response raises post-release,
        # then process_failure runs), so calling release() unconditionally raises
        # RuntimeError("Lock is not acquired."). Skip if already released.
        if self._in_flight.locked():
            self._in_flight.release()


class UserSessionCompletionAPIData(CompletionAPIData):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_session_id: str = Field(exclude=True)
    target_round: int
    prompt_len: Optional[int] = None

    @property
    def user_session(self) -> LocalUserSession:
        return LocalUserSession.get_instance(self.user_session_id)

    async def to_request_body(
        self, effective_model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool
    ) -> RequestBody:
        """
        Constructs the API request payload while managing multi-turn chat history.

        This function:
        1. Acquires the user session concurrency lock to ensure sequential execution of chat rounds.
        2. Evaluates context limitations using the session's maximum token capacity.
        3. Performs a lightweight history pruning (sliding window) if context bounds are exceeded.
        4. In extreme cases (e.g. single prompt exceeds context bounds), performs character-level
           truncation of the system instruction and prompt to avoid model context failures.
        5. Combines the active system prompt, history, and active turn prompt into the final request payload.
        """
        self._session_context = await self.user_session.get_context(self.target_round)

        if self.user_session.tokenizer and self.user_session.max_model_len:
            # 200 token buffer to ensure we stay under model's context length regardless of any tokenization variations
            target_len = self.user_session.max_model_len - max_tokens - 200
            hf_tokenizer = self.user_session.tokenizer.get_tokenizer()

            system_prompt = self.user_session.system_prompt
            history = list(self.user_session.history)
            history_tokens = list(self.user_session.history_tokens)
            if len(history_tokens) != len(history):
                history_tokens = [self.user_session.tokenizer.count_tokens(h) for h in history]
                self.user_session.history_tokens = list(history_tokens)
            current_prompt = self.prompt

            system_tokens = self.user_session.tokenizer.count_tokens(system_prompt)
            current_prompt_tokens = (
                self.prompt_len
                if self.prompt_len is not None
                else self.user_session.tokenizer.count_tokens(current_prompt)
            )

            total_len = system_tokens + sum(history_tokens) + current_prompt_tokens

            # Truncation logic: remove messages from history (oldest first) until the context fits
            if total_len > target_len:
                while history and total_len > target_len:
                    history.pop(0)
                    popped_len = history_tokens.pop(0)
                    total_len -= popped_len

            # If history is fully truncated and still exceeds target_len, truncate system/current prompt
            if total_len > target_len:
                system_ids = hf_tokenizer.encode(system_prompt)
                current_ids = hf_tokenizer.encode(current_prompt)

                available_for_system = target_len - len(current_ids)
                if available_for_system > 0:
                    system_ids = system_ids[:available_for_system]
                    system_prompt = hf_tokenizer.decode(system_ids, skip_special_tokens=True)
                else:
                    system_prompt = ""
                    current_ids = current_ids[:target_len]
                    current_prompt = hf_tokenizer.decode(current_ids, skip_special_tokens=True)

                if system_prompt != self.user_session.system_prompt:
                    self.user_session.system_prompt = system_prompt
                    self.user_session.history_tokens = []

                combined_text = (
                    system_prompt + " " + current_prompt
                    if system_prompt
                    else current_prompt
                )
            else:
                combined_text = join_conversation_context(system_prompt, history, current_prompt)

            self.user_session.history = history
            self.user_session.history_tokens = history_tokens
            self.user_session.context = join_conversation_context(system_prompt, history, "")
            self.prompt = combined_text
        else:
            self.prompt = self._session_context + " " + self.prompt

        return await super().to_request_body(effective_model_name, max_tokens, ignore_eos, streaming)

    def update_inference_info(self, inference_info: InferenceInfo) -> None:
        inference_info.extra_info["user_session"] = self.user_session_id
        inference_info.extra_info["chat_round"] = self.user_session._current_round

    async def process_response(
        self, response: ClientResponse, config: APIConfig, tokenizer: CustomTokenizer, lora_adapter: Optional[str] = None
    ) -> InferenceInfo:
        inference_info = await super().process_response(response, config, tokenizer)
        self.update_inference_info(inference_info)
        total_turn_tokens = (
            inference_info.request_metrics.text.input_tokens +
            (inference_info.response_metrics.output_tokens if inference_info.response_metrics else 0)
        )
        self.user_session.update_context(self.prompt + " " + self.model_response, response_len=total_turn_tokens)
        return inference_info

    async def process_failure(
        self,
        response: Optional[ClientResponse],
        config: APIConfig,
        tokenizer: CustomTokenizer,
        exception: Exception,
        lora_adapter: Optional[str] = None,
    ) -> Optional[InferenceInfo]:
        # no response returned, use context from the last round
        inference_info = InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=0)))
        self.update_inference_info(inference_info)
        self.user_session.update_context(self._session_context, response_len=None)
        return inference_info


# TODO: UserSessionChatAPIData need to be implemented
# class UserSessionChatAPIData(ChatCompletionAPIData):
#     ...
