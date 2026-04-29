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
from typing import Any, Optional
from pydantic import ConfigDict, Field

from aiohttp import ClientResponse
from inference_perf.apis import CompletionAPIData, InferenceInfo
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.config import APIConfig

logger = logging.getLogger(__name__)


class LocalUserSession:
    user_session_id: str
    context: str

    _instances: dict[str, "LocalUserSession"] = {}

    def __init__(self, user_session_id: str, context: str = ""):
        self.user_session_id = user_session_id
        self.contexts = context if context else ""
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
            # entering waiting queue
            future: asyncio.Future[bool] = asyncio.Future()
            self._waiting_rounds.put_nowait(future)
            await future
        await self._in_flight.acquire()
        self._current_round += 1
        return self.contexts

    def update_context(self, response: str) -> None:
        self.contexts = response

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

    @property
    def user_session(self) -> LocalUserSession:
        return LocalUserSession.get_instance(self.user_session_id)

    async def to_payload(
        self, effective_model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool
    ) -> dict[str, Any]:
        self._session_context = await self.user_session.get_context(self.target_round)
        # TODO: Currently, only prompt style (concat messages) support. Adding support for messages style payload.
        self.prompt = self._session_context + " " + self.prompt
        # TODO: The combined prompt (session context + current prompt) might exceed the model's
        #       maximum sequence length. Implement truncation logic/strategy to prevent
        #       errors/failures from the inference server.
        return await super().to_payload(effective_model_name, max_tokens, ignore_eos, streaming)

    def update_inference_info(self, inference_info: InferenceInfo) -> None:
        inference_info.extra_info["user_session"] = self.user_session_id
        inference_info.extra_info["chat_round"] = self.user_session._current_round

    async def process_response(
        self, response: ClientResponse, config: APIConfig, tokenizer: CustomTokenizer, lora_adapter: Optional[str] = None
    ) -> InferenceInfo:
        inference_info = await super().process_response(response, config, tokenizer)
        self.update_inference_info(inference_info)
        self.user_session.update_context(self.prompt + " " + self.model_response)
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
        inference_info = InferenceInfo()
        self.update_inference_info(inference_info)
        self.user_session.update_context(self._session_context)
        return inference_info


# TODO: UserSessionChatAPIData need to be implemented
# class UserSessionChatAPIData(ChatCompletionAPIData):
#     ...
