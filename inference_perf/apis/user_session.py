import logging
import asyncio
from typing import Any, Optional
from pydantic import ConfigDict, Field

from aiohttp import ClientResponse
from inference_perf.apis import CompletionAPIData, InferenceInfo
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.config import APIConfig
from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage

logger = logging.getLogger(__name__)


class LocalUserSession:
    user_session_id: str
    context: list[ChatMessage] | str
    group_id: int

    def __init__(self, user_session_id: str, context: list[ChatMessage] | str = "", group_id: int = 0):
        self.user_session_id = user_session_id
        self.context = context if context else ""
        self.group_id = group_id
        self._current_round = 0
        self._in_flight: asyncio.Lock = asyncio.Lock()
        self._waiting_rounds: asyncio.Queue[asyncio.Future[bool]] = asyncio.Queue()

    async def get_context(self, round: int) -> list[ChatMessage] | str:
        if not self._waiting_rounds.empty() or self._in_flight.locked():
            # entering waiting queue
            future: asyncio.Future[bool] = asyncio.Future()
            self._waiting_rounds.put_nowait(future)
            await future
        await self._in_flight.acquire()
        self._current_round += 1
        return self.context

    def update_context(self, response: list[ChatMessage] | str) -> None:
        self.context = response

        if not self._waiting_rounds.empty():
            future = self._waiting_rounds.get_nowait()
            future.set_result(True)

        self._in_flight.release()


class UserSessionCompletionAPIData(CompletionAPIData):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_session: LocalUserSession = Field(exclude=True)
    target_round: int

    async def to_payload(
        self, effective_model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool
    ) -> dict[str, Any]:
        self._session_context = await self.user_session.get_context(self.target_round)
        # TODO: Currently, only prompt style (concat messages) support. Adding support for messages style payload.
        if not isinstance(self._session_context, str):
            raise TypeError("UserSessionCompletionAPIData expects string context")
        self.prompt = self._session_context + " " + self.prompt
        # TODO: The combined prompt (session context + current prompt) might exceed the model's
        #       maximum sequence length. Implement truncation logic/strategy to prevent
        #       errors/failures from the inference server.
        return await super().to_payload(effective_model_name, max_tokens, ignore_eos, streaming)

    def update_inference_info(self, inference_info: InferenceInfo) -> None:
        inference_info.extra_info["user_session"] = self.user_session.user_session_id
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


class UserSessionChatAPIData(ChatCompletionAPIData):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_session: LocalUserSession = Field(exclude=True)
    target_round: int

    async def to_payload(self, model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool) -> dict[str, Any]:
        self._session_context = await self.user_session.get_context(self.target_round)
        # Append current messages to the session context (history)
        # self.messages contains the new user message for this turn (may include system)
        # self._session_context contains the history (system prompt + previous turns)
        if isinstance(self._session_context, list):
            # History already exists, append only the new user message(s)
            # Remove system from current messages if it exists (already in history)
            new_messages = [msg for msg in self.messages if msg.role != "system"]
            full_messages = self._session_context + new_messages
        else:
            # First turn: context is not a list yet, use all messages (including system)
            full_messages = self.messages

        # We temporarily override self.messages to generate payload, then restore?
        # Or just construct payload manually.
        # ChatCompletionAPIData.to_payload uses self.messages.
        # Let's override self.messages for the payload generation, but we need to be careful.
        # Better to just construct the payload here similar to ChatCompletionAPIData.to_payload

        if self.max_tokens == 0:
            self.max_tokens = max_tokens

        return {
            "model": model_name,
            "messages": [{"role": m.role, "content": m.content} for m in full_messages],
            "max_tokens": self.max_tokens,
            "ignore_eos": ignore_eos,
            "stream": streaming,
        }

    def update_inference_info(self, inference_info: InferenceInfo) -> None:
        inference_info.extra_info["user_session"] = self.user_session.user_session_id
        inference_info.extra_info["chat_round"] = self.user_session._current_round

    async def process_response(self, response: ClientResponse, config: APIConfig, tokenizer: CustomTokenizer) -> InferenceInfo:
        inference_info = await super().process_response(response, config, tokenizer)
        self.update_inference_info(inference_info)

        # Update context with the new turn
        # History <- History + User Message + Assistant Response
        # self._session_context is the history before this turn
        # self.messages is the user message(s) for this turn
        # self.model_response is the assistant response text

        new_history = []
        if isinstance(self._session_context, list):
            # History already exists, extend it
            new_history.extend(self._session_context)
            # Add only new user message(s), excluding system (already in history)
            new_messages = [msg for msg in self.messages if msg.role != "system"]
            new_history.extend(new_messages)
        else:
            # First turn: include all messages (system + user)
            new_history.extend(self.messages)

        # Add assistant response
        new_history.append(ChatMessage(role="assistant", content=self.model_response))

        self.user_session.update_context(new_history)
        return inference_info

    async def process_failure(
        self, response: Optional[ClientResponse], config: APIConfig, tokenizer: CustomTokenizer, exception: Exception
    ) -> Optional[InferenceInfo]:
        # no response returned, use context from the last round (do not add new messages)
        inference_info = InferenceInfo()
        self.update_inference_info(inference_info)
        self.user_session.update_context(self._session_context)
        return inference_info
