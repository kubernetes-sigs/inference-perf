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


class LocalUserSession:
    user_session_id: str
    context: str
    system_prompt: str
    max_model_len: Optional[int]
    history: list[str]

    _instances: dict[str, "LocalUserSession"] = {}

    def __init__(self, user_session_id: str, context: str = "", system_prompt: str = "", tokenizer: Optional[CustomTokenizer] = None, max_model_len: Optional[int] = None):
        self.user_session_id = user_session_id
        self.context = context if context else ""
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer
        self.max_model_len = max_model_len
        self.history = []
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

    def update_context(self, response: str) -> None:
        if self.system_prompt and self.tokenizer and self.max_model_len:
            history_context = " ".join(self.history) if self.history else ""
            base_len = len(self.system_prompt)
            if history_context:
                base_len += len(history_context) + 1
            turn_content = response[base_len:].strip()
            if turn_content:
                self.history.append(turn_content)

            system_tokens = self.tokenizer.count_tokens(self.system_prompt)
            
            while self.history:
                history_str = " ".join(self.history)
                history_tokens = self.tokenizer.count_tokens(history_str)
                if system_tokens + history_tokens <= self.max_model_len:
                    break
                self.history.pop(0)

            self.context = self.system_prompt + " " + " ".join(self.history) if self.history else self.system_prompt
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

    @property
    def user_session(self) -> LocalUserSession:
        return LocalUserSession.get_instance(self.user_session_id)

    async def to_request_body(
        self, effective_model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool
    ) -> RequestBody:
        self._session_context = await self.user_session.get_context(self.target_round)
        
        if self.user_session.tokenizer and self.user_session.max_model_len:
            target_len = self.user_session.max_model_len - max_tokens - 50
            hf_tokenizer = self.user_session.tokenizer.get_tokenizer()
            
            system_prompt = self.user_session.system_prompt
            history = list(self.user_session.history)
            current_prompt = self.prompt
            
            def get_text(sys, hist, curr):
                h_str = " ".join(hist)
                parts = [sys]
                if h_str:
                    parts.append(h_str)
                parts.append(curr)
                return " ".join(parts)
                
            # Truncate history first (from left)
            while history:
                combined_text = get_text(system_prompt, history, current_prompt)
                token_ids = hf_tokenizer.encode(combined_text)
                if len(token_ids) <= target_len:
                    break
                history.pop(0)
                
            combined_text = get_text(system_prompt, history, current_prompt)
            token_ids = hf_tokenizer.encode(combined_text)
            
            if len(token_ids) > target_len:
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
                    
                combined_text = get_text(system_prompt, [], current_prompt)
                
                if system_prompt != self.user_session.system_prompt:
                    self.user_session.system_prompt = system_prompt
                    
            self.user_session.history = history
            self.user_session.context = system_prompt + " " + " ".join(history) if history else system_prompt
            
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
        inference_info = InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=0)))
        self.update_inference_info(inference_info)
        self.user_session.update_context(self._session_context)
        return inference_info


# TODO: UserSessionChatAPIData need to be implemented
# class UserSessionChatAPIData(ChatCompletionAPIData):
#     ...
