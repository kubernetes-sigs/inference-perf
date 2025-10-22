import logging
import asyncio
from typing import Any, Optional, Tuple
from pydantic import ConfigDict, Field

from aiohttp import ClientResponse
from inference_perf.apis import CompletionAPIData, InferenceInfo
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.config import APIConfig

logger = logging.getLogger(__name__)


class LocalUserSession:
    user_session_id: str
    context: str

    def __init__(self, user_session_id: str, context: str = ""):
        self.user_session_id = user_session_id
        self.contexts = context if context else ""
        self._current_round = 0
        self._in_flight: asyncio.Lock = asyncio.Lock()
        self._waiting_rounds: asyncio.PriorityQueue[Tuple[int, asyncio.Future[bool]]] = asyncio.PriorityQueue()

    async def get_context(self, round: int) -> str:
        if not self._waiting_rounds.empty() or self._in_flight.locked():
            # entering waiting queue
            future: asyncio.Future[bool] = asyncio.Future()
            self._waiting_rounds.put_nowait((round, future))
            await future
        await self._in_flight.acquire()
        self._current_round += 1
        return self.contexts

    def update_context(self, response: str) -> None:
        self.contexts = response

        if not self._waiting_rounds.empty():
            _, future = self._waiting_rounds.get_nowait()
            future.set_result(True)

        self._in_flight.release()


class UserSessionCompletionAPIData(CompletionAPIData):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_session: LocalUserSession = Field(exclude=True)
    target_round: int

    async def to_payload(self, model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool) -> dict[str, Any]:
        self._session_context = await self.user_session.get_context(self.target_round)
        self.prompt = self._session_context + " " + self.prompt
        # TODO: The combined prompt (session context + current prompt) might exceed the model's
        #       maximum sequence length. Implement truncation logic/strategy to prevent
        #       errors/failures from the inference server.
        return await super().to_payload(model_name, max_tokens, ignore_eos, streaming)

    def update_inference_info(self, inference_info: InferenceInfo) -> None:
        inference_info.extra_info["user_session"] = self.user_session.user_session_id
        inference_info.extra_info["chat_round"] = self.user_session._current_round

    async def process_response(self, response: ClientResponse, config: APIConfig, tokenizer: CustomTokenizer) -> InferenceInfo:
        inference_info = await super().process_response(response, config, tokenizer)
        self.update_inference_info(inference_info)
        self.user_session.update_context(self.prompt + " " + self.output_token)
        return inference_info

    async def process_failure(
        self, response: Optional[ClientResponse], config: APIConfig, tokenizer: CustomTokenizer, exception: Exception
    ) -> Optional[InferenceInfo]:
        # no response returned, use context from the last round
        inference_info = InferenceInfo()
        self.update_inference_info(inference_info)
        self.user_session.update_context(self._session_context)
        return inference_info
