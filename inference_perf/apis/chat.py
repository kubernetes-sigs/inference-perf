# Copyright 2025 The Kubernetes Authors.
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

from typing import Any, List
from aiohttp import ClientResponse
from pydantic import BaseModel
from inference_perf.apis import InferenceAPIData, InferenceInfo
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.config import APIConfig, APIType


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionAPIData(InferenceAPIData):
    messages: List[ChatMessage]
    max_tokens: int = 0

    def get_api_type(self) -> APIType:
        return APIType.Chat

    def get_route(self) -> str:
        return "/v1/chat/completions"

    def to_payload(self, model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool) -> dict[str, Any]:
        if streaming:
            raise Exception("Generating streaming request payloads for the Chat API is not currently supported.")
        if self.max_tokens == 0:
            self.max_tokens = max_tokens
        return {
            "model": model_name,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "max_tokens": self.max_tokens,
            "ignore_eos": ignore_eos,
        }

    async def process_response(self, response: ClientResponse, config: APIConfig, tokenizer: CustomTokenizer) -> InferenceInfo:
        if config.streaming:
            raise Exception("Decoding streamed responses from the Chat API is not currently supported")
        else:
            data = await response.json()
            prompt_len = tokenizer.count_tokens("".join([m.content for m in self.messages]))
            choices = data.get("choices", [])
            if len(choices) == 0:
                return InferenceInfo(input_tokens=prompt_len)
            output_text = "".join([choice.get("message", {}).get("content", "") for choice in choices])
            output_len = tokenizer.count_tokens(output_text)
            return InferenceInfo(
                input_tokens=prompt_len,
                output_tokens=output_len,
            )
