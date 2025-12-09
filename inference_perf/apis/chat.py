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

import json
import time

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

    async def to_payload(self, model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool) -> dict[str, Any]:
        if self.max_tokens == 0:
            self.max_tokens = max_tokens
        if streaming:
                return {
                    "model": model_name,
                    "messages": [{"role": m.role, "content": m.content} for m in self.messages],
                    "max_tokens": self.max_tokens,
                    "ignore_eos": ignore_eos,
                    "stream": streaming,
                    "stream_options": {"include_usage": "true"},
                }
        else:
                return {
                    "model": model_name,
                    "messages": [{"role": m.role, "content": m.content} for m in self.messages],
                    "max_tokens": self.max_tokens,
                    "ignore_eos": ignore_eos,
                }

    async def process_response(self, response: ClientResponse, config: APIConfig, tokenizer: CustomTokenizer) -> InferenceInfo:
        if config.streaming:
            output_text = ""
            output_token_times: List[float] = []
            buffer = b""
            async for chunk in response.content.iter_any():
                buffer += chunk
                while b"\n\n" in buffer:
                    message, buffer = buffer.split(b"\n\n", 1)
                    output_token_times.append(time.perf_counter())
                    for line in message.split(sep=b"\n"):
                        if line.startswith(b"data:"):
                            data_str = line.removeprefix(b"data: ").strip()
                            if data_str == b"[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                choices = data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content")
                                    if content:
                                        output_text += content
                            except (json.JSONDecodeError, IndexError):
                                continue
                    else:
                        continue
                    break

            prompt_text = "".join([msg.content for msg in self.messages if msg.content])
            prompt_len = tokenizer.count_tokens(prompt_text)
            output_len = tokenizer.count_tokens(output_text)
            return InferenceInfo(
                input_tokens=prompt_len,
                output_tokens=output_len,
                output_token_times=output_token_times,
            )
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
