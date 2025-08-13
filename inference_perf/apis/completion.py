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
from inference_perf.apis import InferenceAPIData, InferenceInfo
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.config import APIConfig, APIType


class CompletionAPIData(InferenceAPIData):
    prompt: str
    max_tokens: int = 0

    def get_api_type(self) -> APIType:
        return APIType.Completion

    def get_route(self) -> str:
        return "/v1/completions"

    def to_payload(self, model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool) -> dict[str, Any]:
        if self.max_tokens == 0:
            self.max_tokens = max_tokens
        return {
            "model": model_name,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "ignore_eos": ignore_eos,
            "stream": streaming,
        }

    async def process_response(self, response: ClientResponse, config: APIConfig, tokenizer: CustomTokenizer) -> InferenceInfo:
        if config.streaming:
            output_text = ""
            output_token_times: List[float] = []
            async for chunk_bytes in response.content:
                chunk_bytes = chunk_bytes.strip()
                output_token_times.append(time.perf_counter())
                if not chunk_bytes:
                    continue
                # After removing the "data: " prefix, each chunk decodes to a response json for a single token or "[DONE]" if end of stream
                chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                if chunk != "[DONE]":
                    data = json.loads(chunk)
                    if choices := data.get("choices"):
                        text = choices[0].get("text")
                        output_text += text
            prompt_len = tokenizer.count_tokens(self.prompt)
            output_len = tokenizer.count_tokens(output_text)
            return InferenceInfo(
                input_tokens=prompt_len,
                output_tokens=output_len,
                output_token_times=output_token_times,
            )
        else:
            data = await response.json()
            prompt_len = tokenizer.count_tokens(self.prompt)
            choices = data.get("choices", [])
            if len(choices) == 0:
                return InferenceInfo(input_tokens=prompt_len)
            output_text = choices[0].get("text", "")
            output_len = tokenizer.count_tokens(output_text)
            return InferenceInfo(input_tokens=prompt_len, output_tokens=output_len)
