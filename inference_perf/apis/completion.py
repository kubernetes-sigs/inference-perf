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


from typing import Any, Optional

from aiohttp import ClientResponse
from inference_perf.apis import InferenceAPIData, InferenceInfo, UnaryInferenceResponseInfo, StreamedInferenceResponseInfo
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.config import APIConfig, APIType
from inference_perf.apis.streaming_parser import parse_sse_stream


class CompletionAPIData(InferenceAPIData):
    prompt: str
    max_tokens: int = 0
    model_response: str = ""

    def get_api_type(self) -> APIType:
        return APIType.Completion

    def get_route(self) -> str:
        return "/v1/completions"

    async def to_payload(
        self, effective_model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool
    ) -> dict[str, Any]:
        if self.max_tokens == 0:
            self.max_tokens = max_tokens
        return {
            "model": effective_model_name,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "ignore_eos": ignore_eos,
            "stream": streaming,
            **({"stream_options": {"include_usage": True}} if streaming else {}),
        }

    async def process_response(
        self, response: ClientResponse, config: APIConfig, tokenizer: CustomTokenizer, lora_adapter: Optional[str] = None
    ) -> InferenceInfo:
        if config.streaming:
            # Use shared streaming parser with completion-specific content extraction
            output_text, chunk_times, raw_content, response_chunks, server_usage = await parse_sse_stream(
                response, extract_content=lambda data: data.get("choices", [{}])[0].get("text")
            )

            prompt_len = tokenizer.count_tokens(self.prompt)
            output_len = tokenizer.count_tokens(output_text)
            return InferenceInfo(
                input_tokens=prompt_len,
                response_info=StreamedInferenceResponseInfo(
                    response_chunks=response_chunks,
                    chunk_times=chunk_times,
                    output_tokens=output_len,
                    output_token_times=chunk_times,
                    server_usage=server_usage,
                ),
                lora_adapter=lora_adapter,
                extra_info={"raw_response": raw_content},
            )
        else:
            data = await response.json()
            prompt_len = tokenizer.count_tokens(self.prompt)
            choices = data.get("choices", [])
            if len(choices) == 0:
                return InferenceInfo(input_tokens=prompt_len, lora_adapter=lora_adapter)
            output_text = choices[0].get("text", "")
            output_len = tokenizer.count_tokens(output_text)
            self.model_response = output_text
            return InferenceInfo(
                input_tokens=prompt_len,
                response_info=UnaryInferenceResponseInfo(output_tokens=output_len),
                lora_adapter=lora_adapter,
            )
