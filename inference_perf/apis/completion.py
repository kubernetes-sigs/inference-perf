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


from typing import Optional

from aiohttp import ClientResponse
from inference_perf.apis import InferenceAPIData, InferenceInfo, UnaryResponseMetrics, StreamedResponseMetrics
from inference_perf.payloads import RequestBody, RequestMetrics, Text
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.config import APIConfig, APIType
from inference_perf.apis.streaming_parser import parse_sse_stream


class CompletionAPIData(InferenceAPIData):
    prompt: str
    max_tokens: int = 0
    # vLLM extension. When set, the server is forced to generate at least
    # this many tokens before honoring EOS or any stop_token_ids. Critical
    # for deterministic output length across model families whose
    # generation_config.eos_token_id lists differ — `ignore_eos` alone only
    # suppresses the primary EOS, leaving chat-template stop tokens (e.g.
    # <|im_end|>) to terminate generation early.
    min_tokens: Optional[int] = None
    model_response: str = ""

    def get_api_type(self) -> APIType:
        return APIType.Completion

    def get_route(self) -> str:
        return "/v1/completions"

    async def to_request_body(
        self, effective_model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool
    ) -> RequestBody:
        if self.max_tokens == 0:
            self.max_tokens = max_tokens
        body: RequestBody = {
            "model": effective_model_name,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "ignore_eos": ignore_eos,
            "stream": streaming,
            **({"stream_options": {"include_usage": True}} if streaming else {}),
        }
        if self.min_tokens is not None:
            body["min_tokens"] = self.min_tokens
        return body

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
            self.model_response = output_text
            return InferenceInfo(
                request_metrics=RequestMetrics(text=Text(input_tokens=prompt_len)),
                response_metrics=StreamedResponseMetrics(
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
                return InferenceInfo(
                    request_metrics=RequestMetrics(text=Text(input_tokens=prompt_len)),
                    lora_adapter=lora_adapter,
                )
            output_text = choices[0].get("text", "")
            output_len = tokenizer.count_tokens(output_text)
            self.model_response = output_text
            return InferenceInfo(
                request_metrics=RequestMetrics(text=Text(input_tokens=prompt_len)),
                response_metrics=UnaryResponseMetrics(output_tokens=output_len),
                lora_adapter=lora_adapter,
            )
