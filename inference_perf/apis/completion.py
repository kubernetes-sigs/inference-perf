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


from typing import Any, Dict, Optional

from aiohttp import ClientResponse
from inference_perf.apis import InferenceAPIData, InferenceInfo, UnaryResponseMetrics, StreamedResponseMetrics
from inference_perf.payloads import RequestBody, RequestMetrics, Text
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.config import APIConfig, APIType
from inference_perf.apis.streaming_parser import parse_sse_stream


class CompletionAPIData(InferenceAPIData):
    prompt: str
    max_tokens: int = 0
    model_response: str = ""
    # None keeps the server's default. False is for prompts that already embed
    # their special tokens (e.g. chat-templated text): it stops the server from
    # prepending another BOS, so its prompt_tokens matches the client's count.
    add_special_tokens: Optional[bool] = None

    def get_api_type(self) -> APIType:
        return APIType.Completion

    def get_route(self) -> str:
        return "/v1/completions"

    def _count_prompt_tokens(self, tokenizer: CustomTokenizer) -> int:
        return tokenizer.count_tokens(
            self.prompt, add_special_tokens=self.add_special_tokens if self.add_special_tokens is not None else True
        )

    def _resolve_prompt_tokens(self, server_usage: Optional[Dict[str, Any]], tokenizer: CustomTokenizer) -> int:
        """Input tokens as reported by the server, falling back to client-side tokenization.

        Server-reported ``usage.prompt_tokens`` is the source of truth when present —
        it reflects exactly what the server tokenized, which client-side tokenization
        can only approximate.
        """
        prompt_tokens = server_usage.get("prompt_tokens") if server_usage else None
        if prompt_tokens is not None:
            return int(prompt_tokens)
        return self._count_prompt_tokens(tokenizer)

    async def to_request_body(
        self, effective_model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool
    ) -> RequestBody:
        if self.max_tokens == 0:
            self.max_tokens = max_tokens
        return {
            "model": effective_model_name,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "ignore_eos": ignore_eos,
            "stream": streaming,
            **({"stream_options": {"include_usage": True}} if streaming else {}),
            **({"add_special_tokens": self.add_special_tokens} if self.add_special_tokens is not None else {}),
        }

    async def process_response(
        self, response: ClientResponse, config: APIConfig, tokenizer: CustomTokenizer, lora_adapter: Optional[str] = None
    ) -> InferenceInfo:
        if config.streaming:
            # Use shared streaming parser with completion-specific content extraction
            output_text, chunk_times, raw_content, response_chunks, server_usage = await parse_sse_stream(
                response, extract_content=lambda data: data.get("choices", [{}])[0].get("text")
            )

            prompt_len = self._resolve_prompt_tokens(server_usage, tokenizer)
            # Generated text is a continuation, not a sequence start: counting it
            # with special tokens would add a BOS the server's completion_tokens
            # never contains.
            output_len = tokenizer.count_tokens(output_text, add_special_tokens=False)
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
            server_usage = data.get("usage")
            prompt_len = self._resolve_prompt_tokens(server_usage, tokenizer)
            choices = data.get("choices", [])
            if len(choices) == 0:
                return InferenceInfo(
                    request_metrics=RequestMetrics(text=Text(input_tokens=prompt_len)),
                    lora_adapter=lora_adapter,
                )
            output_text = choices[0].get("text", "")
            output_len = tokenizer.count_tokens(output_text, add_special_tokens=False)
            self.model_response = output_text
            return InferenceInfo(
                request_metrics=RequestMetrics(text=Text(input_tokens=prompt_len)),
                response_metrics=UnaryResponseMetrics(output_tokens=output_len, server_usage=server_usage),
                lora_adapter=lora_adapter,
            )
