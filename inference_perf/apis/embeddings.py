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

from typing import Any, Dict, List, Optional, Union

from aiohttp import ClientResponse
from inference_perf.apis import InferenceAPIData, InferenceInfo, UnaryResponseMetrics
from inference_perf.payloads import RequestBody, RequestMetrics, Text
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.config import APIConfig, APIType, EmbeddingsEncodingFormat


class EmbeddingsAPIData(InferenceAPIData):
    # A single string, or a batch of strings embedded in one request.
    input: Union[str, List[str]]
    # None leaves these out of the request so the server uses its defaults.
    dimensions: Optional[int] = None
    encoding_format: Optional[EmbeddingsEncodingFormat] = None

    def get_api_type(self) -> APIType:
        return APIType.Embeddings

    def get_route(self) -> str:
        return "/v1/embeddings"

    def _count_prompt_tokens(self, tokenizer: CustomTokenizer) -> int:
        texts = [self.input] if isinstance(self.input, str) else self.input
        return sum(tokenizer.count_tokens(text) for text in texts)

    def _resolve_prompt_tokens(self, server_usage: Optional[Dict[str, Any]], tokenizer: CustomTokenizer) -> int:
        """Input tokens as reported by the server, falling back to client-side tokenization."""
        prompt_tokens = server_usage.get("prompt_tokens") if server_usage else None
        if prompt_tokens is not None:
            return int(prompt_tokens)
        return self._count_prompt_tokens(tokenizer)

    async def to_request_body(
        self, effective_model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool
    ) -> RequestBody:
        # max_tokens, ignore_eos and streaming only apply to generation; an
        # embeddings request generates nothing, so they are not sent.
        return {
            "model": effective_model_name,
            "input": self.input,
            **({"dimensions": self.dimensions} if self.dimensions is not None else {}),
            **({"encoding_format": self.encoding_format.value} if self.encoding_format is not None else {}),
        }

    async def process_response(
        self, response: ClientResponse, config: APIConfig, tokenizer: CustomTokenizer, lora_adapter: Optional[str] = None
    ) -> InferenceInfo:
        data = await response.json()
        server_usage = data.get("usage")
        # The embedding vectors themselves are not kept: only latency and token
        # counts are measured. With no generated tokens there are no token
        # timestamps, so TTFT, TPOT and ITL stay unset in the reports.
        return InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=self._resolve_prompt_tokens(server_usage, tokenizer))),
            response_metrics=UnaryResponseMetrics(output_tokens=0, server_usage=server_usage),
            lora_adapter=lora_adapter,
        )
