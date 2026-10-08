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

from typing import Any, Dict, List, Optional

from aiohttp import ClientResponse
from inference_perf.apis import InferenceAPIData, InferenceInfo, UnaryResponseMetrics
from inference_perf.payloads import RequestBody, RequestMetrics, Text
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.config import APIConfig, APIType, RerankConfig


class RerankAPIData(InferenceAPIData):
    query: str
    documents: List[str]
    route: str = "/v1/rerank"
    query_field: str = "query"
    documents_field: str = "documents"
    top_n: Optional[int] = None

    @classmethod
    def from_query_and_documents(cls, query: str, documents: List[str], options: Optional[RerankConfig]) -> "RerankAPIData":
        """Build a request scoring `documents` against `query` with the configured options."""
        if not documents:
            raise ValueError("a rerank request needs at least one document")
        options = options or RerankConfig()
        return cls(
            query=query,
            documents=documents,
            route=options.route,
            query_field=options.query_field,
            documents_field=options.documents_field,
            top_n=options.top_n,
        )

    def get_api_type(self) -> APIType:
        return APIType.Rerank

    def get_route(self) -> str:
        return self.route

    def _count_prompt_tokens(self, tokenizer: CustomTokenizer) -> int:
        # vLLM scores each query/document pair as a separate scoring input and sums
        # prompt-token usage across them, so the query is counted once per document
        # rather than once per request.
        query_tokens = tokenizer.count_tokens(self.query)
        return sum(query_tokens + tokenizer.count_tokens(document) for document in self.documents)

    def _resolve_prompt_tokens(self, server_usage: Optional[Dict[str, Any]], tokenizer: CustomTokenizer) -> int:
        """Input tokens as reported by the server, falling back to client-side tokenization."""
        if server_usage:
            for key in ("prompt_tokens", "total_tokens"):
                value = server_usage.get(key)
                if value is not None:
                    return int(value)
        return self._count_prompt_tokens(tokenizer)

    async def to_request_body(
        self, effective_model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool
    ) -> RequestBody:
        # max_tokens, ignore_eos and streaming only apply to generation; a rerank
        # request generates nothing, so they are not sent.
        return {
            "model": effective_model_name,
            self.query_field: self.query,
            self.documents_field: self.documents,
            **({"top_n": self.top_n} if self.top_n is not None else {}),
        }

    async def process_response(
        self, response: ClientResponse, config: APIConfig, tokenizer: CustomTokenizer, lora_adapter: Optional[str] = None
    ) -> InferenceInfo:
        data = await response.json()
        server_usage = data.get("usage")
        # Scores and document indices are not kept: only latency and token counts
        # are measured. With no generated tokens there are no token timestamps, so
        # TTFT, TPOT and ITL stay unset in the reports.
        return InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=self._resolve_prompt_tokens(server_usage, tokenizer))),
            response_metrics=UnaryResponseMetrics(output_tokens=0, server_usage=server_usage),
            lora_adapter=lora_adapter,
        )
