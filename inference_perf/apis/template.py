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

import jmespath
from aiohttp import ClientResponse
from inference_perf.apis import InferenceAPIData, InferenceInfo, UnaryResponseMetrics
from inference_perf.payloads import RequestBody, RequestMetrics, Text
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.config import APIConfig, APIType, TemplateConfig


class TemplateAPIData(InferenceAPIData):
    prompt: str
    max_tokens: int = 0
    # Only used to count the prompt on the client, as in CompletionAPIData. The
    # template decides what the server receives.
    add_special_tokens: Optional[bool] = None
    template: TemplateConfig

    def get_api_type(self) -> APIType:
        return APIType.Template

    def get_route(self) -> str:
        return self.template.route

    def _count_prompt_tokens(self, tokenizer: CustomTokenizer) -> int:
        return tokenizer.count_tokens(
            self.prompt, add_special_tokens=self.add_special_tokens if self.add_special_tokens is not None else True
        )

    def _resolve_prompt_tokens(self, server_usage: Optional[Dict[str, Any]], tokenizer: CustomTokenizer) -> int:
        """Input tokens as reported by the server, falling back to client-side tokenization."""
        prompt_tokens = server_usage.get("prompt_tokens") if server_usage else None
        if prompt_tokens is not None:
            return int(prompt_tokens)
        return self._count_prompt_tokens(tokenizer)

    def _server_usage(self, body: Any) -> Optional[Dict[str, Any]]:
        """Token counts read from the response, under the keys of an OpenAI usage object."""
        usage: Dict[str, Any] = {}
        for key, path in (
            ("prompt_tokens", self.template.input_tokens_path),
            ("completion_tokens", self.template.output_tokens_path),
        ):
            value = jmespath.search(path, body) if path is not None else None
            # bool is a subclass of int, but it is never a token count.
            if isinstance(value, int) and not isinstance(value, bool):
                usage[key] = value
        return usage or None

    async def to_request_body(
        self, effective_model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool
    ) -> RequestBody:
        if self.max_tokens == 0:
            self.max_tokens = max_tokens
        # ignore_eos and streaming are not sent. The template holds every option
        # the server takes.
        return self.template.render_body({"prompt": self.prompt, "max_tokens": self.max_tokens, "model": effective_model_name})

    async def process_response(
        self, response: ClientResponse, config: APIConfig, tokenizer: CustomTokenizer, lora_adapter: Optional[str] = None
    ) -> InferenceInfo:
        # A custom server does not always set an application/json content type.
        body = await response.json(content_type=None)
        text = jmespath.search(self.template.text_path, body)
        if not isinstance(text, str):
            raise ValueError(f"text_path '{self.template.text_path}' did not select a string in the response")
        server_usage = self._server_usage(body)
        return InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=self._resolve_prompt_tokens(server_usage, tokenizer))),
            # Generated text is a continuation, so it is counted without special tokens.
            response_metrics=UnaryResponseMetrics(
                output_tokens=tokenizer.count_tokens(text, add_special_tokens=False), server_usage=server_usage
            ),
            lora_adapter=lora_adapter,
        )
