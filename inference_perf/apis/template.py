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

import logging
from typing import Any, Dict, Optional

import jmespath
from aiohttp import ClientResponse
from inference_perf.apis import InferenceAPIData, InferenceInfo, UnaryResponseMetrics
from inference_perf.payloads import RequestBody, RequestMetrics, Text
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.config import APIConfig, APIType, TemplateConfig

logger = logging.getLogger(__name__)

# Count paths that have already logged a warning in this process. A wrong path
# would otherwise log on every request.
_warned_count_paths: set[str] = set()


class TemplateAPIData(InferenceAPIData):
    prompt: str
    max_tokens: int = 0
    # Only used to count the prompt on the client, as in CompletionAPIData. The
    # template decides what the server receives.
    add_special_tokens: Optional[bool] = None
    template: TemplateConfig
    # The route with ${model} filled in. It is set with the request body, because
    # the model name is only known then.
    route: Optional[str] = None

    def get_api_type(self) -> APIType:
        return APIType.Template

    def get_route(self) -> str:
        if self.route is None:
            raise RuntimeError("the template route is filled in by to_request_body(), which has not run yet")
        return self.route

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
        for key, name, path in (
            ("prompt_tokens", "input_tokens_path", self.template.response.input_tokens_path),
            ("completion_tokens", "output_tokens_path", self.template.response.output_tokens_path),
        ):
            if path is None:
                continue
            value = jmespath.search(path, body)
            # bool is a subclass of int, but it is never a token count.
            if isinstance(value, int) and not isinstance(value, bool):
                usage[key] = value
            elif path not in _warned_count_paths:
                _warned_count_paths.add(path)
                found = "nothing" if value is None else f"a {type(value).__name__}"
                logger.warning(
                    "Template response %s '%s' selected %s, not an integer. The tokenizer count is used instead. "
                    "This is logged once per path.",
                    name,
                    path,
                    found,
                )
        return usage or None

    async def to_request_body(
        self, effective_model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool
    ) -> RequestBody:
        if self.max_tokens == 0:
            self.max_tokens = max_tokens
        self.route = self.template.render_route(effective_model_name)
        # The ignore_eos and streaming arguments are not sent. The template holds
        # every option the server takes.
        return self.template.render_body({"prompt": self.prompt, "max_tokens": self.max_tokens, "model": effective_model_name})

    async def process_response(
        self, response: ClientResponse, config: APIConfig, tokenizer: CustomTokenizer, lora_adapter: Optional[str] = None
    ) -> InferenceInfo:
        # A custom server does not always set an application/json content type.
        body = await response.json(content_type=None)
        text_path = self.template.response.text_path
        text = jmespath.search(text_path, body)
        if not isinstance(text, str):
            raise ValueError(f"text_path '{text_path}' did not select a string in the response")
        server_usage = self._server_usage(body)
        return InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=self._resolve_prompt_tokens(server_usage, tokenizer))),
            # Generated text is a continuation, so it is counted without special tokens.
            response_metrics=UnaryResponseMetrics(
                output_tokens=tokenizer.count_tokens(text, add_special_tokens=False), server_usage=server_usage
            ),
            lora_adapter=lora_adapter,
        )
