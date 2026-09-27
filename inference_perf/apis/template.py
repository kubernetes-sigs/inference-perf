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
from inference_perf.apis.base import InferenceAPIData, InferenceInfo, UnaryResponseMetrics
from inference_perf.config import APIConfig, APIType, TemplateConfig
from inference_perf.config.apis.config import compile_response_path, render_request_template
from inference_perf.payloads import RequestBody, RequestMetrics, Text
from inference_perf.utils.custom_tokenizer import CustomTokenizer


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(str(item) for item in value if item is not None)
    return str(value)


def _extract_token_count(expr: Optional[str], data: Any) -> Optional[int]:
    if expr is None:
        return None
    raw = compile_response_path(expr).search(data)
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return None
    return None


class TemplateAPIData(InferenceAPIData):
    prompt: str
    max_tokens: int = 0
    model_response: str = ""
    add_special_tokens: Optional[bool] = None
    template: Optional[TemplateConfig] = None

    def _require_template(self, config: Optional[APIConfig] = None) -> TemplateConfig:
        template = self.template or (config.template if config else None)
        if template is None:
            raise ValueError("template configuration is required for TemplateAPIData")
        return template

    def get_api_type(self) -> APIType:
        return APIType.Template

    def get_route(self) -> str:
        return self._require_template().route

    def _count_prompt_tokens(self, tokenizer: CustomTokenizer) -> int:
        return tokenizer.count_tokens(
            self.prompt, add_special_tokens=self.add_special_tokens if self.add_special_tokens is not None else True
        )

    async def to_request_body(
        self, effective_model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool
    ) -> RequestBody:
        if self.max_tokens == 0:
            self.max_tokens = max_tokens
        template = self._require_template()
        return render_request_template(
            template.request_template,
            prompt=self.prompt,
            max_tokens=self.max_tokens,
            model=effective_model_name,
            ignore_eos=ignore_eos,
            stream=streaming,
        )

    async def process_response(
        self, response: ClientResponse, config: APIConfig, tokenizer: CustomTokenizer, lora_adapter: Optional[str] = None
    ) -> InferenceInfo:
        template = self._require_template(config)
        data = await response.json()

        extracted_output = compile_response_path(template.output_path).search(data)
        output_text = _extract_text(extracted_output)
        self.model_response = output_text

        extracted_input_tokens = _extract_token_count(template.input_tokens_path, data)
        extracted_output_tokens = _extract_token_count(template.output_tokens_path, data)

        prompt_len = extracted_input_tokens if extracted_input_tokens is not None else self._count_prompt_tokens(tokenizer)
        output_len = (
            extracted_output_tokens
            if extracted_output_tokens is not None
            else tokenizer.count_tokens(output_text, add_special_tokens=False)
        )

        server_usage: Optional[dict[str, Any]] = None
        if extracted_input_tokens is not None or extracted_output_tokens is not None:
            server_usage = {}
            if extracted_input_tokens is not None:
                server_usage["prompt_tokens"] = extracted_input_tokens
            if extracted_output_tokens is not None:
                server_usage["completion_tokens"] = extracted_output_tokens

        return InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=prompt_len)),
            response_metrics=UnaryResponseMetrics(output_tokens=output_len, server_usage=server_usage),
            lora_adapter=lora_adapter,
        )
