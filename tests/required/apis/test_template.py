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
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from inference_perf.apis import TemplateAPIData, UnaryResponseMetrics
from inference_perf.config import APIConfig, APIType, TemplateConfig


def _make_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.count_tokens = lambda text, **kw: len((text or "").split())
    return tok


def _make_response(body: dict[str, Any]) -> MagicMock:
    response = MagicMock()
    response.json = AsyncMock(return_value=body)
    return response


_TEMPLATE = TemplateConfig(
    route="/generate",
    request_template='{"model": {{ model }}, "inputs": {{ prompt }}, "parameters": {"max_new_tokens": {{ max_tokens }}, "ignore_eos": {{ ignore_eos }}}}',
    output_path="generated_text",
    input_tokens_path="details.prefill_tokens",
    output_tokens_path="details.generated_tokens",
)

_CONFIG = APIConfig(type=APIType.Template, template=_TEMPLATE)


def test_template_api_type_and_route() -> None:
    data = TemplateAPIData(prompt="hello", template=_TEMPLATE)
    assert data.get_api_type() == APIType.Template
    assert data.get_route() == "/generate"


def test_template_requires_template_config() -> None:
    data = TemplateAPIData(prompt="hello")
    with pytest.raises(ValueError, match="template configuration is required"):
        data.get_route()


@pytest.mark.asyncio
async def test_template_request_body_unquoted_and_quoted_placeholders() -> None:
    unquoted = TemplateAPIData(prompt='Say "hello"\nworld\\!', max_tokens=64, template=_TEMPLATE)
    assert await unquoted.to_request_body("test-model", 128, True, False) == {
        "model": "test-model",
        "inputs": 'Say "hello"\nworld\\!',
        "parameters": {"max_new_tokens": 64, "ignore_eos": True},
    }

    quoted_template = TemplateConfig(
        route="/v1/custom",
        request_template='{"model": "{{ model }}", "prompt": "User: {{ prompt }}", "max_tokens": {{ max_tokens }}}',
        output_path="choices[0].text",
    )
    # Prompt containing quotes, newlines, and a literal placeholder-like string must not break JSON or double-expand.
    quoted = TemplateAPIData(prompt='Line 1 "quoted"\nLine 2 {{ model }}', template=quoted_template)
    assert await quoted.to_request_body("llama-3", 256, False, False) == {
        "model": "llama-3",
        "prompt": 'User: Line 1 "quoted"\nLine 2 {{ model }}',
        "max_tokens": 256,
    }


@pytest.mark.asyncio
async def test_template_process_response_uses_extracted_tokens() -> None:
    data = TemplateAPIData(prompt="one two", template=_TEMPLATE)
    response = _make_response(
        {
            "generated_text": "three four five",
            "details": {"prefill_tokens": 7, "generated_tokens": 11},
        }
    )

    info = await data.process_response(response, _CONFIG, _make_tokenizer(), lora_adapter="lora-a")

    assert data.model_response == "three four five"
    assert info.request_metrics.text.input_tokens == 7
    assert isinstance(info.response_metrics, UnaryResponseMetrics)
    assert info.response_metrics.output_tokens == 11
    assert info.response_metrics.server_usage == {"prompt_tokens": 7, "completion_tokens": 11}
    assert info.lora_adapter == "lora-a"


@pytest.mark.asyncio
async def test_template_process_response_falls_back_to_client_tokenizer_and_jsonpath() -> None:
    jsonpath_template = TemplateConfig(
        route="/v1/generate",
        request_template='{"prompt": {{ prompt }}, "max_tokens": {{ max_tokens }}}',
        output_path="$.predictions[0].output",
    )
    config = APIConfig(type=APIType.Template, template=jsonpath_template)
    data = TemplateAPIData(prompt="one two", template=jsonpath_template)
    response = _make_response({"predictions": [{"output": "alpha beta gamma"}]})

    info = await data.process_response(response, config, _make_tokenizer())

    assert data.model_response == "alpha beta gamma"
    assert info.request_metrics.text.input_tokens == 2
    assert isinstance(info.response_metrics, UnaryResponseMetrics)
    assert info.response_metrics.output_tokens == 3
    assert info.response_metrics.server_usage is None


@pytest.mark.asyncio
async def test_template_process_response_handles_list_and_missing_output() -> None:
    list_template = TemplateConfig(
        route="/v1/generate",
        request_template='{"prompt": {{ prompt }}}',
        output_path="content[].text",
    )
    config = APIConfig(type=APIType.Template, template=list_template)
    data = TemplateAPIData(prompt="hello world")

    info = await data.process_response(
        _make_response({"content": [{"text": "foo "}, {"text": "bar"}]}),
        config,
        _make_tokenizer(),
    )
    assert data.model_response == "foo bar"
    assert info.response_metrics is not None
    assert info.response_metrics.output_tokens == 2

    empty_info = await data.process_response(_make_response({}), config, _make_tokenizer())
    assert data.model_response == ""
    assert empty_info.response_metrics is not None
    assert empty_info.response_metrics.output_tokens == 0
