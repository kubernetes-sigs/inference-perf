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
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from inference_perf.apis import TemplateAPIData, UnaryResponseMetrics
from inference_perf.apis import template as template_module
from inference_perf.config import APIConfig, APIType, TemplateConfig, TemplateResponseConfig


def _make_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.count_tokens = lambda text, **kw: len(text.split())
    return tok


def _make_response(body: Any) -> MagicMock:
    response = MagicMock()
    response.json = AsyncMock(return_value=body)
    return response


# The SGLang native API, as in the example in docs/config.md.
_TEMPLATE = TemplateConfig(
    route="/generate",
    body={"text": "${prompt}", "sampling_params": {"max_new_tokens": "${max_tokens}", "ignore_eos": True}},
    ignore_eos=True,
    response=TemplateResponseConfig(
        text_path="text",
        input_tokens_path="meta_info.prompt_tokens",
        output_tokens_path="meta_info.completion_tokens",
    ),
)
_CONFIG = APIConfig(type=APIType.Template, template=_TEMPLATE)


@pytest.mark.asyncio
async def test_template_route_is_filled_with_the_model() -> None:
    template = TemplateConfig(
        route="/predictions/${model}", body={"text": "${prompt}"}, response=TemplateResponseConfig(text_path="text")
    )
    data = TemplateAPIData(prompt="hello", template=template)

    await data.to_request_body("test-model", 100, False, False)

    assert data.get_api_type() == APIType.Template
    assert data.get_route() == "/predictions/test-model"


def test_template_route_needs_the_request_body_first() -> None:
    # The model name, and with it the route, is only known once the body is built.
    data = TemplateAPIData(prompt="hello", template=_TEMPLATE)

    with pytest.raises(RuntimeError, match="to_request_body"):
        data.get_route()


@pytest.mark.asyncio
async def test_template_request_body_fills_placeholders() -> None:
    data = TemplateAPIData(prompt="Hello, world!", max_tokens=32, template=_TEMPLATE)
    # A string that is only ${max_tokens} is sent as a number. Other values are sent as written.
    assert await data.to_request_body("test-model", 100, True, False) == {
        "text": "Hello, world!",
        "sampling_params": {"max_new_tokens": 32, "ignore_eos": True},
    }


@pytest.mark.asyncio
async def test_template_request_body_fills_model_lists_and_embedded_text() -> None:
    template = TemplateConfig(
        route="/v1/generate",
        body={"model": "${model}", "inputs": ["Question: ${prompt}"], "limit": "${max_tokens}", "note": "costs $$5"},
        response=TemplateResponseConfig(text_path="output"),
    )
    # max_tokens falls back to the client default when the request does not set one.
    data = TemplateAPIData(prompt="hi there", template=template)

    assert await data.to_request_body("test-model", 100, False, False) == {
        "model": "test-model",
        "inputs": ["Question: hi there"],
        "limit": 100,
        "note": "costs $5",
    }


@pytest.mark.asyncio
async def test_template_process_response_reads_text_and_token_counts() -> None:
    data = TemplateAPIData(prompt="one two", template=_TEMPLATE)
    response = _make_response({"text": "a b c", "meta_info": {"prompt_tokens": 9, "completion_tokens": 3}})

    info = await data.process_response(response, _CONFIG, _make_tokenizer())

    assert info.request_metrics.text.input_tokens == 9
    assert isinstance(info.response_metrics, UnaryResponseMetrics)
    assert info.response_metrics.output_tokens == 3
    # The counts are stored under the usage keys the reports read.
    assert info.response_metrics.server_usage == {"prompt_tokens": 9, "completion_tokens": 3}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"text": "a b"},
        {"text": "a b", "meta_info": {"prompt_tokens": "9", "completion_tokens": True}},
    ],
)
async def test_template_process_response_counts_tokens_without_usable_counts(body: dict[str, Any]) -> None:
    # Counts that are missing or not integers are ignored, and the prompt is tokenized on the client.
    data = TemplateAPIData(prompt="one two three", template=_TEMPLATE)

    info = await data.process_response(_make_response(body), _CONFIG, _make_tokenizer())

    assert info.request_metrics.text.input_tokens == 3
    assert info.response_metrics is not None
    assert info.response_metrics.output_tokens == 2
    assert info.response_metrics.server_usage is None


@pytest.mark.asyncio
async def test_template_warns_once_about_a_count_that_is_not_an_integer(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(template_module, "_warned_count_paths", set())
    data = TemplateAPIData(prompt="one two three", template=_TEMPLATE)
    body = {"text": "a b", "meta_info": {"prompt_tokens": 9.0, "completion_tokens": 2}}

    with caplog.at_level(logging.WARNING, logger="inference_perf.apis.template"):
        for _ in range(2):
            info = await data.process_response(_make_response(body), _CONFIG, _make_tokenizer())

    warnings = [record.getMessage() for record in caplog.records if "meta_info.prompt_tokens" in record.getMessage()]
    assert len(warnings) == 1
    assert "selected a float, not an integer" in warnings[0]
    # The float is dropped, so the prompt is counted with the tokenizer.
    assert info.request_metrics.text.input_tokens == 3


@pytest.mark.asyncio
async def test_template_process_response_rejects_a_response_without_text() -> None:
    # A text_path that selects nothing fails the request instead of reporting zero output tokens.
    data = TemplateAPIData(prompt="hello", template=_TEMPLATE)

    with pytest.raises(ValueError, match="text_path 'text' did not select a string"):
        await data.process_response(_make_response({"generated_text": "a b"}), _CONFIG, _make_tokenizer())
