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
"""Unit tests for build_request_body, the one place a request body gets the api config's options."""

from inference_perf.apis import AnthropicMessagesAPIData, ChatCompletionAPIData, ChatMessage
from inference_perf.apis.request_body import build_request_body
from inference_perf.config import APIConfig, APIType
from inference_perf.config.apis.config import ResponseFormat, ResponseFormatType

# Structured output that asks for any JSON object, sent as {"type": "json_object"}.
JSON_OBJECT = ResponseFormat(type=ResponseFormatType.JSON_OBJECT)


# A chat record "hi" built for model "m" with api.response_format json_object gets the shape's own body
# plus "response_format": {"type": "json_object"}.
async def test_response_format_is_added_to_a_chat_body() -> None:
    record = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="hi")])

    body = await build_request_body(
        record,
        APIConfig(type=APIType.Chat, response_format=JSON_OBJECT),
        model_name="m",
        max_tokens=16,
        ignore_eos=False,
        streaming=False,
    )

    assert body == {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
        "ignore_eos": False,
        "stream": False,
        "response_format": {"type": "json_object"},
    }


# The Anthropic Messages API has no response_format parameter, so an Anthropic record built with
# api.response_format json_object gets no "response_format" key.
async def test_response_format_is_not_added_to_an_anthropic_body() -> None:
    record = AnthropicMessagesAPIData(messages=[ChatMessage(role="user", content="hi")])

    body = await build_request_body(
        record,
        APIConfig(type=APIType.AnthropicMessages, response_format=JSON_OBJECT),
        model_name="m",
        max_tokens=16,
        ignore_eos=False,
        streaming=False,
    )

    assert "response_format" not in body
