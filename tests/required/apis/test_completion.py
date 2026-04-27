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
import pytest
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.config import APIType


@pytest.mark.asyncio
async def test_completion_api_data() -> None:
    data = CompletionAPIData(prompt="Hello, world!")
    assert data.get_api_type() == APIType.Completion
    assert data.prompt == "Hello, world!"
    assert await data.to_request_body("test-model", 100, False, True) == {
        "model": "test-model",
        "prompt": "Hello, world!",
        "max_tokens": 100,
        "ignore_eos": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


@pytest.mark.asyncio
async def test_chat_completion_api_data_with_tools() -> None:
    tool_defs = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ]
    data = ChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="What is the weather?")],
        tool_definitions=tool_defs,
    )
    payload = await data.to_payload("test-model", 100, False, False)
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]
    # Other fields unaffected
    assert payload["model"] == "test-model"
    assert payload["messages"] == [{"role": "user", "content": "What is the weather?"}]


@pytest.mark.asyncio
async def test_chat_completion_api_data_without_tools_has_no_tools_key() -> None:
    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="Hello")])
    payload = await data.to_payload("test-model", 100, False, False)
    assert "tools" not in payload


@pytest.mark.asyncio
async def test_chat_message_with_tool_calls_serialized_correctly() -> None:
    """Tool-call assistant messages must use tool_calls key, not content."""
    tool_calls = [
        {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Paris"}'}}
    ]
    data = ChatCompletionAPIData(
        messages=[
            ChatMessage(role="user", content="What is the weather?"),
            ChatMessage(role="assistant", tool_calls=tool_calls),
        ]
    )
    payload = await data.to_payload("test-model", 100, False, False)
    msgs = payload["messages"]
    assert msgs[0] == {"role": "user", "content": "What is the weather?"}
    assert msgs[1] == {"role": "assistant", "tool_calls": tool_calls}
    assert "content" not in msgs[1]


@pytest.mark.asyncio
async def test_chat_message_content_none_treated_as_empty() -> None:
    """content=None should serialize as empty string (not 'None')."""
    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content=None)])
    payload = await data.to_payload("test-model", 100, False, False)
    assert payload["messages"][0] == {"role": "user", "content": ""}
