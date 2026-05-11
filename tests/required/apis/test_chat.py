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
from unittest.mock import MagicMock

import pytest

from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.config import APIType


@pytest.mark.asyncio
async def test_chat_completion_api_data() -> None:
    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="Hello, world!")])
    assert data.get_api_type() == APIType.Chat
    assert len(data.messages) == 1
    assert await data.to_request_body("test-model", 100, False, False) == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "max_tokens": 100,
        "ignore_eos": False,
        "stream": False,
    }


def test_count_prompt_tokens_includes_prefix_text() -> None:
    """``_count_prompt_tokens`` sums prefix_text tokens alongside message
    tokens — the total reflects the actual prompt sent to the model."""
    tokenizer = MagicMock()
    # One token per whitespace-separated word.
    tokenizer.count_tokens.side_effect = lambda s: len(s.split())

    data = ChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="three words here")],
        prefix_text="prefix has four tokens",
    )
    # 4 (prefix) + 3 (message) = 7
    assert data._count_prompt_tokens(tokenizer) == 7


def test_count_prompt_tokens_without_prefix_text_unchanged() -> None:
    """Existing behavior holds when prefix_text is unset."""
    tokenizer = MagicMock()
    tokenizer.count_tokens.side_effect = lambda s: len(s.split())

    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="five tokens in this prompt")])
    assert data._count_prompt_tokens(tokenizer) == 5
