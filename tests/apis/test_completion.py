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
    assert await data.to_payload("test-model", 100, False, True) == {
        "model": "test-model",
        "prompt": "Hello, world!",
        "max_tokens": 100,
        "ignore_eos": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
