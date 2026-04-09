# Copyright 2025 The Kubernetes Authors.
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
from unittest.mock import Mock
from inference_perf.apis.streaming_parser import parse_sse_stream

@pytest.mark.asyncio
async def test_parse_sse_stream():
    mock_response = Mock()
    mock_content = Mock()
    mock_response.content = mock_content
    
    chunks = [
        b"data: {\"choices\": [{\"delta\": {\"content\": \"Hello\"}}]}\n\n",
        b"data: {\"choices\": [{\"delta\": {\"content\": \" world\"}}]}\n\n",
        b"data: [DONE]\n\n"
    ]
    
    async def mock_iter_any():
        for chunk in chunks:
            yield chunk
            
    mock_content.iter_any = mock_iter_any
    
    extract_content = lambda data: data.get("choices", [{}])[0].get("delta", {}).get("content")
    
    output_text, output_token_times, raw_content = await parse_sse_stream(mock_response, extract_content)
    
    assert output_text == "Hello world"
    assert len(output_token_times) == 3
    assert "Hello" in raw_content
    assert "world" in raw_content
    assert "[DONE]" in raw_content
