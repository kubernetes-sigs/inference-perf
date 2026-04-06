#!/usr/bin/env python3

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

"""Tests for otel_trace_utils - LLM output/input reconstruction."""

from inference_perf.datagen.otel_trace_utils import (
    reconstruct_llm_output,
    reconstruct_llm_input,
    estimate_token_count,
    reconstruct_with_token_estimate,
    _extract_message,
    _extract_text_content,
    _extract_tool_calls,
    _format_tool_call,
)


class TestReconstructLLMOutput:
    """Test reconstruct_llm_output with various formats."""

    def test_text_formats(self) -> None:
        """Test text responses in different formats."""
        # JSON string
        assert reconstruct_llm_output('[{"role": "assistant", "content": "Hi"}]') == "Hi"
        # Dict
        assert reconstruct_llm_output({"role": "assistant", "content": "Test"}) == "Test"
        # OpenAI choices
        assert reconstruct_llm_output({"choices": [{"message": {"content": "OK"}}]}) == "OK"
        # Empty/null
        assert reconstruct_llm_output({}) == ""
        assert reconstruct_llm_output({"role": "assistant", "content": None}) == ""

    def test_tool_calls(self) -> None:
        """Test tool call extraction and formatting."""
        response = {
            "role": "assistant",
            "content": "Checking weather",
            "tool_calls": [{"id": "c1", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}],
        }
        result = reconstruct_llm_output(response)
        assert "Checking weather" in result
        assert "<|tool_call|>get_weather" in result
        assert "Paris" in result


class TestReconstructLLMInput:
    """Test reconstruct_llm_input."""

    def test_basic_formats(self) -> None:
        """Test basic input formats."""
        assert reconstruct_llm_input({"content": "Hello"}) == "Hello"
        assert reconstruct_llm_input('{"content": "Test"}') == "Test"
        assert reconstruct_llm_input({}) == ""

    def test_parts_and_lists(self) -> None:
        """Test OTEL parts and content lists."""
        # Parts format
        msg = {"parts": [{"type": "text", "text": "Q1"}, {"type": "text", "content": "Q2"}]}
        result = reconstruct_llm_input(msg)
        assert "Q1" in result and "Q2" in result

        # Content as list
        msg = {"content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]}
        result = reconstruct_llm_input(msg)
        assert "A" in result and "B" in result


class TestHelperFunctions:
    """Test helper functions."""

    def test_extract_message(self) -> None:
        """Test message extraction from various structures."""
        # List
        result = _extract_message([{"role": "assistant", "content": "Hi"}])
        assert result is not None and result["content"] == "Hi"
        # Choices
        result = _extract_message({"choices": [{"message": {"content": "OK"}}]})
        assert result is not None and result["content"] == "OK"
        # Direct
        result = _extract_message({"content": "Test"})
        assert result is not None and result["content"] == "Test"
        # Invalid
        assert _extract_message({"invalid": "data"}) is None

    def test_extract_text_content(self) -> None:
        """Test text content extraction."""
        assert _extract_text_content({"content": "Hello"}) == "Hello"
        assert _extract_text_content({"content": [{"type": "text", "text": "Hi"}]}) == "Hi"
        assert _extract_text_content({"parts": [{"type": "text", "text": "Test"}]}) == "Test"
        assert _extract_text_content({}) == ""

    def test_extract_tool_calls(self) -> None:
        """Test tool call extraction from different formats."""
        # OpenAI format
        msg = {"tool_calls": [{"id": "c1", "function": {"name": "fn", "arguments": "{}"}}]}
        assert _extract_tool_calls(msg)[0]["name"] == "fn"

        # OTEL parts
        msg = {"parts": [{"type": "tool_call", "name": "calc", "arguments": "{}"}]}
        assert _extract_tool_calls(msg)[0]["name"] == "calc"

        # Legacy function_call
        msg = {"function_call": {"name": "old_fn", "arguments": "{}"}}  # type: ignore[dict-item]
        assert _extract_tool_calls(msg)[0]["name"] == "old_fn"

        # No tool calls
        assert _extract_tool_calls({"content": "text"}) == []

    def test_format_tool_call(self) -> None:
        """Test tool call formatting."""
        # String args
        result = _format_tool_call({"name": "fn", "arguments": '{"x": 1}'})
        assert result.startswith("<|tool_call|>fn<|tool_args|>")
        assert result.endswith("<|end|>")

        # Dict args
        result = _format_tool_call({"name": "calc", "arguments": {"x": 5}})
        assert "<|tool_call|>calc" in result
        assert '"x": 5' in result

        # Missing fields
        assert "unknown_function" in _format_tool_call({"arguments": "{}"})
        assert "{}" in _format_tool_call({"name": "fn"})


class TestTokenEstimation:
    """Test token estimation functions."""

    def test_estimate_token_count(self) -> None:
        """Test basic token estimation."""
        assert estimate_token_count("Test") == 1  # 4 chars / 4 = 1
        assert estimate_token_count("This is a test") == 3  # 14 chars / 4 = 3.5 -> 3
        assert estimate_token_count("") == 0
        assert estimate_token_count("Test", chars_per_token=2.0) == 2

    def test_reconstruct_with_token_estimate(self) -> None:
        """Test output reconstruction with token estimate."""
        response = {"role": "assistant", "content": "Test response"}
        result = reconstruct_with_token_estimate(response)
        assert result["raw_output"] == "Test response"
        assert result["estimated_tokens"] > 0
        assert result["character_count"] == 13
