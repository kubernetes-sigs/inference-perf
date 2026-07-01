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

import json
from collections.abc import Callable
from typing import Any

from aiohttp import ClientResponse

from inference_perf.apis.base import InferenceAPIData, InferenceInfo, StreamedResponseMetrics, UnaryResponseMetrics
from inference_perf.apis.chat import ChatMessage, _clean_parameters
from inference_perf.apis.streaming_parser import parse_sse_stream
from inference_perf.config import APIConfig, APIType
from inference_perf.payloads import RequestBody, RequestMetrics, Text
from inference_perf.utils.custom_tokenizer import CustomTokenizer


ANTHROPIC_VERSION = "2023-06-01"


def _content_text(content: str | list[dict[str, Any]] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") in ("text", "input_text")
    )


def _tool_input(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments:
        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"arguments": arguments}
    return {}


def _tool_arguments(tool_input: Any) -> str:
    if isinstance(tool_input, str):
        return tool_input
    return json.dumps(tool_input if isinstance(tool_input, dict) else {}, ensure_ascii=False)


def _anthropic_content(content: str | list[dict[str, Any]] | None) -> str | list[dict[str, Any]]:
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in ("text", "tool_use", "tool_result"):
            blocks.append(block)
        elif block_type == "input_text":
            blocks.append({"type": "text", "text": block.get("text", "")})
    return blocks


def _anthropic_message(message: ChatMessage) -> dict[str, Any] | None:
    if message.role == "system":
        return None

    if message.role == "tool":
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id or "",
                    "content": _content_text(message.content),
                }
            ],
        }

    if message.tool_calls:
        content_blocks: list[dict[str, Any]] = []
        text = _content_text(message.content)
        if text:
            content_blocks.append({"type": "text", "text": text})
        for tool_call in message.tool_calls:
            function = tool_call.get("function") or {}
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id", ""),
                    "name": function.get("name", ""),
                    "input": _tool_input(function.get("arguments")),
                }
            )
        return {"role": "assistant", "content": content_blocks}

    role = "assistant" if message.role == "assistant" else "user"
    return {"role": role, "content": _anthropic_content(message.content)}


def _anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function_candidate = tool.get("function")
    function: dict[str, Any] = function_candidate if isinstance(function_candidate, dict) else tool
    name = function.get("name", "")
    result = {
        "name": name,
        "input_schema": _clean_parameters(function.get("parameters") or {"type": "object", "properties": {}}),
    }
    if function.get("description") is not None:
        result["description"] = function["description"]
    return result


def parse_anthropic_content(content: Any) -> tuple[str, dict[str, Any] | None]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    if not isinstance(content, list):
        return "", None

    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(str(block.get("text", "")))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": _tool_arguments(block.get("input")),
                    },
                }
            )

    output_text = "".join(text_parts)
    if tool_calls:
        output_message: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls}
        if output_text:
            output_message["content"] = output_text
        return output_text, output_message
    return output_text, {"role": "assistant", "content": output_text}


def build_anthropic_request_body(
    messages: list[ChatMessage],
    tool_definitions: list[dict[str, Any]] | None,
    effective_model_name: str,
    max_tokens: int,
    streaming: bool,
) -> RequestBody:
    system_prompt = "\n\n".join(_content_text(message.content) for message in messages if message.role == "system")
    anthropic_messages = [converted for message in messages if (converted := _anthropic_message(message)) is not None]

    payload: dict[str, Any] = {
        "model": effective_model_name,
        "messages": anthropic_messages,
        "max_tokens": max_tokens,
        "stream": streaming,
    }
    if system_prompt:
        payload["system"] = system_prompt
    if tool_definitions:
        payload["tools"] = [_anthropic_tool(tool) for tool in tool_definitions]
    return payload


def count_anthropic_prompt_tokens(messages: list[ChatMessage], tokenizer: CustomTokenizer) -> int:
    return sum(tokenizer.count_tokens(_content_text(message.content)) for message in messages)


def _event_index(data: dict[str, Any], fallback: int) -> int:
    index = data.get("index")
    return index if isinstance(index, int) else fallback


def _build_anthropic_stream_handlers() -> tuple[Callable[[dict[str, Any]], str | None], Callable[[str], dict[str, Any]]]:
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    tool_argument_parts: dict[int, list[str]] = {}

    def extract_content(data: dict[str, Any]) -> str | None:
        event_type = data.get("type")
        if event_type == "content_block_start":
            block = data.get("content_block") or {}
            if not isinstance(block, dict):
                return None

            index = _event_index(data, len(tool_calls_by_index))
            block_type = block.get("type")
            if block_type == "tool_use":
                tool_calls_by_index[index] = {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {"name": block.get("name", ""), "arguments": ""},
                }
                tool_input = block.get("input")
                if tool_input:
                    tool_argument_parts.setdefault(index, []).append(_tool_arguments(tool_input))
            elif block_type == "text" and block.get("text"):
                return str(block["text"])

        elif event_type == "content_block_delta":
            delta = data.get("delta") or {}
            if not isinstance(delta, dict):
                return None

            delta_type = delta.get("type")
            if delta_type == "text_delta":
                text = delta.get("text", "")
                return str(text) if text else None
            if delta_type == "input_json_delta":
                partial_json = delta.get("partial_json", "")
                if partial_json:
                    index = _event_index(data, 0)
                    tool_argument_parts.setdefault(index, []).append(str(partial_json))

        return None

    def output_message(output_text: str) -> dict[str, Any]:
        tool_calls: list[dict[str, Any]] = []
        for index in sorted(tool_calls_by_index):
            tool_call = tool_calls_by_index[index]
            function = dict(tool_call["function"])
            function["arguments"] = "".join(tool_argument_parts.get(index, []))
            tool_calls.append(
                {
                    "id": tool_call["id"],
                    "type": tool_call["type"],
                    "function": function,
                }
            )

        if tool_calls:
            message: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls}
            if output_text:
                message["content"] = output_text
            return message

        return {"role": "assistant", "content": output_text}

    return extract_content, output_message


async def parse_anthropic_stream_response(
    response: ClientResponse,
) -> tuple[str, dict[str, Any], list[float], str, list[str], dict[str, Any] | None]:
    extract_content, build_output_message = _build_anthropic_stream_handlers()
    output_text, chunk_times, raw_content, response_chunks, server_usage = await parse_sse_stream(
        response,
        extract_content=extract_content,
    )
    return output_text, build_output_message(output_text), chunk_times, raw_content, response_chunks, server_usage


class AnthropicMessagesAPIData(InferenceAPIData):
    messages: list[ChatMessage]
    max_tokens: int = 0
    tool_definitions: list[dict[str, Any]] | None = None

    def get_api_type(self) -> APIType:
        return APIType.AnthropicMessages

    def get_route(self) -> str:
        return "/v1/messages"

    async def to_request_body(
        self, effective_model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool
    ) -> RequestBody:
        if self.max_tokens == 0:
            self.max_tokens = max_tokens
        return build_anthropic_request_body(
            self.messages,
            self.tool_definitions,
            effective_model_name,
            self.max_tokens,
            streaming,
        )

    def _count_prompt_tokens(self, tokenizer: CustomTokenizer) -> int:
        return count_anthropic_prompt_tokens(self.messages, tokenizer)

    async def process_response(
        self, response: ClientResponse, config: APIConfig, tokenizer: CustomTokenizer, lora_adapter: str | None = None
    ) -> InferenceInfo:
        if config.streaming:
            (
                output_text,
                output_message,
                chunk_times,
                raw_content,
                response_chunks,
                server_usage,
            ) = await parse_anthropic_stream_response(response)
            input_tokens = (server_usage or {}).get("input_tokens")
            output_tokens = (server_usage or {}).get("output_tokens")
            output_len = int(output_tokens) if output_tokens is not None else tokenizer.count_tokens(output_text)
            return InferenceInfo(
                request_metrics=RequestMetrics(
                    text=Text(
                        input_tokens=int(input_tokens) if input_tokens is not None else self._count_prompt_tokens(tokenizer)
                    )
                ),
                response_metrics=StreamedResponseMetrics(
                    response_chunks=response_chunks,
                    chunk_times=chunk_times,
                    output_tokens=output_len,
                    output_token_times=chunk_times,
                    server_usage=server_usage,
                ),
                lora_adapter=lora_adapter,
                extra_info={"raw_response": raw_content, "output_message": output_message, "output_text": output_text},
            )

        data = await response.json()
        usage = data.get("usage") or {}
        output_text, output_message = parse_anthropic_content(data.get("content"))
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        output_len = int(output_tokens) if output_tokens is not None else tokenizer.count_tokens(output_text)
        extra_info: dict[str, Any] = {
            "stop_reason": data.get("stop_reason"),
            "output_message": output_message,
            "output_text": output_text,
        }
        return InferenceInfo(
            request_metrics=RequestMetrics(
                text=Text(input_tokens=int(input_tokens) if input_tokens is not None else self._count_prompt_tokens(tokenizer))
            ),
            response_metrics=UnaryResponseMetrics(output_tokens=output_len),
            lora_adapter=lora_adapter,
            extra_info=extra_info,
        )
