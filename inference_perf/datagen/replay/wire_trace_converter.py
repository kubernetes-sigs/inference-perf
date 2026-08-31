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

"""
Detect the format of a local trace file and convert agent wire captures to OTel spans.

Agent harnesses (Codex, OpenHands, and similar) capture raw HTTP calls to the OpenAI
APIs as JSONL, one request/response pair per line. Those captures are not OTel traces,
so they are converted here before the replay path consumes them.

Supports four file shapes, detected from the first non-blank line rather than the filename:
- OTel document (.json)          top-level dict with "spans"
- OTel JSONL (.jsonl)            one dict with "spans" per line; 1..N traces per file
- Responses API wire (.jsonl)    path contains "/responses"; response is an SSE stream
- Chat Completions wire (.jsonl) path contains "/chat/completions"; response is plain JSON
- Anthropic Messages wire (.jsonl) path contains "/v1/messages"; response is an SSE stream

Reasoning content is intentionally never emitted -- OpenAI `reasoning` items and
Anthropic `thinking` blocks alike. The replay path flattens message parts and only
consumes text / tool_call / tool_call_response, so reasoning would be silently dropped
downstream anyway. This costs workload fidelity on reasoning models; see
docs/superpowers/notes/reasoning-content-dropped-in-wire-replay.md.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Detected file shapes, returned by detect_trace_format().
FORMAT_OTEL_JSON = "otel_json"
FORMAT_OTEL_JSONL = "otel_jsonl"
FORMAT_RESPONSES_WIRE = "responses_wire"
FORMAT_CHAT_COMPLETIONS_WIRE = "chat_completions_wire"
FORMAT_ANTHROPIC_WIRE = "anthropic_wire"

WIRE_FORMATS = (FORMAT_RESPONSES_WIRE, FORMAT_CHAT_COMPLETIONS_WIRE, FORMAT_ANTHROPIC_WIRE)

# OTel status codes.
_STATUS_OK = 1
_STATUS_ERROR = 2


def _span_id() -> str:
    return uuid.uuid4().hex[:16]


def _new_trace_id() -> str:
    return uuid.uuid4().hex[:32]


def _unix_to_iso(unix_ts: float) -> str:
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()


def _first_json_line(path: Path) -> Optional[Dict[str, Any]]:
    """Parse the first non-blank line of a file as a JSON object.

    Only meaningful for line-delimited files. Returns None for an empty/blank file.
    Raises ValueError naming the file if the first meaningful line is not a JSON object.
    """
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: first line is not valid JSON: {e}") from e
            if not isinstance(parsed, dict):
                raise ValueError(f"{path}: expected a JSON object, got {type(parsed).__name__}")
            return parsed
    return None


def _probe_file(path: Path) -> Optional[Dict[str, Any]]:
    """Return a representative top-level object for format detection.

    A .json file is parsed whole, since it may be pretty-printed across many lines.
    Any other file is treated as line-delimited and probed with its first record.
    """
    if path.suffix == ".json":
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: not valid JSON: {e}") from e
        if not isinstance(parsed, dict):
            raise ValueError(f"{path}: expected a JSON object at the top level, got {type(parsed).__name__}")
        return parsed
    return _first_json_line(path)


def detect_trace_format(path: Path) -> str:
    """Detect a trace file's format from its content.

    Returns one of FORMAT_OTEL_JSON, FORMAT_OTEL_JSONL, FORMAT_RESPONSES_WIRE,
    FORMAT_CHAT_COMPLETIONS_WIRE, FORMAT_ANTHROPIC_WIRE.

    Raises ValueError naming the file when the shape is empty or unrecognizable.
    """
    first = _probe_file(path)
    if first is None:
        raise ValueError(f"{path}: file is empty")

    if "spans" in first:
        # A .json file holding a single OTel document, vs a .jsonl of OTel documents.
        return FORMAT_OTEL_JSONL if path.suffix == ".jsonl" else FORMAT_OTEL_JSON

    api_path = first.get("path")
    if isinstance(api_path, str) and "request" in first:
        if "/chat/completions" in api_path:
            return FORMAT_CHAT_COMPLETIONS_WIRE
        if "/responses" in api_path:
            return FORMAT_RESPONSES_WIRE
        # Matched on the API suffix, not a gateway prefix: captures reach the Anthropic
        # Messages API as "/v1/messages", "/anthropic/v1/messages", "/v1/messages?beta=true", ...
        # A capture's first record can be a rejected probe to the bare gateway root
        # (path="/anthropic", HTTP 405), so the wire dialect is honored as a fallback:
        # detection only ever sees line 1, and rejecting the file on a probe would discard
        # every real /v1/messages record behind it.
        if "/v1/messages" in api_path or first.get("wire") == "anthropic":
            return FORMAT_ANTHROPIC_WIRE
        raise ValueError(f"{path}: unsupported wire capture API path {api_path!r}")

    raise ValueError(
        f"{path}: unrecognized trace format — expected an OTel document with 'spans', "
        "or a wire capture with 'path' and 'request'"
    )


def _parse_sse_completed(response_text: str) -> Optional[Dict[str, Any]]:
    """Return the payload of the response.completed SSE event, or None."""
    for line in response_text.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            event = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        if event.get("type") == "response.completed":
            response = event.get("response")
            return response if isinstance(response, dict) else None
    return None


def _parse_chat_completions_response(response_text: Any) -> Optional[Dict[str, Any]]:
    """Parse a plain-JSON Chat Completions response body."""
    if not response_text:
        return None
    try:
        resp = json.loads(response_text) if isinstance(response_text, str) else response_text
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(resp, dict) or "choices" not in resp:
        return None
    return resp


def _parse_anthropic_sse(response_text: str) -> Optional[Dict[str, Any]]:
    """Reassemble an Anthropic Messages SSE stream into a response-shaped dict.

    Unlike the Responses API, whose response.completed event carries the whole result,
    Anthropic streams the message incrementally and never restates it, so the content
    blocks have to be accumulated:

      message_start        message envelope + prompt-side usage
      content_block_start  opens a block at an index (text / thinking / tool_use)
      content_block_delta  text_delta appends text; input_json_delta appends a fragment
                           of the tool-call arguments as a JSON *string*
      content_block_stop   closes the block at that index
      message_delta        stop_reason + output-side usage
      message_stop         end of stream

    Returns a dict with "content", "usage", "stop_reason" and "id", or None when the
    stream carries no message_start (an empty or truncated body).
    """
    if not response_text:
        return None

    message: Optional[Dict[str, Any]] = None
    usage: Dict[str, Any] = {}
    stop_reason: Optional[str] = None
    # Blocks are addressed by index, and indices need not arrive in order.
    blocks: Dict[int, Dict[str, Any]] = {}
    # Tool-call arguments arrive as JSON fragments that are only valid once concatenated.
    json_fragments: Dict[int, List[str]] = {}

    for line in response_text.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            event = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        event_type = event.get("type")

        if event_type == "message_start":
            msg = event.get("message")
            if isinstance(msg, dict):
                message = msg
                if isinstance(msg.get("usage"), dict):
                    usage.update(msg["usage"])

        elif event_type == "content_block_start":
            index = event.get("index")
            block = event.get("content_block")
            if isinstance(index, int) and isinstance(block, dict):
                blocks[index] = dict(block)
                json_fragments[index] = []

        elif event_type == "content_block_delta":
            index = event.get("index")
            delta = event.get("delta")
            if not isinstance(index, int) or not isinstance(delta, dict):
                continue
            block = blocks.setdefault(index, {"type": "text", "text": ""})
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                block["text"] = (block.get("text") or "") + (delta.get("text") or "")
            elif delta_type == "thinking_delta":
                block["thinking"] = (block.get("thinking") or "") + (delta.get("thinking") or "")
            elif delta_type == "input_json_delta":
                json_fragments.setdefault(index, []).append(delta.get("partial_json") or "")

        elif event_type == "message_delta":
            delta = event.get("delta")
            if isinstance(delta, dict) and delta.get("stop_reason"):
                stop_reason = delta["stop_reason"]
            if isinstance(event.get("usage"), dict):
                usage.update(event["usage"])

    if message is None:
        return None

    # Materialize accumulated tool-call arguments now that every fragment has arrived.
    for index, fragments in json_fragments.items():
        if not fragments or index not in blocks:
            continue
        blocks[index]["input"] = _parse_arguments("".join(fragments))

    content = [blocks[i] for i in sorted(blocks)]
    return {
        "id": message.get("id", ""),
        "content": content,
        "usage": usage,
        "stop_reason": stop_reason or message.get("stop_reason"),
    }


def _parse_arguments(args: Any) -> Dict[str, Any]:
    """Tool-call arguments arrive as a JSON string on the wire; OTel wants a dict."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            return {"_raw": args}
        return parsed if isinstance(parsed, dict) else {"_raw": args}
    return {}


def _text_from_content(content: Any) -> str:
    """Flatten a message content field (str, or a list of content parts) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("input_text", "output_text", "text"):
                texts.append(part.get("text", ""))
            elif isinstance(part, str):
                texts.append(part)
        return "\n".join(texts)
    return "" if content is None else str(content)


def _text_parts(content: Any) -> List[Dict[str, Any]]:
    """Convert a message content field to one OTel text part per source part.

    A structured content list keeps its part boundaries rather than being joined, so the
    converted message mirrors what the harness actually sent.
    """
    if isinstance(content, str):
        return [{"type": "text", "content": content}] if content else []
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("input_text", "output_text", "text"):
                parts.append({"type": "text", "content": part.get("text", "")})
            elif isinstance(part, str) and part:
                parts.append({"type": "text", "content": part})
        return parts
    return []


def _convert_responses_input(input_items: List[Any]) -> List[Dict[str, Any]]:
    """Convert Responses API input[] items to OTel messages in parts format.

    Item types: message, function_call, function_call_output. Reasoning items are skipped.
    """
    messages: List[Dict[str, Any]] = []
    for item in input_items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "")

        if item_type == "message":
            text = _text_from_content(item.get("content", []))
            if text:
                messages.append({"role": item.get("role", ""), "parts": [{"type": "text", "content": text}]})

        elif item_type == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "tool_call",
                            "id": item.get("call_id", item.get("id", "")),
                            "name": item.get("name", ""),
                            "arguments": _parse_arguments(item.get("arguments", "")),
                        }
                    ],
                }
            )

        elif item_type == "function_call_output":
            output = item.get("output", "")
            messages.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "type": "tool_call_response",
                            "id": item.get("call_id", item.get("id", "")),
                            "result": output if isinstance(output, str) else json.dumps(output),
                        }
                    ],
                }
            )

        # type=reasoning: skipped (encrypted_content is unusable, and reasoning is not emitted)

    return messages


def _convert_responses_output(output_items: List[Any], finish_reason: Optional[str]) -> Optional[Dict[str, Any]]:
    """Convert Responses API output[] items to a single OTel output message."""
    parts: List[Dict[str, Any]] = []

    for item in output_items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")

        if item_type == "message":
            for part in item.get("content", []):
                if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                    parts.append({"type": "text", "content": part.get("text", "")})

        elif item_type == "function_call":
            parts.append(
                {
                    "type": "tool_call",
                    "id": item.get("call_id", item.get("id", "")),
                    "name": item.get("name", ""),
                    "arguments": _parse_arguments(item.get("arguments", "")),
                }
            )

        # type=reasoning: skipped

    if not parts:
        return None

    output_msg: Dict[str, Any] = {"role": "assistant", "parts": parts}
    if finish_reason:
        output_msg["finish_reason"] = finish_reason
    return output_msg


def _convert_chat_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """Convert Chat Completions messages[] to OTel messages in parts format."""
    result: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content")
        tool_call_id = msg.get("tool_call_id")

        # A tool result becomes a tool_call_response part carried on a user message.
        if role == "tool" and tool_call_id is not None:
            result.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "type": "tool_call_response",
                            "id": tool_call_id,
                            "result": content if isinstance(content, str) else json.dumps(content),
                        }
                    ],
                }
            )
            continue

        parts: List[Dict[str, Any]] = _text_parts(content)

        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {})
            parts.append(
                {
                    "type": "tool_call",
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": _parse_arguments(fn.get("arguments", "")),
                }
            )

        if parts:
            result.append({"role": role, "parts": parts})

    return result


def _anthropic_blocks_to_parts(content: Any) -> List[Dict[str, Any]]:
    """Convert Anthropic content blocks to OTel parts, preserving block boundaries.

    thinking / redacted_thinking blocks are skipped, matching how the OpenAI converters
    treat reasoning (see the module docstring).
    """
    if isinstance(content, str):
        return [{"type": "text", "content": content}] if content else []
    if not isinstance(content, list):
        return []

    parts: List[Dict[str, Any]] = []
    for block in content:
        if isinstance(block, str):
            if block:
                parts.append({"type": "text", "content": block})
            continue
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")
        if block_type == "text":
            parts.append({"type": "text", "content": block.get("text", "")})
        elif block_type == "tool_use":
            # Anthropic already sends arguments as an object; _parse_arguments passes
            # a dict straight through and repairs the streamed-fragment case.
            parts.append(
                {
                    "type": "tool_call",
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": _parse_arguments(block.get("input", {})),
                }
            )
        elif block_type == "tool_result":
            result = block.get("content")
            parts.append(
                {
                    "type": "tool_call_response",
                    "id": block.get("tool_use_id", ""),
                    "result": result if isinstance(result, str) else json.dumps(result),
                }
            )
        # type=thinking / redacted_thinking: skipped

    return parts


def _convert_anthropic_messages(system: Any, messages: List[Any]) -> List[Dict[str, Any]]:
    """Convert an Anthropic request's system + messages[] to OTel messages.

    `system` is a top-level field rather than a message, and may be a plain string or a
    list of text blocks; it becomes a leading system message so the replayed prompt keeps
    the instructions the harness actually sent.

    A tool_result block is carried on a user message, matching how the Chat Completions
    converter handles role=tool: the replay path keys tool responses off the part, not
    the role.
    """
    result: List[Dict[str, Any]] = []

    system_parts = _anthropic_blocks_to_parts(system)
    if system_parts:
        result.append({"role": "system", "parts": system_parts})

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        parts = _anthropic_blocks_to_parts(msg.get("content"))
        if parts:
            result.append({"role": msg.get("role", ""), "parts": parts})

    return result


def _convert_anthropic_output(resp: Dict[str, Any], finish_reason: Optional[str]) -> Optional[Dict[str, Any]]:
    """Convert a reassembled Anthropic response's content blocks to one OTel message."""
    parts = _anthropic_blocks_to_parts(resp.get("content") or [])
    if not parts:
        return None

    output_msg: Dict[str, Any] = {"role": "assistant", "parts": parts}
    if finish_reason:
        output_msg["finish_reason"] = finish_reason
    return output_msg


def _extract_anthropic_tool_defs(tools: List[Any]) -> List[Dict[str, Any]]:
    """Normalize Anthropic tool definitions, whose schema key is input_schema."""
    result: List[Dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        result.append(
            {
                "type": "function",
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            }
        )
    return result


def _convert_chat_output(resp: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Extract (output_message, finish_reason) from a Chat Completions response.

    message.reasoning / message.reasoning_content are intentionally ignored.
    """
    choices = resp.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return None, None
    choice = choices[0]
    finish_reason = choice.get("finish_reason", "")
    msg = choice.get("message") or {}

    parts: List[Dict[str, Any]] = _text_parts(msg.get("content"))

    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function", {})
        parts.append(
            {
                "type": "tool_call",
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "arguments": _parse_arguments(fn.get("arguments", "")),
            }
        )

    if not parts:
        return None, finish_reason

    output_msg: Dict[str, Any] = {"role": msg.get("role", "assistant"), "parts": parts}
    if finish_reason:
        output_msg["finish_reason"] = finish_reason
    return output_msg, finish_reason


def _extract_tool_defs(tools: List[Any], nested: bool) -> List[Dict[str, Any]]:
    """Normalize tool definitions.

    Chat Completions nests them under "function"; the Responses API keeps them flat.
    """
    result: List[Dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function", t) if nested else t
        result.append(
            {
                "type": "function",
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
        )
    return result


def _timestamps(record: Dict[str, Any]) -> Tuple[str, str]:
    """Return (start_time, end_time) as ISO strings from the wire record's timing."""
    start_unix = record.get("start_unix")
    total_ms = record.get("total_ms") or 0
    if start_unix:
        return _unix_to_iso(start_unix), _unix_to_iso(start_unix + total_ms / 1000.0)
    now = datetime.now(tz=timezone.utc).isoformat()
    return now, now


def _build_span(
    trace_id: str,
    model: str,
    start_time: str,
    end_time: str,
    http_status: int,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    response_id: str,
    finish_reason: Optional[str],
    input_messages: List[Dict[str, Any]],
    output_msg: Optional[Dict[str, Any]],
    tool_defs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble an OTel span from extracted fields."""
    is_error = not (200 <= http_status < 300)

    span: Dict[str, Any] = {
        "trace_id": trace_id,
        "span_id": _span_id(),
        "parent_span_id": None,
        "name": f"chat {model}",
        "kind": "SPAN_KIND_CLIENT",
        "start_time": start_time,
        "end_time": end_time,
        "attributes": {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": model,
            "gen_ai.response.model": model,
            "gen_ai.usage.input_tokens": input_tokens,
            "gen_ai.usage.output_tokens": output_tokens,
        },
        "resource_attributes": {
            "service.name": "wire-capture",
        },
        "status": {
            "code": _STATUS_ERROR if is_error else _STATUS_OK,
            "message": f"HTTP {http_status}" if is_error else "",
        },
    }

    attrs = span["attributes"]
    if cached_tokens:
        attrs["gen_ai.usage.cache_read_tokens"] = cached_tokens
    if response_id:
        attrs["gen_ai.response.id"] = response_id
    if finish_reason:
        attrs["gen_ai.response.finish_reasons"] = [finish_reason]
    if input_messages:
        attrs["gen_ai.input.messages"] = json.dumps(input_messages)
    if output_msg:
        attrs["gen_ai.output.messages"] = json.dumps([output_msg])
    if tool_defs:
        attrs["gen_ai.tool.definitions"] = json.dumps(tool_defs)

    return span


def _load_request(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = record.get("request", "")
    try:
        request = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return None
    return request if isinstance(request, dict) else None


def convert_responses_api_record(record: Dict[str, Any], trace_id: str) -> Optional[Dict[str, Any]]:
    """Convert one Responses API wire record to an OTel span.

    The response body is an SSE stream; usage and output come from response.completed.
    Returns None if the request body is unparseable.
    """
    request = _load_request(record)
    if request is None:
        return None

    completed = _parse_sse_completed(record.get("response") or "") or {}
    usage = completed.get("usage") or {}

    response_status = completed.get("status", "")
    finish_reason = "stop" if response_status == "completed" else response_status

    start_time, end_time = _timestamps(record)

    return _build_span(
        trace_id=trace_id,
        model=record.get("model") or request.get("model", "unknown"),
        start_time=start_time,
        end_time=end_time,
        http_status=int(record.get("status", 200)),
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cached_tokens=(usage.get("input_tokens_details") or {}).get("cached_tokens", 0),
        response_id=completed.get("id", ""),
        finish_reason=finish_reason,
        input_messages=_convert_responses_input(request.get("input") or []),
        output_msg=_convert_responses_output(completed.get("output") or [], finish_reason),
        tool_defs=_extract_tool_defs(request.get("tools") or [], nested=False),
    )


def convert_chat_completions_record(record: Dict[str, Any], trace_id: str) -> Optional[Dict[str, Any]]:
    """Convert one Chat Completions wire record to an OTel span.

    The response body is a plain JSON object with choices[] and usage.
    Returns None if the request body is unparseable.
    """
    request = _load_request(record)
    if request is None:
        return None

    resp = _parse_chat_completions_response(record.get("response"))
    usage = (resp or {}).get("usage") or {}

    output_msg, finish_reason = _convert_chat_output(resp) if resp else (None, None)
    start_time, end_time = _timestamps(record)

    return _build_span(
        trace_id=trace_id,
        model=record.get("model") or request.get("model", "unknown"),
        start_time=start_time,
        end_time=end_time,
        http_status=int(record.get("status", 200)),
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        response_id=(resp or {}).get("id", ""),
        finish_reason=finish_reason,
        input_messages=_convert_chat_messages(request.get("messages") or []),
        output_msg=output_msg,
        tool_defs=_extract_tool_defs(request.get("tools") or [], nested=True),
    )


# Anthropic stop_reason -> the finish_reason vocabulary the OpenAI converters emit.
_ANTHROPIC_STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "refusal": "content_filter",
}


def convert_anthropic_record(record: Dict[str, Any], trace_id: str) -> Optional[Dict[str, Any]]:
    """Convert one Anthropic Messages wire record to an OTel span.

    The response body is an SSE stream that has to be reassembled; usage arrives split
    across message_start (prompt side) and message_delta (output side).
    Returns None if the request body is unparseable.
    """
    request = _load_request(record)
    if request is None:
        return None

    resp = _parse_anthropic_sse(record.get("response") or "") or {}
    usage = resp.get("usage") or {}

    stop_reason = resp.get("stop_reason")
    finish_reason = _ANTHROPIC_STOP_REASONS.get(stop_reason, stop_reason) if stop_reason else None

    start_time, end_time = _timestamps(record)

    # Anthropic reports input_tokens net of cache hits, so the cached count has to be added
    # back to get the full prompt size the other converters report.
    cached_tokens = usage.get("cache_read_input_tokens", 0) or 0
    input_tokens = (usage.get("input_tokens", 0) or 0) + cached_tokens

    return _build_span(
        trace_id=trace_id,
        model=record.get("model") or request.get("model", "unknown"),
        start_time=start_time,
        end_time=end_time,
        http_status=int(record.get("status", 200)),
        input_tokens=input_tokens,
        output_tokens=usage.get("output_tokens", 0) or 0,
        cached_tokens=cached_tokens,
        response_id=resp.get("id", ""),
        finish_reason=finish_reason,
        input_messages=_convert_anthropic_messages(request.get("system"), request.get("messages") or []),
        output_msg=_convert_anthropic_output(resp, finish_reason) if resp else None,
        tool_defs=_extract_anthropic_tool_defs(request.get("tools") or []),
    )


def convert_wire_record(record: Dict[str, Any], trace_id: str) -> Optional[Dict[str, Any]]:
    """Convert one wire record to an OTel span, dispatching on its API path."""
    api_path = record.get("path", "")
    if "/chat/completions" in api_path:
        return convert_chat_completions_record(record, trace_id)
    if "/responses" in api_path:
        return convert_responses_api_record(record, trace_id)
    if "/v1/messages" in api_path or record.get("wire") == "anthropic":
        return convert_anthropic_record(record, trace_id)
    return None


# Request headers that carry an agent session id, in preference order.
_SESSION_ID_HEADERS = (
    "session_id",
    "x-client-request-id",
    "x-litellm-session-id",
)


def derive_trace_id(path: Path) -> str:
    """Derive a stable trace_id for a wire capture file.

    Prefers a session id from the request headers so the id is stable across runs.
    Falls back to a hash of the file path, since capture files are commonly all named
    the same thing (e.g. every session's file is "calls.jsonl") and a bare filename
    stem would collide across sessions.
    """
    try:
        first = _probe_file(path)
    except ValueError:
        first = None
    if first:
        headers = first.get("headers") or {}
        if isinstance(headers, dict):
            for header in _SESSION_ID_HEADERS:
                sid = headers.get(header)
                if isinstance(sid, str) and sid:
                    return sid.replace("-", "")[:32]
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:32]


def iter_wire_spans(path: Path, trace_id: str) -> Iterator[Dict[str, Any]]:
    """Stream a wire capture file, yielding one OTel span per convertible record.

    Reads line by line and drops each parsed record after converting it, so the file's
    records and the resulting spans are never both fully resident.
    """
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            if not isinstance(record, dict):
                continue
            span = convert_wire_record(record, trace_id)
            if span is not None:
                yield span


def convert_wire_file(path: Path, trace_id: Optional[str] = None) -> Dict[str, Any]:
    """Convert a wire capture file to a complete OTel trace document.

    One file is one session, so the result mirrors the local OTel document shape the
    replay loader already understands.
    """
    resolved_trace_id = trace_id or derive_trace_id(path)
    spans = list(iter_wire_spans(path, resolved_trace_id))
    return {
        "trace_id": resolved_trace_id,
        "span_count": len(spans),
        "collected_at": datetime.now(tz=timezone.utc).isoformat(),
        "spans": spans,
    }


def iter_otel_jsonl_traces(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream OTel trace documents from a .jsonl file, one per line."""
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            if not isinstance(doc, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object, got {type(doc).__name__}")
            yield doc


def write_converted_trace(input_path: Path, output_path: Path) -> Dict[str, Any]:
    """Convert one wire capture file and write the OTel trace document to disk."""
    trace = convert_wire_file(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)
    return trace


def flatten_output_name(input_path: Path, input_root: Path) -> str:
    """Flatten a nested capture path into a single output filename.

    Capture trees commonly repeat the same leaf name for every session
    (aime/codex/bifrost/2/model/calls.jsonl), so the relative path is joined into the
    filename to keep outputs distinct:
    aime/codex/bifrost/2/model/calls.jsonl -> aime_codex_bifrost_2_model_calls.json
    """
    rel = input_path.relative_to(input_root)
    return "_".join(rel.with_suffix("").parts) + ".json"


def _cli() -> int:
    """Convert wire capture files to OTel trace documents on disk.

    inference-perf converts these in memory at load time, so this CLI exists for
    inspecting conversions and for producing OTel files to share or upload.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert agent wire capture JSONL files to OpenTelemetry trace JSON.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input .jsonl file or a directory to search recursively")
    parser.add_argument("-o", "--output", type=Path, help="Output .json file (single input) or directory (batch)")
    parser.add_argument("--silent", action="store_true", help="Only report errors and the final summary")
    args = parser.parse_args()

    if args.input.is_file():
        out = args.output or args.input.with_suffix(".json")
        trace = write_converted_trace(args.input, out)
        if not args.silent:
            print(f"{args.input} -> {out}  ({trace['span_count']} spans, trace_id={trace['trace_id']})")
        return 0

    if not args.input.is_dir():
        print(f"Error: {args.input} does not exist")
        return 1

    out_dir = args.output or args.input.parent / f"{args.input.name}_otel"
    files = sorted(args.input.rglob("*.jsonl"))
    if not files:
        print(f"No .jsonl files found under {args.input}")
        return 1
    if not args.silent:
        print(f"Found {len(files)} files")

    ok = 0
    errors = 0
    total_spans = 0
    for f in files:
        try:
            trace = write_converted_trace(f, out_dir / flatten_output_name(f, args.input))
        except Exception as e:
            errors += 1
            print(f"Error on {f}: {e}")
            continue
        ok += 1
        total_spans += trace["span_count"]
        if not args.silent:
            print(f"  {f.relative_to(args.input)} -> {trace['span_count']} spans")

    print(f"Done: {ok} ok, {errors} errors, {total_spans} spans -> {out_dir}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(_cli())
