# Copyright 2025 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for wire capture detection and conversion to OTel traces.

Fixtures are synthetic but mirror the structure of real captures: the wire record
envelope (path/headers/request/response/status/start_unix/total_ms), the Chat
Completions plain-JSON response, and the Responses API SSE stream whose
response.completed event carries usage and output.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from inference_perf.datagen.replay.otel_trace_replay_datagen import (
    _load_files_to_dataset,
    _normalize_file_trace,
    _rows_for_file,
)
from inference_perf.datagen.replay.otel_trace_to_replay_graph import build_graph, build_raw_calls
from inference_perf.datagen.replay.wire_trace_converter import (
    FORMAT_CHAT_COMPLETIONS_WIRE,
    FORMAT_OTEL_JSON,
    FORMAT_OTEL_JSONL,
    FORMAT_RESPONSES_WIRE,
    convert_wire_file,
    derive_trace_id,
    detect_trace_format,
    flatten_output_name,
    iter_otel_jsonl_traces,
)

# --------------------------------------------------------------------------------------
# Fixture builders — synthetic payloads shaped like real captures
# --------------------------------------------------------------------------------------

START_UNIX = 1_700_000_000.0


def chat_record(
    *,
    content: Any = "hello",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    reasoning: Optional[str] = None,
    status: int = 200,
    prompt_tokens: int = 11,
    completion_tokens: int = 5,
    cached_tokens: int = 0,
    session_id: Optional[str] = "sess-chat-1",
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """One Chat Completions wire record. request/response are JSON strings, as on the wire."""
    message: Dict[str, Any] = {"role": "assistant"}
    if reasoning is not None:
        message["reasoning"] = reasoning
        message["reasoning_details"] = [{"index": 0, "type": "reasoning.text", "text": reasoning}]
    message["content"] = content
    if tool_calls is not None:
        message["tool_calls"] = tool_calls

    response: Dict[str, Any] = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "test/model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "message": message,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

    request: Dict[str, Any] = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "you are a test agent"}]},
            {"role": "user", "content": [{"type": "text", "text": "do the thing"}]},
        ],
        "model": "test/model",
    }
    if tools is not None:
        request["tools"] = tools

    headers = {"content-type": "application/json"}
    if session_id:
        headers["x-litellm-session-id"] = session_id

    return {
        "path": "/openai/v1/chat/completions",
        "start_unix": START_UNIX,
        "headers": headers,
        "request": json.dumps(request),
        "model": "test/model",
        "status": status,
        "response": json.dumps(response) if status == 200 else "",
        "total_ms": 250.0,
    }


def _sse(events: List[Dict[str, Any]]) -> str:
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events)


def responses_record(
    *,
    status: int = 200,
    input_items: Optional[List[Dict[str, Any]]] = None,
    output_items: Optional[List[Dict[str, Any]]] = None,
    input_tokens: int = 20,
    output_tokens: int = 7,
    reasoning_tokens: int = 0,
    session_id: Optional[str] = "sess-resp-1",
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """One Responses API wire record; the response body is an SSE stream."""
    if input_items is None:
        input_items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "question"}]}]
    if output_items is None:
        output_items = [{"type": "message", "content": [{"type": "output_text", "text": "answer"}]}]

    completed = {
        "type": "response.completed",
        "response": {
            "id": "resp-test",
            "status": "completed",
            "output": output_items,
            "usage": {
                "input_tokens": input_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": output_tokens,
                "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
                "total_tokens": input_tokens + output_tokens,
            },
        },
    }
    body = _sse([{"type": "response.created", "response": {"id": "resp-test"}}, completed]) if status == 200 else ""

    request: Dict[str, Any] = {"model": "test/model", "input": input_items, "store": True, "stream": True}
    if tools is not None:
        request["tools"] = tools

    headers = {"content-type": "application/json"}
    if session_id:
        headers["session_id"] = session_id

    return {
        "path": "/openai/v1/responses",
        "start_unix": START_UNIX,
        "headers": headers,
        "request": json.dumps(request),
        "model": "test/model",
        "status": status,
        "response": body,
        "total_ms": 500.0,
    }


def otel_trace(trace_id: str = "trace-1", n_spans: int = 2) -> Dict[str, Any]:
    """A minimal OTel trace document with replayable LLM spans."""
    spans = []
    for i in range(n_spans):
        spans.append(
            {
                "trace_id": trace_id,
                "span_id": f"span{i:04d}",
                "parent_span_id": None,
                "name": "chat test/model",
                "kind": "SPAN_KIND_CLIENT",
                "start_time": f"2024-01-01T00:00:{i:02d}+00:00",
                "end_time": f"2024-01-01T00:00:{i:02d}.500000+00:00",
                "attributes": {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.request.model": "test/model",
                    "gen_ai.usage.input_tokens": 10,
                    "gen_ai.usage.output_tokens": 4,
                    "gen_ai.input.messages": json.dumps(
                        [{"role": "user", "parts": [{"type": "text", "content": f"turn {i}"}]}]
                    ),
                    "gen_ai.output.messages": json.dumps(
                        [{"role": "assistant", "parts": [{"type": "text", "content": f"reply {i}"}]}]
                    ),
                },
                "status": {"code": 1, "message": ""},
            }
        )
    return {"trace_id": trace_id, "span_count": len(spans), "collected_at": "2024-01-01T00:00:00+00:00", "spans": spans}


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# Format detection
# --------------------------------------------------------------------------------------


def test_detect_format_chat_completions_wire(tmp_path: Path) -> None:
    f = write_jsonl(tmp_path / "calls.jsonl", [chat_record()])
    assert detect_trace_format(f) == FORMAT_CHAT_COMPLETIONS_WIRE


def test_detect_format_responses_wire(tmp_path: Path) -> None:
    f = write_jsonl(tmp_path / "calls.jsonl", [responses_record()])
    assert detect_trace_format(f) == FORMAT_RESPONSES_WIRE


def test_detect_format_otel_jsonl(tmp_path: Path) -> None:
    f = write_jsonl(tmp_path / "trace.jsonl", [otel_trace()])
    assert detect_trace_format(f) == FORMAT_OTEL_JSONL


def test_detect_format_otel_json(tmp_path: Path) -> None:
    f = tmp_path / "trace.json"
    f.write_text(json.dumps(otel_trace()), encoding="utf-8")
    assert detect_trace_format(f) == FORMAT_OTEL_JSON


def test_detect_format_pretty_printed_json(tmp_path: Path) -> None:
    """A .json document spans many lines, so detection must parse the whole file."""
    f = tmp_path / "trace.json"
    f.write_text(json.dumps(otel_trace(), indent=2), encoding="utf-8")
    assert detect_trace_format(f) == FORMAT_OTEL_JSON


def test_detect_format_skips_leading_blank_lines(tmp_path: Path) -> None:
    f = tmp_path / "calls.jsonl"
    f.write_text("\n\n  \n" + json.dumps(chat_record()) + "\n", encoding="utf-8")
    assert detect_trace_format(f) == FORMAT_CHAT_COMPLETIONS_WIRE


def test_detect_format_empty_file_raises_naming_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.jsonl"
    f.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match=r"empty\.jsonl.*empty"):
        detect_trace_format(f)


def test_detect_format_top_level_array_raises_naming_file(tmp_path: Path) -> None:
    f = tmp_path / "array.json"
    f.write_text(json.dumps([{"spans": []}]), encoding="utf-8")
    with pytest.raises(ValueError, match=r"array\.json"):
        detect_trace_format(f)


def test_detect_format_unknown_shape_raises_naming_file(tmp_path: Path) -> None:
    f = write_jsonl(tmp_path / "mystery.jsonl", [{"something": "else"}])
    with pytest.raises(ValueError, match=r"mystery\.jsonl.*unrecognized"):
        detect_trace_format(f)


def test_detect_format_unsupported_api_path_raises(tmp_path: Path) -> None:
    f = write_jsonl(tmp_path / "embed.jsonl", [{"path": "/openai/v1/embeddings", "request": "{}"}])
    with pytest.raises(ValueError, match=r"embed\.jsonl.*embeddings"):
        detect_trace_format(f)


# --------------------------------------------------------------------------------------
# Chat Completions conversion
# --------------------------------------------------------------------------------------


def test_chat_conversion_basic_span_shape(tmp_path: Path) -> None:
    f = write_jsonl(tmp_path / "calls.jsonl", [chat_record()])
    trace = convert_wire_file(f)

    assert trace["span_count"] == 1
    span = trace["spans"][0]
    assert span["name"] == "chat test/model"
    assert span["kind"] == "SPAN_KIND_CLIENT"
    assert span["status"] == {"code": 1, "message": ""}
    attrs = span["attributes"]
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.usage.input_tokens"] == 11
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert attrs["gen_ai.response.finish_reasons"] == ["stop"]
    # start_unix + total_ms drive real timestamps
    assert span["start_time"].startswith("2023-11-14")
    assert span["start_time"] != span["end_time"]


def test_chat_conversion_drops_reasoning(tmp_path: Path) -> None:
    f = write_jsonl(tmp_path / "calls.jsonl", [chat_record(reasoning="secret chain of thought")])
    trace = convert_wire_file(f)

    blob = json.dumps(trace)
    assert "secret chain of thought" not in blob
    assert "reasoning" not in blob
    parts = json.loads(trace["spans"][0]["attributes"]["gen_ai.output.messages"])[0]["parts"]
    assert [p["type"] for p in parts] == ["text"]


def test_chat_conversion_tool_calls_arguments_parsed_to_dict(tmp_path: Path) -> None:
    tool_calls = [
        {
            "index": 0,
            "type": "function",
            "id": "call-1",
            "function": {"name": "list_files", "arguments": '{"directory": "/tmp"}'},
        }
    ]
    f = write_jsonl(tmp_path / "calls.jsonl", [chat_record(content=None, tool_calls=tool_calls)])
    trace = convert_wire_file(f)

    parts = json.loads(trace["spans"][0]["attributes"]["gen_ai.output.messages"])[0]["parts"]
    assert len(parts) == 1
    assert parts[0]["type"] == "tool_call"
    assert parts[0]["id"] == "call-1"
    assert parts[0]["name"] == "list_files"
    # arguments arrive as a JSON string on the wire and must become a dict
    assert parts[0]["arguments"] == {"directory": "/tmp"}


def test_chat_conversion_malformed_arguments_preserved_raw(tmp_path: Path) -> None:
    tool_calls = [{"type": "function", "id": "call-1", "function": {"name": "f", "arguments": "not json"}}]
    f = write_jsonl(tmp_path / "calls.jsonl", [chat_record(content=None, tool_calls=tool_calls)])
    trace = convert_wire_file(f)

    parts = json.loads(trace["spans"][0]["attributes"]["gen_ai.output.messages"])[0]["parts"]
    assert parts[0]["arguments"] == {"_raw": "not json"}


def test_chat_conversion_keeps_content_part_boundaries(tmp_path: Path) -> None:
    """Structured content keeps one OTel part per source part, not a joined string."""
    content = [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]
    f = write_jsonl(tmp_path / "calls.jsonl", [chat_record(content=content)])
    trace = convert_wire_file(f)

    parts = json.loads(trace["spans"][0]["attributes"]["gen_ai.output.messages"])[0]["parts"]
    assert [p["content"] for p in parts] == ["first", "second"]


def test_chat_conversion_tool_result_becomes_user_message(tmp_path: Path) -> None:
    record = chat_record()
    request = json.loads(record["request"])
    request["messages"].append({"role": "tool", "tool_call_id": "call-1", "content": "the result", "name": "f"})
    record["request"] = json.dumps(request)
    f = write_jsonl(tmp_path / "calls.jsonl", [record])

    messages = json.loads(convert_wire_file(f)["spans"][0]["attributes"]["gen_ai.input.messages"])
    tool_msg = messages[-1]
    # tool results are carried on a user message per the OTel parts convention
    assert tool_msg["role"] == "user"
    assert tool_msg["parts"][0] == {"type": "tool_call_response", "id": "call-1", "result": "the result"}


def test_chat_conversion_tool_definitions_unwrapped(tmp_path: Path) -> None:
    tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}]
    f = write_jsonl(tmp_path / "calls.jsonl", [chat_record(tools=tools)])
    trace = convert_wire_file(f)

    defs = json.loads(trace["spans"][0]["attributes"]["gen_ai.tool.definitions"])
    assert defs == [{"type": "function", "name": "f", "description": "d", "parameters": {"type": "object"}}]


def test_chat_conversion_cached_tokens_recorded(tmp_path: Path) -> None:
    f = write_jsonl(tmp_path / "calls.jsonl", [chat_record(cached_tokens=7)])
    attrs = convert_wire_file(f)["spans"][0]["attributes"]
    assert attrs["gen_ai.usage.cache_read_tokens"] == 7


# --------------------------------------------------------------------------------------
# Responses API conversion
# --------------------------------------------------------------------------------------


def test_responses_conversion_parses_sse_completed(tmp_path: Path) -> None:
    f = write_jsonl(tmp_path / "calls.jsonl", [responses_record()])
    span = convert_wire_file(f)["spans"][0]

    attrs = span["attributes"]
    assert attrs["gen_ai.usage.input_tokens"] == 20
    assert attrs["gen_ai.usage.output_tokens"] == 7
    assert attrs["gen_ai.response.id"] == "resp-test"
    assert attrs["gen_ai.response.finish_reasons"] == ["stop"]
    parts = json.loads(attrs["gen_ai.output.messages"])[0]["parts"]
    assert parts == [{"type": "text", "content": "answer"}]


def test_responses_conversion_input_item_types(tmp_path: Path) -> None:
    input_items: List[Dict[str, Any]] = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "ask"}]},
        {"type": "reasoning", "summary": [], "encrypted_content": "OPAQUE"},
        {"type": "function_call", "name": "run", "arguments": '{"cmd": "ls"}', "call_id": "c1"},
        {"type": "function_call_output", "call_id": "c1", "output": "file.txt"},
    ]
    f = write_jsonl(tmp_path / "calls.jsonl", [responses_record(input_items=input_items)])
    trace = convert_wire_file(f)

    messages = json.loads(trace["spans"][0]["attributes"]["gen_ai.input.messages"])
    assert [m["parts"][0]["type"] for m in messages] == ["text", "tool_call", "tool_call_response"]
    assert messages[1]["parts"][0]["arguments"] == {"cmd": "ls"}
    assert messages[2]["role"] == "user"
    # the encrypted reasoning item is skipped entirely
    assert "OPAQUE" not in json.dumps(trace)


def test_responses_conversion_drops_reasoning_output(tmp_path: Path) -> None:
    output_items = [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking hard"}], "encrypted_content": "X"},
        {"type": "message", "content": [{"type": "output_text", "text": "final"}]},
    ]
    f = write_jsonl(tmp_path / "calls.jsonl", [responses_record(output_items=output_items, reasoning_tokens=42)])
    trace = convert_wire_file(f)

    blob = json.dumps(trace)
    assert "thinking hard" not in blob
    assert "reasoning" not in blob
    parts = json.loads(trace["spans"][0]["attributes"]["gen_ai.output.messages"])[0]["parts"]
    assert parts == [{"type": "text", "content": "final"}]


def test_responses_conversion_flat_tool_definitions(tmp_path: Path) -> None:
    tools = [{"type": "function", "name": "exec", "description": "run", "parameters": {"type": "object"}}]
    f = write_jsonl(tmp_path / "calls.jsonl", [responses_record(tools=tools)])
    defs = json.loads(convert_wire_file(f)["spans"][0]["attributes"]["gen_ai.tool.definitions"])
    assert defs == [{"type": "function", "name": "exec", "description": "run", "parameters": {"type": "object"}}]


# --------------------------------------------------------------------------------------
# Error spans
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("status", [500, 502, 404, 429])
def test_http_error_marks_span_as_error(tmp_path: Path, status: int) -> None:
    f = write_jsonl(tmp_path / "calls.jsonl", [chat_record(status=status)])
    span = convert_wire_file(f)["spans"][0]
    assert span["status"] == {"code": 2, "message": f"HTTP {status}"}


def test_error_spans_excluded_from_replay_calls(tmp_path: Path) -> None:
    # Distinct content per record: identical spans are deduplicated by the graph builder.
    records = [
        chat_record(content="first"),
        chat_record(content="second", status=502),
        chat_record(content="third"),
    ]
    f = write_jsonl(tmp_path / "calls.jsonl", records)
    spans = convert_wire_file(f)["spans"]
    assert len(spans) == 3

    calls, _ = build_raw_calls(spans, include_errors=False)
    assert len(calls) == 2
    calls_with_errors, _ = build_raw_calls(spans, include_errors=True)
    assert len(calls_with_errors) == 3


# --------------------------------------------------------------------------------------
# trace_id derivation
# --------------------------------------------------------------------------------------


def test_trace_id_from_session_header(tmp_path: Path) -> None:
    f = write_jsonl(tmp_path / "calls.jsonl", [chat_record(session_id="abc-def-123")])
    assert derive_trace_id(f) == "abcdef123"


def test_trace_id_unique_per_path_when_no_header(tmp_path: Path) -> None:
    """Capture trees reuse one leaf name, so the fallback must not collide."""
    ids = set()
    for i in range(3):
        d = tmp_path / f"session{i}" / "model"
        d.mkdir(parents=True)
        ids.add(derive_trace_id(write_jsonl(d / "calls.jsonl", [chat_record(session_id=None)])))
    assert len(ids) == 3


def test_trace_id_is_stable_across_calls(tmp_path: Path) -> None:
    f = write_jsonl(tmp_path / "calls.jsonl", [chat_record(session_id=None)])
    assert derive_trace_id(f) == derive_trace_id(f)


# --------------------------------------------------------------------------------------
# Streaming and multi-trace files
# --------------------------------------------------------------------------------------


def test_iter_otel_jsonl_yields_each_trace(tmp_path: Path) -> None:
    f = write_jsonl(tmp_path / "traces.jsonl", [otel_trace("t1"), otel_trace("t2"), otel_trace("t3")])
    assert [d["trace_id"] for d in iter_otel_jsonl_traces(f)] == ["t1", "t2", "t3"]


def test_otel_jsonl_multi_row_session_ids_unique(tmp_path: Path) -> None:
    """Rows lacking trace_id must not collapse onto one filename-derived session_id."""
    docs = []
    for _ in range(3):
        doc = otel_trace()
        doc.pop("trace_id")
        docs.append(doc)
    f = write_jsonl(tmp_path / "traces.jsonl", docs)

    rows = _rows_for_file(f, skip_invalid=False)
    assert len(rows) == 3
    assert len({r["session_id"] for r in rows}) == 3


def test_otel_jsonl_single_row_keeps_plain_session_id(tmp_path: Path) -> None:
    """A single-trace file keeps the unsuffixed id, so existing behavior is unchanged."""
    doc = otel_trace()
    doc.pop("trace_id")
    f = write_jsonl(tmp_path / "solo.jsonl", [doc])
    assert _rows_for_file(f, skip_invalid=False)[0]["session_id"] == "solo"


def test_wire_file_with_no_convertible_records_raises(tmp_path: Path) -> None:
    record = chat_record()
    record["request"] = "{not valid json"
    f = write_jsonl(tmp_path / "calls.jsonl", [record])
    with pytest.raises(ValueError, match="no convertible spans"):
        _rows_for_file(f, skip_invalid=False)


# --------------------------------------------------------------------------------------
# Loader integration
# --------------------------------------------------------------------------------------


def test_load_directory_mixed_json_and_jsonl(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps(otel_trace("otel-doc")), encoding="utf-8")
    write_jsonl(tmp_path / "b.jsonl", [otel_trace("otel-line")])
    write_jsonl(tmp_path / "c.jsonl", [chat_record(session_id="chat-sess")])
    write_jsonl(tmp_path / "d.jsonl", [responses_record(session_id="resp-sess")])

    files = sorted(p for p in tmp_path.glob("*.json*"))
    dataset = _load_files_to_dataset(files, skip_invalid=False)

    assert len(dataset) == 4
    assert len(set(dataset["session_id"])) == 4
    for i in range(len(dataset)):
        assert dataset[i]["spans"], "every row should carry spans"


def test_skip_invalid_files_honored_for_shape_errors(tmp_path: Path) -> None:
    (tmp_path / "good.json").write_text(json.dumps(otel_trace()), encoding="utf-8")
    (tmp_path / "bad.jsonl").write_text("{not json at all", encoding="utf-8")
    files = sorted(p for p in tmp_path.glob("*.json*"))

    assert len(_load_files_to_dataset(files, skip_invalid=True)) == 1
    with pytest.raises(ValueError, match=r"bad\.jsonl"):
        _load_files_to_dataset(files, skip_invalid=False)


def test_json_path_behavior_unchanged(tmp_path: Path) -> None:
    """The pre-existing .json path must keep its exact row shape."""
    doc = otel_trace("keep-me")
    f = tmp_path / "trace.json"
    f.write_text(json.dumps(doc), encoding="utf-8")

    row = _rows_for_file(f, skip_invalid=False)[0]
    expected = _normalize_file_trace(doc, f.name, str(f))
    assert row == expected
    assert row["session_id"] == "keep-me"
    assert row["source_id"] == str(f)
    assert row["harness"] == "unknown"
    assert row["benchmark"] == "unknown"


def test_normalize_rejects_top_level_array(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected an OTel trace object"):
        _normalize_file_trace([], "arr.json", str(tmp_path / "arr.json"))  # type: ignore[arg-type]


def test_converted_wire_spans_build_a_replay_graph(tmp_path: Path) -> None:
    """End to end: wire capture -> spans -> raw calls -> replay graph."""
    tools = [{"type": "function", "function": {"name": "run", "description": "run it", "parameters": {}}}]
    tool_calls = [{"type": "function", "id": "c1", "function": {"name": "run", "arguments": "{}"}}]
    records = [
        chat_record(content=None, tool_calls=tool_calls, tools=tools),
        chat_record(content="done", tools=tools),
    ]
    f = write_jsonl(tmp_path / "calls.jsonl", records)

    trace = convert_wire_file(f)
    calls, _ = build_raw_calls(trace["spans"], include_errors=False)
    graph = build_graph(calls, source_file=str(f))

    assert len(calls) == 2
    assert len(graph.events) == 2
    assert graph.root_event_ids


# --------------------------------------------------------------------------------------
# CLI helpers
# --------------------------------------------------------------------------------------


def test_flatten_output_name_joins_nested_path(tmp_path: Path) -> None:
    root = tmp_path / "captures"
    nested = root / "aime" / "codex" / "bifrost" / "2" / "model"
    nested.mkdir(parents=True)
    f = nested / "calls.jsonl"
    f.touch()
    assert flatten_output_name(f, root) == "aime_codex_bifrost_2_model_calls.json"
