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

"""Recorded images replay as images, end to end.

Covers the pipeline wire capture → OTel span → replay graph → session request, for the
two places Claude Code puts an image: a ``Read`` tool result and a pasted image.
"""

import base64
import io
import json
import logging
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from PIL import Image as PILImage

from inference_perf.apis import LazyLoadInferenceAPIData
from inference_perf.apis.anthropic_messages import build_anthropic_request_body
from inference_perf.apis.chat import ChatMessage
from inference_perf.apis.image_parts import count_missing_images, is_image_part, to_openai_image
from inference_perf.client.modelserver.otel_instrumentation import summarize_media
from inference_perf.config.datagen.replay import BadToolCallHandling, ToolChoiceMode
from inference_perf.datagen.replay.otel_trace_replay_datagen import OTelTraceReplayDataGenerator
from inference_perf.datagen.replay.otel_trace_to_replay_graph import (
    build_graph,
    build_raw_calls,
    message_tokens,
    messages_equal,
)
from inference_perf.datagen.replay.otel_trace_utils import reconstruct_llm_input
from inference_perf.datagen.replay.replay_graph_session_datagen import (
    EventOutputRegistry,
    ReplaySessionEvent,
    SessionChatCompletionAPIData,
    WorkerSessionTracker,
    replayed_images,
)
from inference_perf.datagen.replay.replay_graph_types import ComplexReplayMessage, InputSegment
from inference_perf.datagen.replay.wire_trace_converter import convert_wire_record


def _png(width: int, height: int, color: str = "red") -> str:
    buf = io.BytesIO()
    PILImage.new("RGB", (width, height), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


PNG = _png(3, 2)
OTHER_PNG = _png(3, 2, "blue")
DATA_URL = f"data:image/png;base64,{PNG}"
IMAGE = {"type": "image_url", "image_url": {"url": DATA_URL}}


def _image_block(data: str = PNG) -> Dict[str, Any]:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def _record(request: Dict[str, Any], response: Any, start: float = 1_700_000_000.0) -> Dict[str, Any]:
    return {
        "path": "/v1/messages",
        "wire": "anthropic",
        "start_unix": start,
        "total_ms": 1000,
        "status": 200,
        "request": json.dumps(request),
        "response": response if isinstance(response, str) else json.dumps(response),
    }


def _message_response(text: str = "ok") -> Dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 600, "output_tokens": 2},
    }


# A Claude Code turn: the agent Reads a screenshot, then answers.
READ_REQUEST: Dict[str, Any] = {
    "model": "claude",
    "stream": False,
    "system": "You are an agent.",
    "messages": [
        {"role": "user", "content": "Look at /problem_assets/1.png"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "/problem_assets/1.png"}}
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [_image_block()]}]},
    ],
}

# A pasted image: a top-level image block between text blocks.
PASTE_REQUEST: Dict[str, Any] = {
    "model": "claude",
    "stream": False,
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "[Image #1] what is in the image?"}, _image_block()]},
    ],
}


def _span(request: Dict[str, Any]) -> Dict[str, Any]:
    span = convert_wire_record(_record(request, _message_response()), "t", 1)
    assert span is not None
    return span


def _span_messages(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    span = _span(request)
    messages: List[Dict[str, Any]] = json.loads(span["attributes"]["gen_ai.input.messages"])
    return messages


def _graph_messages(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls, _ = build_raw_calls([_span(request)])
    graph = build_graph(calls, source_file="t")
    messages: List[Dict[str, Any]] = next(iter(graph.events.values())).call.messages
    return messages


class TestWireConversion:
    def test_tool_result_image_becomes_a_blob_part(self) -> None:
        tool_message = _span_messages(READ_REQUEST)[-1]
        result = tool_message["parts"][0]["result"]
        assert result == [{"type": "blob", "modality": "image", "mime_type": "image/png", "content": PNG}]

    def test_pasted_image_becomes_a_blob_part_in_place(self) -> None:
        parts = _span_messages(PASTE_REQUEST)[0]["parts"]
        assert [p["type"] for p in parts] == ["text", "blob"]

    def test_text_only_tool_result_list_keeps_its_json_form(self) -> None:
        request = dict(READ_REQUEST)
        request["messages"] = READ_REQUEST["messages"][:2] + [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "hi"}]}],
            }
        ]
        assert _span_messages(request)[-1]["parts"][0]["result"] == json.dumps([{"type": "text", "text": "hi"}])

    def test_chat_completions_image_url_becomes_a_blob_part(self) -> None:
        record = {
            "path": "/v1/chat/completions",
            "start_unix": 1_700_000_000.0,
            "status": 200,
            "request": json.dumps(
                {
                    "model": "m",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "what?"},
                                IMAGE,
                            ],
                        }
                    ],
                }
            ),
            "response": json.dumps(
                {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}
            ),
        }
        span = convert_wire_record(record, "t", 1)
        assert span is not None
        parts = json.loads(span["attributes"]["gen_ai.input.messages"])[0]["parts"]
        assert parts[1] == {"type": "blob", "modality": "image", "mime_type": "image/png", "content": PNG}

    def test_streamed_request_with_assembled_json_response_converts(self) -> None:
        """Claude Code's raw-body logs store a streamed call's response as one JSON message."""
        request = dict(PASTE_REQUEST, stream=True)
        assert convert_wire_record(_record(request, _message_response()), "t", 1) is not None

    def test_streamed_request_with_truncated_response_is_still_skipped(self) -> None:
        request = dict(PASTE_REQUEST, stream=True)
        truncated = 'event: message_start\ndata: {"type": "message_start", "message": {"usage": {}}}\n'
        assert convert_wire_record(_record(request, truncated), "t", 1) is None


class TestGraph:
    def test_tool_result_image_replays_as_an_image_in_the_tool_message(self) -> None:
        tool_message = _graph_messages(READ_REQUEST)[-1]
        assert tool_message["role"] == "tool"
        assert tool_message["tool_call_id"] == "toolu_1"
        assert tool_message["content"] == [IMAGE]

    def test_pasted_image_keeps_its_position_among_the_text(self) -> None:
        user_message = _graph_messages(PASTE_REQUEST)[0]
        assert user_message["content"] == [
            {"type": "text", "text": "[Image #1] what is in the image?"},
            IMAGE,
        ]

    def test_text_only_messages_stay_strings(self) -> None:
        system, user = _graph_messages(READ_REQUEST)[:2]
        assert system["content"] == "You are an agent."
        assert user["content"] == "Look at /problem_assets/1.png"

    def test_base64_is_not_counted_as_text(self) -> None:
        info = {
            "role": "user",
            "parts": [{"type": "blob", "modality": "image", "mime_type": "image/png", "content": "A" * 400_000}],
        }
        msg = ComplexReplayMessage(role="user", message_info=info, raw_reconstructed_text=reconstruct_llm_input(info))
        assert "AAAA" not in msg.text
        assert message_tokens(msg) < 1_000

    def test_messages_with_different_images_are_not_equal(self) -> None:
        def msg(data: str) -> ComplexReplayMessage:
            info = {
                "role": "user",
                "parts": [{"type": "blob", "modality": "image", "mime_type": "image/png", "content": data}],
            }
            return ComplexReplayMessage(role="user", message_info=info, raw_reconstructed_text=reconstruct_llm_input(info))

        assert messages_equal(msg(PNG), msg(PNG))
        assert not messages_equal(msg(PNG), msg(OTHER_PNG))


def _generator_for(request: Dict[str, Any]) -> Any:
    calls, _ = build_raw_calls([_span(request)])
    graph = build_graph(calls, source_file="t")
    gc = next(iter(graph.events.values()))
    event = ReplaySessionEvent(
        call_id=gc.call.call_id,
        event_id=f"session_0:{gc.event_id}",
        session_index=0,
        t_start_ms=gc.t_start_ms,
        t_end_ms=gc.t_end_ms,
        model=gc.call.model,
        messages=gc.call.messages,
        expected_output=gc.call.expected_output,
        input_segments=gc.call.input_segments,
        expected_output_tokens=gc.call.expected_output_tokens,
        temperature=gc.call.temperature,
        max_tokens_recorded=gc.call.max_tokens_recorded,
        predecessor_event_ids=[],
        wait_ms=0,
    )
    gen = object.__new__(OTelTraceReplayDataGenerator)
    gen.all_events = [event]
    gen._session_events = {0: [event]}
    gen.output_registry = EventOutputRegistry()
    gen.worker_tracker = WorkerSessionTracker()
    gen.session_completion_queue = None
    cfg = MagicMock()
    cfg.bad_tool_call_handling = BadToolCallHandling.NONE
    cfg.tool_choice_mode = ToolChoiceMode.FORCE_RECORDED
    cfg.attribute_to_header_map = {}
    cfg.attribute_to_label_map = {}
    gen.otel_config = cfg
    gen.replay_config = cfg
    gen.session_graph_state = {"session_0": MagicMock(graph=graph, random_string=None)}
    return gen


class TestSessionRequest:
    def test_load_lazy_data_keeps_image_parts(self) -> None:
        api_data = _generator_for(READ_REQUEST).load_lazy_data(LazyLoadInferenceAPIData(data_index=0))
        tool_message = next(m for m in api_data.messages if m.role == "tool")
        assert tool_message.content == [IMAGE]
        assert api_data.original_messages[-1]["content"] == tool_message.content

    @pytest.mark.asyncio
    async def test_random_session_id_is_a_text_part_on_image_content(self) -> None:
        content = [IMAGE]
        api_data = SessionChatCompletionAPIData(
            messages=[ChatMessage(role="user", content=content)],
            max_tokens=10,
            event_id="session_1:event_0",
            registry=EventOutputRegistry(),
            worker_tracker=WorkerSessionTracker(),
            completion_queue=None,
            total_events_in_session=1,
            inject_random_session_id=True,
            session_random_string="abc",
            input_segments=[InputSegment(type="unique", message_count=1, token_count=10, source_event_id=None)],
            original_messages=[{"role": "user", "content": content}],
        )
        await api_data.wait_for_predecessors_and_substitute()
        assert api_data.messages[0].content == [{"type": "text", "text": "[SESS:abc] "}, *content]

    def test_replayed_images_are_measured(self) -> None:
        images = replayed_images([ChatMessage(role="tool", content=[IMAGE])])
        assert images is not None
        assert images.count == 1
        assert images.instances[0].pixels == 6
        assert images.instances[0].bytes == len(base64.b64decode(PNG))
        assert replayed_images([ChatMessage(role="user", content="text")]) is None


class TestAnthropicRenderer:
    def test_images_render_as_native_blocks(self) -> None:
        messages = [
            ChatMessage(role="user", content=[{"type": "text", "text": "what?"}, IMAGE]),
            ChatMessage(role="tool", tool_call_id="toolu_1", content=[IMAGE]),
        ]
        body = build_anthropic_request_body(messages, None, "m", 10, False)
        image = _image_block()
        assert body["messages"][0]["content"] == [{"type": "text", "text": "what?"}, image]
        assert body["messages"][1]["content"] == [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [image]}]

    def test_text_tool_result_stays_a_string(self) -> None:
        body = build_anthropic_request_body([ChatMessage(role="tool", tool_call_id="t", content="done")], None, "m", 10, False)
        assert body["messages"][0]["content"][0]["content"] == "done"


class TestMissingImages:
    def _span(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "trace_id": "t",
            "span_id": "s",
            "name": "chat m",
            "start_time": "2026-01-01T00:00:00",
            "end_time": "2026-01-01T00:00:01",
            "attributes": {"gen_ai.operation.name": "chat", "gen_ai.input.messages": json.dumps(messages)},
        }

    @pytest.mark.parametrize(
        "part",
        [
            {"type": "tool_call_response", "id": "c1", "response": "[image]"},
            {"type": "blob", "modality": "image", "mime_type": "image/png", "content": ""},
            {"type": "text", "content": "see @@@langfuseMedia:type=image/png|id=1|source=base64@@@"},
        ],
    )
    def test_placeholders_are_warned_about(self, part: Dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            build_raw_calls([self._span([{"role": "user", "parts": [part]}])])
        assert "1 image occurrence(s) in 1 call(s) are missing" in caplog.text

    def test_real_images_and_markdown_alt_text_are_not_warned_about(self, caplog: pytest.LogCaptureFixture) -> None:
        parts = [
            {"type": "text", "content": "![image](/problem_assets/1.png) [image] is alt text"},
            {"type": "blob", "modality": "image", "mime_type": "image/png", "content": PNG},
        ]
        with caplog.at_level(logging.WARNING):
            build_raw_calls([self._span([{"role": "user", "parts": parts}])])
        assert "missing from the trace" not in caplog.text


def _responses_record(input_items: List[Dict[str, Any]], tools: Any = None) -> Dict[str, Any]:
    completed = {
        "type": "response.completed",
        "response": {
            "id": "resp_1",
            "status": "completed",
            "output": [{"type": "computer_call", "call_id": "cc_2", "action": {"type": "click", "x": 10, "y": 20}}],
            "usage": {"input_tokens": 900, "output_tokens": 5},
        },
    }
    request: Dict[str, Any] = {"model": "computer-use-preview", "input": input_items, "stream": True}
    if tools is not None:
        request["tools"] = tools
    return {
        "path": "/v1/responses",
        "start_unix": 1_700_000_000.0,
        "total_ms": 500,
        "status": 200,
        "request": json.dumps(request),
        "response": f"data: {json.dumps(completed)}\n\n",
    }


class TestResponsesApi:
    """Responses API shapes: computer use screenshots, input_image messages, image tool outputs."""

    COMPUTER_SESSION: List[Dict[str, Any]] = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Open the settings"}]},
        {"type": "computer_call", "call_id": "cc_1", "action": {"type": "screenshot"}},
        {"type": "computer_call_output", "call_id": "cc_1", "output": {"type": "computer_screenshot", "image_url": DATA_URL}},
    ]

    def _graph(self, items: List[Dict[str, Any]], tools: Any = None) -> Any:
        span = convert_wire_record(_responses_record(items, tools), "t", 1)
        assert span is not None
        calls, _ = build_raw_calls([span])
        return next(iter(build_graph(calls, source_file="t").events.values())).call

    def test_computer_screenshot_replays_as_an_image_answering_a_computer_call(self) -> None:
        call = self._graph(self.COMPUTER_SESSION, tools=[{"type": "computer_use_preview", "display_width": 1024}])
        assistant, tool = call.messages[1], call.messages[2]
        assert assistant["tool_calls"][0]["function"] == {"name": "computer", "arguments": '{"type": "screenshot"}'}
        assert tool == {"role": "tool", "tool_call_id": "cc_1", "content": [IMAGE]}
        assert [t["name"] for t in call.tool_definitions] == ["computer"]
        assert call.expected_output_tool_names == ["computer"]

    def test_input_image_in_a_user_message_keeps_its_place(self) -> None:
        items = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Current screenshot:"},
                    {"type": "input_image", "image_url": DATA_URL},
                ],
            }
        ]
        assert self._graph(items).messages[0]["content"] == [{"type": "text", "text": "Current screenshot:"}, IMAGE]

    def test_function_call_output_with_an_image(self) -> None:
        items: List[Dict[str, Any]] = [
            {"type": "function_call", "call_id": "fc_1", "name": "screenshot", "arguments": "{}"},
            {
                "type": "function_call_output",
                "call_id": "fc_1",
                "output": [{"type": "input_text", "text": "done"}, {"type": "input_image", "image_url": DATA_URL}],
            },
        ]
        assert self._graph(items).messages[1] == {
            "role": "tool",
            "tool_call_id": "fc_1",
            "content": [{"type": "text", "text": "done"}, IMAGE],
        }

    def test_text_only_messages_keep_their_joined_form(self) -> None:
        items = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}],
            }
        ]
        assert self._graph(items).messages[0]["content"] == "a\nb"

    def test_uploaded_file_image_is_warned_about(self, caplog: pytest.LogCaptureFixture) -> None:
        items = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "file_id": "file_1"}, {"type": "input_text", "text": "x"}],
            }
        ]
        with caplog.at_level(logging.WARNING):
            self._graph(items)
        assert "1 image occurrence(s) in 1 call(s) are missing" in caplog.text


def test_builtin_responses_tools_are_not_replayed_as_unnamed_functions() -> None:
    """Codex sends {"type": "web_search"} beside its functions; an empty function name fails the request."""
    tools = [{"type": "function", "name": "view_image", "parameters": {"type": "object"}}, {"type": "web_search"}]
    span = convert_wire_record(_responses_record(TestResponsesApi.COMPUTER_SESSION[:1], tools), "t", 1)
    assert span is not None
    assert [t["name"] for t in json.loads(span["attributes"]["gen_ai.tool.definitions"])] == ["view_image"]


class TestReviewRegressions:
    """Cases found in review: images must never become text, and text-only traces must not change."""

    def test_image_beside_a_tool_result_rides_on_it_as_an_image(self) -> None:
        request: Dict[str, Any] = dict(READ_REQUEST)
        request["messages"] = READ_REQUEST["messages"][:2] + [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "done"}, _image_block()]}
        ]
        tool_message = _graph_messages(request)[-1]
        assert tool_message["content"] == [{"type": "text", "text": "done"}, IMAGE]
        assert PNG not in json.dumps([p for p in tool_message["content"] if p.get("type") == "text"])

    def test_undecodable_image_is_counted_without_raising(self) -> None:
        images = replayed_images(
            [ChatMessage(role="user", content=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}])]
        )
        # Counted, but not listed: an unknown size must not pull the pixel/byte distributions to zero.
        assert images is not None and images.count == 1 and images.instances == []

    def test_image_detail_is_kept(self) -> None:
        part = {"type": "image_url", "image_url": {"url": DATA_URL, "detail": "low"}}
        request = {"model": "m", "messages": [{"role": "user", "content": [{"type": "text", "text": "what?"}, part]}]}
        record = {
            "path": "/v1/chat/completions",
            "start_unix": 1_700_000_000.0,
            "status": 200,
            "request": json.dumps(request),
            "response": json.dumps(
                {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}
            ),
        }
        span = convert_wire_record(record, "t", 1)
        assert span is not None
        calls, _ = build_raw_calls([span])
        message = next(iter(build_graph(calls, source_file="t").events.values())).call.messages[0]
        assert message["content"][1] == part

    def test_anthropic_file_image_is_warned_about(self, caplog: pytest.LogCaptureFixture) -> None:
        file_image = {"type": "image", "source": {"type": "file", "file_id": "file_1"}}
        request = {
            "model": "claude",
            "stream": False,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "look"}, file_image]}],
        }
        with caplog.at_level(logging.WARNING):
            build_raw_calls([_span(request)])
        assert "1 image occurrence(s) in 1 call(s) are missing" in caplog.text

    def test_named_custom_tools_are_kept(self) -> None:
        tools = [{"type": "custom", "name": "apply_patch", "description": "patch"}, {"type": "web_search"}]
        span = convert_wire_record(_responses_record(TestResponsesApi.COMPUTER_SESSION[:1], tools), "t", 1)
        assert span is not None
        assert [t["name"] for t in json.loads(span["attributes"]["gen_ai.tool.definitions"])] == ["apply_patch"]

    def test_text_only_tool_result_list_reconstructs_as_before(self) -> None:
        part = {"type": "tool_call_response", "id": "c1", "response": [{"type": "text", "text": "hello"}, "plain"]}
        text = reconstruct_llm_input({"role": "tool", "parts": [part]})
        assert "[{'type': 'text', 'text': 'hello'}, 'plain']" in text

    def test_assistant_text_blocks_with_tool_calls_replay_as_before(self) -> None:
        span = {
            "trace_id": "t",
            "span_id": "s",
            "name": "chat m",
            "start_time": "2026-01-01T00:00:00",
            "end_time": "2026-01-01T00:00:01",
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.input.messages": json.dumps(
                    [
                        {"role": "user", "content": "hi"},
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Let me check"}],
                            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
                        },
                    ]
                ),
            },
        }
        calls, _ = build_raw_calls([span])
        assistant = next(iter(build_graph(calls, source_file="t").events.values())).call.messages[1]
        assert assistant["content"] == ""


class TestExternalReviewRegressions:
    """Second review round: summarized spans, sibling blocks, image detail, URL images, JPEG headers."""

    def test_summarized_image_is_missing_not_sent(self, caplog: pytest.LogCaptureFixture) -> None:
        """inference-perf's own spans, replayed as a trace, carry summaries instead of bytes."""
        messages = summarize_media([{"role": "user", "content": [{"type": "text", "text": "look"}, IMAGE]}])
        span = {
            "trace_id": "t",
            "span_id": "s",
            "name": "chat m",
            "start_time": "2026-01-01T00:00:00",
            "end_time": "2026-01-01T00:00:01",
            "attributes": {"gen_ai.operation.name": "chat", "gen_ai.input.messages": json.dumps(messages)},
        }
        with caplog.at_level(logging.WARNING):
            calls, _ = build_raw_calls([span])
        assert "1 image occurrence(s) in 1 call(s) are missing" in caplog.text
        assert next(iter(build_graph(calls, source_file="t").events.values())).call.messages[0]["content"] == "look"

    def test_search_result_beside_an_image_keeps_its_text(self) -> None:
        search = {
            "type": "search_result",
            "source": "https://x",
            "title": "T",
            "content": [{"type": "text", "text": "SEARCH TEXT"}],
        }
        request: Dict[str, Any] = dict(READ_REQUEST)
        request["messages"] = READ_REQUEST["messages"][:2] + [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [search, _image_block()]}],
            }
        ]
        content = _graph_messages(request)[-1]["content"]
        assert content[0] == {"type": "text", "text": json.dumps(search)}
        assert content[1] == IMAGE

    def test_image_detail_is_part_of_its_identity(self) -> None:
        def msg(detail: str) -> ComplexReplayMessage:
            info = {
                "role": "user",
                "parts": [{"type": "blob", "modality": "image", "mime_type": "image/png", "content": PNG, "detail": detail}],
            }
            return ComplexReplayMessage(role="user", message_info=info, raw_reconstructed_text=reconstruct_llm_input(info))

        assert not messages_equal(msg("low"), msg("high"))

    def test_url_image_is_counted(self) -> None:
        images = replayed_images(
            [ChatMessage(role="user", content=[{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}])]
        )
        assert images is not None and images.count == 1 and images.instances == []

    def test_jpeg_with_large_metadata_reports_its_size(self) -> None:
        buf = io.BytesIO()
        PILImage.new("RGB", (640, 480)).save(buf, format="JPEG", exif=b"Exif\x00\x00" + b"\x00" * (60 * 1024))
        url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        images = replayed_images([ChatMessage(role="user", content=[{"type": "image_url", "image_url": {"url": url}}])])
        assert images is not None and images.instances[0].pixels == 640 * 480


class TestThirdReviewRegressions:
    """Third review round: image position among parallel results, robustness, non-image files."""

    def test_image_after_parallel_results_stays_after_them(self) -> None:
        request: Dict[str, Any] = dict(READ_REQUEST)
        request["messages"] = [
            READ_REQUEST["messages"][0],
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "A", "name": "f", "input": {}},
                    {"type": "tool_use", "id": "B", "name": "f", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "A", "content": "a"},
                    {"type": "tool_result", "tool_use_id": "B", "content": "b"},
                    _image_block(),
                ],
            },
        ]
        tool_a, tool_b = _graph_messages(request)[-2:]
        assert tool_a == {"role": "tool", "tool_call_id": "A", "content": "a"}
        assert tool_b == {"role": "tool", "tool_call_id": "B", "content": [{"type": "text", "text": "b"}, IMAGE]}

    def test_non_string_blob_content_is_not_an_image(self) -> None:
        assert to_openai_image({"type": "blob", "modality": "image", "content": ["x"]}) is None

    def test_chat_file_part_is_not_an_image(self) -> None:
        pdf = {"type": "file", "file": {"filename": "a.pdf", "file_data": "data:application/pdf;base64,AAAA"}}
        assert not is_image_part(pdf)
        assert count_missing_images(pdf) == 0


class TestPR900ReviewCoverage:
    """Paths found untested by mutation testing on #900."""

    REMINDER = "<system-reminder>Keep the existing style.</system-reminder>"

    def test_text_beside_a_tool_result_image_is_kept(self) -> None:
        """Claude Code sends a <system-reminder> text block next to a tool_result holding a screenshot."""
        request: Dict[str, Any] = dict(READ_REQUEST)
        request["messages"] = READ_REQUEST["messages"][:2] + [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [_image_block()]},
                    {"type": "text", "text": self.REMINDER},
                ],
            }
        ]
        tool_message = _graph_messages(request)[-1]
        assert tool_message["content"] == [{"type": "text", "text": self.REMINDER}, IMAGE]

        body = build_anthropic_request_body(
            [ChatMessage(role="tool", tool_call_id="toolu_1", content=tool_message["content"])], None, "m", 10, False
        )
        assert body["messages"][0]["content"][0]["content"] == [{"type": "text", "text": self.REMINDER}, _image_block()]

    def test_screenshot_tool_result_is_not_counted_as_text(self) -> None:
        """A 192 KB screenshot in a tool result must be estimated as a digest, not as its base64."""
        screenshot = base64.b64encode(b"\x00" * 192 * 1024).decode()
        response = [{"type": "blob", "modality": "image", "mime_type": "image/png", "content": screenshot}]
        info = {"role": "tool", "parts": [{"type": "tool_call_response", "id": "toolu_1", "response": response}]}
        msg = ComplexReplayMessage(role="tool", message_info=info, raw_reconstructed_text=reconstruct_llm_input(info))
        assert screenshot not in msg.text
        assert message_tokens(msg) < 100

    def test_url_image_renders_as_an_anthropic_url_source(self) -> None:
        url = "https://example.com/a.png"
        body = build_anthropic_request_body(
            [ChatMessage(role="user", content=[{"type": "image_url", "image_url": {"url": url}}])], None, "m", 10, False
        )
        assert body["messages"][0]["content"] == [{"type": "image", "source": {"type": "url", "url": url}}]

    def test_raw_dict_tool_results_keep_image_order(self) -> None:
        """An OTel message in raw OpenAI shape, with tool_call_response parts beside an image."""
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "A", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                    {"id": "B", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_call_response", "id": "A", "response": "a"},
                    {"type": "tool_call_response", "id": "B", "response": "b"},
                    IMAGE,
                ],
            },
        ]
        span = {
            "trace_id": "t",
            "span_id": "s",
            "name": "chat m",
            "start_time": "2026-01-01T00:00:00",
            "end_time": "2026-01-01T00:00:01",
            "attributes": {"gen_ai.operation.name": "chat", "gen_ai.input.messages": json.dumps(messages)},
        }
        calls, _ = build_raw_calls([span])
        tool_a, tool_b = next(iter(build_graph(calls, source_file="t").events.values())).call.messages[-2:]
        assert tool_a == {"role": "tool", "tool_call_id": "A", "content": "a"}
        assert tool_b == {"role": "tool", "tool_call_id": "B", "content": [{"type": "text", "text": "b"}, IMAGE]}
