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

"""Regression tests for preserving recorded reasoning effort during replay."""

import json
from copy import copy
from multiprocessing.managers import SyncManager
from pathlib import Path

import pytest
from pydantic import ValidationError

from inference_perf.config import APIConfig, APIType, DataConfig, SessionReplayConfig
from inference_perf.config.datagen.replay import ToolChoiceMode
from inference_perf.datagen.replay.export_replay_graph_to_dot import export_to_dot
from inference_perf.datagen.replay.otel_trace_to_replay_graph import (
    build_graph,
    build_raw_calls,
    graph_to_dict,
    summarize_graph,
)
from inference_perf.datagen.replay.replay_graph_session_datagen import ReplayGraphSessionGeneratorBase, ReplaySession
from inference_perf.datagen.replay.replay_graph_types import ReplayGraph
from inference_perf.datagen.replay.wire_trace_converter import convert_wire_record
from inference_perf.utils.mp_context import MP_CONTEXT


# The three wire formats record "high" in different request fields; all
# must preserve it under the same OTel span attribute.
@pytest.mark.parametrize(
    ("path", "request_body"),
    [
        ("/v1/chat/completions", {"reasoning_effort": "high"}),
        ("/v1/responses", {"reasoning": {"effort": "high"}}),
        ("/v1/messages", {"output_config": {"effort": "high"}}),
    ],
)
def test_wire_preserves_recorded_reasoning_effort(path: str, request_body: dict[str, object]) -> None:
    span = convert_wire_record({"path": path, "request": request_body, "start_unix": 1}, "trace")
    assert span is not None
    assert span["attributes"].get("gen_ai.request.reasoning_effort") == "high"


# Missing and malformed fields leave the span unchanged instead of adding
# a null attribute or failing conversion of an otherwise usable record.
@pytest.mark.parametrize(
    ("path", "request_body"),
    [
        ("/v1/chat/completions", {}),
        ("/v1/chat/completions", {"reasoning_effort": 1}),
        ("/v1/responses", {"reasoning": "high"}),
        ("/v1/responses", {"reasoning": {"effort": None}}),
        ("/v1/messages", {"output_config": {"format": {"type": "json_schema"}}}),
    ],
)
def test_wire_omits_missing_or_malformed_effort(path: str, request_body: dict[str, object]) -> None:
    span = convert_wire_record({"path": path, "request": request_body, "start_unix": 1}, "trace")
    assert span is not None
    assert "gen_ai.request.reasoning_effort" not in span["attributes"]


# Build one native OTel chat call without source API metadata so replay must
# decide compatibility using the target vocabulary, not the provider's name.
def _native_graph(effort: object) -> ReplayGraph:
    attrs: dict[str, object] = {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "test-model",
        "gen_ai.input.messages": json.dumps([{"role": "user", "parts": [{"type": "text", "content": "Hello"}]}]),
        "gen_ai.usage.output_tokens": 8,
    }
    if effort is not None:
        attrs["gen_ai.request.reasoning_effort"] = effort
    calls, _ = build_raw_calls(
        [{"span_id": "call", "trace_id": "trace", "start_time": "2026-01-01T00:00:00Z", "attributes": attrs}]
    )
    return build_graph(calls)


# A recorded effort appears in each graph export; unrecorded and non-string
# values do not change the call's exported decoding fields.
@pytest.mark.parametrize("effort", ["high", None, 1])
def test_native_effort_graph_exports(effort: object, tmp_path: Path) -> None:
    graph = _native_graph(effort)
    result = graph_to_dict(graph)
    event = next(iter(result["events"].values()))
    dot_path = tmp_path / "graph.dot"
    export_to_dot(result, str(dot_path))
    if isinstance(effort, str):
        assert event["call"]["reasoning_effort"] == effort
        assert f"reasoning_effort={effort}" in summarize_graph(graph)
        assert f"reasoning_effort={effort}" in dot_path.read_text(encoding="utf-8")
    else:
        assert "reasoning_effort" not in event["call"]
        assert "reasoning_effort=" not in summarize_graph(graph)
        assert "reasoning_effort=" not in dot_path.read_text(encoding="utf-8")


# Schedule one graph-backed call, optionally marking it as a recorded tool
# turn so the replay builder takes the tool-choice path.
def _replay_generator(
    target: APIType,
    config: SessionReplayConfig,
    graph: ReplayGraph,
    *,
    tool_call: bool = False,
    manager: SyncManager | None = None,
) -> ReplayGraphSessionGeneratorBase:
    if tool_call:
        call = next(iter(graph.events.values())).call
        call.expected_output_is_tool_call = True
        call.tool_definitions = [
            {"type": "function", "name": "lookup", "description": "Lookup", "parameters": {"type": "object"}}
        ]
    generator = ReplayGraphSessionGeneratorBase(
        APIConfig(type=target, streaming=False), DataConfig(), None, mp_manager=manager, replay_config=config
    )
    generator.initialize_sessions([ReplaySession(session_id="s", source_id="trace", session_index=0, graph=graph)])
    return generator


# Serialize the request through the same lazy materialization used by replay
# workers, without contacting a model server.
async def _replay_body(generator: ReplayGraphSessionGeneratorBase) -> dict[str, object]:
    (item,) = generator.get_session_events(0)
    request = generator.load_lazy_data(item)
    return await request.to_request_body("target-model", max_tokens=20, ignore_eos=False, streaming=False)


# The matching value is forwarded unchanged for each target, even when
# as_recorded takes the early return that leaves tool_choice untouched.
@pytest.mark.parametrize(
    ("target", "key", "value"),
    [(APIType.Chat, "reasoning_effort", "high"), (APIType.AnthropicMessages, "output_config", {"effort": "high"})],
)
async def test_opt_in_forwards_recorded_effort(target: APIType, key: str, value: object) -> None:
    config = SessionReplayConfig(replay_request_params=["reasoning_effort"], tool_choice_mode=ToolChoiceMode.AS_RECORDED)
    body = await _replay_body(_replay_generator(target, config, _native_graph("high"), tool_call=True))
    assert body[key] == value
    assert "tool_choice" not in body


# A trace with effort replayed at defaults must serialize exactly like the
# same trace without effort. Opt-in cannot invent a value for a missing one.
@pytest.mark.parametrize("target", [APIType.Chat, APIType.AnthropicMessages])
async def test_default_and_missing_effort_leave_payload_unchanged(target: APIType) -> None:
    default = SessionReplayConfig()
    opt_in = SessionReplayConfig(replay_request_params=["reasoning_effort"])
    baseline = await _replay_body(_replay_generator(target, default, _native_graph(None)))
    with_recorded_effort = await _replay_body(_replay_generator(target, default, _native_graph("high")))
    with_missing_effort = await _replay_body(_replay_generator(target, opt_in, _native_graph(None)))
    assert json.dumps(with_recorded_effort) == json.dumps(baseline) == json.dumps(with_missing_effort)
    assert "reasoning_effort" not in baseline
    assert "output_config" not in baseline


# "minimal" can go to Chat but not Anthropic. Two copies of a generator
# sharing its manager proxy must omit the Anthropic field and warn only once.
async def test_incompatible_effort_skips_and_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    config = SessionReplayConfig(replay_request_params=["reasoning_effort"])
    graph = _native_graph("minimal")
    chat = await _replay_body(_replay_generator(APIType.Chat, config, graph))
    assert chat["reasoning_effort"] == "minimal"
    with MP_CONTEXT.Manager() as manager:
        generator = _replay_generator(APIType.AnthropicMessages, config, graph, manager=manager)
        worker_copy = copy(generator)
        first = await _replay_body(generator)
        second = await _replay_body(worker_copy)
    assert "output_config" not in first
    assert "output_config" not in second
    assert caplog.text.count("Skipping recorded reasoning_effort 'minimal'") == 1


# Without a worker manager, repeated unsupported values still warn only once
# and never enter the request body.
async def test_unknown_effort_without_workers(caplog: pytest.LogCaptureFixture) -> None:
    config = SessionReplayConfig(replay_request_params=["reasoning_effort"])
    generator = _replay_generator(APIType.Chat, config, _native_graph("custom"))
    assert "reasoning_effort" not in await _replay_body(generator)
    assert "reasoning_effort" not in await _replay_body(generator)
    assert caplog.text.count("Skipping recorded reasoning_effort 'custom'") == 1


# Request parameters not yet implemented must fail config validation rather
# than silently having no effect.
def test_unknown_replay_parameter_is_rejected() -> None:
    with pytest.raises(ValidationError, match="replay_request_params"):
        SessionReplayConfig.model_validate({"replay_request_params": ["temperature"]})
