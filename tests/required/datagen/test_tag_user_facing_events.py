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

"""Tests for tag_user_facing_events (TFUT user-facing event tagging)."""

from inference_perf.datagen.replay.replay_graph_types import GraphCall, GraphEvent, ReplayGraph
from inference_perf.datagen.replay.replay_graph_builder import tag_user_facing_events


def _make_call(
    call_id: str = "span1",
    expected_output_is_tool_call: bool = False,
    attributes: dict[str, object] | None = None,
) -> GraphCall:
    from inference_perf.datagen.replay.replay_graph_types import InputSegment

    return GraphCall(
        call_id=call_id,
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        expected_output="response",
        input_segments=[InputSegment(type="unique", message_count=1, token_count=5)],
        total_input_tokens=5,
        expected_output_tokens=10,
        temperature=0.0,
        max_tokens_recorded=100,
        expected_output_is_tool_call=expected_output_is_tool_call,
        attributes=attributes,
    )


def _make_event(
    event_id: str,
    call: GraphCall | None = None,
    predecessor_event_ids: list[str] | None = None,
    predecessor_dependency_types: dict[str, str] | None = None,
) -> GraphEvent:
    return GraphEvent(
        event_id=event_id,
        call=call or _make_call(call_id=event_id),
        predecessor_event_ids=predecessor_event_ids or [],
        predecessor_dependency_types=predecessor_dependency_types or {},
        wait_ms=0,
        t_start_ms=0,
        t_end_ms=100,
    )


def _make_graph(events: list[GraphEvent]) -> ReplayGraph:
    event_map = {e.event_id: e for e in events}
    root_ids = [e.event_id for e in events if not e.predecessor_event_ids]
    return ReplayGraph(events=event_map, root_event_ids=root_ids, source_file="test")


# ---------------------------------------------------------------------------
# Graph position: successors do NOT exclude (only tool-internal does)
# ---------------------------------------------------------------------------


class TestGraphPosition:
    def test_single_event_is_user_facing(self) -> None:
        """A single event with no successors is user-facing."""
        e = _make_event("e1")
        graph = _make_graph([e])
        tag_user_facing_events(graph)
        assert graph.events["e1"].is_user_facing is True

    def test_non_leaf_with_temporal_successor_is_user_facing(self) -> None:
        """In a multi-turn conversation, all events are user-facing (temporal deps)."""
        e1 = _make_event("e1")
        e2 = _make_event("e2", predecessor_event_ids=["e1"], predecessor_dependency_types={"e1": "temporal"})
        graph = _make_graph([e1, e2])
        tag_user_facing_events(graph)
        assert graph.events["e1"].is_user_facing is True
        assert graph.events["e2"].is_user_facing is True

    def test_tool_call_with_causal_successor_is_not_user_facing(self) -> None:
        """A tool-call event whose output is consumed by a successor is not user-facing."""
        e1 = _make_event("e1", call=_make_call(expected_output_is_tool_call=True))
        e2 = _make_event("e2", predecessor_event_ids=["e1"], predecessor_dependency_types={"e1": "causal"})
        graph = _make_graph([e1, e2])
        tag_user_facing_events(graph)
        assert graph.events["e1"].is_user_facing is False
        assert graph.events["e2"].is_user_facing is True

    def test_prose_with_causal_successor_is_user_facing(self) -> None:
        """A prose event with a causal successor is still user-facing (multi-turn)."""
        e1 = _make_event("e1", call=_make_call(expected_output_is_tool_call=False))
        e2 = _make_event("e2", predecessor_event_ids=["e1"], predecessor_dependency_types={"e1": "causal"})
        graph = _make_graph([e1, e2])
        tag_user_facing_events(graph)
        assert graph.events["e1"].is_user_facing is True
        assert graph.events["e2"].is_user_facing is True


# ---------------------------------------------------------------------------
# Condition 2: not a tool call
# ---------------------------------------------------------------------------


class TestConditionNotToolCall:
    def test_expected_tool_call_excludes(self) -> None:
        """expected_output_is_tool_call=True excludes from leaf."""
        e = _make_event("e1", call=_make_call(expected_output_is_tool_call=True))
        graph = _make_graph([e])
        tag_user_facing_events(graph)
        assert graph.events["e1"].is_user_facing is False

    def test_tool_call_finish_reason_excludes(self) -> None:
        """finish_reason='tool_calls' excludes even when expected_output_is_tool_call=False."""
        e = _make_event(
            "e1",
            call=_make_call(
                expected_output_is_tool_call=False,
                attributes={"gen_ai.response.finish_reasons": ["tool_calls"]},
            ),
        )
        graph = _make_graph([e])
        tag_user_facing_events(graph)
        assert graph.events["e1"].is_user_facing is False

    def test_tool_use_finish_reason_excludes(self) -> None:
        """finish_reason='tool_use' (Anthropic style) also excludes."""
        e = _make_event(
            "e1",
            call=_make_call(
                expected_output_is_tool_call=False,
                attributes={"gen_ai.response.finish_reasons": ["tool_use"]},
            ),
        )
        graph = _make_graph([e])
        tag_user_facing_events(graph)
        assert graph.events["e1"].is_user_facing is False

    def test_stop_finish_reason_does_not_exclude(self) -> None:
        """finish_reason='stop' does not trigger the tool-call exclusion."""
        e = _make_event(
            "e1",
            call=_make_call(
                expected_output_is_tool_call=False,
                attributes={"gen_ai.response.finish_reasons": ["stop"]},
            ),
        )
        graph = _make_graph([e])
        tag_user_facing_events(graph)
        assert graph.events["e1"].is_user_facing is True


# ---------------------------------------------------------------------------
# Condition 3: not a structured-output call
# ---------------------------------------------------------------------------


class TestConditionNotStructuredOutput:
    def test_output_schema_excludes(self) -> None:
        e = _make_event(
            "e1",
            call=_make_call(attributes={"gen_ai.request.output_schema": '{"type": "object"}'}),
        )
        graph = _make_graph([e])
        tag_user_facing_events(graph)
        assert graph.events["e1"].is_user_facing is False
        assert graph.events["e1"].is_structured_output_call is True

    def test_json_output_type_excludes(self) -> None:
        e = _make_event(
            "e1",
            call=_make_call(attributes={"gen_ai.output.type": "json"}),
        )
        graph = _make_graph([e])
        tag_user_facing_events(graph)
        assert graph.events["e1"].is_user_facing is False
        assert graph.events["e1"].is_structured_output_call is True

    def test_text_output_type_does_not_exclude(self) -> None:
        e = _make_event(
            "e1",
            call=_make_call(attributes={"gen_ai.output.type": "text"}),
        )
        graph = _make_graph([e])
        tag_user_facing_events(graph)
        assert graph.events["e1"].is_user_facing is True
        assert graph.events["e1"].is_structured_output_call is False


# ---------------------------------------------------------------------------
# Condition 4: not tool-internal
# ---------------------------------------------------------------------------


class TestConditionNotToolInternal:
    def test_structural_tool_internal_via_spans(self) -> None:
        """Event nested under a tool.execution span is marked tool-internal."""
        e = _make_event("e1", call=_make_call(call_id="child_span"))
        graph = _make_graph([e])
        spans = [
            {"span_id": "tool_exec_1", "name": "tool.execution", "parent_span_id": "root"},
            {"span_id": "child_span", "name": "llm.call", "parent_span_id": "tool_exec_1"},
        ]
        tag_user_facing_events(graph, all_spans=spans)
        assert graph.events["e1"].is_tool_internal is True
        assert graph.events["e1"].is_user_facing is False

    def test_deeply_nested_tool_internal(self) -> None:
        """Event nested two levels deep under tool.execution is still tool-internal."""
        e = _make_event("e1", call=_make_call(call_id="deep_span"))
        graph = _make_graph([e])
        spans = [
            {"span_id": "tool_exec_1", "name": "tool.execution", "parent_span_id": "root"},
            {"span_id": "mid_span", "name": "processing", "parent_span_id": "tool_exec_1"},
            {"span_id": "deep_span", "name": "llm.call", "parent_span_id": "mid_span"},
        ]
        tag_user_facing_events(graph, all_spans=spans)
        assert graph.events["e1"].is_tool_internal is True

    def test_fallback_tool_call_with_causal_successor(self) -> None:
        """Without span info, a tool-call event with a causal successor is tool-internal."""
        e1 = _make_event("e1", call=_make_call(expected_output_is_tool_call=True))
        e2 = _make_event("e2", predecessor_event_ids=["e1"], predecessor_dependency_types={"e1": "causal"})
        graph = _make_graph([e1, e2])
        tag_user_facing_events(graph, all_spans=None)
        assert graph.events["e1"].is_tool_internal is True
        assert graph.events["e1"].is_user_facing is False

    def test_fallback_prose_with_causal_successor_not_tool_internal(self) -> None:
        """Without span info, a prose event with a causal successor is NOT tool-internal (multi-turn)."""
        e1 = _make_event("e1", call=_make_call(expected_output_is_tool_call=False))
        e2 = _make_event("e2", predecessor_event_ids=["e1"], predecessor_dependency_types={"e1": "causal"})
        graph = _make_graph([e1, e2])
        tag_user_facing_events(graph, all_spans=None)
        assert graph.events["e1"].is_tool_internal is False
        assert graph.events["e1"].is_user_facing is True

    def test_temporal_successor_not_tool_internal(self) -> None:
        """A temporal-only successor does NOT trigger tool-internal fallback."""
        e1 = _make_event("e1")
        e2 = _make_event("e2", predecessor_event_ids=["e1"], predecessor_dependency_types={"e1": "temporal"})
        graph = _make_graph([e1, e2])
        tag_user_facing_events(graph, all_spans=None)
        assert graph.events["e1"].is_tool_internal is False

    def test_structural_info_disables_fallback(self) -> None:
        """When span info is available, causal-successor fallback is disabled."""
        e1 = _make_event("e1", call=_make_call(call_id="span_a"))
        e2 = _make_event(
            "e2",
            call=_make_call(call_id="span_b"),
            predecessor_event_ids=["e1"],
            predecessor_dependency_types={"e1": "causal"},
        )
        graph = _make_graph([e1, e2])
        # Provide spans but e1's span is NOT under a tool.execution
        spans = [
            {"span_id": "tool_exec_1", "name": "tool.execution", "parent_span_id": "root"},
            {"span_id": "span_a", "name": "llm.call", "parent_span_id": "root"},
            {"span_id": "span_b", "name": "llm.call", "parent_span_id": "root"},
        ]
        tag_user_facing_events(graph, all_spans=spans)
        # e1 has a causal successor but structural info says it's NOT under tool.execution
        assert graph.events["e1"].is_tool_internal is False


# ---------------------------------------------------------------------------
# Combined conditions
# ---------------------------------------------------------------------------


class TestCombinedConditions:
    def test_all_conditions_met(self) -> None:
        """Not tool call + not structured output + not tool-internal = user-facing."""
        e = _make_event(
            "e1",
            call=_make_call(
                expected_output_is_tool_call=False,
                attributes={"gen_ai.response.finish_reasons": ["stop"]},
            ),
        )
        graph = _make_graph([e])
        tag_user_facing_events(graph)
        assert graph.events["e1"].is_user_facing is True

    def test_multiple_exclusion_reasons(self) -> None:
        """An event that fails multiple conditions is still not user-facing."""
        e = _make_event(
            "e1",
            call=_make_call(
                expected_output_is_tool_call=True,
                attributes={"gen_ai.request.output_schema": "{}"},
            ),
        )
        graph = _make_graph([e])
        tag_user_facing_events(graph)
        assert graph.events["e1"].is_user_facing is False
        assert graph.events["e1"].is_structured_output_call is True
