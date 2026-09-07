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

"""Integration test: synthetic agentic graph has correct TFUT user-facing tagging."""

from typing import cast

from inference_perf.datagen.replay.replay_graph_types import ReplayGraph
from inference_perf.datagen.synthetic_agentic.synthetic_agentic_datagen import build_graph_for_session
from inference_perf.datagen.synthetic_agentic.synthetic_themes import GENERIC_THEME
from inference_perf.datagen.replay.otel_trace_to_replay_graph import tag_user_facing_events
from inference_perf.config.common import Distribution
from inference_perf.config.datagen.replay import SyntheticAgenticConfig
from inference_perf.utils.custom_tokenizer import CustomTokenizer


class _WordTok:
    """Simple word-counting tokenizer for tests (no HF dependency)."""

    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        return max(1, len(str(text).split()))

    def get_tokenizer(self) -> None:
        raise NotImplementedError


def _build_graph(turns: int = 2, tool_loop_depth: int = 2, fanout: float = 0.0, seed: int = 42) -> ReplayGraph:
    cfg = SyntheticAgenticConfig(
        num_sessions=1,
        input_tokens_per_turn=Distribution(type="fixed", mean=20),
        output_tokens_per_turn=Distribution(type="fixed", mean=10),
        turns_per_session=Distribution(type="fixed", mean=turns),
        tool_loop_depth=Distribution(type="fixed", mean=tool_loop_depth),
        tool_call_latency_sec=Distribution(type="fixed", mean=1),
        fanout_probability=fanout,
        seed=seed,
    )
    graph = build_graph_for_session(cfg, GENERIC_THEME, cast(CustomTokenizer, _WordTok()), session_index=0)
    tag_user_facing_events(graph)
    return graph


class TestSyntheticGraphUserFacingTagging:
    """Verify user-facing tagging on synthetic agentic graphs (no span info, fallback logic)."""

    def test_at_least_one_user_facing_event(self) -> None:
        """Every synthetic session must produce at least one user-facing event."""
        graph = _build_graph()
        uf = [eid for eid, ev in graph.events.items() if ev.is_user_facing]
        assert len(uf) >= 1

    def test_user_facing_is_not_a_tool_call(self) -> None:
        """User-facing events must not be tool-call events."""
        graph = _build_graph()
        for eid, ev in graph.events.items():
            if ev.is_user_facing:
                assert ev.call.expected_output_is_tool_call is False, f"{eid} is user-facing but also a tool call"

    def test_user_facing_is_not_tool_internal(self) -> None:
        """User-facing events must not be tool-internal."""
        graph = _build_graph()
        for eid, ev in graph.events.items():
            if ev.is_user_facing:
                assert ev.is_tool_internal is False, f"{eid} is user-facing but also tool-internal"

    def test_tool_call_events_are_not_user_facing(self) -> None:
        """Events with expected_output_is_tool_call=True must never be user-facing."""
        graph = _build_graph()
        for eid, ev in graph.events.items():
            if ev.call.expected_output_is_tool_call:
                assert ev.is_user_facing is False, f"{eid} is a tool call but tagged as user-facing"

    def test_tool_calls_are_majority(self) -> None:
        """In a session with tool loops, most events should be tool calls (not user-facing)."""
        graph = _build_graph(turns=2, tool_loop_depth=4)
        tool_calls = [ev for ev in graph.events.values() if ev.call.expected_output_is_tool_call]
        user_facing = [ev for ev in graph.events.values() if ev.is_user_facing]
        assert len(tool_calls) > len(user_facing)

    def test_fanout_graph_has_user_facing(self) -> None:
        """A graph with fan-out still produces user-facing events."""
        graph = _build_graph(turns=1, tool_loop_depth=1, fanout=1.0, seed=7)
        user_facing = [eid for eid, ev in graph.events.items() if ev.is_user_facing]
        assert len(user_facing) >= 1

    def test_fanout_graph_user_facing_count_bounded(self) -> None:
        """Fan-out produces user-facing events — but not more than total events."""
        graph = _build_graph(turns=1, tool_loop_depth=1, fanout=1.0, seed=7)
        user_facing = [eid for eid, ev in graph.events.items() if ev.is_user_facing]
        assert len(user_facing) <= len(graph.events)

    def test_multi_turn_has_user_facing_per_round(self) -> None:
        """Multi-turn: each round's terminal answer is user-facing."""
        graph = _build_graph(turns=3, tool_loop_depth=2, fanout=0.0)
        user_facing = [eid for eid, ev in graph.events.items() if ev.is_user_facing]
        # Each round produces one user-facing answer event (the terminal prose response).
        # With 3 turns, we expect at least 3 user-facing events.
        assert len(user_facing) >= 3

    def test_deep_tool_loop_single_user_facing(self) -> None:
        """A single-turn session with a deep tool loop has exactly one user-facing event."""
        graph = _build_graph(turns=1, tool_loop_depth=8, fanout=0.0)
        user_facing = [eid for eid, ev in graph.events.items() if ev.is_user_facing]
        assert len(user_facing) == 1

    def test_no_structured_output_in_synthetic(self) -> None:
        """Synthetic graphs don't use structured output — none should be excluded."""
        graph = _build_graph()
        structured = [ev for ev in graph.events.values() if ev.is_structured_output_call]
        assert len(structured) == 0

    def test_tool_internal_via_causal_fallback(self) -> None:
        """Without span info, tool-call events with causal successors are marked tool-internal."""
        graph = _build_graph(turns=1, tool_loop_depth=3)
        # Tool-call events that have a causal successor should be tool_internal
        # (fallback logic, since no spans). Prose events with causal successors are NOT.
        successors: dict[str, list[str]] = {eid: [] for eid in graph.events}
        for _eid, ev in graph.events.items():
            for pred_id, dep_type in ev.predecessor_dependency_types.items():
                if pred_id in successors:
                    successors[pred_id].append(dep_type)
        for eid, ev in graph.events.items():
            has_causal_succ = any(dt != "temporal" for dt in successors[eid])
            if has_causal_succ and ev.call.expected_output_is_tool_call:
                assert ev.is_tool_internal is True, f"{eid} is tool-call with causal successor but not tool_internal"

    def test_user_facing_and_tool_internal_are_disjoint(self) -> None:
        """No event can be both user-facing and tool-internal."""
        graph = _build_graph(turns=2, tool_loop_depth=3, fanout=0.0)
        for eid, ev in graph.events.items():
            if ev.is_user_facing:
                assert ev.is_tool_internal is False, f"{eid} is both user-facing and tool_internal"

    def test_deterministic_across_rebuilds(self) -> None:
        """Same seed produces identical user-facing tagging."""
        g1 = _build_graph(seed=99)
        g2 = _build_graph(seed=99)
        for eid in g1.events:
            assert g1.events[eid].is_user_facing == g2.events[eid].is_user_facing
            assert g1.events[eid].is_tool_internal == g2.events[eid].is_tool_internal
            assert g1.events[eid].is_structured_output_call == g2.events[eid].is_structured_output_call

    def test_multiple_seeds_all_have_user_facing(self) -> None:
        """Various seeds all produce at least one user-facing event."""
        for seed in range(10):
            graph = _build_graph(seed=seed)
            uf = [ev for ev in graph.events.values() if ev.is_user_facing]
            assert len(uf) >= 1, f"seed={seed} produced no user-facing events"
