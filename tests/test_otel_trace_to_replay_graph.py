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
Unit tests for build_graph() in otel_trace_to_replay_csv.py.

Three scenarios, each self-contained with inline span data:

  1. test_sequential_chain      — 3 turns of a conversation, each causally depends on the previous
  2. test_parallel_fan_out      — 1 root call fans out to 2 parallel sub-calls, then 1 final call
  3. test_growing_prefix        — 4 turns where the shared prefix grows with each turn

Each test prints a human-readable graph summary so you can see the full output when
running under the debugger or with `pytest -s`.

Key structural properties of the new graph format:
  - Each GraphEvent contains exactly ONE call (event.call, not event.calls)
  - Predecessor relationships are explicit: event.predecessor_event_ids (list of event_ids)
  - event.wait_ms = delay after last predecessor finishes before this event starts
  - InputSegments are at MESSAGE granularity (not character level):
      SHARED  = N leading messages identical to a predecessor's messages (KV cache hit)
      OUTPUT  = 1 assistant message whose content matches a predecessor's output
      UNIQUE  = remaining messages unique to this call
"""

import json
from typing import Any, Dict, List, Optional

from inference_perf.datagen.otel_trace_to_replay_graph import (
    build_graph,
    build_raw_calls,
    print_graph,
    summarize_graph,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_span(
    span_id: str,
    start_time: str,
    end_time: str,
    input_messages: List[Dict[str, Any]],
    output_text: str,
    model: str = "gpt-4",
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a minimal OTel-style span dict for testing."""
    attrs: Dict[str, Any] = {
        "gen_ai.request.model": model,
        "gen_ai.input.messages": json.dumps(input_messages),
        "gen_ai.output.text": output_text,
    }
    if prompt_tokens is not None:
        attrs["gen_ai.usage.prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        attrs["gen_ai.usage.completion_tokens"] = completion_tokens

    return {
        "trace_id": "trace_test",
        "span_id": span_id,
        "parent_span_id": None,
        "name": f"chat {model}",
        "start_time": start_time,
        "end_time": end_time,
        "attributes": attrs,
        "status": {"code": 1, "message": ""},
    }


# ---------------------------------------------------------------------------
# Test 1: Sequential chain (3-turn conversation)
# ---------------------------------------------------------------------------
#
# Timeline:
#   span_A  [0s → 2s]   user asks "What is the capital of France?"
#                        output: "The capital of France is Paris. It is a beautiful city."
#
#   span_B  [3s → 5s]   multi-turn: span_A's output appears as an assistant message
#                        → causal dep on A
#                        output: "Paris has many landmarks: the Eiffel Tower, the Louvre, Notre-Dame."
#
#   span_C  [6s → 8s]   multi-turn: span_B's output appears as an assistant message
#                        → causal dep on B
#                        output: "The Eiffel Tower is 330 meters tall."
#
# Expected graph:
#   event_000 (span_A) → [wait 1000ms] → event_001 (span_B) → [wait 1000ms] → event_002 (span_C)
#   root_event_ids = ["event_000"]
#
# Expected segments (message-level, no " assistant: " artifact):
#   span_A: [UNIQUE(1msg)]
#   span_B: [SHARED(1msg from event_000), OUTPUT(1msg from event_000), UNIQUE(1msg)]
#   span_C: [SHARED(3msg from event_001), OUTPUT(1msg from event_001), UNIQUE(1msg)]

OUTPUT_A = "The capital of France is Paris. It is a beautiful city full of history and culture."
OUTPUT_B = "Paris has many landmarks: the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral."
OUTPUT_C = "The Eiffel Tower is exactly 330 meters tall, including its broadcast antenna."

SPAN_A = make_span(
    span_id="span_A",
    start_time="2026-01-01T10:00:00.000000",
    end_time="2026-01-01T10:00:02.000000",
    input_messages=[
        {"role": "user", "content": "What is the capital of France?"},
    ],
    output_text=OUTPUT_A,
    prompt_tokens=10,
    completion_tokens=20,
)

SPAN_B = make_span(
    span_id="span_B",
    start_time="2026-01-01T10:00:03.000000",
    end_time="2026-01-01T10:00:05.000000",
    input_messages=[
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": OUTPUT_A},
        {"role": "user", "content": "Tell me about Paris landmarks."},
    ],
    output_text=OUTPUT_B,
    prompt_tokens=40,
    completion_tokens=22,
)

SPAN_C = make_span(
    span_id="span_C",
    start_time="2026-01-01T10:00:06.000000",
    end_time="2026-01-01T10:00:08.000000",
    input_messages=[
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": OUTPUT_A},
        {"role": "user", "content": "Tell me about Paris landmarks."},
        {"role": "assistant", "content": OUTPUT_B},
        {"role": "user", "content": "How tall is the Eiffel Tower?"},
    ],
    output_text=OUTPUT_C,
    prompt_tokens=75,
    completion_tokens=18,
)


def test_sequential_chain() -> None:
    """
    3-turn conversation: each turn causally depends on the previous.

    INPUT (spans):
        span_A  t=0s→2s    [user: "What is the capital of France?"]
        span_B  t=3s→5s    [user: Q, assistant: OUTPUT_A, user: "Tell me about Paris landmarks."]
        span_C  t=6s→8s    [user: Q, assistant: OUTPUT_A, user: Q2, assistant: OUTPUT_B, user: Q3]

    EXPECTED GRAPH (one event per call, linear chain):
        event_000 (span_A) --[wait 1000ms]--> event_001 (span_B) --[wait 1000ms]--> event_002 (span_C)

    EXPECTED SEGMENTS (message-level, no role-label artifacts):
        span_A: UNIQUE(1msg)
        span_B: SHARED(1msg ← event_000) + OUTPUT(1msg ← event_000) + UNIQUE(1msg)
        span_C: SHARED(3msg ← event_001) + OUTPUT(1msg ← event_001) + UNIQUE(1msg)
    """
    spans = [SPAN_A, SPAN_B, SPAN_C]
    calls = build_raw_calls(spans)
    graph = build_graph(calls)

    # Print for visual inspection when running with pytest -s or debugger
    print("\n" + "=" * 70)
    print("TEST: test_sequential_chain")
    print("=" * 70)
    print("\nINPUT calls:")
    for c in calls:
        print(f"  {c.call_id}  t={c.t_start_ms}ms→{c.t_end_ms}ms  messages={len(c.messages)}")
        print(f"    output: {(c.out_message.text if c.out_message else '')[:60]}...")
    print("\nOUTPUT graph:")
    print_graph(graph)
    print("\nSummary:")
    print(summarize_graph(graph))

    # --- Structural assertions ---
    assert len(graph.events) == 3, f"Expected 3 events, got {len(graph.events)}"
    assert graph.root_event_ids == ["event_000_span_A"], f"Expected root=event_000, got {graph.root_event_ids}"

    event_000 = graph.events["event_000_span_A"]
    event_001 = graph.events["event_001_span_B"]
    event_002 = graph.events["event_002_span_C"]

    # Each event has exactly 1 call (new structure: event.call, not event.calls)
    assert event_000.call.call_id == "span_A"
    assert event_001.call.call_id == "span_B"
    assert event_002.call.call_id == "span_C"

    # Predecessor relationships (new structure: predecessor_event_ids)
    assert event_000.predecessor_event_ids == []
    assert event_001.predecessor_event_ids == ["event_000_span_A"]
    assert event_002.predecessor_event_ids == ["event_001_span_B"]

    # Wait times
    assert event_000.wait_ms == 0
    assert event_001.wait_ms == 1000  # 3000ms start - 2000ms end
    assert event_002.wait_ms == 1000  # 6000ms start - 5000ms end

    # Token counts (from recorded values)
    assert event_000.call.total_input_tokens == 10
    assert event_000.call.expected_output_tokens == 20
    assert event_001.call.total_input_tokens == 40
    assert event_001.call.expected_output_tokens == 22
    assert event_002.call.total_input_tokens == 75
    assert event_002.call.expected_output_tokens == 18

    # Segment types for span_A: should be all unique (no predecessors)
    segs_A = event_000.call.input_segments
    assert all(s.type == "unique" for s in segs_A), f"span_A should be all unique, got {segs_A}"

    # Segment types for span_B: SHARED(1) + OUTPUT(1) + UNIQUE(1)
    segs_B = event_001.call.input_segments
    seg_types_B = [s.type for s in segs_B]
    print(f"\n  span_B segments: {[(s.type, s.message_count, s.token_count, s.source_event_id) for s in segs_B]}")
    assert seg_types_B[0] == "shared", f"span_B first segment should be shared, got {seg_types_B}"
    assert segs_B[0].source_event_id == "event_000_span_A"
    assert segs_B[0].message_count == 1  # only "user: What is the capital of France?"
    assert "output" in seg_types_B, f"span_B should have an output segment, got {seg_types_B}"
    # No UNIQUE(" assistant: ") artifact — segments are whole messages
    output_seg_B = next(s for s in segs_B if s.type == "output")
    assert output_seg_B.message_count == 1  # exactly 1 assistant message
    assert output_seg_B.source_event_id == "event_000_span_A"

    # Segment types for span_C: SHARED(3) + OUTPUT(1) + UNIQUE(1)
    segs_C = event_002.call.input_segments
    seg_types_C = [s.type for s in segs_C]
    print(f"  span_C segments: {[(s.type, s.message_count, s.token_count, s.source_event_id) for s in segs_C]}")
    assert seg_types_C[0] == "shared", f"span_C first segment should be shared, got {seg_types_C}"
    assert segs_C[0].source_event_id == "event_001_span_B"
    assert segs_C[0].message_count == 3  # user Q + assistant OUTPUT_A + user Q2
    assert "output" in seg_types_C, f"span_C should have an output segment, got {seg_types_C}"
    output_seg_C = next(s for s in segs_C if s.type == "output")
    assert output_seg_C.message_count == 1
    assert output_seg_C.source_event_id == "event_001_span_B"

    # Total message counts must add up
    total_msgs_B = sum(s.message_count for s in segs_B)
    assert total_msgs_B == len(SPAN_B["attributes"]["gen_ai.input.messages"]) or total_msgs_B == 3
    total_msgs_C = sum(s.message_count for s in segs_C)
    assert total_msgs_C == 5


# ---------------------------------------------------------------------------
# Test 2: Parallel fan-out
# ---------------------------------------------------------------------------
#
# Timeline:
#   span_ROOT  [0s → 2s]   "Analyze this document: <doc>"
#                           output: "I will analyze it from two angles: technical and business."
#
#   span_P1    [3s → 5s]   "Technical analysis of <doc>"
#                           Does NOT contain ROOT's output as an assistant message
#                           → no causal dep on ROOT via output injection
#                           → timing fallback: predecessor = event_000
#
#   span_P2    [3.2s → 5.2s] "Business analysis of <doc>"
#                           Does NOT contain ROOT's output as an assistant message
#                           → timing fallback: predecessor = event_001 (immediately preceding)
#
#   span_FINAL [6s → 8s]   Contains P1's output AND P2's output as assistant messages
#                           → causal dep on both P1 and P2
#
# Expected graph (one event per call):
#   event_000 (ROOT) → event_001 (P1) → event_003 (FINAL)
#                    ↘ event_002 (P2) ↗
#   event_003.predecessor_event_ids = ["event_001", "event_002"]

SYSTEM_PROMPT = "You are a helpful document analysis assistant."
DOCUMENT = "The quarterly revenue increased by 15% driven by cloud services growth."

OUTPUT_ROOT = "I will analyze this document from two angles: technical infrastructure and business impact."
OUTPUT_P1 = (
    "Technical analysis: The cloud services growth indicates strong infrastructure scaling capabilities and DevOps maturity."
)
OUTPUT_P2 = (
    "Business analysis: A 15% revenue increase is significant, suggesting strong market position and customer retention."
)
OUTPUT_FINAL = (
    "Combined analysis complete. Both technical and business perspectives confirm strong organizational performance."
)

SPAN_ROOT = make_span(
    span_id="span_ROOT",
    start_time="2026-01-01T10:00:00.000000",
    end_time="2026-01-01T10:00:02.000000",
    input_messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze this document: {DOCUMENT}"},
    ],
    output_text=OUTPUT_ROOT,
    prompt_tokens=30,
    completion_tokens=25,
)

SPAN_P1 = make_span(
    span_id="span_P1",
    start_time="2026-01-01T10:00:03.000000",
    end_time="2026-01-01T10:00:05.000000",
    input_messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Provide a technical analysis of: {DOCUMENT}"},
    ],
    output_text=OUTPUT_P1,
    prompt_tokens=28,
    completion_tokens=30,
)

SPAN_P2 = make_span(
    span_id="span_P2",
    start_time="2026-01-01T10:00:03.200000",  # 200ms after P1
    end_time="2026-01-01T10:00:05.200000",
    input_messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Provide a business analysis of: {DOCUMENT}"},
    ],
    output_text=OUTPUT_P2,
    prompt_tokens=28,
    completion_tokens=32,
)

SPAN_FINAL = make_span(
    span_id="span_FINAL",
    start_time="2026-01-01T10:00:06.000000",
    end_time="2026-01-01T10:00:08.000000",
    input_messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Combine these analyses into a unified summary."},
        {"role": "assistant", "content": OUTPUT_P1},  # P1's output injected
        {"role": "assistant", "content": OUTPUT_P2},  # P2's output injected
    ],
    output_text=OUTPUT_FINAL,
    prompt_tokens=90,
    completion_tokens=20,
)


def test_parallel_fan_out() -> None:
    """
    Fan-out pattern: root → 2 parallel calls → final aggregation.

    INPUT (spans):
        span_ROOT  t=0s→2s     [sys, user: "Analyze document"]
        span_P1    t=3s→5s     [sys, user: "Technical analysis"]  (no causal dep on ROOT)
        span_P2    t=3.2s→5.2s [sys, user: "Business analysis"]   (no causal dep on ROOT)
        span_FINAL t=6s→8s     [sys, user, assistant:P1_out, assistant:P2_out]

    EXPECTED GRAPH (one event per call):
        event_000 (ROOT)
        event_001 (P1)    predecessor_event_ids=["event_000"]  (timing fallback)
        event_002 (P2)    predecessor_event_ids=["event_000"]  (timing fallback)
        event_003 (FINAL) predecessor_event_ids=["event_001", "event_002"]  (causal: P1+P2 outputs)

    EXPECTED SEGMENTS for span_FINAL:
        SHARED(1msg ← event_001 or event_002)  [sys]
        UNIQUE(1msg)                            ["Combine these analyses into a unified summary."]
        OUTPUT(1msg ← event_001)               [assistant: OUTPUT_P1]
        OUTPUT(1msg ← event_002)               [assistant: OUTPUT_P2]
    """
    spans = [SPAN_ROOT, SPAN_P1, SPAN_P2, SPAN_FINAL]
    calls = build_raw_calls(spans)
    graph = build_graph(calls)

    print("\n" + "=" * 70)
    print("TEST: test_parallel_fan_out")
    print("=" * 70)
    print("\nINPUT calls:")
    for c in calls:
        print(f"  {c.call_id}  t={c.t_start_ms}ms→{c.t_end_ms}ms  messages={len(c.messages)}")
        print(f"    output: {(c.out_message.text if c.out_message else '')[:60]}...")
    print("\nOUTPUT graph:")
    print_graph(graph)
    print("\nSummary:")
    print(summarize_graph(graph))

    # --- Structural assertions ---
    assert len(graph.events) == 4, f"Expected 4 events, got {len(graph.events)}: {list(graph.events.keys())}"
    assert graph.root_event_ids == ["event_000_span_ROOT"]

    event_000 = graph.events["event_000_span_ROOT"]
    event_001 = graph.events["event_001_span_P1"]
    event_002 = graph.events["event_002_span_P2"]
    event_003 = graph.events["event_003_span_FINAL"]

    # Each event has exactly 1 call
    assert event_000.call.call_id == "span_ROOT"
    assert event_001.call.call_id == "span_P1"
    assert event_002.call.call_id == "span_P2"
    assert event_003.call.call_id == "span_FINAL"

    # FINAL depends on both P1 and P2 (causal: their outputs appear in FINAL's messages)
    assert set(event_003.predecessor_event_ids) == {"event_001_span_P1", "event_002_span_P2"}, (
        f"FINAL should depend on P1 and P2, got {event_003.predecessor_event_ids}"
    )

    # ROOT has no predecessors
    assert event_000.predecessor_event_ids == []

    assert event_001.predecessor_event_ids == event_002.predecessor_event_ids, (
        "event_001 and event_002 are run in parallel and have the same predecessor"
    )

    # span_FINAL should have OUTPUT segments from P1 and P2
    segs_final = event_003.call.input_segments
    seg_types_final = [s.type for s in segs_final]
    output_sources = {s.source_event_id for s in segs_final if s.type == "output"}
    print(f"\n  span_FINAL segments: {[(s.type, s.message_count, s.source_event_id) for s in segs_final]}")
    assert "output" in seg_types_final, f"span_FINAL should have output segments, got {seg_types_final}"
    assert "event_001_span_P1" in output_sources, f"span_FINAL should reference P1 output, got {output_sources}"
    assert "event_002_span_P2" in output_sources, f"span_FINAL should reference P2 output, got {output_sources}"

    # Total message count for FINAL must be 4
    total_msgs_final = sum(s.message_count for s in segs_final)
    assert total_msgs_final == 4, f"FINAL has 4 messages total, got {total_msgs_final}"


# ---------------------------------------------------------------------------
# Test 3: Growing prefix (system prompt + growing conversation)
# ---------------------------------------------------------------------------
#
# All 4 calls share the same long system prompt at the start.
# Each subsequent call adds the previous assistant response to the context.
# The shared prefix (in messages) grows with each turn.
#
# Timeline:
#   span_T1  [0s → 2s]    [sys, user Q1]
#   span_T2  [3s → 5s]    [sys, user Q1, assistant A1, user Q2]
#   span_T3  [6s → 8s]    [sys, user Q1, assistant A1, user Q2, assistant A2, user Q3]
#   span_T4  [9s → 11s]   [sys, Q1, A1, Q2, A2, Q3, assistant A3, user Q4]
#
# Expected: shared prefix message count grows T2(1) < T3(3) < T4(5)
#   span_T1: UNIQUE(2msg)
#   span_T2: SHARED(2msg from T1) + OUTPUT(1msg from T1) + UNIQUE(1msg)
#   span_T3: SHARED(4msg from T2) + OUTPUT(1msg from T2) + UNIQUE(1msg)
#   span_T4: SHARED(6msg from T3) + OUTPUT(1msg from T3) + UNIQUE(1msg)

LONG_SYSTEM = (
    "You are an expert Python programming tutor. "
    "You explain concepts clearly with examples. "
    "Always provide runnable code snippets. "
    "Be concise but thorough."
)

OUT_T1 = "A list comprehension is a concise way to create lists: `[x*2 for x in range(10)]` creates [0,2,4,...,18]."
OUT_T2 = "Generator expressions are like list comprehensions but lazy: `(x*2 for x in range(10))` — use when you don't need all values at once."
OUT_T3 = "The key difference: list comprehensions create the full list in memory immediately; generators produce values one at a time, saving memory for large datasets."

SPAN_T1 = make_span(
    span_id="span_T1",
    start_time="2026-01-01T10:00:00.000000",
    end_time="2026-01-01T10:00:02.000000",
    input_messages=[
        {"role": "system", "content": LONG_SYSTEM},
        {"role": "user", "content": "What is a list comprehension in Python?"},
    ],
    output_text=OUT_T1,
    prompt_tokens=50,
    completion_tokens=28,
)

SPAN_T2 = make_span(
    span_id="span_T2",
    start_time="2026-01-01T10:00:03.000000",
    end_time="2026-01-01T10:00:05.000000",
    input_messages=[
        {"role": "system", "content": LONG_SYSTEM},
        {"role": "user", "content": "What is a list comprehension in Python?"},
        {"role": "assistant", "content": OUT_T1},
        {"role": "user", "content": "What about generator expressions?"},
    ],
    output_text=OUT_T2,
    prompt_tokens=95,
    completion_tokens=35,
)

SPAN_T3 = make_span(
    span_id="span_T3",
    start_time="2026-01-01T10:00:06.000000",
    end_time="2026-01-01T10:00:08.000000",
    input_messages=[
        {"role": "system", "content": LONG_SYSTEM},
        {"role": "user", "content": "What is a list comprehension in Python?"},
        {"role": "assistant", "content": OUT_T1},
        {"role": "user", "content": "What about generator expressions?"},
        {"role": "assistant", "content": OUT_T2},
        {"role": "user", "content": "What is the key difference between them?"},
    ],
    output_text=OUT_T3,
    prompt_tokens=148,
    completion_tokens=40,
)

SPAN_T4 = make_span(
    span_id="span_T4",
    start_time="2026-01-01T10:00:09.000000",
    end_time="2026-01-01T10:00:11.000000",
    input_messages=[
        {"role": "system", "content": LONG_SYSTEM},
        {"role": "user", "content": "What is a list comprehension in Python?"},
        {"role": "assistant", "content": OUT_T1},
        {"role": "user", "content": "What about generator expressions?"},
        {"role": "assistant", "content": OUT_T2},
        {"role": "user", "content": "What is the key difference between them?"},
        {"role": "assistant", "content": OUT_T3},
        {"role": "user", "content": "When should I use one over the other?"},
    ],
    output_text="Use list comprehensions when you need random access or multiple iterations. Use generators for large datasets or streaming pipelines.",
    prompt_tokens=210,
    completion_tokens=25,
)


def test_growing_prefix() -> None:
    """
    4-turn conversation where the shared prefix grows with each turn.

    INPUT (spans):
        span_T1  t=0s→2s    [sys, user Q1]                                    (2 messages)
        span_T2  t=3s→5s    [sys, user Q1, asst A1, user Q2]                  (4 messages)
        span_T3  t=6s→8s    [sys, Q1, A1, Q2, asst A2, user Q3]               (6 messages)
        span_T4  t=9s→11s   [sys, Q1, A1, Q2, A2, Q3, asst A3, user Q4]       (8 messages)

    EXPECTED GRAPH:
        event_000 → event_001 → event_002 → event_003  (linear chain)

    EXPECTED SEGMENTS (message-level, no role-label artifacts):
        span_T1: UNIQUE(2msg)
        span_T2: SHARED(2msg ← event_000) + OUTPUT(1msg ← event_000) + UNIQUE(1msg)
        span_T3: SHARED(4msg ← event_001) + OUTPUT(1msg ← event_001) + UNIQUE(1msg)
        span_T4: SHARED(6msg ← event_002) + OUTPUT(1msg ← event_002) + UNIQUE(1msg)

    KEY PROPERTY: shared prefix message count grows monotonically: 2 < 4 < 6
    """
    spans = [SPAN_T1, SPAN_T2, SPAN_T3, SPAN_T4]
    calls = build_raw_calls(spans)
    graph = build_graph(calls)

    print("\n" + "=" * 70)
    print("TEST: test_growing_prefix")
    print("=" * 70)
    print("\nINPUT calls:")
    for c in calls:
        print(f"  {c.call_id}  t={c.t_start_ms}ms→{c.t_end_ms}ms  messages={len(c.messages)}  prompt_tokens={c.prompt_tokens}")
    print("\nOUTPUT graph:")
    print_graph(graph)
    print("\nSummary:")
    print(summarize_graph(graph))

    # --- Structural assertions ---
    assert len(graph.events) == 4, f"Expected 4 events, got {len(graph.events)}"
    assert graph.root_event_ids == ["event_000_span_T1"]

    # Linear chain
    assert graph.events["event_000_span_T1"].predecessor_event_ids == []
    assert graph.events["event_001_span_T2"].predecessor_event_ids == ["event_000_span_T1"]
    assert graph.events["event_002_span_T3"].predecessor_event_ids == ["event_001_span_T2"]
    assert graph.events["event_003_span_T4"].predecessor_event_ids == ["event_002_span_T3"]

    # Wait times: each gap is 1000ms (3s-2s, 6s-5s, 9s-8s)
    assert graph.events["event_000_span_T1"].wait_ms == 0
    assert graph.events["event_001_span_T2"].wait_ms == 1000
    assert graph.events["event_002_span_T3"].wait_ms == 1000
    assert graph.events["event_003_span_T4"].wait_ms == 1000

    # span_T1: all unique (no predecessors)
    segs_T1 = graph.events["event_000_span_T1"].call.input_segments
    assert all(s.type == "unique" for s in segs_T1)
    assert sum(s.message_count for s in segs_T1) == 2

    # span_T2: SHARED(2) + OUTPUT(1) + UNIQUE(1)
    segs_T2 = graph.events["event_001_span_T2"].call.input_segments
    assert segs_T2[0].type == "shared"
    assert segs_T2[0].source_event_id == "event_000_span_T1"
    shared_msgs_T2 = segs_T2[0].message_count
    print(f"\n  span_T2 shared prefix: {shared_msgs_T2} messages (source: {segs_T2[0].source_event_id})")
    print(f"  span_T2 segments: {[(s.type, s.message_count) for s in segs_T2]}")
    assert shared_msgs_T2 == 2, f"T2 should share 2 messages with T1, got {shared_msgs_T2}"
    assert sum(s.message_count for s in segs_T2) == 4

    # span_T3: SHARED(4) + OUTPUT(1) + UNIQUE(1)
    segs_T3 = graph.events["event_002_span_T3"].call.input_segments
    assert segs_T3[0].type == "shared"
    assert segs_T3[0].source_event_id == "event_001_span_T2"
    shared_msgs_T3 = segs_T3[0].message_count
    print(f"  span_T3 shared prefix: {shared_msgs_T3} messages (source: {segs_T3[0].source_event_id})")
    print(f"  span_T3 segments: {[(s.type, s.message_count) for s in segs_T3]}")
    assert shared_msgs_T3 == 4, f"T3 should share 4 messages with T2, got {shared_msgs_T3}"
    assert sum(s.message_count for s in segs_T3) == 6

    # span_T4: SHARED(6) + OUTPUT(1) + UNIQUE(1)
    segs_T4 = graph.events["event_003_span_T4"].call.input_segments
    assert segs_T4[0].type == "shared"
    assert segs_T4[0].source_event_id == "event_002_span_T3"
    shared_msgs_T4 = segs_T4[0].message_count
    print(f"  span_T4 shared prefix: {shared_msgs_T4} messages (source: {segs_T4[0].source_event_id})")
    print(f"  span_T4 segments: {[(s.type, s.message_count) for s in segs_T4]}")
    assert shared_msgs_T4 == 6, f"T4 should share 6 messages with T3, got {shared_msgs_T4}"
    assert sum(s.message_count for s in segs_T4) == 8

    # Shared prefix grows monotonically
    assert shared_msgs_T2 < shared_msgs_T3 < shared_msgs_T4, (
        f"Shared prefix should grow: {shared_msgs_T2} < {shared_msgs_T3} < {shared_msgs_T4}"
    )

    # All turns T2-T4 have an output segment
    for event_id, call_id in [
        ("event_001_span_T2", "span_T2"),
        ("event_002_span_T3", "span_T3"),
        ("event_003_span_T4", "span_T4"),
    ]:
        segs = graph.events[event_id].call.input_segments
        seg_types = [s.type for s in segs]
        assert "output" in seg_types, f"{call_id} should have an output segment, got {seg_types}"
