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
"""TraceLab coding-agent traces (github.com/uw-syfi/TraceLab, CC BY 4.0).

One row per LLM round, read from JSONL, a JSON array, either of them gzipped,
or the Hugging Face tables. The fields read here:

    session_id           the session the round belongs to
    round_index          the round's order within its session
    input_tokens_total   the whole prompt, = prefix_tokens + newly_append_tokens
    prefix_tokens        the leading part the engine served from cache
    output_tokens        generated tokens
    model                the model the round ran on
    timing_events[]      typed events with ISO-8601 timestamps
    tools[]              the round's tool calls: latency and, when recorded,
                         the wall-clock window each one ran in
    round_pk             join key of the Hugging Face tool_calls and
                         timing_events tables

No text and no content hashes are recorded. A round records its whole prompt,
not the messages that were added, so every round is its own record of its
session: a user message whose text spec is that prompt, with the recorded
output length to ask for. Block ids are session-scoped and minted so
that the first `prefix_tokens` are the same blocks the previous round sent and
the rest are new. That reproduces the recorded prefix reuse to block
granularity, and when an agent compacts its context (prefix smaller than the
previous prompt) the later blocks are simply not reused. Sharing between
sessions cannot be reproduced: the corpus carries nothing to identify it.

Each round waits for the previous round of its session. A round
is submitted when its last input event (`user_message`, `tool_result`) arrives
and completes at its last model-output event (`reasoning`, `text`,
`tool_call`), and the think time before a round is the gap between the
previous round's completion and its own submission. `usage_report` rows are
written after the tool ran, so they bound neither. A round without usable
timestamps waits for as long as the previous round's tools ran, with
overlapping tool windows merged. Subagent sessions are recorded as separate
sessions with no link to their parent and are replayed as such.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field

from inference_perf.workload.format import Row, TraceFormat
from inference_perf.workload.record import (
    Delivery,
    Message,
    Record,
    Schedule,
    Session,
    TextPart,
    TextSpec,
)

# The corpus records no block structure, so the size is a choice; 64 matches
# what the engines it was captured against use.
TRACELAB_BLOCK_SIZE = 64

# TraceLab event types (see the dataset's README): the LLM request is submitted
# when the last input event arrives and finishes at the last model-output event.
INPUT_EVENT_TYPES = frozenset({"user_message", "tool_result"})
OUTPUT_EVENT_TYPES = frozenset({"reasoning", "text", "tool_call"})

# Sent for a round that names no model.
DEFAULT_MODEL = "tracelab-model"


class TraceLabToolCall(BaseModel):
    model_config = {"extra": "ignore"}

    tool_name: Optional[str] = None
    tool_wall_latency_ms: Optional[float] = None
    tool_internal_latency_ms: Optional[float] = None
    # Wall-clock window of the tool execution. Used to merge overlapping
    # intervals when parallel tools ran concurrently.
    emitted_at: Optional[str] = None
    result_at: Optional[str] = None


class TraceLabTimingEvent(BaseModel):
    model_config = {"extra": "ignore"}

    event_type: Optional[str] = None
    timestamp: Optional[str] = None


class TraceLabRound(BaseModel):
    """One TraceLab JSONL row (one LLM invocation). Extra fields ignored."""

    model_config = {"extra": "ignore", "populate_by_name": True}

    session_id: str
    round_id: Optional[str] = None
    round_pk: Optional[Union[int, str]] = None  # HF join key for tool_calls/timing_events
    round_index: int
    provider: Optional[str] = None
    model: Optional[str] = None
    input_tokens_total: int
    prefix_tokens: int = 0
    newly_append_tokens: int = 0
    output_tokens: int
    tools: List[TraceLabToolCall] = Field(default_factory=list)
    timing_events: List[TraceLabTimingEvent] = Field(default_factory=list)


def _parse_ts(ts: Optional[str]) -> Optional[float]:
    """Parse an ISO-8601 timestamp to epoch seconds, None when unparseable."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _round_time_bounds(rnd: TraceLabRound) -> Tuple[Optional[float], Optional[float]]:
    """Recorded (submission, completion) time of a round.

    The LLM request for a round is submitted when its last *input* event
    arrives: ``user_message`` on a fresh turn, or ``tool_result`` once the last
    of several parallel tools has returned. Falling back to ``min`` of the
    model-output events keeps rounds that recorded no input event usable.

    The round completes at its last *model-output* event. ``usage_report`` rows
    (Codex writes them after the tool call has finished) describe the response
    that was already produced, so including them would push ``prev_end`` past
    the tool execution and collapse the measured think time to zero.

    Returns ``(None, None)`` when the round carries no usable event.
    """
    events = [(e.event_type or "", _parse_ts(e.timestamp)) for e in rnd.timing_events or []]
    timed = [(event_type, stamp) for event_type, stamp in events if stamp is not None]
    inputs = [stamp for event_type, stamp in timed if event_type in INPUT_EVENT_TYPES]
    outputs = [stamp for event_type, stamp in timed if event_type in OUTPUT_EVENT_TYPES]
    if not inputs and not outputs:
        return (None, None)
    start = max(inputs) if inputs else min(outputs)
    end = max(outputs) if outputs else max(inputs)
    return (start, end)


def round_gap_ms(prev: TraceLabRound, curr: TraceLabRound) -> Optional[int]:
    """Recorded think time before ``curr``: its submission time minus the
    previous round's completion time, in milliseconds.

    Returns None when either round lacks usable ``timing_events``
    timestamps so the caller can fall back to tool latency.
    """
    _, prev_end = _round_time_bounds(prev)
    curr_start, _ = _round_time_bounds(curr)
    if prev_end is None or curr_start is None:
        return None
    return max(int((curr_start - prev_end) * 1000.0), 0)


def tool_latency_ms(rnd: TraceLabRound) -> int:
    """Wall time the round's tools occupied before the next round started.

    Parallel tools run concurrently, so overlapping ``[emitted_at,
    result_at]`` intervals are merged instead of summed: N concurrent 5s
    tools cost 5s, not 5×N. Tools that lack both timestamps fall back to
    their recorded latency, which is added on top of the merged total.
    """
    intervals: List[Tuple[float, float]] = []
    serial_ms = 0.0
    for t in rnd.tools or []:
        emitted = _parse_ts(t.emitted_at)
        result = _parse_ts(t.result_at)
        if emitted is not None and result is not None and result >= emitted:
            intervals.append((emitted, result))
            continue
        lat = t.tool_wall_latency_ms if t.tool_wall_latency_ms is not None else t.tool_internal_latency_ms
        if lat:
            serial_ms += max(float(lat), 0.0)

    merged_ms = 0.0
    if intervals:
        intervals.sort()
        cur_start, cur_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                merged_ms += (cur_end - cur_start) * 1000.0
                cur_start, cur_end = start, end
        merged_ms += (cur_end - cur_start) * 1000.0
    return max(int(merged_ms + serial_ms), 0)


def _in_order(rounds: List[TraceLabRound]) -> List[TraceLabRound]:
    """A session's rounds in the order they ran.

    Real traces carry occasional duplicate ``(session_id, round_index)``
    pairs and rows whose ``round_index`` disagrees with their timestamps.
    When every round has a submission timestamp, the timestamps decide
    (tie-breaking on round_index/round_id); otherwise round_index does.
    """
    stamped: List[Tuple[Optional[float], TraceLabRound]] = [(_round_time_bounds(r)[0], r) for r in rounds]

    def stamped_key(item: Tuple[Optional[float], TraceLabRound]) -> Tuple[float, int, str]:
        start, rnd = item
        assert start is not None
        return (start, rnd.round_index, rnd.round_id or "")

    if all(start is not None for start, _ in stamped):
        return [rnd for _, rnd in sorted(stamped, key=stamped_key)]
    return sorted(rounds, key=lambda r: (r.round_index, r.round_id or ""))


class TraceLabFormat(TraceFormat):
    name = "TraceLab"
    session_key = "session_id"
    # on Hugging Face a round's tool calls and timing events are tables of their own
    side_tables = {"tools": "tool_calls", "timing_events": "timing_events"}
    join_key = "round_pk"
    default_block_size = TRACELAB_BLOCK_SIZE

    def check(self, row: Row) -> None:
        TraceLabRound.model_validate(row)

    def summary(self, key: str, rows: Sequence[Row]) -> Dict[str, object]:
        rounds = [TraceLabRound.model_validate(row) for row in rows]
        per_round = [(r.input_tokens_total or 0) + (r.output_tokens or 0) for r in rounds]
        providers = sorted({r.provider for r in rounds if r.provider})
        return {
            "session_id": key,
            "provider": providers[0] if len(providers) == 1 else providers,
            "providers": providers,
            "models": sorted({r.model for r in rounds if r.model}),
            "max_tokens": max(per_round, default=0),
            "total_tokens": sum(per_round),
            "num_rounds": len(rounds),
            "num_turns": len(rounds),
        }

    def session(self, key: str, rows: Sequence[Row]) -> Session:
        rounds = _in_order([TraceLabRound.model_validate(row) for row in rows])
        records: List[Record] = []
        schedules: Dict[str, Schedule] = {}
        blocks: List[int] = []  # the previous round's block ids, in prompt order
        next_id = 0
        for index, rnd in enumerate(rounds):
            input_tokens = max(rnd.input_tokens_total, 0)
            # a prefix can be recorded larger than the prompt it leads, after a compaction
            ids = blocks[: min(max(rnd.prefix_tokens, 0), input_tokens) // self.block_size]
            while len(ids) < math.ceil(input_tokens / self.block_size):
                ids.append(next_id)
                next_id += 1
            blocks = ids
            # a round that recorded no input still has to send a message
            prompt = (
                TextPart(spec=TextSpec(num_tokens=input_tokens, block_ids=ids, block_size=self.block_size, scope="session"))
                if input_tokens
                else TextPart(text=" ")
            )
            start, end = _round_time_bounds(rnd)
            wait_ms = 0
            if index > 0:
                gap = round_gap_ms(rounds[index - 1], rnd)
                wait_ms = gap if gap is not None else tool_latency_ms(rounds[index - 1])
            record_id = f"{key}:round_{index}"
            schedules[record_id] = Schedule(
                start_ms=int(round(start * 1000)) if start is not None else None,
                duration_ms=max(int(round((end - start) * 1000)), 0) if start is not None and end is not None else None,
                wait_ms=wait_ms,
                after=[records[-1].id] if records else [],
            )
            records.append(
                Record(
                    id=record_id,
                    messages=[Message(role="user", parts=[prompt])],
                    model=rnd.model or DEFAULT_MODEL,
                    output_tokens=max(rnd.output_tokens, 0),
                )
            )
        return Session(id=key, records=records, delivery=Delivery(schedules=schedules))
