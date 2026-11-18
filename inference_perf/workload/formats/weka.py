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
"""Weka agentic traces.

One trace per JSON file (a directory of them), or one per line of a
`.jsonl` file. A trace is one agent session: a list of requests, each with
a send time `t`, its input and output lengths, the hash ids of its full
prefix blocks and its `api_time`, plus subagent entries that make requests
of their own. `tool_tokens + system_tokens` of the leading blocks are the
system prompt.

A trace records each round's whole prompt, not the messages that were
added, so every round is its own record: the prompt rebuilt as role
messages the way the Weka generator rebuilds it (shared blocks up to the
longest common prefix with the previous round, then the reply to the
previous round as one or more assistant blocks, then the new user blocks
and the partial tail), with the recorded output length to ask for. The
record's reply carries the blocks the next round recorded for it, so the session runtime finds them in the next prompt and sends
the live reply in their place. A round whose reply the next round did not
record (it produced no output, or the next round only rewinds) takes the last
assistant message of the next prompt instead, as the Weka generator does.

A subagent's requests are packed into streams that never overlap, and each
stream is a conversation of its own inside the trace's session. Block ids
are session-scoped: two traces that hash to the same id share nothing.

Dependencies are the Weka generator's, computed on block ids instead of
text: a request depends on every earlier request of the session whose
recorded reply its prompt carries (minus those already reachable through
another), and on the latest request that finished before it was sent. Its
think time is the gap between its send time and the end of the latest of
those.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Annotated, Dict, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Union

from pydantic import BaseModel, Field, model_validator

from inference_perf.workload.format import Row, TraceFormat
from inference_perf.workload.record import (
    Delivery,
    Message,
    Part,
    Record,
    Schedule,
    Session,
    TextPart,
    TextSpec,
    ToolCallPart,
    ToolResultPart,
)

# A segment's identity for matching a recorded reply to the assistant
# message that carries it: equal ids and length is equal text.
_SegmentKey = Tuple[Tuple[int, ...], int]


# The rows of a trace file.


class WekaNormalRequest(BaseModel):
    t: float
    type: Literal["n"]
    model: str
    input_length: int = Field(alias="in")
    output_length: int = Field(alias="out")
    hash_ids: List[int] = Field(default_factory=list)
    api_time: Optional[float] = None
    think_time: Optional[float] = None
    # why the reply ended: "tool_use" when it was a tool call
    stop: Optional[str] = None
    # what the round added to the prompt: "tool_result" when it answers a tool call
    input_types: Optional[List[str]] = None


class WekaStreamingRequest(BaseModel):
    t: float
    type: Literal["s"]
    model: str
    input_length: int = Field(alias="in")
    output_length: int = Field(alias="out")
    hash_ids: List[int] = Field(default_factory=list)
    api_time: Optional[float] = None
    think_time: Optional[float] = None
    # why the reply ended: "tool_use" when it was a tool call
    stop: Optional[str] = None
    # what the round added to the prompt: "tool_result" when it answers a tool call
    input_types: Optional[List[str]] = None
    ttft: Optional[float] = None


class WekaSubagentEntry(BaseModel):
    t: float
    type: Literal["subagent"]
    agent_id: str
    subagent_type: str
    duration_ms: Optional[int] = None
    requests: List[WekaNormalRequest] = Field(default_factory=list)
    # the subagents this one spawned, which the file nests among its requests
    subagents: List["WekaSubagentEntry"] = Field(default_factory=list)
    # for a nested subagent, how many of the enclosing subagent's own calls the file lists before it
    calls_before: Optional[int] = None
    tool_tokens: int = 0
    system_tokens: int = 0

    @model_validator(mode="before")
    @classmethod
    def _split_nested(cls, data: object) -> object:
        # `requests` stays the subagent's own calls, so its readers see what they always did
        if not isinstance(data, dict) or not isinstance(data.get("requests"), list):
            return data
        own: List[object] = []
        nested: List[object] = []
        for r in data["requests"]:
            if isinstance(r, dict) and r.get("type") == "subagent":
                nested.append({**r, "calls_before": len(own)})
            else:
                own.append(r)
        if not nested:
            return data
        return {**data, "requests": own, "subagents": [*data.get("subagents", []), *nested]}


WekaRequest = Annotated[Union[WekaNormalRequest, WekaStreamingRequest, WekaSubagentEntry], Field(discriminator="type")]


class WekaTrace(BaseModel):
    id: str
    models: List[str]
    block_size: int
    tool_tokens: int = 0
    system_tokens: int = 0
    # "global" when a hash id names the same block in every trace of the file
    hash_id_scope: Optional[Literal["local", "global"]] = None
    requests: List[WekaRequest]


def longest_common_prefix(prev_hash_ids: List[int], curr_hash_ids: List[int]) -> int:
    n = min(len(prev_hash_ids), len(curr_hash_ids))
    for i in range(n):
        if prev_hash_ids[i] != curr_hash_ids[i]:
            return i
    return n


def subagent_request_absolute_t(entry: WekaSubagentEntry, req: WekaNormalRequest) -> float:
    return _absolute_t(entry.t, req.t)


def _absolute_t(spawn_t: float, t: float) -> float:
    """A time under a subagent on the trace's clock: one before the spawn is an offset from it."""
    if t + 1e-6 < spawn_t:
        return spawn_t + t
    return t


def pack_into_streams(requests: List[WekaNormalRequest]) -> List[List[WekaNormalRequest]]:
    sorted_reqs = sorted(requests, key=lambda r: r.t)
    streams: List[List[WekaNormalRequest]] = []
    stream_ends: List[float] = []
    for r in sorted_reqs:
        r_end = r.t + (r.api_time or 0.0)
        placed = False
        for i, end in enumerate(stream_ends):
            if end <= r.t:
                streams[i].append(r)
                stream_ends[i] = r_end
                placed = True
                break
        if not placed:
            streams.append([r])
            stream_ends.append(r_end)
    return streams


_Request = Union[WekaNormalRequest, WekaStreamingRequest]


@dataclass
class _Segment:
    role: Literal["system", "user", "assistant"]
    block_ids: List[int]
    # the part's length: the named blocks plus any tail past them
    num_tokens: int
    # what the message holds: plain text, a tool call the model made, or what a tool returned
    kind: Literal["text", "tool_call", "tool_result"] = "text"
    # a tool call's id: the round whose reply made it
    call_id: Optional[str] = None

    def key(self) -> _SegmentKey:
        return (tuple(self.block_ids), self.num_tokens)


def _added_kind(prev: Optional[_Request], curr: _Request) -> Literal["text", "tool_result"]:
    """What the user message a round adds holds. A round that says it added a
    tool result did. A round that says nothing about what it added, sent right
    after a reply that was a tool call, is taken to answer that call."""
    added = curr.input_types or []
    if "tool_result" in added:
        return "tool_result"
    if not added and prev is not None and prev.stop == "tool_use":
        return "tool_result"
    return "text"


class _Prompt:
    """The Weka generator's `ConversationReconstructor`, on block ids and
    lengths rather than tokens."""

    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self.segments: List[_Segment] = []

    def first_round(self, first: _Request, tool_tokens: int, system_tokens: int) -> None:
        hash_ids, in_tokens = first.hash_ids, first.input_length
        bs = self.block_size
        full_blocks = in_tokens // bs
        covered = min(full_blocks, len(hash_ids))
        tail = (full_blocks - covered) * bs + in_tokens - full_blocks * bs
        cursor = 0
        self.segments = []
        system_blocks = min(math.ceil((tool_tokens + system_tokens) / bs), len(hash_ids))
        if system_blocks > 0:
            self.segments.append(_Segment("system", hash_ids[:system_blocks], system_blocks * bs))
            cursor = system_blocks
        user_ids = hash_ids[cursor:covered]
        self.segments.append(_Segment("user", user_ids, len(user_ids) * bs + tail, _added_kind(None, first)))

    def next_round(self, prev: _Request, curr: _Request, prev_id: str) -> Optional[_Segment]:
        """Advance to `curr`'s prompt; returns what the Weka generator takes as
        `prev`'s recorded reply: the last assistant segment of that prompt."""
        bs = self.block_size
        lcp = longest_common_prefix(prev.hash_ids, curr.hash_ids)
        self._truncate(lcp)
        new_ids = curr.hash_ids[lcp:]
        tail = max(0, (curr.input_length // bs - len(curr.hash_ids)) * bs) + curr.input_length % bs
        reply_blocks = math.ceil(prev.output_length / bs) if prev.output_length > 0 else 0
        reply_blocks = min(reply_blocks, len(new_ids))
        if reply_blocks > 0:
            reply_kind: Literal["text", "tool_call"] = "tool_call" if prev.stop == "tool_use" else "text"
            made = prev_id if reply_kind == "tool_call" else None
            self.segments.append(_Segment("assistant", new_ids[:reply_blocks], reply_blocks * bs, reply_kind, made))
        user_ids = new_ids[reply_blocks:]
        if user_ids or tail > 0:
            self.segments.append(_Segment("user", user_ids, len(user_ids) * bs + tail, _added_kind(prev, curr)))
        # Usually the segment just placed. When none was placed (`prev` recorded
        # no output, or `curr` only rewinds), it is an earlier reply or what a
        # rewind left of one, and `prev` is given it all the same.
        last = next((seg for seg in reversed(self.segments) if seg.role == "assistant"), None)
        # a copy: a later rewind that cuts into the reply edits the prompt's segment in place
        return (
            _Segment(last.role, list(last.block_ids), last.num_tokens, last.kind, last.call_id) if last is not None else None
        )

    def _truncate(self, target_blocks: int) -> None:
        """Keep the first `target_blocks` blocks and nothing past them. A kept
        segment holds whole blocks only: a partial tail sat after the last
        block, and the next round counts those tokens again in its own blocks."""
        kept: List[_Segment] = []
        cursor = 0
        for seg in self.segments:
            take = min(len(seg.block_ids), target_blocks - cursor)
            if take <= 0:
                break
            seg.block_ids = seg.block_ids[:take]
            seg.num_tokens = min(seg.num_tokens, take * self.block_size)
            kept.append(seg)
            cursor += take
        self.segments = kept

    def snapshot(self) -> List[_Segment]:
        return [_Segment(s.role, list(s.block_ids), s.num_tokens, s.kind, s.call_id) for s in self.segments]


@dataclass
class _Call:
    # The Weka generator's call id: with the send time, it orders the
    # session the way that generator does, which the temporal edges follow.
    call_id: str
    t_start_ms: int
    t_end_ms: int
    request: _Request
    prompt: List[_Segment]
    reply: Optional[_Segment] = None
    depends_on: List[int] = field(default_factory=list)
    # the subagent that made the call, None for the session's own agent
    agent: Optional[str] = None
    # for a subagent's call, the request the file lists just before the subagent
    spawner: Optional[_Request] = None

    def made_call(self) -> Optional[str]:
        """The id of the tool call this call's reply made, when it made one."""
        if self.reply is not None:
            return self.reply.call_id
        # the last round of a conversation: nothing after it recorded the reply's blocks
        return self.call_id if self.request.stop == "tool_use" else None


class WekaFormat(TraceFormat):
    name = "Weka"
    # a row is one trace, and a trace is one agent session
    session_key = "id"
    # a round's messages add up to its recorded input length
    counts_special_tokens = False
    # The Weka generator's default, for traces that do not record one.
    default_block_size = 64
    file_suffixes = (".json", ".jsonl")

    def check(self, row: Row) -> None:
        self._trace(row)

    @staticmethod
    def _trace(row: Row) -> WekaTrace:
        try:
            return WekaTrace.model_validate(row)
        except ValueError as e:
            raise ValueError(f"not a Weka trace: {e}") from e

    def summary(self, key: str, rows: Sequence[Row]) -> Dict[str, object]:
        """The trace itself plus its request count and its largest and total
        request sizes, subagent requests included. Input and output both
        occupy the context window, so a size is their sum."""
        trace = rows[0]
        flat: List[Mapping[str, object]] = []
        # subagents nest to any depth
        pending = [trace.get("requests")]
        while pending:
            requests = pending.pop()
            for entry in requests if isinstance(requests, list) else []:
                if not isinstance(entry, dict):
                    continue
                if entry.get("type") == "subagent":
                    pending.append(entry.get("requests"))
                else:
                    flat.append(entry)
        sizes = [sum(v for v in (r.get("in"), r.get("out")) if isinstance(v, int)) for r in flat]
        return {
            **trace,
            "session_id": key,
            "max_tokens": max(sizes, default=0),
            "total_tokens": sum(sizes),
            "num_turns": len(flat),
        }

    def session(self, key: str, rows: Sequence[Row]) -> Session:
        if len(rows) != 1:
            raise ValueError(f"duplicate trace id {key!r}")
        trace = self._trace(rows[0])
        bs = trace.block_size or self.block_size
        calls: List[_Call] = []

        parent = [(req, req.t) for req in trace.requests if isinstance(req, (WekaNormalRequest, WekaStreamingRequest))]
        calls += self._stream(parent, "parent_turn_{k}", bs, trace.tool_tokens, trace.system_tokens)
        # (subagent, when it was spawned on the trace's clock, the request listed just before it),
        # nested ones after the one that spawned them
        spawned: List[Tuple[WekaSubagentEntry, float, Optional[_Request]]] = []
        above: Optional[_Request] = None
        for req in trace.requests:
            if isinstance(req, WekaSubagentEntry):
                spawned.append((req, req.t, above))
            else:
                above = req
        while spawned:
            entry, spawn_t, spawner = spawned.pop(0)
            for nested in entry.subagents:
                before = nested.calls_before or 0
                spawned.append((nested, _absolute_t(spawn_t, nested.t), entry.requests[before - 1] if before else None))
            for s, stream in enumerate(pack_into_streams(list(entry.requests))):
                timed: List[Tuple[_Request, float]] = [(req, _absolute_t(spawn_t, req.t)) for req in stream]
                call_ids = f"sa_{entry.agent_id}_s{s}_turn_{{k}}"
                for call in self._stream(timed, call_ids, bs, entry.tool_tokens, entry.system_tokens):
                    call.agent = entry.agent_id
                    call.spawner = spawner
                    calls.append(call)

        # calls sent in the same millisecond keep their round order: turn 7 before turn 10
        calls.sort(key=lambda c: (c.t_start_ms, *_stream_and_round(c.call_id)))
        _find_dependencies(calls)

        # a trace that says its ids are global shares block text with every other trace of the run
        scope: Literal["trace", "session"] = "trace" if trace.hash_id_scope == "global" else "session"
        # A subagent is started by a tool call. When the round listed above it
        # says its reply was one, the subagent's requests name that call.
        made = {id(call.request): call.made_call() for call in calls}
        records: List[Record] = []
        schedules: Dict[str, Schedule] = {}
        for call in calls:
            messages = [Message(role=seg.role, parts=[self._part(seg, bs, scope)]) for seg in call.prompt]
            deps = [calls[j] for j in call.depends_on]
            record_id = f"{key}/{call.call_id}"
            schedules[record_id] = Schedule(
                start_ms=call.t_start_ms,
                duration_ms=int(call.request.api_time * 1000.0) if call.request.api_time is not None else None,
                wait_ms=max(0, call.t_start_ms - max(d.t_end_ms for d in deps)) if deps else 0,
                after=[f"{key}/{d.call_id}" for d in deps],
            )
            records.append(
                Record(
                    id=record_id,
                    messages=messages,
                    model=call.request.model,
                    output_tokens=call.request.output_length,
                    reply=self._reply(call, bs, scope),
                    agent=call.agent or None,
                    call_id=made.get(id(call.spawner)) if call.spawner is not None else None,
                )
            )
        return Session(id=key, records=records, delivery=Delivery(schedules=schedules))

    @staticmethod
    def _part(seg: _Segment, block_size: int, scope: Literal["trace", "session"]) -> Part:
        spec = TextSpec(num_tokens=seg.num_tokens, block_ids=seg.block_ids, block_size=block_size, scope=scope)
        text = TextPart(spec=spec)
        if seg.kind == "tool_call":
            # no name is recorded, only how long the call was; the id is the round that made it
            return ToolCallPart(id=seg.call_id, arguments=text)
        if seg.kind == "tool_result":
            return ToolResultPart(content=[text])
        return text

    @classmethod
    def _reply(cls, call: _Call, block_size: int, scope: Literal["trace", "session"]) -> List[Part]:
        if call.reply is not None:
            return [cls._part(call.reply, block_size, scope)]
        # the last round of a conversation: nothing after it recorded the reply's blocks
        return [ToolCallPart(id=call.call_id)] if call.request.stop == "tool_use" else []

    @staticmethod
    def _stream(
        timed: Sequence[Tuple[_Request, float]],
        call_id: str,
        block_size: int,
        tool_tokens: int,
        system_tokens: int,
    ) -> List[_Call]:
        """One conversation's rounds, each with its prompt and the blocks the
        next round recorded for its reply."""
        prompt = _Prompt(block_size)
        calls: List[_Call] = []
        for k, (req, t) in enumerate(timed):
            if k == 0:
                prompt.first_round(req, tool_tokens, system_tokens)
            else:
                calls[-1].reply = prompt.next_round(timed[k - 1][0], req, calls[-1].call_id)
            t_start_ms = int(t * 1000.0)
            calls.append(
                _Call(
                    call_id=call_id.format(k=k),
                    t_start_ms=t_start_ms,
                    t_end_ms=t_start_ms + int((req.api_time or 0.0) * 1000.0),
                    request=req,
                    prompt=prompt.snapshot(),
                )
            )
        return calls


def _stream_and_round(call_id: str) -> Tuple[str, int]:
    """A call id as its stream's name and its round's number."""
    stream, _, k = call_id.rpartition("_")
    return stream, int(k)


def _find_dependencies(calls: List[_Call]) -> None:
    """The Weka generator's predecessor finder on segment keys: for each call,
    the earlier calls whose recorded reply is one of its assistant messages,
    latest first, skipping any already reachable through one found; then
    the latest earlier call that ended by the time this one was sent."""
    causal: List[List[int]] = [[] for _ in calls]
    replies: List[Optional[_SegmentKey]] = [c.reply.key() if c.reply is not None else None for c in calls]

    def ancestors(i: int) -> Set[int]:
        seen: Set[int] = set()
        stack = list(causal[i])
        while stack:
            j = stack.pop()
            if j not in seen:
                seen.add(j)
                stack.extend(causal[j])
        return seen

    for i, call in enumerate(calls):
        carried = {seg.key() for seg in call.prompt if seg.role == "assistant"}
        reachable: Set[int] = set()
        for j in range(i - 1, -1, -1):
            if j in reachable:
                continue
            reply = replies[j]
            if reply is not None and reply in carried:
                causal[i].append(j)
                reachable |= ancestors(j)
        call.depends_on = list(causal[i])
        for j in range(i - 1, -1, -1):
            if calls[j].t_end_ms <= call.t_start_ms:
                if j not in call.depends_on:
                    call.depends_on.append(j)
                break
