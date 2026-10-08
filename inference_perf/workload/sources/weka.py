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
and the partial tail), and an assistant turn that asks for the recorded
output. That assistant turn carries the blocks the next round recorded for
the reply, so the session runtime finds them in the next prompt and sends
the live reply in their place.

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

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Set, Tuple, Union

from inference_perf.datagen.replay.weka_trace_replay_datagen import (
    WekaNormalRequest,
    WekaStreamingRequest,
    WekaSubagentEntry,
    WekaTrace,
    _pack_into_streams,
    _subagent_request_absolute_t,
    longest_common_prefix,
)
from inference_perf.workload.arrangement import Arrangement, Node
from inference_perf.workload.record import Record, SyntheticPart, Turn
from inference_perf.workload.sources.base import Workload, WorkloadSource

# A segment's identity for matching a recorded reply to the assistant
# message that carries it: equal ids and length is equal text.
_SegmentKey = Tuple[Tuple[int, ...], int]


_Request = Union[WekaNormalRequest, WekaStreamingRequest]


@dataclass
class _Segment:
    role: Literal["system", "user", "assistant"]
    block_ids: List[int]
    # The part's length: the named blocks plus any tail past them. Fewer
    # tokens than the blocks cover when the previous round's partial tail
    # was stripped off a block segment.
    num_tokens: int

    def key(self) -> _SegmentKey:
        return (tuple(self.block_ids), self.num_tokens)


class _Prompt:
    """The Weka generator's `ConversationReconstructor`, on block ids and
    lengths rather than tokens."""

    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self.segments: List[_Segment] = []

    def first_round(self, hash_ids: List[int], in_tokens: int, tool_tokens: int, system_tokens: int) -> None:
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
        self.segments.append(_Segment("user", user_ids, len(user_ids) * bs + tail))

    def next_round(self, prev: _Request, curr: _Request) -> Optional[_Segment]:
        """Advance to `curr`'s prompt; returns the assistant segment it placed
        for `prev`'s reply, if it placed one."""
        bs = self.block_size
        lcp = longest_common_prefix(prev.hash_ids, curr.hash_ids)
        self._truncate(lcp, prev.input_length % bs)
        new_ids = curr.hash_ids[lcp:]
        tail = max(0, (curr.input_length // bs - len(curr.hash_ids)) * bs) + curr.input_length % bs
        reply_blocks = math.ceil(prev.output_length / bs) if prev.output_length > 0 else 0
        reply_blocks = min(reply_blocks, len(new_ids))
        placed: Optional[_Segment] = None
        if reply_blocks > 0:
            placed = _Segment("assistant", new_ids[:reply_blocks], reply_blocks * bs)
            self.segments.append(placed)
        user_ids = new_ids[reply_blocks:]
        if user_ids or tail > 0:
            self.segments.append(_Segment("user", user_ids, len(user_ids) * bs + tail))
        return placed

    def _truncate(self, target_blocks: int, prev_partial_tail: int) -> None:
        """Keep the first `target_blocks` blocks. Cut exactly at the end of a
        segment, that segment loses the previous round's partial tail."""
        if target_blocks <= 0:
            self.segments = []
            return
        cursor = 0
        for i, seg in enumerate(self.segments):
            count = len(seg.block_ids)
            if cursor + count < target_blocks:
                cursor += count
                continue
            if cursor + count == target_blocks:
                if prev_partial_tail > 0 and seg.num_tokens > 0:
                    seg.num_tokens -= min(prev_partial_tail, seg.num_tokens)
                    seg.block_ids = seg.block_ids[: math.ceil(seg.num_tokens / self.block_size)]
                del self.segments[i + 1 :]
                return
            if cursor == target_blocks:
                del self.segments[i:]
                return
            kept = target_blocks - cursor
            seg.num_tokens = min(seg.num_tokens, kept * self.block_size)
            seg.block_ids = seg.block_ids[:kept]
            del self.segments[i + 1 :]
            return

    def snapshot(self) -> List[_Segment]:
        return [_Segment(s.role, list(s.block_ids), s.num_tokens) for s in self.segments]


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


class WekaSource(WorkloadSource):
    format = "Weka"
    # The Weka generator's default, for traces that do not record one.
    default_block_size = 64

    def load(self, path: Path) -> Workload:
        traces = self._read(path)
        records: List[Record] = []
        nodes: List[Node] = []
        seen: Set[str] = set()
        for trace in traces:
            if trace.id in seen:
                raise ValueError(f"{path}: duplicate trace id {trace.id!r}")
            seen.add(trace.id)
            self._add_trace(trace, records, nodes)
        # A round's messages add up to its recorded input length.
        return Workload(
            source_id=path.name,
            records=records,
            arrangement=Arrangement(nodes=nodes),
            special_tokens_in_lengths=False,
        )

    def _read(self, path: Path) -> List[WekaTrace]:
        if path.is_dir():
            files = sorted(path.glob("*.json"))
            if not files:
                raise ValueError(f"No JSON trace files in {path}")
            return [self._parse(f.read_text(encoding="utf-8"), str(f)) for f in files]
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                return [self._parse(line, f"{path}:{n}") for n, line in enumerate(f, 1) if line.strip()]
        return [self._parse(path.read_text(encoding="utf-8"), str(path))]

    @staticmethod
    def _parse(text: str, where: str) -> WekaTrace:
        try:
            return WekaTrace.model_validate(json.loads(text))
        except ValueError as e:
            raise ValueError(f"{where}: not a Weka trace: {e}") from e

    def _add_trace(self, trace: WekaTrace, records: List[Record], nodes: List[Node]) -> None:
        bs = trace.block_size or self.block_size
        calls: List[_Call] = []

        parent = [(req, req.t) for req in trace.requests if isinstance(req, (WekaNormalRequest, WekaStreamingRequest))]
        calls += self._stream(parent, "parent_turn_{k}", bs, trace.tool_tokens, trace.system_tokens)
        for entry in (req for req in trace.requests if isinstance(req, WekaSubagentEntry)):
            for s, stream in enumerate(_pack_into_streams(list(entry.requests))):
                timed: List[Tuple[_Request, float]] = [(req, _subagent_request_absolute_t(entry, req)) for req in stream]
                call_ids = f"sa_{entry.agent_id}_s{s}_turn_{{k}}"
                calls += self._stream(timed, call_ids, bs, entry.tool_tokens, entry.system_tokens)

        calls.sort(key=lambda c: (c.t_start_ms, c.call_id))
        _find_dependencies(calls)

        for call in calls:
            node_id = f"{trace.id}/{call.call_id}"
            turns = [Turn(role=seg.role, parts=[self._part(seg, bs)]) for seg in call.prompt]
            reply_parts = [self._part(call.reply, bs)] if call.reply is not None else []
            turns.append(Turn(role="assistant", parts=reply_parts, output_tokens=call.request.output_length))
            records.append(Record(id=node_id, session_id=trace.id, turns=turns, metadata={"model": call.request.model}))
            deps = [calls[j] for j in call.depends_on]
            think_ms = max(0, call.t_start_ms - max(d.t_end_ms for d in deps)) if deps else 0
            nodes.append(
                Node(
                    id=node_id,
                    record_id=node_id,
                    turn=len(turns) - 1,
                    send_at_ms=call.t_start_ms,
                    depends_on=[f"{trace.id}/{d.call_id}" for d in deps],
                    think_ms=think_ms,
                )
            )

    @staticmethod
    def _part(seg: _Segment, block_size: int) -> SyntheticPart:
        return SyntheticPart(num_tokens=seg.num_tokens, block_ids=seg.block_ids, block_size=block_size, scope="session")

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
                prompt.first_round(req.hash_ids, req.input_length, tool_tokens, system_tokens)
            else:
                calls[-1].reply = prompt.next_round(timed[k - 1][0], req)
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
