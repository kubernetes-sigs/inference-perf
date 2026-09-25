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

One JSON object per line, one LLM round each. The fields read here:

    session_id           the session the round belongs to
    round_index          the round's order within its session
    input_tokens_total   the whole prompt, = prefix_tokens + newly_append_tokens
    prefix_tokens        the leading part the engine served from cache
    output_tokens        generated tokens
    timing_events[]      ordered events with absolute ISO-8601 timestamps; the
                         first one is when the round's input arrived, the last
                         one is when the model finished

No text and no content hashes are recorded. A session becomes one record with a
user and an assistant turn per round. Each user turn is self-contained: its
synthetic part is the whole prompt of that round, with session-scoped block ids
minted so that the first `prefix_tokens` are the same blocks the previous round
sent and the rest are new. That reproduces the recorded prefix reuse to block
granularity, and when an agent compacts its context (prefix smaller than the
previous prompt) the later blocks are simply not reused. Sharing between
sessions cannot be reproduced: the corpus carries nothing to identify it.

Each round is one node, sent at its recorded offset, depending on the previous
round of its session, with the recorded gap between that round's end and this
round's start as think time. Subagent sessions are recorded as separate
sessions with no link to their parent and are replayed as such.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from inference_perf.workload.arrangement import Arrangement, Node
from inference_perf.workload.record import Record, SyntheticPart, Turn
from inference_perf.workload.sources.base import Workload, WorkloadSource

# The corpus records no block structure, so the size is a choice; 64 matches
# what the engines it was captured against use.
TRACELAB_BLOCK_SIZE = 64
_FIELDS = ("session_id", "round_index", "input_tokens_total", "prefix_tokens", "output_tokens", "timing_events")


def _parse_iso_ms(ts: str) -> int:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(round(dt.timestamp() * 1000))


class _Round:
    """One parsed line, kept until its session is complete."""

    def __init__(self, entry: Dict[str, Any]) -> None:
        self.index: int = entry["round_index"]
        self.input_tokens: int = entry["input_tokens_total"]
        self.prefix_tokens: int = entry["prefix_tokens"]
        self.output_tokens: int = entry["output_tokens"]
        stamps = [_parse_iso_ms(e["timestamp"]) for e in entry["timing_events"]]
        self.start_ms: int = min(stamps)
        self.end_ms: int = max(stamps)
        self.model: Optional[str] = entry.get("model")


class TraceLabSource(WorkloadSource):
    format = "TraceLab"
    default_block_size = TRACELAB_BLOCK_SIZE

    def load(self, path: Path) -> Workload:
        sessions: Dict[str, List[_Round]] = defaultdict(list)
        order: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                entry = self._parse_line(line, line_num, path)
                session_id = entry["session_id"]
                if session_id not in sessions:
                    order.append(session_id)
                sessions[session_id].append(_Round(entry))

        origin_ms = min(r.start_ms for rounds in sessions.values() for r in rounds) if sessions else 0
        records: List[Record] = []
        nodes: List[Node] = []
        for session_id in order:
            rounds = sorted(sessions[session_id], key=lambda r: r.index)
            records.append(self._record(session_id, rounds))
            nodes.extend(self._nodes(session_id, rounds, origin_ms))
        return Workload(source_id=path.name, records=records, arrangement=Arrangement(nodes=nodes))

    def _record(self, session_id: str, rounds: List[_Round]) -> Record:
        bs = self.block_size
        turns: List[Turn] = []
        blocks: List[int] = []  # the previous round's block ids, in prompt order
        next_id = 0
        for r in rounds:
            reused = min(r.prefix_tokens // bs, len(blocks))
            needed = math.ceil(r.input_tokens / bs)
            ids = blocks[:reused]
            while len(ids) < needed:
                ids.append(next_id)
                next_id += 1
            blocks = ids
            turns.append(
                Turn(
                    role="user",
                    parts=[SyntheticPart(num_tokens=r.input_tokens, block_ids=ids, block_size=bs, scope="session")],
                    self_contained=True,
                )
            )
            turns.append(Turn(role="assistant", output_tokens=r.output_tokens))
        models = sorted({r.model for r in rounds if r.model})
        metadata: Dict[str, Any] = {"rounds": len(rounds)}
        if models:
            metadata["models"] = models
        return Record(id=session_id, session_id=session_id, turns=turns, metadata=metadata)

    def _nodes(self, session_id: str, rounds: List[_Round], origin_ms: int) -> List[Node]:
        nodes: List[Node] = []
        for k, r in enumerate(rounds):
            node_id = f"{session_id}:{r.index}"
            depends_on = [nodes[-1].id] if nodes else []
            think_ms = max(0, r.start_ms - rounds[k - 1].end_ms) if k > 0 else 0
            nodes.append(
                Node(
                    id=node_id,
                    record_id=session_id,
                    turn=2 * k + 1,
                    send_at_ms=r.start_ms - origin_ms,
                    depends_on=depends_on,
                    think_ms=think_ms,
                )
            )
        return nodes

    def _parse_line(self, line: str, line_num: int, path: Path) -> Dict[str, Any]:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{line_num}: not JSON: {e}") from e
        if not isinstance(entry, dict):
            raise ValueError(f"{path}:{line_num}: expected an object, got {type(entry).__name__}")
        missing = [k for k in _FIELDS if k not in entry]
        if missing:
            raise ValueError(f"{path}:{line_num}: missing {', '.join(missing)}")
        if not isinstance(entry["session_id"], str) or not entry["session_id"]:
            raise ValueError(f"{path}:{line_num}: session_id must be a non-empty string")
        for k in ("round_index", "input_tokens_total", "prefix_tokens", "output_tokens"):
            if not isinstance(entry[k], int) or isinstance(entry[k], bool) or entry[k] < 0:
                raise ValueError(f"{path}:{line_num}: {k} must be a non-negative integer, got {entry[k]!r}")
        if entry["prefix_tokens"] > entry["input_tokens_total"]:
            raise ValueError(f"{path}:{line_num}: prefix_tokens exceeds input_tokens_total")
        events = entry["timing_events"]
        if not isinstance(events, list) or not events:
            raise ValueError(f"{path}:{line_num}: timing_events must be a non-empty list")
        for event in events:
            if not isinstance(event, dict) or not isinstance(event.get("timestamp"), str):
                raise ValueError(f"{path}:{line_num}: every timing event needs a timestamp string")
            try:
                _parse_iso_ms(event["timestamp"])
            except ValueError as err:
                raise ValueError(f"{path}:{line_num}: bad timestamp {event['timestamp']!r}") from err
        return entry
