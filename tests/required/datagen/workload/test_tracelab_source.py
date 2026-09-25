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

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pytest

from inference_perf.workload import SyntheticPart
from inference_perf.workload.sources.tracelab import TraceLabSource


def rnd(
    session: str,
    index: int,
    total: int,
    prefix: int,
    out: int,
    start: str,
    end: str,
    model: str = "claude-opus-4-8",
) -> Dict[str, Any]:
    return {
        "session_id": session,
        "round_index": index,
        "model": model,
        "input_tokens_total": total,
        "prefix_tokens": prefix,
        "newly_append_tokens": total - prefix,
        "output_tokens": out,
        "timing_events": [
            {"event_type": "user_message", "timestamp": start},
            {"event_type": "text", "timestamp": end},
        ],
        "tools": [],
    }


# Session A grows: round 1 reuses 192 of round 0's 200 tokens. Session B
# compacts: round 1 keeps only 64 of round 0's 300 tokens. B's rounds are
# written out of order to check sorting by round_index.
LINES = [
    rnd("claude:A", 0, 200, 0, 30, "2026-06-01T10:00:00.000Z", "2026-06-01T10:00:02.000Z"),
    rnd("claude:B", 1, 130, 64, 12, "2026-06-01T10:01:00.000Z", "2026-06-01T10:01:01.000Z"),
    rnd("claude:A", 1, 260, 192, 40, "2026-06-01T10:00:05.000Z", "2026-06-01T10:00:08.000Z"),
    rnd("claude:B", 0, 300, 0, 20, "2026-06-01T10:00:30.000Z", "2026-06-01T10:00:31.500Z"),
]


def write_trace(tmp_path: Path, lines: Sequence[object]) -> Path:
    path = tmp_path / "syfi_coding_trace.jsonl"
    path.write_text("\n".join(json.dumps(line) if not isinstance(line, str) else line for line in lines) + "\n")
    return path


def parts(record_turns: List[Any]) -> List[SyntheticPart]:
    out = []
    for turn in record_turns:
        if turn.role == "user":
            part = turn.parts[0]
            assert isinstance(part, SyntheticPart)
            out.append(part)
    return out


# Each session becomes one record with a user and an assistant turn per
# round, in round_index order regardless of file order. User turns are
# self-contained and session-scoped; assistant turns carry output_tokens.
def test_sessions_become_records_with_a_turn_pair_per_round(tmp_path: Path) -> None:
    workload = TraceLabSource(block_size=64).load(write_trace(tmp_path, LINES))
    assert [r.id for r in workload.records] == ["claude:A", "claude:B"]
    a = workload.records[0]
    assert a.session_id == "claude:A"
    assert [t.role for t in a.turns] == ["user", "assistant", "user", "assistant"]
    assert all(t.self_contained for t in a.turns if t.role == "user")
    assert [t.output_tokens for t in a.turns if t.role == "assistant"] == [30, 40]
    assert [p.num_tokens for p in parts(a.turns)] == [200, 260]
    assert all(p.scope == "session" and p.block_size == 64 for p in parts(a.turns))
    assert a.metadata == {"rounds": 2, "models": ["claude-opus-4-8"]}
    b = workload.records[1]
    assert [p.num_tokens for p in parts(b.turns)] == [300, 130]


# Block ids are minted per session. Session A round 0 needs 4 blocks for 200
# tokens ([0, 1, 2, 3]); round 1 reuses the 192 // 64 = 3 leading blocks and
# mints 2 more for its 260 tokens ([0, 1, 2, 4, 5]). Block 3, round 0's
# partial tail, is never reused because the prefix stopped short of it.
def test_growing_session_reuses_leading_blocks(tmp_path: Path) -> None:
    workload = TraceLabSource(block_size=64).load(write_trace(tmp_path, LINES))
    r0, r1 = parts(workload.records[0].turns)
    assert r0.block_ids == [0, 1, 2, 3]
    assert r1.block_ids == [0, 1, 2, 4, 5]
    assert r1.prefix_tokens == 260


# Session B compacts: round 0 sent 5 blocks ([0..4]); round 1's prefix of 64
# tokens keeps only block 0, and its 130 tokens need 3 blocks, so it gets
# [0, 5, 6]. Blocks 1 to 4 are not reused.
def test_compacted_session_drops_later_blocks(tmp_path: Path) -> None:
    workload = TraceLabSource(block_size=64).load(write_trace(tmp_path, LINES))
    r0, r1 = parts(workload.records[1].turns)
    assert r0.block_ids == [0, 1, 2, 3, 4]
    assert r1.block_ids == [0, 5, 6]


# Two sessions mint the same ids ([0, 1, 2, 3] and [0, 1, 2, 3, 4] both start
# at 0), which is fine because the scope is the session: the materializer
# keys block text by session, so the ids do not collide across sessions.
def test_block_ids_are_per_session(tmp_path: Path) -> None:
    workload = TraceLabSource(block_size=64).load(write_trace(tmp_path, LINES))
    a0 = parts(workload.records[0].turns)[0]
    b0 = parts(workload.records[1].turns)[0]
    assert a0.block_ids[:4] == b0.block_ids[:4] == [0, 1, 2, 3]
    assert a0.scope == b0.scope == "session"


# One node per round, sent at its offset from the earliest event in the file
# (A0 at 0 ms, A1 at 5000 ms, B0 at 30000 ms, B1 at 60000 ms). A round
# depends on the previous round of its session, with the gap between that
# round's last event and this round's first event as think time: A1 waits
# 5000 - 2000 = 3000 ms, B1 waits 60000 - 31500 = 28500 ms.
def test_nodes_chain_rounds_with_recorded_think_time(tmp_path: Path) -> None:
    workload = TraceLabSource(block_size=64).load(write_trace(tmp_path, LINES))
    nodes = {n.id: n for n in workload.arrangement.nodes}
    assert list(nodes) == ["claude:A:0", "claude:A:1", "claude:B:0", "claude:B:1"]
    assert [nodes[i].send_at_ms for i in nodes] == [0, 5000, 30000, 60000]
    assert nodes["claude:A:0"].depends_on == [] and nodes["claude:A:0"].think_ms == 0
    assert nodes["claude:A:1"].depends_on == ["claude:A:0"] and nodes["claude:A:1"].think_ms == 3000
    assert nodes["claude:B:1"].depends_on == ["claude:B:0"] and nodes["claude:B:1"].think_ms == 28500
    assert [nodes[i].turn for i in nodes] == [1, 3, 1, 3]
    assert workload.arrangement.has_dependencies()


# Blank lines are skipped. A line that is not JSON, one without timing
# events, one whose prefix exceeds its total and one with a bad timestamp
# each fail with the file name and line number in the message.
def test_malformed_lines_fail_with_line_numbers(tmp_path: Path) -> None:
    ok = json.dumps(LINES[0])
    assert len(TraceLabSource().load(write_trace(tmp_path, [ok, "", ok.replace("claude:A", "claude:C")])).records) == 2
    no_events = dict(LINES[0], timing_events=[])
    bad_prefix = dict(LINES[0], prefix_tokens=999)
    bad_stamp = dict(LINES[0], timing_events=[{"event_type": "text", "timestamp": "yesterday"}])
    bad_cases = [
        ("{not json", "not JSON"),
        (json.dumps(no_events), "timing_events"),
        (json.dumps(bad_prefix), "prefix_tokens exceeds"),
        (json.dumps(bad_stamp), "bad timestamp"),
    ]
    for line, message in bad_cases:
        path = write_trace(tmp_path, [ok, line])
        with pytest.raises(ValueError, match=message) as info:
            TraceLabSource().load(path)
        assert f"{path}:2" in str(info.value)
