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
from typing import Dict, List, Sequence

import pytest

from inference_perf.workload import SyntheticPart, TextPart
from inference_perf.workload.sources import Workload
from inference_perf.workload.sources.tracelab import TraceLabSource


# One TraceLab line: rnd("claude:A", 0, 200, 0, 30, t0, t1) is round 0 of session claude:A,
# a 200-token prompt with nothing cached and 30 output tokens, arriving at t0 and done at t1.
def rnd(
    session: str,
    index: int,
    total: int,
    prefix: int,
    out: int,
    start: str,
    end: str,
    model: str = "claude-opus-4-8",
) -> Dict[str, object]:
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
# compacts: round 1 keeps only 64 of round 0's 300 tokens. A session's rows
# sit together, as the reader requires. B's are written out of order to
# check that the rounds are put back in the order they were submitted.
LINES = [
    rnd("claude:A", 0, 200, 0, 30, "2026-06-01T10:00:00.000Z", "2026-06-01T10:00:02.000Z"),
    rnd("claude:A", 1, 260, 192, 40, "2026-06-01T10:00:05.000Z", "2026-06-01T10:00:08.000Z"),
    rnd("claude:B", 1, 130, 64, 12, "2026-06-01T10:01:00.000Z", "2026-06-01T10:01:01.000Z"),
    rnd("claude:B", 0, 300, 0, 20, "2026-06-01T10:00:30.000Z", "2026-06-01T10:00:31.500Z"),
]


def write_trace(tmp_path: Path, lines: Sequence[object]) -> Path:
    path = tmp_path / "syfi_coding_trace.jsonl"
    path.write_text("\n".join(json.dumps(line) if not isinstance(line, str) else line for line in lines) + "\n")
    return path


# The prompt of each round of one session, in round order: the synthetic
# part of the user turn of each of the session's records.
def prompts(workload: Workload, session: str) -> List[SyntheticPart]:
    out = []
    for record in workload.records:
        if record.session_id == session:
            part = record.turns[0].parts[0]
            assert isinstance(part, SyntheticPart)
            out.append(part)
    return out


# Each round becomes its own record of its session, in submission order
# regardless of file order: a user turn holding the round's whole prompt
# (200 then 260 tokens for session A, 300 then 130 for B) as a
# session-scoped synthetic part, and an assistant turn carrying
# output_tokens (30 then 40 for A).
def test_rounds_become_records_of_their_session(tmp_path: Path) -> None:
    workload = TraceLabSource(block_size=64).load(write_trace(tmp_path, LINES))
    assert [r.id for r in workload.records] == ["claude:A:round_0", "claude:A:round_1", "claude:B:round_0", "claude:B:round_1"]
    assert [r.session_id for r in workload.records] == ["claude:A", "claude:A", "claude:B", "claude:B"]
    assert all([t.role for t in r.turns] == ["user", "assistant"] for r in workload.records)
    assert [r.turns[1].output_tokens for r in workload.records[:2]] == [30, 40]
    assert [p.num_tokens for p in prompts(workload, "claude:A")] == [200, 260]
    assert all(p.scope == "session" and p.block_size == 64 for p in prompts(workload, "claude:A"))
    assert workload.records[0].metadata == {"model": "claude-opus-4-8"}
    assert [p.num_tokens for p in prompts(workload, "claude:B")] == [300, 130]


# Block ids are minted per session. Session A round 0 needs 4 blocks for 200
# tokens ([0, 1, 2, 3]); round 1 reuses the 192 // 64 = 3 leading blocks and
# mints 2 more for its 260 tokens ([0, 1, 2, 4, 5]). Block 3, round 0's
# partial tail, is never reused because the prefix stopped short of it.
def test_growing_session_reuses_leading_blocks(tmp_path: Path) -> None:
    workload = TraceLabSource(block_size=64).load(write_trace(tmp_path, LINES))
    r0, r1 = prompts(workload, "claude:A")
    assert r0.block_ids == [0, 1, 2, 3]
    assert r1.block_ids == [0, 1, 2, 4, 5]
    assert r1.prefix_tokens == 260


# Session B compacts: round 0 sent 5 blocks ([0..4]); round 1's prefix of 64
# tokens keeps only block 0, and its 130 tokens need 3 blocks, so it gets
# [0, 5, 6]. Blocks 1 to 4 are not reused.
def test_compacted_session_drops_later_blocks(tmp_path: Path) -> None:
    workload = TraceLabSource(block_size=64).load(write_trace(tmp_path, LINES))
    r0, r1 = prompts(workload, "claude:B")
    assert r0.block_ids == [0, 1, 2, 3, 4]
    assert r1.block_ids == [0, 5, 6]


# Two sessions mint the same ids ([0, 1, 2, 3] and [0, 1, 2, 3, 4] both start
# at 0), which is fine because the scope is the session: the materializer
# keys block text by session, so the ids do not collide across sessions.
def test_block_ids_are_per_session(tmp_path: Path) -> None:
    workload = TraceLabSource(block_size=64).load(write_trace(tmp_path, LINES))
    a0 = prompts(workload, "claude:A")[0]
    b0 = prompts(workload, "claude:B")[0]
    assert a0.block_ids[:4] == b0.block_ids[:4] == [0, 1, 2, 3]
    assert a0.scope == b0.scope == "session"


# One node per round, eliciting the assistant turn of that round's record
# (turn 1), sent at its offset from the earliest submission in the file
# (A0 at 0 ms, A1 at 5000 ms, B0 at 30000 ms, B1 at 60000 ms). A round
# depends on the previous round of its session, with the gap between that
# round's last output event and this round's last input event as think
# time: A1 waits 5000 - 2000 = 3000 ms, B1 waits 60000 - 31500 = 28500 ms.
def test_nodes_chain_rounds_with_recorded_think_time(tmp_path: Path) -> None:
    workload = TraceLabSource(block_size=64).load(write_trace(tmp_path, LINES))
    nodes = {n.id: n for n in workload.arrangement.nodes}
    assert list(nodes) == ["claude:A:round_0", "claude:A:round_1", "claude:B:round_0", "claude:B:round_1"]
    assert [nodes[i].send_at_ms for i in nodes] == [0, 5000, 30000, 60000]
    assert nodes["claude:A:round_0"].depends_on == [] and nodes["claude:A:round_0"].think_ms == 0
    assert nodes["claude:A:round_1"].depends_on == ["claude:A:round_0"] and nodes["claude:A:round_1"].think_ms == 3000
    assert nodes["claude:B:round_1"].depends_on == ["claude:B:round_0"] and nodes["claude:B:round_1"].think_ms == 28500
    assert [nodes[i].record_id for i in nodes] == list(nodes)
    assert [nodes[i].turn for i in nodes] == [1, 1, 1, 1]
    assert workload.arrangement.has_dependencies()


# A prefix can only reuse blocks the previous round sent and blocks that fit
# in this round. Round 0 is 100 tokens (blocks [0, 1]). Round 1 records a
# 192-token prefix of 260, which names three blocks but finds two to reuse
# ([0, 1, 2, 3, 4]). Round 2 records a 999-token prefix of a 70-token prompt,
# which is capped at the one whole block the prompt holds ([0, 5]).
def test_prefix_reuse_is_bounded_by_both_rounds(tmp_path: Path) -> None:
    lines = [
        rnd("s", 0, 100, 0, 5, "2026-06-01T10:00:00.000Z", "2026-06-01T10:00:01.000Z"),
        rnd("s", 1, 260, 192, 5, "2026-06-01T10:00:02.000Z", "2026-06-01T10:00:03.000Z"),
        rnd("s", 2, 70, 999, 5, "2026-06-01T10:00:04.000Z", "2026-06-01T10:00:05.000Z"),
    ]
    workload = TraceLabSource(block_size=64).load(write_trace(tmp_path, lines))
    assert [p.block_ids for p in prompts(workload, "s")] == [[0, 1], [0, 1, 2, 3, 4], [0, 5]]


# A round that recorded no input tokens still sends a message: its user turn
# is a single literal space instead of a zero-length synthetic part.
def test_round_without_input_sends_a_space(tmp_path: Path) -> None:
    line = rnd("s", 0, 0, 0, 5, "2026-06-01T10:00:00.000Z", "2026-06-01T10:00:01.000Z")
    (record,) = TraceLabSource().load(write_trace(tmp_path, [line])).records
    assert record.turns[0].parts == [TextPart(text=" ")]
    assert record.turns[1].output_tokens == 5


# Rounds without a usable timestamp still load. Round 1 has no events and
# round 2 has an unparseable one, so neither has a send offset. Round 1 waits
# for round 0's tool (1200 ms), round 2 for round 1's tools (none, 0 ms).
def test_rounds_without_timestamps_wait_for_the_previous_tools(tmp_path: Path) -> None:
    first = dict(
        rnd("s", 0, 100, 0, 5, "2026-06-01T10:00:00.000Z", "2026-06-01T10:00:01.000Z"),
        tools=[{"tool_name": "bash", "tool_wall_latency_ms": 1200.0}],
    )
    no_events = dict(rnd("s", 1, 120, 64, 5, "", ""), timing_events=[])
    bad_stamp = dict(rnd("s", 2, 140, 64, 5, "", ""), timing_events=[{"event_type": "text", "timestamp": "yesterday"}])
    nodes = TraceLabSource().load(write_trace(tmp_path, [first, no_events, bad_stamp])).arrangement.nodes
    assert [n.send_at_ms for n in nodes] == [0, None, None]
    assert [n.think_ms for n in nodes] == [0, 1200, 0]


# A directory is read as every trace file in it, in name order: two files of
# one session each give two sessions, and the notes file is ignored.
def test_a_directory_loads_every_trace_file(tmp_path: Path) -> None:
    (tmp_path / "b.jsonl").write_text(json.dumps(LINES[2]) + "\n" + json.dumps(LINES[3]) + "\n")
    (tmp_path / "a.jsonl").write_text(json.dumps(LINES[0]) + "\n" + json.dumps(LINES[1]) + "\n")
    (tmp_path / "notes.txt").write_text("not a trace")
    workload = TraceLabSource().load(tmp_path)
    assert [r.session_id for r in workload.records] == ["claude:A", "claude:A", "claude:B", "claude:B"]
    assert workload.source_id == tmp_path.name


# Blank lines are skipped. A line that is not JSON, one missing a required
# count and one whose count is not a number each fail with the file name and
# line number in the message.
def test_unreadable_lines_fail_with_line_numbers(tmp_path: Path) -> None:
    ok = json.dumps(LINES[0])
    assert len(TraceLabSource().load(write_trace(tmp_path, [ok, "", ok.replace("claude:A", "claude:C")])).records) == 2
    no_count = {k: v for k, v in LINES[0].items() if k != "input_tokens_total"}
    bad_count = dict(LINES[0], output_tokens="many")
    bad_cases = [
        ("{not json", "not JSON"),
        (json.dumps(no_count), "input_tokens_total"),
        (json.dumps(bad_count), "output_tokens"),
    ]
    for line, message in bad_cases:
        path = write_trace(tmp_path, [ok, line])
        with pytest.raises(ValueError, match=message) as info:
            TraceLabSource().load(path)
        assert f"{path}:2" in str(info.value)


# Every row of a session has to sit together: a session that comes back
# after another one started is an error, not a silent merge.
def test_a_session_split_by_another_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reappears"):
        TraceLabSource().load(write_trace(tmp_path, [LINES[0], LINES[2], LINES[1]]))
