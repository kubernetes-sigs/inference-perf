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
from typing import Any, Dict, List, Tuple

import pytest

from inference_perf.workload import Record, SyntheticPart
from inference_perf.workload.sources import load_workload
from inference_perf.workload.sources.weka import WekaSource


def call(t: float, n_in: int, n_out: int, hash_ids: List[int], api_time: float) -> Dict[str, Any]:
    return {"t": t, "type": "n", "model": "m", "in": n_in, "out": n_out, "hash_ids": hash_ids, "api_time": api_time}


def trace(trace_id: str, requests: List[Dict[str, Any]], tool_tokens: int = 0, system_tokens: int = 0) -> Dict[str, Any]:
    return {
        "id": trace_id,
        "models": ["m"],
        "block_size": 4,
        "tool_tokens": tool_tokens,
        "system_tokens": system_tokens,
        "requests": requests,
    }


def write(tmp_path: Path, *traces: Dict[str, Any]) -> Path:
    for t in traces:
        (tmp_path / f"{t['id']}.json").write_text(json.dumps(t))
    return tmp_path


def shape(record: Record) -> List[Tuple[str, int, List[int]]]:
    """(role, num_tokens, block_ids) per turn; an assistant turn with no
    recorded reply blocks shows as (assistant, 0, [])."""
    out: List[Tuple[str, int, List[int]]] = []
    for turn in record.turns:
        part = turn.parts[0] if turn.parts else SyntheticPart(num_tokens=0)
        assert isinstance(part, SyntheticPart)
        out.append((turn.role, part.num_tokens, part.block_ids))
    return out


# Block size 4, a one-block system prompt (2 tool + 2 system tokens), and
# three rounds:
# - round 0: 14 tokens over blocks [1, 2, 3] = the system block, a user
#   message of blocks [2, 3] plus a 2-token tail; its reply is block 4,
#   the block round 1 recorded for it.
# - round 1: 22 tokens over [1..5]. It keeps round 0's prompt less round
#   0's 2-token tail, then the reply (block 4, ceil(3 / 4) = 1 block), then
#   block 5 plus a 2-token tail. Round 2 carries no reply for it.
# - round 2: 9 tokens over [1, 2], a rewind: the prompt is cut back to two
#   blocks (the system block and block 2 of the user message) plus a
#   1-token user tail.
# Every round is its own record in session T, every part session-scoped,
# and each prompt adds up to the recorded input length.
def test_rounds_become_records_with_the_rebuilt_prompt(tmp_path: Path) -> None:
    rounds = [call(0.0, 14, 3, [1, 2, 3], 0.5), call(2.0, 22, 5, [1, 2, 3, 4, 5], 1.0), call(4.0, 9, 2, [1, 2], 0.2)]
    workload = WekaSource().load(write(tmp_path, trace("T", rounds, tool_tokens=2, system_tokens=2)))
    r0, r1, r2 = workload.records
    assert [r.id for r in workload.records] == ["T/parent_turn_0", "T/parent_turn_1", "T/parent_turn_2"]
    assert {r.session_id for r in workload.records} == {"T"}
    assert shape(r0) == [("system", 4, [1]), ("user", 10, [2, 3]), ("assistant", 4, [4])]
    assert shape(r1) == [
        ("system", 4, [1]),
        ("user", 8, [2, 3]),
        ("assistant", 4, [4]),
        ("user", 6, [5]),
        ("assistant", 0, []),
    ]
    assert shape(r2) == [("system", 4, [1]), ("user", 4, [2]), ("user", 1, []), ("assistant", 0, [])]
    assert [r.turns[-1].output_tokens for r in workload.records] == [3, 5, 2]
    for record, n_in in zip(workload.records, [14, 22, 9], strict=True):
        assert sum(n for role, n, _ in shape(record)[:-1]) == n_in
    parts = [p for r in workload.records for t in r.turns for p in t.parts]
    assert all(isinstance(p, SyntheticPart) and p.scope == "session" and p.block_size == 4 for p in parts)
    assert workload.special_tokens_in_lengths is False


# Round 1 carries round 0's reply, so it depends on round 0 and waits the
# recorded gap after it: sent at 2000 ms, round 0 ended at 500 ms. Round 2
# carries no reply (the rewind dropped round 1's), so it depends only on
# the latest call that had finished when it was sent, round 1, and waits
# 4000 - 3000 ms.
def test_rounds_depend_on_the_reply_they_carry_or_the_last_finished_call(tmp_path: Path) -> None:
    rounds = [call(0.0, 14, 3, [1, 2, 3], 0.5), call(2.0, 22, 5, [1, 2, 3, 4, 5], 1.0), call(4.0, 9, 2, [1, 2], 0.2)]
    nodes = WekaSource().load(write(tmp_path, trace("T", rounds))).arrangement.nodes
    assert [(n.id, n.depends_on, n.think_ms, n.send_at_ms) for n in nodes] == [
        ("T/parent_turn_0", [], 0, 0),
        ("T/parent_turn_1", ["T/parent_turn_0"], 1500, 2000),
        ("T/parent_turn_2", ["T/parent_turn_1"], 1000, 4000),
    ]
    assert all(n.turn == len(r.turns) - 1 for n, r in zip(nodes, WekaSource().load(tmp_path).records, strict=True))


# A subagent spawned at 1.0 s makes two requests whose times are relative
# to the spawn (0.0 and 0.5) and which overlap (the first runs until 2.0 s),
# so they are two streams of one request each. Both are spawned from the
# parent's round 0 (the last call finished when each was sent), and the
# parent's round 1 depends on round 0 (whose reply it carries, block 3) and
# on the stream that finished last before it, s1, waiting 3000 - 2000 ms.
def test_subagent_streams_spawn_from_and_join_the_parent(tmp_path: Path) -> None:
    subagent = {
        "t": 1.0,
        "type": "subagent",
        "agent_id": "x",
        "subagent_type": "worker",
        "requests": [call(0.0, 6, 1, [7], 1.0), call(0.5, 4, 1, [8], 0.5)],
    }
    requests = [call(0.0, 8, 2, [1, 2], 0.5), subagent, call(3.0, 16, 1, [1, 2, 3, 4], 0.1)]
    workload = WekaSource().load(write(tmp_path, trace("T", requests)))
    assert {r.session_id for r in workload.records} == {"T"}
    assert [(n.id, n.depends_on, n.think_ms) for n in workload.arrangement.nodes] == [
        ("T/parent_turn_0", [], 0),
        ("T/sa_x_s0_turn_0", ["T/parent_turn_0"], 500),
        ("T/sa_x_s1_turn_0", ["T/parent_turn_0"], 1000),
        ("T/parent_turn_1", ["T/parent_turn_0", "T/sa_x_s1_turn_0"], 1000),
    ]
    by_id = {r.id: r for r in workload.records}
    assert shape(by_id["T/sa_x_s0_turn_0"]) == [("user", 6, [7]), ("assistant", 0, [])]
    assert shape(by_id["T/parent_turn_0"])[-1] == ("assistant", 4, [3])


# The registry loads a directory of traces as one workload, one session per
# trace; a .jsonl file holds one trace per line; a repeated trace id and a
# file that is not a trace fail with the file named.
def test_loading_directories_jsonl_and_bad_input(tmp_path: Path) -> None:
    a = trace("A", [call(0.0, 4, 1, [1], 0.1)])
    b = trace("B", [call(0.0, 4, 1, [1], 0.1)])
    (tmp_path / "dir").mkdir()
    workload = load_workload("Weka", str(write(tmp_path / "dir", a, b)))
    assert [r.session_id for r in workload.records] == ["A", "B"]

    jsonl = tmp_path / "traces.jsonl"
    jsonl.write_text(json.dumps(a) + "\n\n" + json.dumps(b) + "\n")
    assert [r.session_id for r in WekaSource().load(jsonl).records] == ["A", "B"]

    dup = tmp_path / "dup.jsonl"
    dup.write_text(json.dumps(a) + "\n" + json.dumps(a) + "\n")
    with pytest.raises(ValueError, match="duplicate trace id 'A'"):
        WekaSource().load(dup)

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"id": "C", "requests": []}))
    with pytest.raises(ValueError, match="bad.json: not a Weka trace"):
        WekaSource().load(bad)
