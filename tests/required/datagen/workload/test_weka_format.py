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
from typing import Dict, List, Set, Tuple

import pytest

from inference_perf.workload import (
    BlockTextMaterializer,
    Prompter,
    Record,
    Session,
    TextPart,
    TextSpec,
    ToolCallPart,
    ToolResultPart,
    TraceLocation,
    continues,
    format_class,
    read_sessions,
)
from inference_perf.workload.formats.weka import WekaFormat
from workload_fixtures import new_word_tokenizer, write_corpus


# call(0.0, 14, 3, [1, 2, 3], 0.5) is one Weka call sent at t=0s with 14 input
# and 3 output tokens over prefix blocks [1, 2, 3], taking 0.5s.
def call(t: float, n_in: int, n_out: int, hash_ids: List[int], api_time: float) -> Dict[str, object]:
    return {"t": t, "type": "n", "model": "m", "in": n_in, "out": n_out, "hash_ids": hash_ids, "api_time": api_time}


# trace("T", rounds) is the Weka trace "T" holding those requests, block size 4, model "m".
def trace(trace_id: str, requests: List[Dict[str, object]], tool_tokens: int = 0, system_tokens: int = 0) -> Dict[str, object]:
    return {
        "id": trace_id,
        "models": ["m"],
        "block_size": 4,
        "tool_tokens": tool_tokens,
        "system_tokens": system_tokens,
        "requests": requests,
    }


# write(tmp_path, a, b) writes each trace to <id>.json under tmp_path and returns tmp_path.
def write(tmp_path: Path, *traces: Dict[str, object]) -> Path:
    for t in traces:
        (tmp_path / f"{t['id']}.json").write_text(json.dumps(t))
    return tmp_path


# The sessions of a directory of traces, or of one .jsonl file of them.
def load(path: Path) -> List[Session]:
    where = TraceLocation(directory=str(path)) if path.is_dir() else TraceLocation(files=[str(path)])
    return read_sessions(WekaFormat(), where)


def shape(record: Record) -> List[Tuple[str, int, List[int]]]:
    """(role, num_tokens, block_ids) per message, then the reply; a record
    with no recorded reply blocks ends in (assistant, 0, [])."""
    out: List[Tuple[str, int, List[int]]] = []
    for role, parts in [*((m.role, m.parts) for m in record.messages), ("assistant", record.reply)]:
        if not parts:
            out.append((role, 0, []))
            continue
        part = parts[0]
        assert isinstance(part, TextPart) and part.spec is not None
        out.append((role, part.spec.num_tokens, part.spec.block_ids))
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
# Every round is its own record of session T, every part session-scoped,
# and each prompt adds up to the recorded input length.
def test_rounds_become_records_with_the_rebuilt_prompt(tmp_path: Path) -> None:
    rounds = [call(0.0, 14, 3, [1, 2, 3], 0.5), call(2.0, 22, 5, [1, 2, 3, 4, 5], 1.0), call(4.0, 9, 2, [1, 2], 0.2)]
    (session,) = load(write(tmp_path, trace("T", rounds, tool_tokens=2, system_tokens=2)))
    r0, r1, r2 = session.records
    assert session.id == "T"
    assert [r.id for r in session.records] == ["T/parent_turn_0", "T/parent_turn_1", "T/parent_turn_2"]
    assert shape(r0) == [("system", 4, [1]), ("user", 10, [2, 3]), ("assistant", 4, [4])]
    assert shape(r1) == [
        ("system", 4, [1]),
        ("user", 8, [2, 3]),
        ("assistant", 4, [4]),
        ("user", 6, [5]),
        ("assistant", 0, []),
    ]
    assert shape(r2) == [("system", 4, [1]), ("user", 4, [2]), ("user", 1, []), ("assistant", 0, [])]
    assert [r.output_tokens for r in session.records] == [3, 5, 2]
    for record, n_in in zip(session.records, [14, 22, 9], strict=True):
        assert sum(n for role, n, _ in shape(record)[:-1]) == n_in
    parts = [p for r in session.records for m in r.messages for p in m.parts] + [p for r in session.records for p in r.reply]
    specs = [p.spec for p in parts if isinstance(p, TextPart) and p.spec is not None]
    assert len(specs) == len(parts)
    assert all(spec.scope == "session" and spec.block_size == 4 for spec in specs)
    assert WekaFormat.counts_special_tokens is False


# (id, the records it waited for, the wait in ms, the send time in ms) of
# every request of a session.
def schedules(session: Session) -> List[Tuple[str, List[str], int, int | None]]:
    return [
        (r.id, session.delivery.of(r.id).after, session.delivery.of(r.id).wait_ms, session.delivery.of(r.id).start_ms)
        for r in session.records
    ]


# Round 1 carries round 0's reply, so it waits for round 0 and for the
# recorded gap after it: sent at 2000 ms, round 0 ended at 500 ms. Round 2
# carries no reply (the rewind dropped round 1's), so it waits only for the
# latest call that had finished when it was sent, round 1, and for
# 4000 - 3000 ms. A call's duration is its api_time (500, 1000 and 200 ms),
# and a call that recorded no api_time has none.
def test_rounds_wait_for_the_reply_they_carry_or_the_last_finished_call(tmp_path: Path) -> None:
    rounds = [call(0.0, 14, 3, [1, 2, 3], 0.5), call(2.0, 22, 5, [1, 2, 3, 4, 5], 1.0), call(4.0, 9, 2, [1, 2], 0.2)]
    (session,) = load(write(tmp_path, trace("T", rounds)))
    assert schedules(session) == [
        ("T/parent_turn_0", [], 0, 0),
        ("T/parent_turn_1", ["T/parent_turn_0"], 1500, 2000),
        ("T/parent_turn_2", ["T/parent_turn_1"], 1000, 4000),
    ]
    assert [session.delivery.of(r.id).duration_ms for r in session.records] == [500, 1000, 200]
    untimed = {k: v for k, v in call(0.0, 14, 3, [1, 2, 3], 0.5).items() if k != "api_time"}
    (tmp_path / "untimed").mkdir()
    (alone,) = load(write(tmp_path / "untimed", trace("U", [untimed])))
    assert alone.delivery.of(alone.records[0].id).duration_ms is None


# Round 0 is answered with 8 tokens, which round 1 carries as blocks [3, 4].
# Round 2 only rewinds, into the middle of that reply: it keeps block 3 and
# adds 2 tokens. Round 0's reply is still the two blocks round 1 carries, so
# round 1 continues round 0. Round 2 placed no reply for round 1, so round 1
# takes what the rewind left, block 3, and round 2 waits for round 1 alone.
def test_a_later_rewind_does_not_rewrite_an_earlier_reply(tmp_path: Path) -> None:
    rounds = [call(0.0, 8, 8, [1, 2], 0.5), call(2.0, 20, 4, [1, 2, 3, 4, 5], 0.5), call(4.0, 14, 4, [1, 2, 3], 0.5)]
    (session,) = load(write(tmp_path, trace("T", rounds)))
    first, second, third = session.records
    assert shape(first)[-1] == ("assistant", 8, [3, 4])
    assert shape(second) == [("user", 8, [1, 2]), ("assistant", 8, [3, 4]), ("user", 4, [5]), ("assistant", 4, [3])]
    assert continues(session, second) is first
    assert shape(third)[:3] == [("user", 8, [1, 2]), ("assistant", 4, [3]), ("user", 2, [])]
    assert session.delivery.of(third.id).after == [second.id]


# Round 1 recorded no output, so round 2 places no reply for it. Its reply is
# then the last assistant message of round 2's prompt, block 3, which is round
# 0's reply. Round 2 waits for round 1 alone: round 0 is behind round 1.
def test_a_round_with_no_output_takes_the_reply_before_it(tmp_path: Path) -> None:
    rounds = [call(0.0, 8, 4, [1, 2], 0.5), call(2.0, 16, 0, [1, 2, 3, 4], 0.5), call(4.0, 16, 4, [1, 2, 3, 9], 0.5)]
    (session,) = load(write(tmp_path, trace("T", rounds)))
    first, second, third = session.records
    assert shape(first)[-1] == shape(second)[-1] == ("assistant", 4, [3])
    assert second.output_tokens == 0
    assert shape(third)[:3] == [("user", 8, [1, 2]), ("assistant", 4, [3]), ("user", 4, [9])]
    assert session.delivery.of(third.id).after == [second.id]


# A subagent spawned at 1.0 s makes two requests whose times are relative
# to the spawn (0.0 and 0.5) and which overlap (the first runs until 2.0 s),
# so they are two streams of one request each. Both are spawned from the
# parent's round 0 (the last call finished when each was sent), and the
# parent's round 1 waits for round 0 (whose reply it carries, block 3) and
# for the stream that finished last before it, s1, 3000 - 2000 ms. A
# subagent's records name it as their agent; the parent's name no agent.
def test_subagent_streams_spawn_from_and_join_the_parent(tmp_path: Path) -> None:
    subagent = {
        "t": 1.0,
        "type": "subagent",
        "agent_id": "x",
        "subagent_type": "worker",
        "requests": [call(0.0, 6, 1, [7], 1.0), call(0.5, 4, 1, [8], 0.5)],
    }
    requests = [call(0.0, 8, 2, [1, 2], 0.5), subagent, call(3.0, 16, 1, [1, 2, 3, 4], 0.1)]
    (session,) = load(write(tmp_path, trace("T", requests)))
    assert [entry[:3] for entry in schedules(session)] == [
        ("T/parent_turn_0", [], 0),
        ("T/sa_x_s0_turn_0", ["T/parent_turn_0"], 500),
        ("T/sa_x_s1_turn_0", ["T/parent_turn_0"], 1000),
        ("T/parent_turn_1", ["T/parent_turn_0", "T/sa_x_s1_turn_0"], 1000),
    ]
    by_id = {r.id: r for r in session.records}
    assert shape(by_id["T/sa_x_s0_turn_0"]) == [("user", 6, [7]), ("assistant", 0, [])]
    assert shape(by_id["T/parent_turn_0"])[-1] == ("assistant", 4, [3])
    assert (by_id["T/sa_x_s0_turn_0"].model, by_id["T/sa_x_s0_turn_0"].agent) == ("m", "x")
    assert (by_id["T/parent_turn_0"].model, by_id["T/parent_turn_0"].agent) == ("m", None)


# Subagent x is spawned at 5.0 s and calls at 5.0 s and 6.0 s (offsets 0.0 and
# 1.0). Between them it spawns y at offset 0.5, so at 5.5 s, and y calls once
# at offset 0.2, so at 5.7 s, for 0.2 s. y waits for x's first call, the last
# one finished when y's was sent. x's second call carries its first reply
# (block 8) and joins y, 6000 - 5900 ms after it. The parent's round 1 then
# joins x. y's record names y as its agent.
def test_a_subagent_spawned_by_a_subagent_is_read_and_joined(tmp_path: Path) -> None:
    y = {"t": 0.5, "type": "subagent", "agent_id": "y", "subagent_type": "worker", "requests": [call(0.2, 4, 1, [20], 0.2)]}
    x = {
        "t": 5.0,
        "type": "subagent",
        "agent_id": "x",
        "subagent_type": "worker",
        "requests": [call(0.0, 4, 1, [7], 0.5), y, call(1.0, 12, 1, [7, 8, 9], 0.2)],
    }
    requests = [call(0.0, 8, 2, [1, 2], 0.5), x, call(8.0, 16, 1, [1, 2, 3, 4], 0.1)]
    (session,) = load(write(tmp_path, trace("T", requests)))
    assert schedules(session) == [
        ("T/parent_turn_0", [], 0, 0),
        ("T/sa_x_s0_turn_0", ["T/parent_turn_0"], 4500, 5000),
        ("T/sa_y_s0_turn_0", ["T/sa_x_s0_turn_0"], 200, 5700),
        ("T/sa_x_s0_turn_1", ["T/sa_x_s0_turn_0", "T/sa_y_s0_turn_0"], 100, 6000),
        ("T/parent_turn_1", ["T/parent_turn_0", "T/sa_x_s0_turn_1"], 1800, 8000),
    ]
    assert {r.id: r.agent for r in session.records}["T/sa_y_s0_turn_0"] == "y"


# The summary of a trace whose subagent spawns another counts all three
# calls: the parent's 8 + 2, x's 4 + 1 and y's 6 + 1 make 3 turns, a largest
# of 10 and a total of 22.
def test_summary_counts_calls_of_nested_subagents() -> None:
    y = {"t": 0.5, "type": "subagent", "agent_id": "y", "subagent_type": "worker", "requests": [call(0.2, 6, 1, [20], 0.2)]}
    x = {"t": 1.0, "type": "subagent", "agent_id": "x", "subagent_type": "worker", "requests": [call(0.0, 4, 1, [7], 0.5), y]}
    summary = WekaFormat().summary("T", [trace("T", [call(0.0, 8, 2, [1, 2], 0.5), x])])
    assert (summary["num_turns"], summary["max_tokens"], summary["total_tokens"]) == (3, 10, 22)


# The scope every text part of a trace's one call was given.
def scopes(row: Dict[str, object]) -> Set[str]:
    session = WekaFormat().session("T", [row])
    return {
        part.spec.scope for m in session.records[0].messages for part in m.parts if isinstance(part, TextPart) and part.spec
    }


# A trace that says hash_id_scope "global" names the same blocks as every
# other trace, so its parts are trace-scoped.
def test_global_hash_ids_are_trace_scoped() -> None:
    assert scopes({**trace("T", [call(0.0, 8, 2, [1, 2], 0.5)]), "hash_id_scope": "global"}) == {"trace"}


# A trace that says hash_id_scope "local" keeps its blocks to itself.
def test_local_hash_ids_are_session_scoped() -> None:
    assert scopes({**trace("T", [call(0.0, 8, 2, [1, 2], 0.5)]), "hash_id_scope": "local"}) == {"session"}


# A trace that does not say is read as local, as weka_trace_replay reads it.
def test_hash_ids_without_a_stated_scope_are_session_scoped() -> None:
    assert scopes(trace("T", [call(0.0, 8, 2, [1, 2], 0.5)])) == {"session"}


# Block size 4. Round 0 sends 10 tokens over blocks [1, 2], round 1 grows to
# 22 over [1..5], and round 2 rewinds to 9 over [1, 2]: the cut lands exactly
# at the end of round 0's user message, which has no partial tail to lose.
# Every round's messages still add up to its recorded input: 10, 22 and 9.
def test_a_rewind_to_the_end_of_an_earlier_message_keeps_the_recorded_length(tmp_path: Path) -> None:
    rounds = [call(0.0, 10, 3, [1, 2], 0.5), call(2.0, 22, 1, [1, 2, 3, 4, 5], 0.5), call(4.0, 9, 1, [1, 2], 0.5)]
    (session,) = load(write(tmp_path, trace("T", rounds)))
    assert [sum(tokens for _, tokens, _ in shape(record)[:-1]) for record in session.records] == [10, 22, 9]


# Eleven rounds all recorded at 0.0 s. They stay in round order, so round 10
# comes last and not between rounds 1 and 2.
def test_rounds_sent_at_the_same_time_keep_their_order(tmp_path: Path) -> None:
    rounds = [call(0.0, 4 * (k + 1), 0, list(range(1, k + 2)), 0.0) for k in range(11)]
    (session,) = load(write(tmp_path, trace("T", rounds)))
    assert [record.id for record in session.records] == [f"T/parent_turn_{k}" for k in range(11)]


# subagent("x", 5.0, calls) is the Weka subagent x spawned at 5.0 s making those calls.
def subagent(agent_id: str, t: float, requests: List[Dict[str, object]]) -> Dict[str, object]:
    return {"t": t, "type": "subagent", "agent_id": agent_id, "subagent_type": "worker", "requests": requests}


# The call each record of a session says it carries out, by record id.
def carried_out(session: Session) -> Dict[str, str | None]:
    return {record.id.split("/")[1]: record.call_id for record in session.records}


# The parent's round 0 stopped on a tool call and subagent x is listed right
# after it. Both of x's requests name that call, "parent_turn_0", which is
# the id on round 0's reply. The parent's own requests carry out nothing.
def test_a_subagent_names_the_tool_call_listed_above_it(tmp_path: Path) -> None:
    first = {**call(0.0, 8, 2, [1, 2], 0.5), "stop": "tool_use"}
    x = subagent("x", 5.0, [call(0.0, 4, 1, [7], 0.5), call(1.0, 12, 1, [7, 8, 9], 0.2)])
    (session,) = load(write(tmp_path, trace("T", [first, x, call(8.0, 16, 1, [1, 2, 3, 4], 0.1)])))
    assert session.records[0].reply == [ToolCallPart(id="parent_turn_0", arguments=one_block(3))]
    assert carried_out(session) == {
        "parent_turn_0": None,
        "sa_x_s0_turn_0": "parent_turn_0",
        "sa_x_s0_turn_1": "parent_turn_0",
        "parent_turn_1": None,
    }


# The same trace with a round 0 that does not say how it stopped. Nothing
# records a tool call, so x's requests name none.
def test_a_subagent_under_a_round_that_recorded_no_tool_call_names_none(tmp_path: Path) -> None:
    x = subagent("x", 5.0, [call(0.0, 4, 1, [7], 0.5)])
    (session,) = load(write(tmp_path, trace("T", [call(0.0, 8, 2, [1, 2], 0.5), x, call(8.0, 16, 1, [1, 2, 3, 4], 0.1)])))
    assert carried_out(session)["sa_x_s0_turn_0"] is None


# Subagent y is nested in x, right after x's first call, which stopped on a
# tool call. y's request names x's call, "sa_x_s0_turn_0", not the parent's.
def test_a_nested_subagent_names_the_call_of_the_subagent_that_spawned_it(tmp_path: Path) -> None:
    spawns = {**call(0.0, 4, 1, [7], 0.5), "stop": "tool_use"}
    y = subagent("y", 0.5, [call(0.2, 4, 1, [20], 0.2)])
    x = subagent("x", 5.0, [spawns, y, call(1.0, 12, 1, [7, 8, 9], 0.2)])
    first = {**call(0.0, 8, 2, [1, 2], 0.5), "stop": "tool_use"}
    (session,) = load(write(tmp_path, trace("T", [first, x, call(8.0, 16, 1, [1, 2, 3, 4], 0.1)])))
    assert carried_out(session)["sa_y_s0_turn_0"] == "sa_x_s0_turn_0"
    assert carried_out(session)["sa_x_s0_turn_0"] == "parent_turn_0"


# A trace's filter summary is the trace plus its request count and sizes,
# subagent requests included: a parent call of 14 + 3 tokens and a subagent
# call of 6 + 1 make 2 turns, a largest of 17 and a total of 24.
def test_summary_counts_parent_and_subagent_requests() -> None:
    subagent = {"t": 1.0, "type": "subagent", "agent_id": "x", "requests": [call(0.0, 6, 1, [7], 1.0)]}
    summary = WekaFormat().summary("T", [trace("T", [call(0.0, 14, 3, [1, 2, 3], 0.5), subagent])])
    assert (summary["num_turns"], summary["max_tokens"], summary["total_tokens"]) == (2, 17, 24)
    assert summary["session_id"] == "T" and summary["models"] == ["m"]


# A directory of traces is one session per trace; a .jsonl file holds one
# trace per line; a repeated trace id and a file that is not a trace fail
# with the file named.
def test_loading_directories_jsonl_and_bad_input(tmp_path: Path) -> None:
    a = trace("A", [call(0.0, 4, 1, [1], 0.1)])
    b = trace("B", [call(0.0, 4, 1, [1], 0.1)])
    (tmp_path / "dir").mkdir()
    assert [session.id for session in load(write(tmp_path / "dir", a, b))] == ["A", "B"]

    jsonl = tmp_path / "traces.jsonl"
    jsonl.write_text(json.dumps(a) + "\n\n" + json.dumps(b) + "\n")
    assert [session.id for session in load(jsonl)] == ["A", "B"]

    dup = tmp_path / "dup.jsonl"
    dup.write_text(json.dumps(a) + "\n" + json.dumps(a) + "\n")
    with pytest.raises(ValueError, match="duplicate trace id 'A'"):
        load(dup)

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"id": "C", "requests": []}))
    with pytest.raises(ValueError, match=r"bad\.json:0: not a Weka trace"):
        load(bad)


# Three rounds of 8, 16 and 24 input tokens over blocks [1, 2], [1..4] and
# [1..6], each answered with 4 tokens, so every round adds one reply block
# and one user block. `extra` adds fields to each round by its index.
def tool_rounds(extra: Dict[int, Dict[str, object]]) -> List[Dict[str, object]]:
    rounds = [call(0.0, 8, 4, [1, 2], 0.5), call(2.0, 16, 4, [1, 2, 3, 4], 0.5), call(4.0, 24, 4, [1, 2, 3, 4, 5, 6], 0.5)]
    return [{**rnd, **extra.get(index, {})} for index, rnd in enumerate(rounds)]


# A 4-token text spec over one session-scoped block.
def one_block(block_id: int) -> TextPart:
    return TextPart(spec=TextSpec(num_tokens=4, block_ids=[block_id], block_size=4, scope="session"))


# Round 0 stopped on a tool call and round 1 says it added a tool result. So
# round 0's reply is a tool call, named after the round that made it, whose
# arguments are its recorded block 3,
# and round 1 sends that call back as the assistant message followed by a
# tool result holding block 4. Round 1 still continues round 0. Round 1
# itself ended normally, so its reply and round 2's new message are text.
def test_a_tool_call_and_its_result_are_read_as_parts(tmp_path: Path) -> None:
    rounds = tool_rounds({0: {"stop": "tool_use"}, 1: {"stop": "end_turn", "input_types": ["tool_result"]}})
    (session,) = load(write(tmp_path, trace("T", rounds)))
    first, second, third = session.records
    called = ToolCallPart(id="parent_turn_0", arguments=one_block(3))
    assert first.reply == [called]
    assert [m.role for m in second.messages] == ["user", "assistant", "user"]
    assert second.messages[1].parts == [called]
    assert second.messages[2].parts == [ToolResultPart(content=[one_block(4)])]
    assert continues(session, second) is first
    assert second.reply == [one_block(5)]
    assert third.messages[3].parts == [one_block(5)] and third.messages[4].parts == [one_block(6)]


# What round 1 adds, by what the trace says. It is a tool result when round
# 1 says so, whatever round 0 stopped on. When round 1 says nothing it is a
# tool result only if round 0 stopped on a tool call. When round 1 says it
# added text it is text, even right after a tool call.
def test_what_a_round_adds_is_a_tool_result_when_the_trace_says_so(tmp_path: Path) -> None:
    cases: List[Tuple[Dict[int, Dict[str, object]], type]] = [
        ({1: {"input_types": ["tool_result"]}}, ToolResultPart),
        ({0: {"stop": "end_turn"}, 1: {"input_types": ["text", "tool_result"]}}, ToolResultPart),
        ({0: {"stop": "tool_use"}}, ToolResultPart),
        ({0: {"stop": "tool_use"}, 1: {"input_types": []}}, ToolResultPart),
        ({0: {"stop": "tool_use"}, 1: {"input_types": ["text"]}}, TextPart),
        ({0: {"stop": "end_turn"}}, TextPart),
        ({}, TextPart),
    ]
    for index, (extra, expected) in enumerate(cases):
        folder = tmp_path / str(index)
        folder.mkdir()
        (session,) = load(write(folder, trace("T", tool_rounds(extra))))
        assert type(session.records[1].messages[2].parts[0]) is expected, extra


# The first round of a trace can itself say it sends a tool result. The last
# round's reply has no later round to record its blocks: it is an empty tool
# call when the round stopped on one and nothing otherwise.
def test_tool_parts_on_the_first_and_last_round(tmp_path: Path) -> None:
    rounds = tool_rounds({0: {"input_types": ["tool_result"]}, 2: {"stop": "tool_use"}})
    (session,) = load(write(tmp_path, trace("T", rounds)))
    assert type(session.records[0].messages[0].parts[0]) is ToolResultPart
    assert session.records[2].reply == [ToolCallPart(id="parent_turn_2")]
    (tmp_path / "plain").mkdir()
    (plain,) = load(write(tmp_path / "plain", trace("T", tool_rounds({}))))
    assert type(plain.records[0].messages[0].parts[0]) is TextPart
    assert plain.records[2].reply == []


# The same three rounds with and without tool calls recorded build the same
# message text, role by role: a tool call's arguments and a tool result's
# content go out as plain text until an API shapes them.
def test_tool_parts_send_the_same_text_as_plain_rounds(tmp_path: Path) -> None:
    (tmp_path / "plain").mkdir()
    (tmp_path / "tools").mkdir()
    (plain,) = load(write(tmp_path / "plain", trace("T", tool_rounds({}))))
    (tools,) = load(write(tmp_path / "tools", trace("T", tool_rounds({0: {"stop": "tool_use"}, 1: {"stop": "tool_use"}}))))
    assert type(tools.records[2].messages[3].parts[0]) is ToolCallPart
    assert type(tools.records[2].messages[4].parts[0]) is ToolResultPart
    tokenizer = new_word_tokenizer().get_tokenizer()
    prompter = Prompter("t", BlockTextMaterializer(tokenizer, base_seed=1, corpus_path=write_corpus(tmp_path)), False)
    for a, b in zip(plain.records, tools.records, strict=True):
        built_a, built_b = prompter.prompt(a, "T"), prompter.prompt(b, "T")
        assert [(m.role, m.text, m.declared_tokens) for m in built_a] == [(m.role, m.text, m.declared_tokens) for m in built_b]
        assert all(len(m.text.split()) == m.declared_tokens for m in built_b)
        assert prompter.reply(a, "T").text == prompter.reply(b, "T").text


# A config that says `format: Weka` gets this format.
def test_the_format_is_built_in_under_its_name() -> None:
    assert format_class("Weka") is WekaFormat
