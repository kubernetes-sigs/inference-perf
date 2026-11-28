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

from pathlib import Path
from typing import List, Optional

from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType, SessionReplayConfig, WorkloadReplayConfig
from inference_perf.datagen.replay.replay_graph_session_datagen import SessionChatCompletionAPIData
from inference_perf.datagen.workload import Trace, WorkloadSessionGenerator
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.workload import Delivery, Message, Record, Schedule, Session, TextPart, TextSpec
from workload_fixtures import new_word_tokenizer, write_corpus


# One round that recorded its whole prompt: a user message of `total` tokens
# whose leading blocks are `blocks` (16 tokens each, session-scoped) and a
# record asking for `out` tokens.
def round_record(
    session: str,
    index: int,
    total: int,
    blocks: List[int],
    out: int,
) -> Record:
    return Record(
        id=f"{session}:{index}",
        messages=[
            Message(
                role="user",
                parts=[TextPart(spec=TextSpec(num_tokens=total, block_ids=blocks, block_size=16, scope="session"))],
            )
        ],
        output_tokens=out,
    )


# The delivery of a two-request session: request 0 went out at `first_ms`,
# request 1 at `second_ms` after waiting `wait_ms` for request 0.
def chain_delivery(session: str, first_ms: int, second_ms: int, wait_ms: int) -> Delivery:
    return Delivery(
        schedules={
            f"{session}:0": Schedule(start_ms=first_ms),
            f"{session}:1": Schedule(start_ms=second_ms, wait_ms=wait_ms, after=[f"{session}:0"]),
        }
    )


# Two sessions of the TraceLab shape (each round its own record holding the
# whole prompt, with session-scoped blocks; session B reuses blocks [0, 1]
# of its round 0 in round 1) and one recorded text conversation whose
# second request carries the whole exchange. Ids contain ':' the way
# TraceLab's do.
def session_trace() -> Trace:
    hello = Message(role="user", parts=[TextPart(text="w1 w2")])
    sessions = [
        Session(
            id="claude:A",
            records=[round_record("claude:A", 0, 40, [0, 1, 2], 5), round_record("claude:A", 1, 50, [0, 1, 2, 3], 6)],
            delivery=chain_delivery("claude:A", 0, 5000, 3000),
        ),
        Session(
            id="claude:B",
            records=[round_record("claude:B", 0, 48, [0, 1, 2], 4), round_record("claude:B", 1, 40, [0, 1, 9], 2)],
            delivery=chain_delivery("claude:B", 1000, 9000, 7500),
        ),
        Session(
            id="chat",
            records=[
                Record(
                    id="chat:0",
                    messages=[hello],
                    output_tokens=1,
                    reply=[TextPart(text="w3")],
                ),
                Record(
                    id="chat:1",
                    messages=[
                        hello,
                        Message(role="assistant", parts=[TextPart(text="w3")]),
                        Message(role="user", parts=[TextPart(text="w4 w5 w6")]),
                    ],
                    output_tokens=2,
                ),
            ],
            delivery=chain_delivery("chat", 0, 100, 50),
        ),
    ]
    return Trace(source_id="trace.jsonl", sessions=sessions)


# data.workload for a trace handed to the generator directly, with the
# session settings given.
def data_config(corpus_path: Path, **settings: object) -> DataConfig:
    block = WorkloadReplayConfig.model_validate({"format": "test", "trace_files": ["unused"], **settings})
    return DataConfig(type=DataGenType.WorkloadReplay, workload=block, corpus_file_path=str(corpus_path))


def make_generator(
    corpus_path: Path, tokenizer: CustomTokenizer, trace: Optional[Trace] = None, **settings: object
) -> WorkloadSessionGenerator:
    return WorkloadSessionGenerator(
        APIConfig(type=APIType.Chat),
        data_config(corpus_path, **settings),
        tokenizer,
        base_seed=1,
        trace=trace or session_trace(),
    )


def events(gen: WorkloadSessionGenerator, session_index: int) -> List[SessionChatCompletionAPIData]:
    assert gen.is_session_buildable(session_index)
    out = []
    for lazy in gen.get_session_events(session_index):
        data = gen.load_lazy_data(lazy)
        assert isinstance(data, SessionChatCompletionAPIData)
        out.append(data)
    return out


# One replay session per session of the trace, in the order the trace lists them.
# Session ids have no ':' because the session runtime joins ids with it,
# and a session reports its own id in the trace as its source.
def test_session_records_become_sessions(tmp_path: Path) -> None:
    gen = make_generator(write_corpus(tmp_path), new_word_tokenizer())
    assert gen.get_session_count() == 3
    assert [gen.get_session_info(i)["session_id"] for i in range(3)] == ["s0_claude_A", "s1_claude_B", "s2_chat"]
    assert [gen.get_session_info(i)["source_id"] for i in range(3)] == ["claude:A", "claude:B", "chat"]


# Every event names the record it was built from, with the record's own id
# (':' included) and not the call id the runtime uses. A duplicated session
# (slot 3 copies session 0) names the same records as its source.
def test_events_carry_their_record_ids(tmp_path: Path) -> None:
    gen = make_generator(write_corpus(tmp_path), new_word_tokenizer(), duplicate_sessions_target=4)
    assert [[e.record_id for e in events(gen, i)] for i in range(3)] == [
        ["claude:A:0", "claude:A:1"],
        ["claude:B:0", "claude:B:1"],
        ["chat:0", "chat:1"],
    ]
    assert [e.record_id for e in events(gen, 3)] == ["claude:A:0", "claude:A:1"]


# Session A's two events: the first has no predecessors and waits 0 ms; the
# second depends on the first and waits the recorded 3000 ms, not the
# 5000 ms gap between the send times. Each prompt is one user message of
# the declared length, and max_tokens is the record's output_tokens.
def test_session_events_follow_the_schedule(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    gen = make_generator(write_corpus(tmp_path), word_tokenizer)
    first, second = events(gen, 0)
    assert first.predecessor_event_ids == [] and first.wait_ms == 0
    assert second.predecessor_event_ids == [first.event_id] and second.wait_ms == 3000
    assert len(first.messages) == 1 and first.messages[0].role == "user"
    assert word_tokenizer.count_tokens(str(first.messages[0].content)) == 40 and first.max_tokens == 5
    assert word_tokenizer.count_tokens(str(second.messages[0].content)) == 50 and second.max_tokens == 6
    graph_events = sorted(gen._get_session(0).graph.events.values(), key=lambda e: e.t_start_ms)
    assert [e.call.total_input_tokens for e in graph_events] == [40, 50]
    assert [e.t_start_ms for e in graph_events] == [0, 5000]


# Within a session, round 1 reuses blocks [0, 1, 2] of round 0, so its
# prompt starts with the same 48 words. Across sessions the ids are the
# same numbers but the scope is the session, so A's block 0 and B's block 0
# differ.
def test_session_scoped_prefix_reuse(tmp_path: Path) -> None:
    gen = make_generator(write_corpus(tmp_path), new_word_tokenizer())
    a0, a1 = (str(e.messages[0].content).split() for e in events(gen, 0))
    b0, b1 = (str(e.messages[0].content).split() for e in events(gen, 1))
    assert a0[:40] == a1[:40] and a1[40:48] != a0[:8]
    assert b0[:32] == b1[:32] and b1[32:] != b0[32:]
    assert a0[:16] != b0[:16]


# A recorded conversation is one record per request: the second carries the
# first user message, the recorded assistant reply and the new user message, and
# waits the recorded 50 ms for the first.
def test_recorded_conversation_accumulates(tmp_path: Path) -> None:
    gen = make_generator(write_corpus(tmp_path), new_word_tokenizer())
    first, second = events(gen, 2)
    assert [(m.role, m.content) for m in first.messages] == [("user", "w1 w2")]
    assert [(m.role, m.content) for m in second.messages] == [("user", "w1 w2"), ("assistant", "w3"), ("user", "w4 w5 w6")]
    assert second.predecessor_event_ids == [first.event_id] and second.wait_ms == 50


# The session settings in data.workload reach the runtime: a max_wait_ms of
# 1000 caps session B's 7500 ms wait.
def test_session_settings_cap_waits(tmp_path: Path) -> None:
    gen = make_generator(write_corpus(tmp_path), new_word_tokenizer(), max_wait_ms=1000)
    _, second = events(gen, 1)
    assert second.wait_ms == 1000


# A format with a config block of its own passes that block in. Its
# max_wait_ms of 1000 caps session B's 7500 ms wait although data.workload
# leaves the clamp at its default, which alone would leave the wait at 7500.
def test_a_replay_config_argument_sets_the_session_settings(tmp_path: Path) -> None:
    corpus_path = write_corpus(tmp_path)
    gen = WorkloadSessionGenerator(
        APIConfig(type=APIType.Chat),
        data_config(corpus_path),
        new_word_tokenizer(),
        base_seed=1,
        trace=session_trace(),
        replay_config=SessionReplayConfig(max_wait_ms=1000),
    )
    _, second = events(gen, 1)
    assert second.wait_ms == 1000
    _, unset = events(make_generator(corpus_path, new_word_tokenizer()), 1)
    assert unset.wait_ms == 7500


# duplicate_sessions_target of 5 over the 3 sessions adds copies of the first
# two, named after them with a _dup suffix and asking for what they ask for
# (max_tokens 5 then 6, and 4 then 2). A target of 2 is already met and adds
# nothing.
def test_duplicate_sessions_target_adds_copies(tmp_path: Path) -> None:
    corpus_path = write_corpus(tmp_path)
    gen = make_generator(corpus_path, new_word_tokenizer(), duplicate_sessions_target=5)
    assert [gen.get_session_info(i)["session_id"] for i in range(gen.get_session_count())] == [
        "s0_claude_A",
        "s1_claude_B",
        "s2_chat",
        "s0_claude_A_dup1",
        "s1_claude_B_dup2",
    ]
    assert [[e.max_tokens for e in events(gen, i)] for i in (3, 4)] == [[5, 6], [4, 2]]
    met = make_generator(corpus_path, new_word_tokenizer(), duplicate_sessions_target=2)
    assert met.get_session_count() == 3


# A stage bounded by time asks for sessions past the end of the trace.
# Index 3 of 3 sessions replays session 0 (max_tokens 5 then 6), index 4
# session 1 (4 then 2) and index 5 session 2 (1 then 2), each under its
# source's id plus _dup1. With two duplicates added first (5 slots), index 6
# replays slot 1 and index 8 replays slot 3, itself a copy of session 0.
def test_indices_past_the_trace_replay_it(tmp_path: Path) -> None:
    corpus_path = write_corpus(tmp_path)
    gen = make_generator(corpus_path, new_word_tokenizer())
    assert [[e.max_tokens for e in events(gen, i)] for i in (3, 4, 5)] == [[5, 6], [4, 2], [1, 2]]
    assert [gen.get_session_info(i)["session_id"] for i in (3, 5)] == ["s0_claude_A_dup1", "s2_chat_dup1"]
    padded = make_generator(corpus_path, new_word_tokenizer(), duplicate_sessions_target=5)
    assert [[e.max_tokens for e in events(padded, i)] for i in (6, 8)] == [[4, 2], [5, 6]]
    assert padded.get_session_info(8)["session_id"] == "s0_claude_A_dup1_dup1"


# A session of one request "w1 w2" that recorded `out` output tokens on
# `model`.
def one_request(session: str, out: int, model: Optional[str] = None) -> Session:
    record = Record(
        id=session,
        messages=[Message(role="user", parts=[TextPart(text="w1 w2")])],
        output_tokens=out,
        model=model,
    )
    return Session(id=session, records=[record])


# Two one-request sessions recorded on models "recorded" and "other". The
# graph names the recorded model by default, the mapped one where
# model_mapping has an entry ("recorded" -> "served", "other" unchanged), and
# static_model_name for every call when use_static_model is set.
def test_session_settings_name_the_model_in_the_graph(tmp_path: Path) -> None:
    corpus_path = write_corpus(tmp_path)
    trace = Trace(source_id="t", sessions=[one_request("a", 3, "recorded"), one_request("b", 3, "other")])

    # the model the graph names for the one call of session a, then of session b
    def models(**settings: object) -> List[str]:
        gen = make_generator(corpus_path, new_word_tokenizer(), trace=trace, **settings)
        return [gen._get_session(i).graph.events["event_000_" + key].call.model for i, key in enumerate(("a", "b"))]

    assert models() == ["recorded", "other"]
    assert models(model_mapping={"recorded": "served"}) == ["served", "other"]
    assert models(use_static_model=True, static_model_name="static") == ["static", "static"]


# The graph's recorded max_tokens is the record's output_tokens (3), or the
# session default_max_tokens (77) when the turn recorded zero.
def test_a_zero_length_output_records_the_default_max_tokens(tmp_path: Path) -> None:
    trace = Trace(source_id="t", sessions=[one_request("a", 3), one_request("b", 0)])
    gen = make_generator(write_corpus(tmp_path), new_word_tokenizer(), trace=trace, default_max_tokens=77)
    recorded = [
        gen._get_session(i).graph.events["event_000_" + key].call.max_tokens_recorded for i, key in enumerate(("a", "b"))
    ]
    assert recorded == [3, 77]


# Text named whole by `ids`: 8 tokens per id, scoped to the session.
def block(ids: List[int]) -> TextPart:
    return TextPart(spec=TextSpec(num_tokens=8 * len(ids), block_ids=ids, block_size=8, scope="session"))


# Two records of one session "t", the Weka shape: round 0 asks for 3 tokens
# and recorded them as block 5; round 1's prompt is round 0's user blocks,
# that assistant block, and a new user block. They run as one session of
# two events. Round 0's recorded output is the same text as round 1's
# assistant message, so round 1's input is a shared segment (round 0's
# user message), an output segment from round 0 (the live reply goes
# there, counted as round 0's 3 tokens) and a unique segment.
def test_recorded_outputs_are_substituted(tmp_path: Path) -> None:
    records = [
        Record(
            id="r0",
            messages=[Message(role="user", parts=[block([1, 2])])],
            output_tokens=3,
            reply=[block([5])],
        ),
        Record(
            id="r1",
            messages=[
                Message(role="user", parts=[block([1, 2])]),
                Message(role="assistant", parts=[block([5])]),
                Message(role="user", parts=[block([6])]),
            ],
            output_tokens=4,
        ),
    ]
    delivery = Delivery(schedules={"r1": Schedule(wait_ms=20, after=["r0"])})
    trace = Trace(source_id="t.json", sessions=[Session(id="t", records=records, delivery=delivery)])
    gen = make_generator(write_corpus(tmp_path), new_word_tokenizer(), trace=trace)
    assert gen.get_session_count() == 1
    first, second = events(gen, 0)
    assert [m.role for m in second.messages] == ["user", "assistant", "user"]
    assert second.messages[0].content == first.messages[0].content
    assert [(s.type, s.message_count) for s in second.input_segments] == [("shared", 1), ("output", 1), ("unique", 1)]
    output = second.input_segments[1]
    assert output.source_event_id == first.event_id and output.token_count == 3
    assert second.wait_ms == 20


# With a tokenizer that adds one BOS token, a trace that declares its
# lengths without special tokens gets messages of exactly the declared
# words (16 and 8), not one fewer each; the default counts the BOS, so the
# same 16-token part lands as 15 words.
def test_lengths_without_special_tokens(tmp_path: Path) -> None:
    bos_tokenizer = new_word_tokenizer(bos=1)
    corpus_path = write_corpus(tmp_path)
    record = Record(
        id="r",
        messages=[
            Message(role="user", parts=[block([1, 2])]),
            Message(role="assistant", parts=[block([3])]),
            Message(role="user", parts=[TextPart(spec=TextSpec(num_tokens=8, block_size=512))]),
        ],
        output_tokens=1,
    )
    for counted, words in ((False, [16, 8, 8]), (True, [15, 7, 7])):
        trace = Trace(source_id="t", sessions=[Session(id="r", records=[record])], counts_special_tokens=counted)
        (only,) = events(make_generator(corpus_path, bos_tokenizer, trace=trace), 0)
        assert [len(str(m.content).split()) for m in only.messages] == words
