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
from typing import List

import pytest

from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType, SessionReplayConfig, WorkloadReplayConfig
from inference_perf.datagen.replay.replay_graph_session_datagen import SessionChatCompletionAPIData
from inference_perf.datagen.workload import WorkloadSessionGenerator
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.workload import Arrangement, Node, Record, SyntheticPart, TextPart, Turn
from inference_perf.workload.sources import Workload
from workload_fixtures import new_word_tokenizer, write_corpus


# One round that recorded its whole prompt: a record of its session holding
# a user turn of `total` tokens whose leading blocks are `blocks` (16 tokens
# each, session-scoped) and an assistant turn asking for `out` tokens.
def round_record(session: str, index: int, total: int, blocks: List[int], out: int) -> Record:
    return Record(
        id=f"{session}:{index}",
        session_id=session,
        turns=[
            Turn(role="user", parts=[SyntheticPart(num_tokens=total, block_ids=blocks, block_size=16, scope="session")]),
            Turn(role="assistant", output_tokens=out),
        ],
    )


# Two sessions of the TraceLab shape (each round its own record holding the
# whole prompt, with session-scoped blocks; session B reuses blocks [0, 1]
# of its round 0 in round 1) and one recorded text conversation whose
# second request must carry the whole exchange. Ids contain ':' the way
# TraceLab's do.
def session_workload() -> Workload:
    records = [
        round_record("claude:A", 0, 40, [0, 1, 2], 5),
        round_record("claude:A", 1, 50, [0, 1, 2, 3], 6),
        round_record("claude:B", 0, 48, [0, 1, 2], 4),
        round_record("claude:B", 1, 40, [0, 1, 9], 2),
        Record(
            id="chat",
            turns=[
                Turn(role="user", parts=[TextPart(text="w1 w2")]),
                Turn(role="assistant", parts=[TextPart(text="w3")], output_tokens=1),
                Turn(role="user", parts=[TextPart(text="w4 w5 w6")]),
                Turn(role="assistant", output_tokens=2),
            ],
        ),
    ]
    nodes = [
        Node(id="claude:A:0", record_id="claude:A:0", turn=1, send_at_ms=0),
        Node(id="claude:A:1", record_id="claude:A:1", turn=1, send_at_ms=5000, depends_on=["claude:A:0"], think_ms=3000),
        Node(id="claude:B:0", record_id="claude:B:0", turn=1, send_at_ms=1000),
        Node(id="claude:B:1", record_id="claude:B:1", turn=1, send_at_ms=9000, depends_on=["claude:B:0"], think_ms=7500),
        Node(id="chat:0", record_id="chat", turn=1, send_at_ms=0),
        Node(id="chat:1", record_id="chat", turn=3, send_at_ms=100, depends_on=["chat:0"], think_ms=50),
    ]
    return Workload(source_id="trace.jsonl", records=records, arrangement=Arrangement(nodes=nodes))


def make_generator(
    corpus_path: Path, tokenizer: CustomTokenizer, session: SessionReplayConfig | None = None
) -> WorkloadSessionGenerator:
    data = DataConfig(
        type=DataGenType.WorkloadReplay,
        workload=WorkloadReplayConfig(format="test", file="unused", session=session),
        corpus_file_path=str(corpus_path),
    )
    return WorkloadSessionGenerator(APIConfig(type=APIType.Chat), data, tokenizer, base_seed=1, workload=session_workload())


def events(gen: WorkloadSessionGenerator, session_index: int) -> List[SessionChatCompletionAPIData]:
    assert gen.is_session_buildable(session_index)
    out = []
    for lazy in gen.get_session_events(session_index):
        data = gen.load_lazy_data(lazy)
        assert isinstance(data, SessionChatCompletionAPIData)
        out.append(data)
    return out


# One session per session id (the two rounds of A and of B share theirs; the
# conversation has none, so it is its own), in arrangement order. Session
# ids have no ':' because the session runtime joins ids with it.
def test_records_become_sessions(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    gen = make_generator(corpus_path, word_tokenizer)
    assert gen.get_session_count() == 3
    assert [gen.get_session_info(i)["session_id"] for i in range(3)] == ["s0_claude_A", "s1_claude_B", "s2_chat"]


# Session A's two events: the first has no predecessors and waits 0 ms; the
# second depends on the first and waits the node's think time, 3000 ms, not
# the 5000 ms gap between the timestamps. Each prompt is one user message
# of the declared length, and max_tokens is the assistant turn's count.
def test_session_events_follow_the_arrangement(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    gen = make_generator(corpus_path, word_tokenizer)
    first, second = events(gen, 0)
    assert first.predecessor_event_ids == [] and first.wait_ms == 0
    assert second.predecessor_event_ids == [first.event_id] and second.wait_ms == 3000
    assert len(first.messages) == 1 and first.messages[0].role == "user"
    assert word_tokenizer.count_tokens(str(first.messages[0].content)) == 40 and first.max_tokens == 5
    assert word_tokenizer.count_tokens(str(second.messages[0].content)) == 50 and second.max_tokens == 6
    graph_events = sorted(gen._get_session(0).graph.events.values(), key=lambda e: e.t_start_ms)
    assert [e.call.total_input_tokens for e in graph_events] == [40, 50]


# Within a session, round 1 reuses blocks [0, 1, 2] of round 0, so its
# prompt starts with the same 48 words. Across sessions the ids are the
# same numbers but the scope is the session, so A's block 0 and B's block 0
# differ.
def test_session_scoped_prefix_reuse(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    gen = make_generator(corpus_path, word_tokenizer)
    a0, a1 = (str(e.messages[0].content).split() for e in events(gen, 0))
    b0, b1 = (str(e.messages[0].content).split() for e in events(gen, 1))
    assert a0[:40] == a1[:40] and a1[40:48] != a0[:8]
    assert b0[:32] == b1[:32] and b1[32:] != b0[32:]
    assert a0[:16] != b0[:16]


# A recorded conversation accumulates: its second request carries
# the first user turn, the recorded assistant reply and the new user turn.
def test_recorded_conversation_accumulates(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    gen = make_generator(corpus_path, word_tokenizer)
    first, second = events(gen, 2)
    assert [(m.role, m.content) for m in first.messages] == [("user", "w1 w2")]
    assert [(m.role, m.content) for m in second.messages] == [("user", "w1 w2"), ("assistant", "w3"), ("user", "w4 w5 w6")]
    assert second.wait_ms == 50


# The session settings under data.workload.session reach the runtime: a
# max_wait_ms of 1000 caps session B's 7500 ms think time.
def test_session_config_caps_waits(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    gen = make_generator(corpus_path, word_tokenizer, session=SessionReplayConfig(max_wait_ms=1000))
    _, second = events(gen, 1)
    assert second.wait_ms == 1000


# A node that depends on a node of another session is refused: claude:A
# and claude:B are different session ids.
def test_cross_session_dependencies_are_refused() -> None:
    word_tokenizer = new_word_tokenizer()
    workload = session_workload()
    crossed = Workload(
        source_id="t",
        records=workload.records,
        arrangement=Arrangement(
            nodes=[
                Node(id="a", record_id="claude:A:0", turn=1),
                Node(id="b", record_id="claude:B:0", turn=1, depends_on=["a"]),
            ]
        ),
    )
    data = DataConfig(type=DataGenType.WorkloadReplay, workload=WorkloadReplayConfig(format="test", file="unused"))
    with pytest.raises(ValueError, match="outside its session"):
        WorkloadSessionGenerator(APIConfig(type=APIType.Chat), data, word_tokenizer, workload=crossed)


def block(ids: List[int]) -> SyntheticPart:
    return SyntheticPart(num_tokens=8 * len(ids), block_ids=ids, block_size=8, scope="session")


# Two records of one session "t", the Weka shape: round 0 asks for 3 tokens
# and recorded them as block 5; round 1's prompt is round 0's user blocks,
# that assistant block, and a new user block. They run as one session of
# two events. Round 0's recorded output is the same text as round 1's
# assistant message, so round 1's input is a shared segment (round 0's
# user message), an output segment from round 0 (the live reply goes
# there, counted as round 0's 3 tokens) and a unique segment.
def test_records_of_one_session_share_it_and_recorded_outputs_are_substituted(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    records = [
        Record(
            id="r0",
            session_id="t",
            turns=[Turn(role="user", parts=[block([1, 2])]), Turn(role="assistant", parts=[block([5])], output_tokens=3)],
        ),
        Record(
            id="r1",
            session_id="t",
            turns=[
                Turn(role="user", parts=[block([1, 2])]),
                Turn(role="assistant", parts=[block([5])]),
                Turn(role="user", parts=[block([6])]),
                Turn(role="assistant", output_tokens=4),
            ],
        ),
    ]
    nodes = [
        Node(id="n0", record_id="r0", turn=1),
        Node(id="n1", record_id="r1", turn=3, depends_on=["n0"], think_ms=20),
    ]
    workload = Workload(source_id="t.json", records=records, arrangement=Arrangement(nodes=nodes))
    data = DataConfig(
        type=DataGenType.WorkloadReplay,
        workload=WorkloadReplayConfig(format="test", file="unused"),
        corpus_file_path=str(corpus_path),
    )
    gen = WorkloadSessionGenerator(APIConfig(type=APIType.Chat), data, word_tokenizer, base_seed=1, workload=workload)
    assert gen.get_session_count() == 1
    first, second = events(gen, 0)
    assert [m.role for m in second.messages] == ["user", "assistant", "user"]
    assert second.messages[0].content == first.messages[0].content
    assert [(s.type, s.message_count) for s in second.input_segments] == [("shared", 1), ("output", 1), ("unique", 1)]
    output = second.input_segments[1]
    assert output.source_event_id == first.event_id and output.token_count == 3
    assert second.wait_ms == 20


# With a tokenizer that adds one BOS token, a workload that declares its
# lengths without special tokens gets messages of exactly the declared
# words (16 and 8), not one fewer each; the default counts the BOS, so the
# same 16-token part lands as 15 words.
def test_lengths_without_special_tokens(tmp_path: Path) -> None:
    bos_tokenizer = new_word_tokenizer(bos=1)
    corpus_path = write_corpus(tmp_path)
    records = [
        Record(
            id="r",
            turns=[
                Turn(role="user", parts=[block([1, 2])]),
                Turn(role="assistant", parts=[block([3])]),
                Turn(role="user", parts=[SyntheticPart(num_tokens=8)]),
                Turn(role="assistant", output_tokens=1),
            ],
        )
    ]
    nodes = [Node(id="n", record_id="r", turn=3)]
    data = DataConfig(
        type=DataGenType.WorkloadReplay,
        workload=WorkloadReplayConfig(format="test", file="unused"),
        corpus_file_path=str(corpus_path),
    )
    for declared, words in ((False, [16, 8, 8]), (True, [15, 7, 7])):
        workload = Workload(
            source_id="t", records=records, arrangement=Arrangement(nodes=nodes), special_tokens_in_lengths=declared
        )
        gen = WorkloadSessionGenerator(APIConfig(type=APIType.Chat), data, bos_tokenizer, base_seed=1, workload=workload)
        (only,) = events(gen, 0)
        assert [len(str(m.content).split()) for m in only.messages] == words
