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


def synthetic_round(total: int, blocks: List[int], out: int) -> List[Turn]:
    return [
        Turn(
            role="user",
            parts=[SyntheticPart(num_tokens=total, block_ids=blocks, block_size=16, scope="session")],
            self_contained=True,
        ),
        Turn(role="assistant", output_tokens=out),
    ]


# Two sessions of the TraceLab shape (self-contained rounds with
# session-scoped blocks; session B reuses blocks [0, 1] of its round 0 in
# round 1) and one recorded text conversation whose second request must
# carry the whole exchange. Ids contain ':' the way TraceLab's do.
def session_workload() -> Workload:
    records = [
        Record(
            id="claude:A",
            session_id="claude:A",
            turns=synthetic_round(40, [0, 1, 2], 5) + synthetic_round(50, [0, 1, 2, 3], 6),
        ),
        Record(
            id="claude:B", session_id="claude:B", turns=synthetic_round(48, [0, 1, 2], 4) + synthetic_round(40, [0, 1, 9], 2)
        ),
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
        Node(id="claude:A:0", record_id="claude:A", turn=1, send_at_ms=0),
        Node(id="claude:A:1", record_id="claude:A", turn=3, send_at_ms=5000, depends_on=["claude:A:0"], think_ms=3000),
        Node(id="claude:B:0", record_id="claude:B", turn=1, send_at_ms=1000),
        Node(id="claude:B:1", record_id="claude:B", turn=3, send_at_ms=9000, depends_on=["claude:B:0"], think_ms=7500),
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


# One session per record with nodes, in arrangement order; session ids have
# no ':' because the session runtime joins ids with it.
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


# A recorded conversation is not self-contained: its second request carries
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


# A node that depends on a node of another record is refused: a session is
# one record.
def test_cross_record_dependencies_are_refused() -> None:
    word_tokenizer = new_word_tokenizer()
    workload = session_workload()
    crossed = Workload(
        source_id="t",
        records=workload.records,
        arrangement=Arrangement(
            nodes=[
                Node(id="a", record_id="claude:A", turn=1),
                Node(id="b", record_id="claude:B", turn=1, depends_on=["a"]),
            ]
        ),
    )
    data = DataConfig(type=DataGenType.WorkloadReplay, workload=WorkloadReplayConfig(format="test", file="unused"))
    with pytest.raises(ValueError, match="outside its record"):
        WorkloadSessionGenerator(APIConfig(type=APIType.Chat), data, word_tokenizer, workload=crossed)
