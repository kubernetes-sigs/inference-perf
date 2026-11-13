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
from typing import Dict, List

from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType, TraceLabTraceReplayConfig, WorkloadReplayConfig
from inference_perf.datagen.replay.replay_graph_session_datagen import SessionChatCompletionAPIData
from inference_perf.datagen.replay.tracelab_trace_replay_datagen import TraceLabTraceReplayDataGenerator
from inference_perf.datagen.workload import WorkloadSessionGenerator
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from workload_fixtures import new_word_tokenizer, write_corpus


# One TraceLab line: rnd("claude:A", 0, 200, 0, 30, t0, t1) is round 0 of session claude:A,
# a 200-token prompt with nothing cached and 30 output tokens, arriving at t0 and done at t1.
def rnd(session: str, index: int, total: int, prefix: int, out: int, start: str, end: str) -> Dict[str, object]:
    return {
        "session_id": session,
        "round_index": index,
        "model": "claude-opus-4-8",
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


# Session A grows (round 1 reuses 192 of round 0's 200 tokens); session B
# compacts (round 1 keeps 64 of round 0's 300 tokens).
LINES = [
    rnd("claude:A", 0, 200, 0, 30, "2026-06-01T10:00:00.000Z", "2026-06-01T10:00:02.000Z"),
    rnd("claude:A", 1, 260, 192, 40, "2026-06-01T10:00:05.000Z", "2026-06-01T10:00:08.000Z"),
    rnd("claude:B", 0, 300, 0, 20, "2026-06-01T10:00:30.000Z", "2026-06-01T10:00:31.500Z"),
    rnd("claude:B", 1, 130, 64, 12, "2026-06-01T10:01:00.000Z", "2026-06-01T10:01:01.000Z"),
]


def make_generator(tmp_path: Path, corpus_path: Path, tokenizer: CustomTokenizer) -> WorkloadSessionGenerator:
    trace = tmp_path / "syfi_coding_trace.jsonl"
    trace.write_text("\n".join(json.dumps(line) for line in LINES) + "\n")
    data = DataConfig(
        type=DataGenType.WorkloadReplay,
        workload=WorkloadReplayConfig(format="TraceLab", file=str(trace), block_size=64),
        corpus_file_path=str(corpus_path),
    )
    return WorkloadSessionGenerator(APIConfig(type=APIType.Chat), data, tokenizer, base_seed=1)


def events(gen: WorkloadSessionGenerator, session_index: int) -> List[SessionChatCompletionAPIData]:
    assert gen.is_session_buildable(session_index)
    out = []
    for lazy in gen.get_session_events(session_index):
        data = gen.load_lazy_data(lazy)
        assert isinstance(data, SessionChatCompletionAPIData)
        out.append(data)
    return out


# Picked by name through the registry, the two-session trace gives the
# session generator two sessions. Session A's rounds are 200 and 260 tokens
# with max_tokens 30 and 40; round 1 depends on round 0 and waits the
# recorded 3000 ms think time; its prompt starts with round 0's first 192
# tokens (the 3 reused blocks of 64) and differs after them.
def test_tracelab_replays_through_the_registry(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    gen = make_generator(tmp_path, corpus_path, word_tokenizer)
    assert gen.get_session_count() == 2
    r0, r1 = events(gen, 0)
    assert word_tokenizer.count_tokens(str(r0.messages[0].content)) == 200 and r0.max_tokens == 30
    assert word_tokenizer.count_tokens(str(r1.messages[0].content)) == 260 and r1.max_tokens == 40
    assert r1.predecessor_event_ids == [r0.event_id] and r1.wait_ms == 3000
    a0 = str(r0.messages[0].content).split()
    a1 = str(r1.messages[0].content).split()
    assert a1[:192] == a0[:192]
    assert a1[192:200] != a0[192:200]


# Session B compacted: round 1 keeps only block 0 of round 0, so the two
# prompts share their first 64 tokens and nothing after.
def test_compaction_keeps_only_the_covered_blocks(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    gen = make_generator(tmp_path, corpus_path, word_tokenizer)
    b0, b1 = (str(e.messages[0].content).split() for e in events(gen, 1))
    assert b1[:64] == b0[:64]
    assert b1[64:128] != b0[64:128]


# The generator the tracelab_trace_replay block builds over `lines`, seeded
# with 1 like make_generator.
def make_block_generator(
    tmp_path: Path, corpus_path: Path, tokenizer: CustomTokenizer, lines: List[Dict[str, object]]
) -> TraceLabTraceReplayDataGenerator:
    trace = tmp_path / "block_trace.jsonl"
    trace.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    data = DataConfig(
        type=DataGenType.TraceLabTraceReplay,
        tracelab_trace_replay=TraceLabTraceReplayConfig(trace_files=[str(trace)]),
        corpus_file_path=str(corpus_path),
    )
    return TraceLabTraceReplayDataGenerator(APIConfig(type=APIType.Chat), data, tokenizer, base_seed=1)


# The tracelab_trace_replay block and the TraceLab format name the same
# sessions: for A and for B, both spellings send the same prompts and ask
# for the same lengths. Sessions are matched by the id they report as their
# source under the block, since the block runs them in its own order.
def test_the_block_and_the_format_send_the_same_requests(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    by_format = make_generator(tmp_path, corpus_path, word_tokenizer)
    by_block = make_block_generator(tmp_path, corpus_path, word_tokenizer, LINES)
    slot_of = {by_block.get_session_info(i)["source_id"]: i for i in range(by_block.get_session_count())}
    assert sorted(slot_of) == ["claude:A", "claude:B"]
    for format_slot, session in enumerate(("claude:A", "claude:B")):
        expected = [(str(e.messages[0].content), e.max_tokens) for e in events(by_format, format_slot)]
        sent = [(str(e.messages[0].content), e.max_tokens) for e in events(by_block, slot_of[session])]
        assert sent == expected and len(sent) == 2


# Six one-round sessions s0 to s5 run in a shuffle of their sorted ids seeded
# with the base seed (1 here), whatever order the file lists them in. A
# session's runtime id carries its place among the sorted ids, with no ':'.
def test_the_block_runs_sessions_in_a_seeded_shuffle(tmp_path: Path) -> None:
    lines = [rnd(f"s{i}", 0, 100, 0, 5, "2026-06-01T10:00:00.000Z", "2026-06-01T10:00:01.000Z") for i in (5, 4, 3, 2, 1, 0)]
    gen = make_block_generator(tmp_path, write_corpus(tmp_path), new_word_tokenizer(), lines)
    order = [gen.get_session_info(i)["source_id"] for i in range(gen.get_session_count())]
    assert order == ["s2", "s3", "s5", "s0", "s4", "s1"]
    assert [gen.get_session_info(i)["session_id"] for i in range(3)] == [
        "tracelabtrace2_s2",
        "tracelabtrace3_s3",
        "tracelabtrace5_s5",
    ]
