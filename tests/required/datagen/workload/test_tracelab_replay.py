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


def write_trace(tmp_path: Path, name: str, lines: List[Dict[str, object]]) -> Path:
    trace = tmp_path / name
    trace.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return trace


# The generator `data.type: workload_replay` builds with `format: TraceLab`
# over LINES, seeded with 1.
def make_generator(tmp_path: Path, corpus_path: Path, tokenizer: CustomTokenizer) -> WorkloadSessionGenerator:
    trace = write_trace(tmp_path, "syfi_coding_trace.jsonl", LINES)
    data = DataConfig(
        type=DataGenType.WorkloadReplay,
        workload=WorkloadReplayConfig(format="TraceLab", trace_files=[str(trace)], block_size=64),
        corpus_file_path=str(corpus_path),
    )
    return WorkloadSessionGenerator(APIConfig(type=APIType.Chat), data, tokenizer, base_seed=1)


# The generator the tracelab_trace_replay block builds over `lines`, seeded
# with 1 like make_generator.
def make_block_generator(
    tmp_path: Path, corpus_path: Path, tokenizer: CustomTokenizer, lines: List[Dict[str, object]]
) -> TraceLabTraceReplayDataGenerator:
    trace = write_trace(tmp_path, "block_trace.jsonl", lines)
    data = DataConfig(
        type=DataGenType.TraceLabTraceReplay,
        tracelab_trace_replay=TraceLabTraceReplayConfig(trace_files=[str(trace)]),
        corpus_file_path=str(corpus_path),
    )
    return TraceLabTraceReplayDataGenerator(APIConfig(type=APIType.Chat), data, tokenizer, base_seed=1)


# The requests of the session a trace calls `session`, whichever slot it runs in.
def events(gen: WorkloadSessionGenerator, session: str) -> List[SessionChatCompletionAPIData]:
    slots = [i for i in range(gen.get_session_count()) if gen.get_session_info(i)["source_id"] == session]
    assert len(slots) == 1, f"expected one session {session!r}, found {len(slots)}"
    assert gen.is_session_buildable(slots[0])
    out = []
    for lazy in gen.get_session_events(slots[0]):
        data = gen.load_lazy_data(lazy)
        assert isinstance(data, SessionChatCompletionAPIData)
        out.append(data)
    return out


# Picked by name, the two-session trace gives the session generator two
# sessions. Session A's rounds are 200 and 260 tokens with max_tokens 30 and
# 40; round 1 depends on round 0 and waits the recorded 3000 ms think time;
# its prompt starts with round 0's first 192 tokens (the 3 reused blocks of
# 64) and differs after them.
def test_tracelab_replays_by_format_name(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    gen = make_generator(tmp_path, write_corpus(tmp_path), word_tokenizer)
    assert gen.get_session_count() == 2
    r0, r1 = events(gen, "claude:A")
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
    gen = make_generator(tmp_path, write_corpus(tmp_path), new_word_tokenizer())
    b0, b1 = (str(e.messages[0].content).split() for e in events(gen, "claude:B"))
    assert b1[:64] == b0[:64]
    assert b1[64:128] != b0[64:128]


# The tracelab_trace_replay block and the TraceLab format name are one
# implementation: for A and for B, both spellings send the same prompts,
# ask for the same lengths, and run the sessions in the same slots.
def test_the_block_and_the_format_send_the_same_requests(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    by_format = make_generator(tmp_path, corpus_path, word_tokenizer)
    by_block = make_block_generator(tmp_path, corpus_path, word_tokenizer, LINES)
    for session in ("claude:A", "claude:B"):
        expected = [(str(e.messages[0].content), e.max_tokens) for e in events(by_format, session)]
        sent = [(str(e.messages[0].content), e.max_tokens) for e in events(by_block, session)]
        assert sent == expected and len(sent) == 2
    slots = [[gen.get_session_info(i)["session_id"] for i in range(2)] for gen in (by_format, by_block)]
    assert slots[0] == slots[1]


# The block's own defaults differ from the bare format's: B's round 1
# recorded a 28500 ms wait, which the block replays in full (its 60 s gap
# cap raises the 15 s runtime clamp) and the format name clamps to 15000.
def test_the_block_raises_the_wait_clamp_to_its_gap_cap(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    by_block = make_block_generator(tmp_path, corpus_path, word_tokenizer, LINES)
    by_format = make_generator(tmp_path, corpus_path, word_tokenizer)
    assert events(by_block, "claude:B")[1].wait_ms == 28500
    assert events(by_format, "claude:B")[1].wait_ms == 15000


# Six one-round sessions s0 to s5 run in a shuffle of their sorted ids seeded
# with the base seed (1 here), whatever order the file lists them in. A
# session's runtime id is its slot and its id, with no ':'.
def test_sessions_run_in_a_seeded_shuffle(tmp_path: Path) -> None:
    lines = [rnd(f"s{i}", 0, 100, 0, 5, "2026-06-01T10:00:00.000Z", "2026-06-01T10:00:01.000Z") for i in (5, 4, 3, 2, 1, 0)]
    gen = make_block_generator(tmp_path, write_corpus(tmp_path), new_word_tokenizer(), lines)
    order = [gen.get_session_info(i)["source_id"] for i in range(gen.get_session_count())]
    assert order == ["s2", "s3", "s5", "s0", "s4", "s1"]
    assert [gen.get_session_info(i)["session_id"] for i in range(3)] == ["s0_s2", "s1_s3", "s2_s5"]
