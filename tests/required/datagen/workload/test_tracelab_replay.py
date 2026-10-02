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
from typing import Any, Dict, List

from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType, WorkloadReplayConfig
from inference_perf.datagen.replay.replay_graph_session_datagen import SessionChatCompletionAPIData
from inference_perf.datagen.workload import WorkloadSessionGenerator
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from workload_fixtures import new_word_tokenizer, write_corpus


def rnd(session: str, index: int, total: int, prefix: int, out: int, start: str, end: str) -> Dict[str, Any]:
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
