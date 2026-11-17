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
"""The Weka format on the record layer builds the replay graph the Weka generator builds."""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType, WorkloadReplayConfig
from inference_perf.config.datagen.replay import WekaTraceReplayConfig
from inference_perf.datagen.replay.replay_graph_types import ReplayGraph
from inference_perf.datagen.replay.weka_trace_replay_datagen import WekaTraceReplayDataGenerator
from inference_perf.datagen.workload import WorkloadSessionGenerator, load_trace
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from workload_fixtures import new_word_tokenizer, write_corpus


# call(t, n_in, n_out, hash_ids) is one Weka request sent at t seconds that took 0.5 s.
def call(t: float, n_in: int, n_out: int, hash_ids: List[int]) -> Dict[str, object]:
    return {"t": t, "type": "n", "model": "m", "in": n_in, "out": n_out, "hash_ids": hash_ids, "api_time": 0.5}


# One session, block size 4, with a one-block system prompt and every shape
# the two paths have to agree on: partial tails (10, 22, 34 and 26 tokens), a
# subagent that makes two calls between rounds 0 and 1, a round that recorded
# no output (round 1), a reply of two blocks (round 2's, blocks 10 and 11) and
# a last round that only rewinds into the middle of that reply.
TRACE: Dict[str, object] = {
    "id": "T",
    "models": ["m"],
    "block_size": 4,
    "tool_tokens": 4,
    "system_tokens": 0,
    "requests": [
        call(0.0, 10, 4, [1, 2]),
        {
            "t": 0.8,
            "type": "subagent",
            "agent_id": "sa1",
            "subagent_type": "worker",
            "duration_ms": 900,
            "tool_tokens": 0,
            "system_tokens": 0,
            "requests": [call(0.8, 8, 2, [20, 21]), call(1.4, 12, 3, [20, 21, 22])],
        },
        call(2.0, 22, 0, [1, 2, 3, 4, 5]),
        call(4.0, 20, 8, [1, 2, 3, 4, 9]),
        call(6.0, 34, 4, [1, 2, 3, 4, 9, 10, 11, 12]),
        call(8.0, 26, 4, [1, 2, 3, 4, 9, 10]),
    ],
}


# What the scheduler and the request builder read off one event of a graph:
# the calls it waits for, the wait, the output and input lengths, the role
# and token count of every message, and how the prompt splits into shared,
# live-output and unique messages.
@dataclass
class Seen:
    after: List[str]
    wait_ms: int
    out: int
    total_in: int
    messages: List[Tuple[str, int]]
    segments: List[Tuple[str, int, Optional[str]]]


# Every event of a graph as a Seen, keyed by call id.
def view(graph: ReplayGraph, tokenizer: CustomTokenizer) -> Dict[str, Seen]:
    call_of = {event.event_id: event.call.call_id.split("/")[-1] for event in graph.events.values()}
    seen: Dict[str, Seen] = {}
    for event in graph.events.values():
        made = event.call
        seen[call_of[event.event_id]] = Seen(
            after=sorted(call_of[p] for p in event.predecessor_event_ids),
            wait_ms=event.wait_ms,
            out=made.expected_output_tokens,
            total_in=made.total_input_tokens,
            messages=[(m["role"], tokenizer.count_tokens(m["content"], add_special_tokens=False)) for m in made.messages],
            segments=[
                (s.type, s.message_count, call_of[s.source_event_id] if s.source_event_id else None)
                for s in made.input_segments
            ],
        )
    return seen


# The trace above through weka_trace_replay and through the Weka format on
# the record layer, same tokenizer, corpus and seed, with the old path's idle
# gap cap raised out of the way. Both graphs have the same seven calls, and
# every call waits for the same calls for the same time, asks for the same
# output and sends messages of the same roles, lengths and live-reply slots.
def test_the_record_layer_builds_the_weka_generators_graph(tmp_path: Path) -> None:
    trace_file = tmp_path / "T.json"
    trace_file.write_text(json.dumps(TRACE))
    corpus = str(write_corpus(tmp_path))
    tokenizer = new_word_tokenizer()
    api = APIConfig(type=APIType.Chat)

    old_block = WekaTraceReplayConfig(trace_files=[str(trace_file)], datagen_workers=1, trace_idle_gap_cap_seconds=1e9)
    old_config = DataConfig(type=DataGenType.WekaTraceReplay, weka_trace_replay=old_block, corpus_file_path=corpus)
    old = WekaTraceReplayDataGenerator(api, old_config, tokenizer, base_seed=1, num_workers=1)

    new_block = WorkloadReplayConfig(format="Weka", trace_files=[str(trace_file)])
    new_config = DataConfig(type=DataGenType.WorkloadReplay, workload=new_block, corpus_file_path=corpus)
    new = WorkloadSessionGenerator(api, new_config, tokenizer, base_seed=1, trace=load_trace("Weka", new_block, 1))

    old_graph, new_graph = old._get_session(0).graph, new._get_session(0).graph
    expected = view(old_graph, tokenizer)
    assert len(expected) == 7
    assert view(new_graph, tokenizer) == expected
    # the fixture does exercise the shapes it claims to: the subagent joins
    # round 1, the round with no output is waited for alone, and what the
    # rewind left of round 2's reply goes out as round 3's live reply
    assert expected["parent_turn_1"].after == ["parent_turn_0", "sa_sa1_s0_turn_1"]
    assert expected["parent_turn_2"].after == ["parent_turn_1"]
    assert expected["parent_turn_4"].segments[1] == ("output", 1, "parent_turn_3")


# The trace above with a subagent "sa2" nested inside "sa1", holding one call.
# The two paths differ here on purpose: weka_trace_replay refuses the trace
# and names it and the subagent, and the record layer reads all eight calls.
def test_only_the_record_layer_reads_a_subagent_inside_a_subagent(tmp_path: Path) -> None:
    nested = {"t": 1.0, "type": "subagent", "agent_id": "sa2", "subagent_type": "worker", "requests": [call(1.0, 4, 1, [30])]}
    trace = copy.deepcopy(TRACE)
    requests = trace["requests"]
    assert isinstance(requests, list) and isinstance(requests[1], dict)
    requests[1]["requests"].append(nested)
    trace_file = tmp_path / "T.json"
    trace_file.write_text(json.dumps(trace))
    corpus = str(write_corpus(tmp_path))
    api = APIConfig(type=APIType.Chat)

    old_block = WekaTraceReplayConfig(trace_files=[str(trace_file)], datagen_workers=1, skip_invalid_files=False)
    old_config = DataConfig(type=DataGenType.WekaTraceReplay, weka_trace_replay=old_block, corpus_file_path=corpus)
    with pytest.raises(Exception, match="trace T: subagent sa1 spawns subagents"):
        WekaTraceReplayDataGenerator(api, old_config, new_word_tokenizer(), base_seed=1, num_workers=1)

    new_block = WorkloadReplayConfig(format="Weka", trace_files=[str(trace_file)])
    (session,) = load_trace("Weka", new_block, 1).sessions
    assert len(session.records) == 8
