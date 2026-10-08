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

import pytest

from inference_perf.apis import ChatCompletionAPIData, CompletionAPIData, LazyLoadInferenceAPIData
from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType, WorkloadReplayConfig
from inference_perf.datagen.workload import WorkloadRequestDataGenerator
from inference_perf.loadgen.load_timer import ArrangementLoadTimer
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.workload import Arrangement, Node, Record, SyntheticPart, TextPart, Turn
from inference_perf.workload.sources import Workload
from workload_fixtures import new_word_tokenizer, write_corpus


# Three timestamped independent requests, the Mooncake shape: two share
# their first two 16-token blocks, the third is literal text.
def timed_workload() -> Workload:
    records = [
        Record(
            id="0",
            turns=[
                Turn(role="user", parts=[SyntheticPart(num_tokens=40, block_ids=[1, 2, 3], block_size=16)]),
                Turn(role="assistant", output_tokens=7),
            ],
        ),
        Record(
            id="1",
            turns=[
                Turn(role="user", parts=[SyntheticPart(num_tokens=36, block_ids=[1, 2, 9], block_size=16)]),
                Turn(role="assistant", output_tokens=3),
            ],
        ),
        Record(
            id="2",
            turns=[
                Turn(role="system", parts=[TextPart(text="w1 w2")]),
                Turn(role="user", parts=[TextPart(text="w3 w4 w5")]),
                Turn(role="assistant", output_tokens=2),
            ],
        ),
    ]
    nodes = [
        Node(id="b", record_id="1", turn=1, send_at_ms=500),
        Node(id="a", record_id="0", turn=1, send_at_ms=0),
        Node(id="c", record_id="2", turn=2, send_at_ms=900),
    ]
    return Workload(source_id="trace.jsonl", records=records, arrangement=Arrangement(nodes=nodes))


def data_config(corpus_path: Path) -> DataConfig:
    return DataConfig(
        type=DataGenType.WorkloadReplay,
        workload=WorkloadReplayConfig(format="test", file="unused"),
        corpus_file_path=str(corpus_path),
    )


def make_generator(api: APIType, corpus_path: Path, tokenizer: CustomTokenizer, seed: int = 1) -> WorkloadRequestDataGenerator:
    return WorkloadRequestDataGenerator(
        APIConfig(type=api), data_config(corpus_path), tokenizer, seed=seed, workload=timed_workload()
    )


def prompt(gen: WorkloadRequestDataGenerator, n: int) -> CompletionAPIData:
    data = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=n))
    assert isinstance(data, CompletionAPIData)
    return data


# Nodes are served in send-time order (a at 0 ms, b at 500, c at 900), not
# arrangement order, and the offsets the load timer gets follow that order.
def test_nodes_are_ordered_by_send_time(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    gen = make_generator(APIType.Completion, corpus_path, word_tokenizer)
    assert gen.get_request_count() == 3
    assert gen.send_offsets_ms() == [0, 500, 900]
    assert [prompt(gen, n).max_tokens for n in range(3)] == [7, 3, 2]


# On the completion path each synthetic prompt tokenizes to its declared
# length with max_tokens from the assistant turn; the two requests that
# share blocks [1, 2] share their first 32 words. The text record's prompt
# is its system and user turns joined by a newline.
def test_completion_prompts(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    gen = make_generator(APIType.Completion, corpus_path, word_tokenizer)
    a, b, c = (prompt(gen, n) for n in range(3))
    assert word_tokenizer.count_tokens(a.prompt) == 40 and a.max_tokens == 7
    assert word_tokenizer.count_tokens(b.prompt) == 36 and b.max_tokens == 3
    assert a.prompt.split()[:32] == b.prompt.split()[:32]
    assert a.prompt.split()[32:36] != b.prompt.split()[32:36]
    assert c.prompt == "w1 w2\nw3 w4 w5" and c.max_tokens == 2


# On the chat path the text record becomes two messages with their roles,
# and a synthetic record becomes one user message of the declared length.
def test_chat_messages(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    gen = make_generator(APIType.Chat, corpus_path, word_tokenizer)
    c = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=2))
    assert isinstance(c, ChatCompletionAPIData)
    assert [(m.role, m.content) for m in c.messages] == [("system", "w1 w2"), ("user", "w3 w4 w5")]
    a = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0))
    assert isinstance(a, ChatCompletionAPIData)
    assert len(a.messages) == 1 and a.messages[0].role == "user"
    assert word_tokenizer.count_tokens(str(a.messages[0].content)) == 40


# get_data cycles through the nodes, so a rate stage longer than the trace
# wraps around: the fourth item is node index 0 again.
def test_get_data_cycles(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    gen = make_generator(APIType.Completion, corpus_path, word_tokenizer)
    stream = gen.get_data()
    indices = [next(stream) for _ in range(4)]
    assert [d.data_index for d in indices if isinstance(d, LazyLoadInferenceAPIData)] == [0, 1, 2, 0]


# An arrangement with a dependency is refused: it needs the session runner.
def test_dependencies_are_refused(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    workload = timed_workload()
    chained = Workload(
        source_id="t",
        records=workload.records,
        arrangement=Arrangement(
            nodes=[Node(id="a", record_id="0", turn=1), Node(id="b", record_id="1", turn=1, depends_on=["a"])]
        ),
    )
    with pytest.raises(ValueError, match="trace_session_replay"):
        WorkloadRequestDataGenerator(
            APIConfig(type=APIType.Completion), data_config(corpus_path), word_tokenizer, workload=chained
        )


# The arrangement timer yields the start time plus each offset: 0, 0.5 and
# 0.9 seconds after a start of 100.
def test_arrangement_timer_offsets() -> None:
    assert list(ArrangementLoadTimer([0, 500, 900]).start_timer(initial=100.0)) == [100.0, 100.5, 100.9]
