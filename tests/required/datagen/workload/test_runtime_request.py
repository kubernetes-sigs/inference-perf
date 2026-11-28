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
from pydantic import ValidationError

from inference_perf.apis import ChatCompletionAPIData, CompletionAPIData, LazyLoadInferenceAPIData
from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType, WorkloadReplayConfig
from inference_perf.datagen.workload import Trace, WorkloadRequestDataGenerator
from inference_perf.loadgen.load_timer import RecordedOffsetsLoadTimer
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.workload import Delivery, Message, Record, Schedule, Session, TextPart, TextSpec
from workload_fixtures import new_word_tokenizer, write_corpus


# A session of one request, called `record_id`, that was sent at `start_ms`.
# It sends `messages` and asks for `out` output tokens.
def alone(record_id: str, messages: List[Message], out: int, start_ms: int) -> Session:
    delivery = Delivery(schedules={record_id: Schedule(start_ms=start_ms)})
    record = Record(id=record_id, messages=messages, output_tokens=out)
    return Session(id=record_id, records=[record], delivery=delivery)


# Three timestamped independent requests, the Mooncake shape: two share
# their first two 16-token blocks, the third is literal text. They are
# listed out of send order.
def timed_trace() -> Trace:
    sessions = [
        alone(
            "1",
            [Message(role="user", parts=[TextPart(spec=TextSpec(num_tokens=36, block_ids=[1, 2, 9], block_size=16))])],
            3,
            500,
        ),
        alone(
            "0",
            [Message(role="user", parts=[TextPart(spec=TextSpec(num_tokens=40, block_ids=[1, 2, 3], block_size=16))])],
            7,
            0,
        ),
        alone(
            "2",
            [
                Message(role="system", parts=[TextPart(text="w1 w2")]),
                Message(role="user", parts=[TextPart(text="w3 w4 w5")]),
            ],
            2,
            900,
        ),
    ]
    return Trace(source_id="trace.jsonl", sessions=sessions, records_sessions=False)


def data_config(corpus_path: Path) -> DataConfig:
    return DataConfig(
        type=DataGenType.WorkloadReplay,
        workload=WorkloadReplayConfig(format="test", trace_files=["unused"]),
        corpus_file_path=str(corpus_path),
    )


def make_generator(api: APIType, corpus_path: Path, tokenizer: CustomTokenizer, seed: int = 1) -> WorkloadRequestDataGenerator:
    return WorkloadRequestDataGenerator(
        APIConfig(type=api), data_config(corpus_path), tokenizer, seed=seed, trace=timed_trace()
    )


def prompt(gen: WorkloadRequestDataGenerator, n: int) -> CompletionAPIData:
    data = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=n))
    assert isinstance(data, CompletionAPIData)
    return data


# Requests are served in the order they were sent (0 ms, 500, 900), not the
# order the trace lists them, and the offsets the load timer gets follow
# that order.
def test_requests_are_ordered_by_send_time(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    gen = make_generator(APIType.Completion, corpus_path, word_tokenizer)
    assert gen.get_request_count() == 3
    assert gen.send_offsets_ms() == [0, 500, 900]
    assert [prompt(gen, n).max_tokens for n in range(3)] == [7, 3, 2]


# Each request names the record it was built from, on the completion and the
# chat path alike, so its metric can be matched to what the trace recorded.
# In send order (0, 500 and 900 ms) those are records "0", "1" and "2".
def test_a_request_carries_its_record_id(tmp_path: Path) -> None:
    corpus_path = write_corpus(tmp_path)
    for api in (APIType.Completion, APIType.Chat):
        gen = make_generator(api, corpus_path, new_word_tokenizer())
        loaded = [gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=n)).record_id for n in range(3)]
        assert loaded == ["0", "1", "2"]


# On the completion path each synthetic prompt tokenizes to its declared
# length with max_tokens from its output_tokens; the two requests that
# share blocks [1, 2] share their first 32 words. The text record's prompt
# is its system and user messages joined by a newline.
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


# get_data cycles through the requests, so a rate stage longer than the
# trace wraps around: the fourth item is request index 0 again.
def test_get_data_cycles(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    gen = make_generator(APIType.Completion, corpus_path, word_tokenizer)
    stream = gen.get_data()
    indices = [next(stream) for _ in range(4)]
    assert [d.data_index for d in indices if isinstance(d, LazyLoadInferenceAPIData)] == [0, 1, 2, 0]


# A request with no recorded send time leaves the trace with no offsets to
# hand the load timer.
def test_a_request_without_a_send_time_gives_no_offsets(tmp_path: Path) -> None:
    trace = timed_trace()
    untimed = Session(id="3", records=[trace.sessions[0].records[0].model_copy(update={"id": "3"})])
    gen = WorkloadRequestDataGenerator(
        APIConfig(type=APIType.Completion),
        data_config(write_corpus(tmp_path)),
        new_word_tokenizer(),
        trace=Trace(source_id="t", sessions=[*trace.sessions, untimed], records_sessions=False),
    )
    assert gen.send_offsets_ms() is None


# A trace that records sessions is refused: it needs the session runner.
def test_a_trace_of_sessions_is_refused(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    sessions = Trace(source_id="t", sessions=timed_trace().sessions, records_sessions=True)
    with pytest.raises(ValueError, match="trace_session_replay"):
        WorkloadRequestDataGenerator(
            APIConfig(type=APIType.Completion), data_config(corpus_path), word_tokenizer, trace=sessions
        )


# The timer yields the start time plus each recorded offset: 0, 0.5 and 0.9
# seconds after a start of 100.
def test_recorded_offsets_timer() -> None:
    assert list(RecordedOffsetsLoadTimer([0, 500, 900]).start_timer(initial=100.0)) == [100.0, 100.5, 100.9]


# data.type workload_replay without data.workload is a config error, and
# data.workload needs a format and exactly one place to read from.
def test_workload_replay_config_requirements() -> None:
    with pytest.raises(ValidationError, match="data.workload"):
        DataConfig(type=DataGenType.WorkloadReplay)
    with pytest.raises(ValidationError, match="must be provided"):
        WorkloadReplayConfig.model_validate({"format": "Mooncake"})
    with pytest.raises(ValidationError, match="format"):
        WorkloadReplayConfig.model_validate({"trace_files": ["t.jsonl"]})
    config = DataConfig(
        type=DataGenType.WorkloadReplay, workload=WorkloadReplayConfig(format="Mooncake", trace_files=["t.jsonl"])
    )
    assert config.workload is not None and config.workload.block_size is None
    assert config.workload.num_dataset_entries is None and config.workload.trace_idle_gap_cap_seconds is None
