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

from inference_perf.apis import CompletionAPIData, LazyLoadInferenceAPIData
from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType, WorkloadReplayConfig
from inference_perf.datagen.workload import WorkloadRequestDataGenerator
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from workload_fixtures import new_word_tokenizer, write_corpus

# Two requests sharing their first 12 of 14 blocks, and one with no blocks.
# Block size is 8 here so the prompts stay small: the shared prefix is
# 12 * 8 = 96 tokens.
LINES = [
    {"timestamp": 0, "input_length": 120, "output_length": 7, "hash_ids": list(range(12)) + [100, 101]},
    {"timestamp": 500, "input_length": 110, "output_length": 3, "hash_ids": list(range(12)) + [200, 201]},
    {"timestamp": 900, "input_length": 9, "output_length": 2, "hash_ids": []},
]


def make_generator(
    tmp_path: Path, corpus_path: Path, tokenizer: CustomTokenizer, seed: int = 1
) -> WorkloadRequestDataGenerator:
    trace = tmp_path / "trace.jsonl"
    trace.write_text("\n".join(json.dumps(line) for line in LINES) + "\n")
    data = DataConfig(
        type=DataGenType.WorkloadReplay,
        workload=WorkloadReplayConfig(format="Mooncake", file=str(trace), block_size=8),
        corpus_file_path=str(corpus_path),
    )
    return WorkloadRequestDataGenerator(APIConfig(type=APIType.Completion), data, tokenizer, seed=seed)


def prompt(gen: WorkloadRequestDataGenerator, n: int) -> CompletionAPIData:
    data = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=n))
    assert isinstance(data, CompletionAPIData)
    return data


# Picked by name through the registry, a Mooncake trace gives the request
# generator three timestamped requests whose prompts tokenize to the
# recorded input_length with max_tokens set to output_length.
def test_mooncake_replays_through_the_registry(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    gen = make_generator(tmp_path, corpus_path, word_tokenizer)
    assert gen.get_request_count() == 3
    assert gen.send_offsets_ms() == [0, 500, 900]
    for n, line in enumerate(LINES):
        data = prompt(gen, n)
        assert word_tokenizer.count_tokens(data.prompt) == line["input_length"]
        assert data.max_tokens == line["output_length"]


# Requests 0 and 1 share 12 leading blocks of 8, so their prompts start
# with the same 96 words and differ from word 97 on; request 2 shares
# nothing. The same seed gives the same prompts in a second generator.
def test_shared_blocks_give_a_shared_prompt_prefix(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    gen = make_generator(tmp_path, corpus_path, word_tokenizer)
    a, b, c = (prompt(gen, n).prompt.split() for n in range(3))
    assert a[:96] == b[:96]
    assert a[96:104] != b[96:104]
    assert c[:9] != a[:9]
    again = make_generator(tmp_path, corpus_path, word_tokenizer)
    assert prompt(again, 0).prompt.split() == a
    other = make_generator(tmp_path, corpus_path, word_tokenizer, seed=2)
    assert prompt(other, 0).prompt.split() != a
