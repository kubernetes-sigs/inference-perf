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
"""Helpers shared by the workload tests, imported by name like the other
test helpers in this tree (a second conftest.py would collide with
tests/optional's under mypy)."""

import tempfile
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from inference_perf.utils.custom_tokenizer import CustomTokenizer


# A word-level tokenizer with no Hub download: token id i is the word "w{i}"
# for i < 2000, encode splits on whitespace, decode joins with spaces, and
# count_tokens is the word count plus `bos` (0 by default) so a test can make
# the count disagree with the id list the way a real BOS token does.
class WordTokenizer(CustomTokenizer):
    def __init__(self, bos: int = 0) -> None:
        # skips CustomTokenizer.__init__, which loads from the Hub
        words = Tokenizer(WordLevel({f"w{i}": i for i in range(2000)}, unk_token="w0"))
        words.pre_tokenizer = WhitespaceSplit()
        # the constructor is untyped, so load through from_pretrained like a real tokenizer
        with tempfile.TemporaryDirectory() as directory:
            words.save(str(Path(directory) / "tokenizer.json"))
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(directory)
        self.bos = bos

    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        return len(text.split()) + (self.bos if add_special_tokens else 0)


# new_word_tokenizer() counts "w1 w2" as 2 tokens and encodes it as [1, 2].
# new_word_tokenizer(bos=1) counts it as 3 and still encodes it as [1, 2].
def new_word_tokenizer(bos: int = 0) -> CustomTokenizer:
    return WordTokenizer(bos=bos)


# A corpus of 2000 distinct words, written to disk so the materializer loads
# it the way it loads the default Shakespeare corpus.
def write_corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.txt"
    path.write_text(" ".join(f"w{i}" for i in range(2000)))
    return path
