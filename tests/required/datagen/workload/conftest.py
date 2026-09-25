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
from typing import Any, Callable, List, cast

import pytest

from inference_perf.utils.custom_tokenizer import CustomTokenizer


# A word-level tokenizer with no HF dependency: token id i is the word "w{i}",
# encode splits on whitespace, decode joins with spaces, and count_tokens is
# the word count plus `bos` (0 by default) so a test can make the count
# disagree with the id list the way a real BOS token does.
class WordTokenizer:
    vocab_size = 2000
    all_special_ids: List[int] = []

    def __init__(self, bos: int = 0) -> None:
        self.bos = bos

    def has_chat_template(self) -> bool:
        return False

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return [int(w[1:]) for w in text.split()]

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"w{i}" for i in ids)

    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        return len(text.split()) + (self.bos if add_special_tokens else 0)

    def get_tokenizer(self) -> Any:
        return self


# A corpus of 2000 distinct words, written to disk so the materializer loads
# it the way it loads the default Shakespeare corpus.
@pytest.fixture
def corpus_path(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.txt"
    path.write_text(" ".join(f"w{i}" for i in range(2000)))
    return path


@pytest.fixture
def word_tokenizer() -> CustomTokenizer:
    return cast(CustomTokenizer, WordTokenizer())


# The same tokenizer with a chosen BOS count, for tests that need the token
# count and the id list to disagree.
@pytest.fixture
def make_word_tokenizer() -> Callable[[int], CustomTokenizer]:
    return lambda bos: cast(CustomTokenizer, WordTokenizer(bos=bos))
