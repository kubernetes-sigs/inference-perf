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

from inference_perf.workload import (
    BlockTextMaterializer,
    ImagePart,
    Message,
    Prompter,
    ReasoningPart,
    Record,
    TextPart,
    TextSpec,
    ToolCallPart,
    ToolResultPart,
)
from workload_fixtures import new_word_tokenizer, write_corpus


# Two materializers built with the same seed give the same 16 tokens for
# block 7 of scope "trace-a"; the same block id under scope "trace-b", or the
# same scope under a different seed, gives different tokens.
def test_block_tokens_are_a_function_of_seed_scope_and_id(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    a = BlockTextMaterializer(word_tokenizer.get_tokenizer(), base_seed=1, corpus_path=corpus_path)
    b = BlockTextMaterializer(word_tokenizer.get_tokenizer(), base_seed=1, corpus_path=corpus_path)
    assert a.block_tokens("trace-a", 7, 16) == b.block_tokens("trace-a", 7, 16)
    assert len(a.block_tokens("trace-a", 7, 16)) == 16
    assert a.block_tokens("trace-a", 7, 16) != a.block_tokens("trace-b", 7, 16)
    other_seed = BlockTextMaterializer(word_tokenizer.get_tokenizer(), base_seed=2, corpus_path=corpus_path)
    assert a.block_tokens("trace-a", 7, 16) != other_seed.block_tokens("trace-a", 7, 16)


# Two parts of 100 tokens whose block ids start [1, 2] and then differ
# ([1, 2, 3] vs [1, 2, 4], block size 32) materialize to texts whose first
# 64 tokens are identical and whose next 32 differ. Their unique tails, past
# the 96 covered tokens, differ too because they are keyed by record id.
def test_shared_leading_blocks_give_a_shared_prefix(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    m = BlockTextMaterializer(word_tokenizer.get_tokenizer(), base_seed=3, corpus_path=corpus_path)
    x = m.materialize(TextSpec(num_tokens=100, block_ids=[1, 2, 3], block_size=32), "t", "rec-x")
    y = m.materialize(TextSpec(num_tokens=100, block_ids=[1, 2, 4], block_size=32), "t", "rec-y")
    assert len(x.token_ids) == len(y.token_ids) == 100
    assert x.prefix_tokens == y.prefix_tokens == 96
    assert x.token_ids[:64] == y.token_ids[:64]
    assert x.token_ids[64:96] != y.token_ids[64:96]
    assert x.token_ids[96:] != y.token_ids[96:]
    assert word_tokenizer.count_tokens(x.text) == 100


# With a tokenizer that counts one extra token (a BOS) the text has to be one
# word shorter than the target to count as exactly 50. The trim comes off the
# unique tail: the 32 prefix tokens of block 9 are untouched.
def test_exact_count_is_landed_by_trimming_the_tail(tmp_path: Path) -> None:
    corpus_path = write_corpus(tmp_path)
    make_word_tokenizer = new_word_tokenizer
    tok = make_word_tokenizer(1)
    m = BlockTextMaterializer(tok.get_tokenizer(), base_seed=3, corpus_path=corpus_path)
    out = m.materialize(TextSpec(num_tokens=50, block_ids=[9], block_size=32), "t", "r")
    assert tok.count_tokens(out.text) == 50
    assert len(out.token_ids) == 49
    assert out.token_ids[:32] == m.block_tokens("t", 9, 32)
    assert out.prefix_tokens == 32


# A second pass with salt "pass-2" produces different tokens for block 5 than
# the unsalted pass, so replaying the trace twice does not re-hit the cache.
def test_salt_changes_block_text(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    first = BlockTextMaterializer(word_tokenizer.get_tokenizer(), base_seed=3, corpus_path=corpus_path)
    second = BlockTextMaterializer(word_tokenizer.get_tokenizer(), base_seed=3, corpus_path=corpus_path, salt="pass-2")
    assert first.block_tokens("t", 5, 16) != second.block_tokens("t", 5, 16)


# The tail for record "a" is the same every time and differs from the tail
# for record "b"; a zero-token part materializes to nothing at all.
def test_tails_and_empty_parts(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    corpus_path = write_corpus(tmp_path)
    m = BlockTextMaterializer(word_tokenizer.get_tokenizer(), base_seed=3, corpus_path=corpus_path)
    assert m.tail_tokens(10, "a") == m.tail_tokens(10, "a")
    assert m.tail_tokens(10, "a") != m.tail_tokens(10, "b")
    assert m.tail_tokens(0, "a") == []
    empty = m.materialize(TextSpec(num_tokens=0, block_size=32), "t", "r")
    assert empty.text == "" and empty.token_ids == [] and empty.prefix_tokens == 0


# A corpus of 5 words still yields a 12-token block: the slice wraps around
# the corpus until it is long enough.
def test_block_longer_than_corpus_wraps(tmp_path: Path) -> None:
    word_tokenizer = new_word_tokenizer()
    small = tmp_path / "small.txt"
    small.write_text("w0 w1 w2 w3 w4")
    m = BlockTextMaterializer(word_tokenizer.get_tokenizer(), base_seed=3, corpus_path=small)
    block = m.block_tokens("t", 1, 12)
    assert len(block) == 12
    assert set(block) <= {0, 1, 2, 3, 4}


# A message's text is its text parts, a tool call's arguments and a tool
# result's text content, in order, joined by newlines. A tool call with no
# arguments adds nothing. An image is not sent yet, on its own or inside a
# tool result, and says so. Neither is the model's thinking.
def test_tool_parts_are_sent_as_text_and_media_is_refused(tmp_path: Path) -> None:
    tokenizer = new_word_tokenizer().get_tokenizer()
    prompter = Prompter("t", BlockTextMaterializer(tokenizer, base_seed=1, corpus_path=write_corpus(tmp_path)))
    parts = [
        TextPart(text="w1"),
        ToolCallPart(name="look", arguments=TextPart(text="w2 w3")),
        ToolCallPart(name="look"),
        ToolResultPart(content=[TextPart(text="w4"), TextPart(text="w5")]),
    ]
    record = Record(id="r", messages=[Message(role="user", parts=parts)], reply=[ToolCallPart(arguments=TextPart(text="w6"))])
    (message,) = prompter.prompt(record, "s")
    assert message.text == "w1\nw2 w3\nw4\nw5"
    assert prompter.reply(record, "s").text == "w6"
    image = ImagePart(ref="file:///cat.png")
    thought = ReasoningPart(content=[TextPart(text="w1")])
    for refused in ([image], [ToolResultPart(content=[TextPart(text="w1"), image])], [thought]):
        with pytest.raises(NotImplementedError, match="parts are not sent"):
            prompter.prompt(Record(id="r", messages=[Message(role="user", parts=refused)]), "s")
