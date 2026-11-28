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
"""From a record to the messages that go on the wire.

This is the one place a record's content rules are applied, for every format
and every API: the prompt of a request is its messages, with every text spec
built under the scope the spec asked for.

The text for a prefix block is a slice of a tokenized corpus, and where that
slice starts is decided by hashing (base seed, salt, scope key, block id). So
the same block id in the same scope always decodes to the same tokens, on any
worker and in any run with the same seed, which is what makes a recorded
prefix hit reproducible. The tail past the named blocks is seeded from the
record, so it is stable but unique.

Blocks are cached per (scope, id, size), so a trace with heavy prefix reuse
tokenizes each shared block once, not once per request.

The `salt` is for repeat passes over the same trace: a different salt gives
different block text for the same ids, so a second pass does not report cache
hits the server earned on the first one.

The tokenizer is anything that encodes and decodes: a Hugging Face tokenizer
is one as it stands.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, Union

from inference_perf.workload.record import Part, Record, TextPart, TextSpec, ToolCallPart, ToolResultPart

logger = logging.getLogger(__name__)

DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / "assets" / "shakespeare.txt"

# How many times a built text is re-measured and adjusted before giving up.
_MAX_ATTEMPTS = 20


class Tokenizer(Protocol):
    """The two calls text building needs."""

    def encode(self, text: str, *, add_special_tokens: bool = ...) -> List[int]: ...

    def decode(self, token_ids: List[int], *, skip_special_tokens: bool = ...) -> Union[str, List[str]]: ...


@dataclass
class MaterializedText:
    text: str
    token_ids: List[int]
    # Tokens at the front that came from named prefix blocks: what the
    # part declared as shareable, before any chat template.
    prefix_tokens: int


def _seed_offset(key: str, modulus: int) -> int:
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % max(modulus, 1)


class BlockTextMaterializer:
    def __init__(self, tokenizer: Tokenizer, base_seed: int, corpus_path: Optional[Path] = None, salt: str = "") -> None:
        self.tokenizer = tokenizer
        self.base_seed = base_seed
        self.salt = salt
        path = corpus_path if corpus_path is not None else DEFAULT_CORPUS
        if not path.is_file():
            raise FileNotFoundError(f"Prompt corpus file not found: {path}")
        text = path.read_text(encoding="utf-8")
        self._corpus: List[int] = list(tokenizer.encode(text, add_special_tokens=False))
        if not self._corpus:
            raise ValueError(f"Prompt corpus {path} tokenized to nothing")
        logger.info(f"Loaded prompt corpus from {path} ({len(self._corpus)} tokens)")
        self._blocks: Dict[Tuple[str, int, int], List[int]] = {}

    def _corpus_slice(self, start: int, n: int) -> List[int]:
        size = len(self._corpus)
        start %= size
        end = start + n
        if end <= size:
            return self._corpus[start:end]
        out = self._corpus[start:]
        # Wrap as many times as it takes for a slice longer than the corpus.
        while len(out) < n:
            out.extend(self._corpus[: n - len(out)])
        return out

    def block_tokens(self, scope_key: str, block_id: int, block_size: int) -> List[int]:
        """The tokens of one prefix block. Same arguments, same tokens."""
        cache_key = (scope_key, block_id, block_size)
        cached = self._blocks.get(cache_key)
        if cached is None:
            start = _seed_offset(f"{self.base_seed}:{self.salt}:{scope_key}:{block_id}", len(self._corpus))
            cached = self._corpus_slice(start, block_size)
            self._blocks[cache_key] = cached
        return list(cached)

    def tail_tokens(self, n: int, tail_key: str) -> List[int]:
        if n <= 0:
            return []
        start = _seed_offset(f"{self.base_seed}:{self.salt}:tail:{tail_key}", len(self._corpus))
        return self._corpus_slice(start, n)

    def materialize(
        self, part: TextSpec, scope_key: str, tail_key: str, count_special_tokens: bool = True
    ) -> MaterializedText:
        """Build text that tokenizes to `part.num_tokens`.

        The named blocks come first and are never edited, so the shared prefix
        survives; the unique tail is what gets extended or trimmed to land the
        exact count after decode and re-encode. The count includes the special
        tokens the tokenizer adds unless `count_special_tokens` is off, for a
        trace whose lengths add up across the messages of one prompt.
        """
        target = part.num_tokens
        if target == 0:
            return MaterializedText(text="", token_ids=[], prefix_tokens=0)

        tokens: List[int] = []
        for block_id in part.block_ids:
            tokens.extend(self.block_tokens(scope_key, block_id, part.block_size))
        prefix_tokens = min(len(tokens), target)
        tokens = tokens[:target]
        tokens.extend(self.tail_tokens(target - len(tokens), tail_key))

        text, ids = self._land(tokens, target, tail_key, count_special_tokens)
        if len(ids) < prefix_tokens:
            logger.debug(f"Landing {target} tokens for {tail_key} trimmed {prefix_tokens - len(ids)} off its last named block")
            prefix_tokens = len(ids)
        return MaterializedText(text=text, token_ids=ids, prefix_tokens=prefix_tokens)

    def _land(self, tokens: List[int], target: int, tail_key: str, count_special_tokens: bool) -> Tuple[str, List[int]]:
        """Decode, measure, and extend or trim the tail until the text measures `target`."""
        measured = -1
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
            text = decoded if isinstance(decoded, str) else " ".join(decoded)
            measured = len(self.tokenizer.encode(text, add_special_tokens=count_special_tokens)) if text else 0
            if measured == target:
                return text, tokens
            if measured < target:
                tokens = tokens + self.tail_tokens(target - measured, f"{tail_key}:{attempt}")
            else:
                excess = measured - target
                tokens = tokens[:-excess] if excess < len(tokens) else []
        raise ValueError(
            f"Could not build a prompt of exactly {target} tokens after {_MAX_ATTEMPTS} attempts (got {measured}). "
            "Either re-tokenization keeps merging tokens across the edited boundary, the length is smaller than the "
            "special tokens the tokenizer adds, or the tokenizer does not match the model the server is running."
        )


def _texts(parts: Sequence[Part]) -> List[TextPart]:
    """The text of a message, in order. A tool call gives its arguments and a
    tool result its content, so both go out as plain message text for now:
    shaping them as tool messages is an API's job. Media is not sent yet."""
    texts: List[TextPart] = []
    for part in parts:
        if isinstance(part, TextPart):
            texts.append(part)
        elif isinstance(part, ToolCallPart):
            texts.extend([part.arguments] if part.arguments is not None else [])
        elif isinstance(part, ToolResultPart) and all(isinstance(inner, TextPart) for inner in part.content):
            texts.extend(inner for inner in part.content if isinstance(inner, TextPart))
        else:
            raise NotImplementedError(f"{part.type} parts are not sent by the workload generators yet")
    return texts


@dataclass
class BuiltMessage:
    """A message with its text built, ready for an API to shape."""

    role: str
    text: str
    # Token count the record declared for this message, when every part
    # declared one; None when the count has to be measured.
    declared_tokens: Optional[int]


class Prompter:
    """Builds the messages of a record. `source_id` is the scope of blocks
    that mean the same text across a whole trace."""

    def __init__(self, source_id: str, materializer: BlockTextMaterializer, count_special_tokens: bool = True) -> None:
        self.source_id = source_id
        self.materializer = materializer
        self.count_special_tokens = count_special_tokens

    def prompt(self, record: Record, session_id: str) -> List[BuiltMessage]:
        """The messages that go on the wire for `record`."""
        return [
            self._built(record, index, message.role, message.parts, session_id)
            for index, message in enumerate(record.messages)
        ]

    def reply(self, record: Record, session_id: str) -> BuiltMessage:
        """The recorded reply, built the way the same message is built when a
        later record carries it."""
        return self._built(record, len(record.messages), "assistant", record.reply, session_id)

    def _built(self, record: Record, index: int, role: str, parts: Sequence[Part], session_id: str) -> BuiltMessage:
        pieces: List[str] = []
        declared = 0
        all_declared = True
        for part_index, part in enumerate(_texts(parts)):
            if part.text is not None:
                pieces.append(part.text)
                all_declared = False
            elif part.spec is not None:
                spec = part.spec
                scope_key = self.source_id if spec.scope == "trace" else session_id
                if spec.num_tokens == len(spec.block_ids) * spec.block_size:
                    # Named entirely by its blocks: the same text in every
                    # record of the scope, even when landing the exact count
                    # needs a tail.
                    tail_key = f"{scope_key}:blocks:{spec.block_size}:{','.join(map(str, spec.block_ids))}"
                else:
                    tail_key = f"{record.id}:{index}:{part_index}"
                built = self.materializer.materialize(spec, scope_key, tail_key, self.count_special_tokens)
                pieces.append(built.text)
                declared += spec.num_tokens
        return BuiltMessage(
            role=role,
            text="\n".join(pieces),
            declared_tokens=declared if all_declared and pieces else None,
        )
