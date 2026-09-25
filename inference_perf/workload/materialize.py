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
"""Turn a `SyntheticPart` into text.

The text for a prefix block is a slice of a tokenized corpus, and where that
slice starts is decided by hashing (base seed, salt, scope key, block id). So
the same block id in the same scope always decodes to the same tokens, on any
worker and in any run with the same seed, which is what makes a recorded
prefix hit reproducible. The tail past the named blocks is seeded from a key
the caller picks (the record id, usually), so it is stable but unique.

Blocks are cached per (scope, id, size), so a trace with heavy prefix reuse
tokenizes each shared block once, not once per request.

The `salt` is for repeat passes over the same trace: a different salt gives
different block text for the same ids, so a second pass does not report cache
hits the server earned on the first one.

This is the Weka generator's prompt synthesis lifted out and made scope-aware.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from inference_perf.datagen.datagen_utils import converge_to_exact_length_text
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.workload.record import SyntheticPart

logger = logging.getLogger(__name__)

DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / "assets" / "shakespeare.txt"


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
    def __init__(
        self,
        tokenizer: CustomTokenizer,
        base_seed: int,
        corpus_path: Optional[Path] = None,
        salt: str = "",
    ) -> None:
        self.tokenizer = tokenizer
        self.base_seed = base_seed
        self.salt = salt
        path = corpus_path if corpus_path is not None else DEFAULT_CORPUS
        if not path.is_file():
            raise FileNotFoundError(f"Prompt corpus file not found: {path}")
        text = path.read_text(encoding="utf-8")
        self._corpus: List[int] = list(tokenizer.get_tokenizer().encode(text, add_special_tokens=False))
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
        self,
        part: SyntheticPart,
        scope_key: str,
        tail_key: str,
        wrap_fn: Optional[Callable[[str], str]] = None,
    ) -> MaterializedText:
        """Build text that tokenizes to `part.num_tokens`.

        The named blocks come first and are never edited, so the shared prefix
        survives; the unique tail is what gets extended or trimmed to land the
        exact count after decode and re-encode. With `wrap_fn` (a chat
        template) the count applies to the wrapped text, as it does everywhere
        else in this project.
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

        extensions = 0

        def adjust(current: List[int], current_len: int, target_len: int) -> List[int]:
            nonlocal extensions
            if current_len < target_len:
                extensions += 1
                current.extend(self.tail_tokens(target_len - current_len, f"{tail_key}:{extensions}"))
                return current
            diff = current_len - target_len
            return current[:-diff] if diff < len(current) else []

        text, ids = converge_to_exact_length_text(
            tokenizer=self.tokenizer,
            target_len=target,
            initial_tokens=tokens,
            adjust_tokens_fn=adjust,
            wrap_fn=wrap_fn,
        )
        if len(ids) < prefix_tokens:
            logger.warning(
                f"Landing {target} tokens trimmed into the shared prefix of {tail_key} "
                f"({prefix_tokens} declared, {len(ids)} kept); the tokenizer and the trace's block size disagree"
            )
            prefix_tokens = len(ids)
        return MaterializedText(text=text, token_ids=ids, prefix_tokens=prefix_tokens)
