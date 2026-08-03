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

from typing import Callable, List, Optional, Set, Tuple

import numpy as np

from inference_perf.utils.custom_tokenizer import CustomTokenizer


def init_vocab_sampling(tokenizer: CustomTokenizer) -> Tuple[int, Set[int], np.ndarray]:
    """Resolve a tokenizer's vocab size and build the valid-token-id pool for random sampling.

    Returns:
        (vocab_size, special_token_ids, valid_token_ids) where valid_token_ids excludes
        the tokenizer's special tokens.

    Raises:
        ValueError: If the tokenizer exposes no usable vocab-size signal, or if the
          resolved vocab size is non-positive.
    """
    hf_tokenizer = tokenizer.get_tokenizer()
    if hasattr(hf_tokenizer, "vocab_size") and hf_tokenizer.vocab_size is not None:
        vocab_size: int = hf_tokenizer.vocab_size
    elif hasattr(hf_tokenizer, "get_vocab") and callable(hf_tokenizer.get_vocab):
        vocab_size = len(hf_tokenizer.get_vocab())
    else:
        try:
            vocab_size = len(hf_tokenizer)
        except TypeError as e:
            raise ValueError(
                "Tokenizer does not have a 'vocab_size' attribute, 'get_vocab()' method, "
                "or support len() for vocabulary size. Cannot use random token generation."
            ) from e
    if vocab_size <= 0:
        raise ValueError(f"Tokenizer vocabulary size must be positive, got {vocab_size}.")

    special_token_ids: Set[int] = set(getattr(hf_tokenizer, "all_special_ids", None) or [])
    valid_token_ids = np.array([i for i in range(vocab_size) if i not in special_token_ids], dtype=np.int64)
    return vocab_size, special_token_ids, valid_token_ids


def random_token_ids(rng: np.random.Generator, valid_token_ids: np.ndarray, length: int) -> List[int]:
    """Sample `length` token IDs uniformly from `valid_token_ids` using `rng`.

    Returns an empty list when length <= 0. The returned list is plain python
    ints so callers can pass it straight to HF tokenizer decode().
    """
    if length <= 0:
        return []
    return rng.choice(valid_token_ids, size=length).tolist()  # type: ignore[no-any-return]


def build_word_start_token_ids(tokenizer: CustomTokenizer, valid_token_ids: np.ndarray) -> np.ndarray:
    """Token IDs whose decoded form starts with whitespace.

    Used to pin the first token of a suffix when appending it to a prefix:
    a whitespace-prefixed token prevents BPE merges across the boundary,
    so ``len(decode(prefix_ids + suffix_ids))`` retokenizes exactly and the
    prefix's server-side tokens stay stable. Falls back to the full vocab
    when no such tokens exist.
    """
    hf_tokenizer = tokenizer.get_tokenizer()
    word_starts: List[int] = []
    for tid in valid_token_ids:
        decoded = hf_tokenizer.decode([int(tid)], skip_special_tokens=True)
        if decoded and decoded[0].isspace():
            word_starts.append(int(tid))
    if not word_starts:
        return valid_token_ids
    return np.array(word_starts, dtype=np.int64)


def converge_to_exact_length_text(
    tokenizer: CustomTokenizer,
    target_len: int,
    initial_tokens: List[int],
    adjust_tokens_fn: Callable[[List[int], int, int], List[int]],
    wrap_fn: Optional[Callable[[str], str]] = None,
) -> Tuple[str, List[int]]:
    """Generate text tokenizing to exactly target_len; return (text, ids).

    ``adjust_tokens_fn`` takes ``(current_tokens, current_len, target_len)``
    and returns new tokens. The returned ids let callers compose with another
    chunk at the token level instead of by string concat (which would
    re-tokenize across the boundary).

    When ``wrap_fn`` is given (e.g. a chat template), each candidate text is
    wrapped before counting and the WRAPPED text is returned, so target_len
    applies to the final string that goes on the wire. The wrapped text is
    counted with add_special_tokens=False because the wrapper embeds its own
    special tokens; the returned ids remain the unwrapped content ids.
    Callers must ensure target_len exceeds the wrapper's fixed overhead.
    """
    if target_len <= 0:
        return "", []

    hf_tokenizer = tokenizer.get_tokenizer()

    current_tokens = initial_tokens

    max_iterations = 20
    last_len = -1
    for _ in range(max_iterations):
        text = hf_tokenizer.decode(current_tokens, skip_special_tokens=True)
        if isinstance(text, list):
            text = " ".join(text)

        if wrap_fn is None:
            current_len = tokenizer.count_tokens(text)
        else:
            text = wrap_fn(text)
            current_len = tokenizer.count_tokens(text, add_special_tokens=False)

        if current_len == target_len:
            return text, current_tokens

        last_len = current_len
        current_tokens = adjust_tokens_fn(current_tokens, current_len, target_len)

    raise ValueError(
        f"Could not generate a prompt of exactly {target_len} tokens after {max_iterations} "
        f"attempts (got {last_len}). This usually means tokenizer.pretrained_model_name_or_path "
        f"does not match the model the server is running."
    )


def generate_random_exact_length_text(
    rng: np.random.Generator,
    valid_token_ids: np.ndarray,
    tokenizer: CustomTokenizer,
    target_len: int,
    wrap_fn: Optional[Callable[[str], str]] = None,
) -> Tuple[str, List[int]]:
    """Generate random text tokenizing to exactly target_len; return (text, ids).

    With ``wrap_fn`` set, target_len applies to the wrapped text; see
    converge_to_exact_length_text.
    """
    if target_len <= 0:
        return "", []

    initial_tokens = random_token_ids(rng, valid_token_ids, target_len)

    def adjust_tokens(current_tokens: List[int], current_len: int, target_len: int) -> List[int]:
        if current_len < target_len:
            current_tokens.extend(random_token_ids(rng, valid_token_ids, target_len - current_len))
            return current_tokens
        diff = current_len - target_len
        if diff < len(current_tokens):
            return current_tokens[:-diff]
        return []

    return converge_to_exact_length_text(
        tokenizer=tokenizer,
        target_len=target_len,
        initial_tokens=initial_tokens,
        adjust_tokens_fn=adjust_tokens,
        wrap_fn=wrap_fn,
    )
