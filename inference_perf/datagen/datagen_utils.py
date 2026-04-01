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

from typing import Callable, List, Set, Tuple

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


def converge_to_exact_length_text(
    tokenizer: CustomTokenizer,
    target_len: int,
    prefix_text: str,
    initial_tokens: List[int],
    adjust_tokens_fn: Callable[[List[int], int, int], List[int]],
) -> str:
    """Generates a string that tokenizes to exactly target_len, optionally prefixed.

    Args:
        tokenizer: The custom tokenizer.
        target_len: The target token length.
        prefix_text: Optional prefix to include in the length calculation.
        initial_tokens: The initial list of token IDs to start with.
        adjust_tokens_fn: A callback function to adjust the token list when the
          length doesn't match. It takes (current_tokens, current_len,
          target_len) and returns new_tokens.

    Raises:
        ValueError: If we cannot land on exactly `target_len` within the
          iteration budget. Most often this happens when the tokenizer's BPE
          merges around the prefix/suffix boundary keep flipping the count by
          more than one token per adjustment, or when the requested length is
          near the tokenizer's vocabulary edge cases (e.g. very small
          target_len with a tokenizer that always prepends a BOS). Mitigation:
          try a slightly different `target_len`, verify the tokenizer config
          matches the model server's, or pre-tokenize the prefix once and
          adjust around it instead of regenerating each iteration.
    """
    if target_len <= 0:
        return ""

    hf_tokenizer = tokenizer.get_tokenizer()

    prefix_len = 0
    if prefix_text:
        prefix_len = tokenizer.count_tokens(prefix_text)
        assert prefix_len < target_len, (
            f"target_len ({target_len}) must be > prefix_len ({prefix_len}). "
            f"This helper generates suffix tokens such that prefix_len + suffix_len == target_len; "
            f"the caller is responsible for choosing target_len with room for a non-empty suffix."
        )

    current_tokens = initial_tokens

    max_iterations = 20
    last_len = -1
    for _ in range(max_iterations):
        text = hf_tokenizer.decode(current_tokens, skip_special_tokens=True)

        full_text = prefix_text + " " + text if prefix_text else text

        current_len = tokenizer.count_tokens(full_text)

        if current_len == target_len:
            return full_text if prefix_text else text

        last_len = current_len
        current_tokens = adjust_tokens_fn(current_tokens, current_len, target_len)

    raise ValueError(
        f"Could not generate a prompt of exactly {target_len} tokens after {max_iterations} "
        f"attempts (got {last_len}). This is usually a configuration mismatch — try one of: "
        f"(1) increase or decrease the requested length by a few tokens in your data config "
        f"(e.g. data.input_distribution.mean / .std_dev, or data.shared_prefix.system_prompt_len "
        f"/ .question_len); (2) ensure tokenizer.pretrained_model_name_or_path matches the model "
        f"the server is running."
    )


def generate_random_exact_length_text(
    rng: np.random.Generator,
    valid_token_ids: np.ndarray,
    tokenizer: CustomTokenizer,
    target_len: int,
    prefix_text: str = "",
) -> str:
    """Generate text that tokenizes to exactly target_len, optionally with a prefix.

    Combines random token sampling with the convergence loop in
    `generate_exact_length_text`. When prefix_text is provided the TOTAL length
    including prefix will be target_len; the returned string is the full
    combined text. Otherwise just the generated suffix is returned.
    """
    if target_len <= 0:
        return ""

    prefix_len = 0
    if prefix_text:
        prefix_len = tokenizer.count_tokens(prefix_text)
        assert prefix_len < target_len, (
            f"target_len ({target_len}) must be > prefix_len ({prefix_len}). "
            f"This helper generates suffix tokens such that prefix_len + suffix_len == target_len; "
            f"the caller is responsible for choosing target_len with room for a non-empty suffix."
        )

    initial_tokens = random_token_ids(rng, valid_token_ids, target_len - prefix_len)

    def adjust_tokens(current_tokens: List[int], current_len: int, target_len: int) -> List[int]:
        if current_len < target_len:
            current_tokens.extend(random_token_ids(rng, valid_token_ids, target_len - current_len))
        else:
            diff = current_len - target_len
            current_tokens = current_tokens[:-diff] if len(current_tokens) > diff else current_tokens[:1]
        return current_tokens

    return converge_to_exact_length_text(
        tokenizer=tokenizer,
        target_len=target_len,
        prefix_text=prefix_text,
        initial_tokens=initial_tokens,
        adjust_tokens_fn=adjust_tokens,
    )
