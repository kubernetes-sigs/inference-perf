import pytest
from unittest.mock import MagicMock
import numpy as np
from inference_perf.config import APIConfig, APIType, CustomTokenizerConfig, DataConfig, Distribution, DataGenType
from inference_perf.apis import LazyLoadInferenceAPIData
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.apis.user_session import UserSessionCompletionAPIData
from inference_perf.datagen.random_datagen import RandomDataGenerator
from inference_perf.datagen.synthetic_datagen import SyntheticDataGenerator
from inference_perf.config import SharedPrefix
from inference_perf.datagen.base import DataGenerator, LazyLoadDataMixin
from inference_perf.datagen.shared_prefix_datagen import SharedPrefixDataGenerator
from inference_perf.utils.custom_tokenizer import CustomTokenizer


def _make_mock_tokenizer(vocab_size: int = 1000) -> MagicMock:
    """Create a mock tokenizer that returns predictable text."""
    mock_tokenizer = MagicMock()
    hf_tok = MagicMock()
    hf_tok.vocab_size = vocab_size
    hf_tok.decode = MagicMock(side_effect=lambda ids, **kw: f"text_{len(ids)}")
    hf_tok.batch_decode = MagicMock(side_effect=lambda batch, **kw: [f"text_{len(ids)}" for ids in batch])
    hf_tok.encode = MagicMock(side_effect=lambda text, **kw: [1] * max(1000, len(text.split())))
    mock_tokenizer.get_tokenizer.return_value = hf_tok

    def count_tokens(text: str) -> int:
        parts = text.split()
        total = 0
        for p in parts:
            if p.startswith("text_"):
                total += int(p[5:])
            else:
                total += 1
        return total

    mock_tokenizer.count_tokens.side_effect = count_tokens
    return mock_tokenizer


@pytest.mark.parametrize("gen_type", [DataGenType.Random, DataGenType.Synthetic, DataGenType.SharedPrefix])
def test_datagen_length_fuzz(gen_type: DataGenType) -> None:
    rng = np.random.default_rng(42)
    mock_tokenizer = _make_mock_tokenizer()

    # Run 10 random configurations
    for _ in range(10):
        target_len = int(rng.integers(10, 100))

        api_config = APIConfig(type=APIType.Completion)

        if gen_type == DataGenType.Random:
            data_config = DataConfig(
                type=DataGenType.Random,
                input_distribution=Distribution(
                    min=target_len, max=target_len, mean=float(target_len), std_dev=0.0, total_count=10
                ),
                output_distribution=Distribution(min=10, max=10, mean=10.0, std_dev=0.0, total_count=10),
            )
            gen: DataGenerator = RandomDataGenerator(api_config, data_config, mock_tokenizer)
        elif gen_type == DataGenType.Synthetic:
            data_config = DataConfig(
                type=DataGenType.Synthetic,
                input_distribution=Distribution(
                    min=target_len, max=target_len, mean=float(target_len), std_dev=0.0, total_count=10
                ),
                output_distribution=Distribution(min=10, max=10, mean=10.0, std_dev=0.0, total_count=10),
            )
            gen = SyntheticDataGenerator(api_config, data_config, mock_tokenizer)
        elif gen_type == DataGenType.SharedPrefix:
            sp = SharedPrefix(
                num_groups=2,
                num_prompts_per_group=5,
                question_len=target_len,
                output_len=10,
                system_prompt_len=10,
            )
            data_config = DataConfig(type=DataGenType.SharedPrefix, shared_prefix=sp)
            gen = SharedPrefixDataGenerator(api_config, data_config, mock_tokenizer)

        # Generate prompts and check length
        assert isinstance(gen, LazyLoadDataMixin)
        prompts = []
        for i, p in enumerate(gen.get_data()):
            if isinstance(p, LazyLoadInferenceAPIData):
                p = gen.load_lazy_data(p)
            prompts.append(p)
            if i >= 9:
                break

        assert len(prompts) == 10

        for p in prompts:
            assert isinstance(p, CompletionAPIData)
            actual_len = mock_tokenizer.count_tokens(p.prompt)

            if gen_type == DataGenType.SharedPrefix:
                # For SharedPrefix, the prompt is prefix + question.
                # Expected length is system_prompt_len + question_len = 10 + target_len
                expected_len = 10 + target_len
            else:
                expected_len = target_len

            print(f"Type: {gen_type}, Expected: {expected_len}, Actual: {actual_len}")
            assert actual_len == expected_len, f"Failed for {gen_type}, expected {expected_len}, got {actual_len}"


# Regression for kubernetes-sigs/inference-perf#490: with a real BPE tokenizer,
# `shared_prefix` with tiny question lengths used to wedge the convergence loop
# at target_len+1 because the prefix/suffix boundary inflated the count by one
# and the adjustment step couldn't shrink past a 1-token suffix. The mock
# tokenizer used in the test above counts by whitespace so it can't surface
# this — exercise gpt2 here to cover the boundary case.
@pytest.mark.parametrize("question_len_min", [1, 2, 3])
def test_shared_prefix_real_tokenizer_small_question_len(question_len_min: int) -> None:
    tokenizer = CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path="gpt2"))
    api_config = APIConfig(type=APIType.Completion)
    shared_prefix_cfg = SharedPrefix(
        num_groups=3,
        num_prompts_per_group=50,
        system_prompt_len=100,
        question_distribution=Distribution(mean=float(question_len_min + 2), std_dev=2.0, min=question_len_min, max=20),
        output_distribution=Distribution(mean=10.0, std_dev=1.0, min=1, max=20),
        enable_multi_turn_chat=True,
    )
    data_config = DataConfig(type=DataGenType.SharedPrefix, shared_prefix=shared_prefix_cfg)

    gen = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

    # Every generated prompt must tokenize to exactly system_prompt_len + sampled q_len.
    # We don't know the per-prompt sampled q_len, but we know prefix is constant per group
    # and full length should be in the valid sampled range.
    for prompt, prefix in zip(gen.prompts, gen.prefix_texts, strict=True):
        prefix_count = tokenizer.count_tokens(prefix)
        full_count = tokenizer.count_tokens(prompt)
        q_count = full_count - prefix_count
        assert prefix_count == 100, f"prefix tokenized to {prefix_count}, expected 100"
        assert question_len_min <= q_count <= 20, f"question length {q_count} outside [{question_len_min}, 20]"

    # Also confirm get_data() returns the right shape via load_lazy_data.
    n = 5
    seen = 0
    for lazy in gen.get_data():
        if isinstance(lazy, LazyLoadInferenceAPIData):
            p = gen.load_lazy_data(lazy)
            assert isinstance(p, UserSessionCompletionAPIData)
        seen += 1
        if seen >= n:
            break


# Prefix-cache invariant: the BPE-tokenized form of every prompt in a group
# must agree on its first len(encode(prefix)) tokens. If the suffix's first
# token were drawn from the full vocab it could merge across the boundary and
# shift the prefix's tokenization — pinning to word-start tokens prevents
# that. Exercise gpt2 (the same tokenizer that surfaces the off-by-one count
# bug under boundary merges) so a regression in the pinning logic would also
# fail here.
def test_shared_prefix_server_tokens_stable_across_group() -> None:
    tokenizer = CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path="gpt2"))
    hf = tokenizer.get_tokenizer()
    api_config = APIConfig(type=APIType.Completion)
    shared_prefix_cfg = SharedPrefix(
        num_groups=3,
        num_prompts_per_group=30,
        system_prompt_len=100,
        question_distribution=Distribution(mean=5.0, std_dev=2.0, min=1, max=15),
        output_distribution=Distribution(mean=10.0, std_dev=1.0, min=1, max=20),
        enable_multi_turn_chat=True,
    )
    data_config = DataConfig(type=DataGenType.SharedPrefix, shared_prefix=shared_prefix_cfg)
    gen = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

    by_group: dict[int, list[tuple[str, str]]] = {}
    for prompt, prefix, gid in zip(gen.prompts, gen.prefix_texts, gen.prompt_groups, strict=True):
        by_group.setdefault(gid, []).append((prompt, prefix))

    for gid, entries in by_group.items():
        canonical_prefix = entries[0][1]
        # Server-side tokenization length we care about. Use add_special_tokens=False
        # so the comparison is over the prefix content itself, not the BOS prefix.
        canonical_prefix_ids = hf(canonical_prefix, add_special_tokens=False).input_ids
        n_prefix = len(canonical_prefix_ids)
        assert n_prefix > 0

        first_n = {tuple(hf(prompt, add_special_tokens=False).input_ids[:n_prefix]) for prompt, _ in entries}
        assert len(first_n) == 1, (
            f"group {gid}: prefix tokenization differs across {len(entries)} prompts; "
            f"got {len(first_n)} distinct token-prefix sequences"
        )
