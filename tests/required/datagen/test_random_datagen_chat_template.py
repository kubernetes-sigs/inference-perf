"""Chat-template fidelity tests against a real HF tokenizer.

The #564 postmortem's central finding was that a low-fidelity mock tokenizer
let token-accounting bugs run silent, so the apply_chat_template +
add_special_tokens interplay is asserted here with the real vendored gemma
tokenizer, not a dummy. The vendored tokenizer is a base model without a chat
template, so the test injects a Gemma-style template into an extracted copy:
the template, tokenizer, and counting all stay real.
"""

import json
import pathlib
import tarfile

import pytest

from inference_perf.apis import CompletionAPIData, LazyLoadInferenceAPIData
from inference_perf.config import (
    APIConfig,
    APIType,
    CustomTokenizerConfig,
    DataConfig,
    DataGenType,
)
from inference_perf.datagen.synthetic.random_datagen import RandomDataGenerator
from inference_perf.utils.custom_tokenizer import CustomTokenizer

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
MODEL_TARBALL = REPO_ROOT / "e2e" / "testdata" / "models" / "google_gemma-3-270m.tar.gz"

GEMMA_CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "<start_of_turn>{{ message['role'] }}\n{{ message['content'] }}<end_of_turn>\n"
    "{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)


@pytest.fixture(scope="module")
def chat_tokenizer_path(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    dest = tmp_path_factory.mktemp("gemma-chat")
    with tarfile.open(MODEL_TARBALL) as tar:
        tar.extractall(dest, filter="data")
    config_path = dest / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["chat_template"] = GEMMA_CHAT_TEMPLATE
    config_path.write_text(json.dumps(config))
    return dest


@pytest.fixture(scope="module")
def chat_tokenizer(chat_tokenizer_path: pathlib.Path) -> CustomTokenizer:
    return CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path=str(chat_tokenizer_path)))


def test_apply_chat_template_embeds_special_tokens(chat_tokenizer: CustomTokenizer) -> None:
    assert chat_tokenizer.has_chat_template()
    rendered = chat_tokenizer.apply_chat_template("hello")
    assert rendered.startswith("<bos><start_of_turn>user")
    assert rendered.endswith("<start_of_turn>model\n")
    # The rendered text already contains BOS, so counting it with the default
    # add_special_tokens=True would double-count it by exactly one.
    without = chat_tokenizer.count_tokens(rendered, add_special_tokens=False)
    with_specials = chat_tokenizer.count_tokens(rendered)
    assert with_specials == without + 1


def test_random_datagen_chat_template_exact_prefill_length(chat_tokenizer: CustomTokenizer) -> None:
    api_config = APIConfig(type=APIType.Completion)
    data_config = DataConfig(
        type=DataGenType.Random,
        use_chat_template=True,
        input_distribution="Min(Max(Normal(100, 40), 20), 200)",
        output_distribution="Min(Max(Normal(7, 1), 5), 10)",
    )
    generator = RandomDataGenerator(api_config, data_config, chat_tokenizer, seed=42, total_count=10)

    hf_tokenizer = chat_tokenizer.get_tokenizer()
    bos_id = hf_tokenizer.convert_tokens_to_ids("<bos>")
    for i in range(10):
        item = generator.load_lazy_data(LazyLoadInferenceAPIData(data_index=i))
        assert isinstance(item, CompletionAPIData)
        assert item.prompt.startswith("<bos><start_of_turn>user")
        assert item.add_special_tokens is False
        # The configured input length is the server-side prefill: the fully
        # templated prompt re-encodes to exactly the target, with a single BOS.
        ids = hf_tokenizer(item.prompt, add_special_tokens=False).input_ids
        assert len(ids) == generator.input_lengths[i]
        assert ids.count(bos_id) == 1


def test_random_datagen_chat_template_min_length_guard(chat_tokenizer: CustomTokenizer) -> None:
    overhead = chat_tokenizer.count_tokens(chat_tokenizer.apply_chat_template(""), add_special_tokens=False)
    api_config = APIConfig(type=APIType.Completion)
    data_config = DataConfig(
        type=DataGenType.Random,
        use_chat_template=True,
        input_distribution=overhead,
        output_distribution=5,
    )
    with pytest.raises(ValueError, match="chat template overhead"):
        RandomDataGenerator(api_config, data_config, chat_tokenizer, seed=42, total_count=5)
