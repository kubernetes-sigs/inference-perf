from unittest.mock import MagicMock
from inference_perf.datagen.shared_prefix_datagen import SharedPrefixDataGenerator
from inference_perf.config import (
    APIConfig,
    DataConfig,
    SharedPrefix,
    SyntheticMultimodalDatagenConfig,
    ImageDatagenConfig,
    APIType,
    DataGenType,
    Distribution,
)
from typing import cast
from inference_perf.apis.base import LazyLoadInferenceAPIData
from inference_perf.apis.chat import ChatCompletionAPIData
from inference_perf.utils.custom_tokenizer import CustomTokenizer


def _make_mock_tokenizer(vocab_size: int = 1000) -> MagicMock:
    """Mock tokenizer compatible with the exact-length text generator (#383):
    decode returns "text_N" markers and count_tokens sums those markers, so
    a single decoded chunk of N tokens reliably reports as N tokens."""
    mock_tokenizer = MagicMock(spec=CustomTokenizer)
    hf_tok = MagicMock()
    hf_tok.vocab_size = vocab_size
    hf_tok.decode = MagicMock(side_effect=lambda ids, **kw: f"text_{len(ids)}")
    hf_tok.batch_decode = MagicMock(side_effect=lambda batch, **kw: [f"text_{len(ids)}" for ids in batch])
    mock_tokenizer.get_tokenizer.return_value = hf_tok

    def count_tokens(text: str) -> int:
        total = 0
        for p in text.split():
            if p.startswith("text_"):
                total += int(p[5:])
            else:
                total += 1
        return total

    mock_tokenizer.count_tokens.side_effect = count_tokens
    return mock_tokenizer


def test_shared_prefix_multimodal_generation() -> None:
    mock_tokenizer = _make_mock_tokenizer()

    api_config = APIConfig(type="chat")

    # Config with both prefix and payload media
    multimodal_config = SyntheticMultimodalDatagenConfig(
        image=ImageDatagenConfig(count=Distribution(min=1, max=1, mean=1, std_dev=0), insertion_point=0.0)
    )

    shared_prefix_multimodal = SyntheticMultimodalDatagenConfig(
        image=ImageDatagenConfig(count=Distribution(min=1, max=1, mean=1, std_dev=0), insertion_point=0.0)
    )

    data_config = DataConfig(
        type=DataGenType.SharedPrefix,
        multimodal=multimodal_config,  # Payload
        shared_prefix=SharedPrefix(
            num_groups=1,
            num_prompts_per_group=2,
            multimodal=shared_prefix_multimodal,  # Prefix
        ),
    )

    generator = SharedPrefixDataGenerator(api_config, data_config, mock_tokenizer)

    # Verify that Chat is supported
    assert APIType.Chat in generator.get_supported_apis()

    # Get data
    data_iter = generator.get_data()
    lazy_data = next(data_iter)

    # Load lazy data
    api_data = generator.load_lazy_data(cast(LazyLoadInferenceAPIData, lazy_data))

    assert isinstance(api_data, ChatCompletionAPIData)
    assert len(api_data.messages) == 1
    assert api_data.messages[0].role == "user"
    content = api_data.messages[0].content
    assert isinstance(content, list)

    # We expect structured content with images
    image_parts = [part for part in content if part.get("type") == "image_url"]
    assert len(image_parts) == 2  # 1 from prefix, 1 from payload
