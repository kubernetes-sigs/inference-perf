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
from typing import cast
from unittest.mock import MagicMock

from inference_perf.config import (
    APIConfig,
    APIType,
    DataConfig,
    DataGenType,
    SyntheticMultimodalDatagenConfig,
    ImageDatagenConfig,
    VideoDatagenConfig,
    AudioDatagenConfig,
    Distribution,
    DistributionType,
)
from inference_perf.datagen.multimodal_datagen import MultimodalDataGenerator
from inference_perf.apis.chat import ChatCompletionAPIData
from inference_perf.apis.base import LazyLoadInferenceAPIData


def _make_mock_tokenizer() -> MagicMock:
    mock_tokenizer = MagicMock()
    mock_tokenizer.count_tokens.return_value = 10
    hf_tokenizer = MagicMock()
    hf_tokenizer.vocab_size = 1000
    hf_tokenizer.decode.return_value = "decoded prompt text"
    mock_tokenizer.get_tokenizer.return_value = hf_tokenizer
    return mock_tokenizer


def test_multimodal_datagen_prefix_mode() -> None:
    api_config = APIConfig(type=APIType.Chat)
    data_config = DataConfig(
        type=DataGenType.Synthetic,
        multimodal=SyntheticMultimodalDatagenConfig(
            image=ImageDatagenConfig(
                count=Distribution(type=DistributionType.UNIFORM, min=2, max=2, mean=2),
                insertion_point=0.0,
            ),
        ),
    )
    gen = MultimodalDataGenerator(api_config, data_config, _make_mock_tokenizer())

    # Get first item
    data_iter = gen.get_data()
    lazy_item = next(data_iter)
    assert isinstance(lazy_item, LazyLoadInferenceAPIData)

    # Load data
    item = gen.load_lazy_data(lazy_item)

    assert isinstance(item, ChatCompletionAPIData)
    assert len(item.messages) == 1
    content = item.messages[0].content
    assert isinstance(content, list)
    assert len(content) == 3  # 2 images + 1 text
    assert content[0]["type"] == "image_url"
    assert content[1]["type"] == "image_url"
    assert content[2]["type"] == "text"

    assert item.multimodal_metrics is not None
    assert item.multimodal_metrics["images"] == 2
    assert item.multimodal_metrics["videos"] == 0


def test_multimodal_datagen_suffix_mode() -> None:
    api_config = APIConfig(type=APIType.Chat)
    data_config = DataConfig(
        type=DataGenType.Synthetic,
        multimodal=SyntheticMultimodalDatagenConfig(
            image=ImageDatagenConfig(
                count=Distribution(type=DistributionType.UNIFORM, min=1, max=1, mean=1),
                insertion_point=1.0,
            ),
        ),
    )
    gen = MultimodalDataGenerator(api_config, data_config, _make_mock_tokenizer())

    data_iter = gen.get_data()
    lazy_item = next(data_iter)
    assert isinstance(lazy_item, LazyLoadInferenceAPIData)
    item = gen.load_lazy_data(lazy_item)

    assert isinstance(item, ChatCompletionAPIData)
    content = item.messages[0].content
    assert isinstance(content, list)
    assert len(content) == 2  # 1 text + 1 image
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"


async def test_multimodal_datagen_video() -> None:
    api_config = APIConfig(type=APIType.Chat)
    data_config = DataConfig(
        type=DataGenType.Synthetic,
        multimodal=SyntheticMultimodalDatagenConfig(
            video=VideoDatagenConfig(
                count=Distribution(type=DistributionType.UNIFORM, min=1, max=1, mean=1),
                insertion_point=0.0,
            ),
        ),
    )
    gen = MultimodalDataGenerator(api_config, data_config, _make_mock_tokenizer())

    data_iter = gen.get_data()
    lazy_item = next(data_iter)
    assert isinstance(lazy_item, LazyLoadInferenceAPIData)
    item = gen.load_lazy_data(lazy_item)

    assert isinstance(item, ChatCompletionAPIData)
    content = item.messages[0].content
    assert isinstance(content, list)
    # We expect 1 video block + 1 text block
    assert len(content) == 2
    assert content[0]["type"] == "video_url"
    assert content[1]["type"] == "text"
    assert item.multimodal_metrics is not None
    assert item.multimodal_metrics["videos"] == 1
    assert item.multimodal_metrics["frames"] == 10  # Default profile frames

    # Video URL should now carry a real MP4 data URI whose byte size scales with config.
    video_url = content[0]["video_url"]["url"]
    assert video_url.startswith("data:video/mp4;base64,")
    assert len(video_url) > len("data:video/mp4;base64,") + 100  # non-trivial payload

    # Check payload shape
    payload = await item.to_payload(effective_model_name="gpt-video", max_tokens=100, ignore_eos=False, streaming=False)
    assert payload["messages"][0]["content"][0]["type"] == "video_url"


def test_multimodal_datagen_interleaved_center() -> None:
    api_config = APIConfig(type=APIType.Chat)
    data_config = DataConfig(
        type=DataGenType.Synthetic,
        multimodal=SyntheticMultimodalDatagenConfig(
            image=ImageDatagenConfig(
                count=Distribution(type=DistributionType.UNIFORM, min=1, max=1, mean=1),
                insertion_point=0.5,
            ),
        ),
    )
    gen = MultimodalDataGenerator(api_config, data_config, _make_mock_tokenizer())

    data_iter = gen.get_data()
    lazy_item = next(data_iter)
    item = gen.load_lazy_data(cast(LazyLoadInferenceAPIData, lazy_item))

    assert isinstance(item, ChatCompletionAPIData)
    content = item.messages[0].content
    assert isinstance(content, list)
    # With insertion_point=0.5, text should be split in half.
    # We expect: Text(part 1), Image, Text(part 2)
    assert len(content) == 3
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[2]["type"] == "text"


async def test_multimodal_datagen_audio() -> None:
    api_config = APIConfig(type=APIType.Chat)
    data_config = DataConfig(
        type=DataGenType.Synthetic,
        multimodal=SyntheticMultimodalDatagenConfig(
            audio=AudioDatagenConfig(
                count=Distribution(type=DistributionType.UNIFORM, min=1, max=1, mean=1),
                insertion_point=0.0,
            ),
        ),
    )
    gen = MultimodalDataGenerator(api_config, data_config, _make_mock_tokenizer())

    data_iter = gen.get_data()
    lazy_item = next(data_iter)
    assert isinstance(lazy_item, LazyLoadInferenceAPIData)
    item = gen.load_lazy_data(lazy_item)

    assert isinstance(item, ChatCompletionAPIData)
    content = item.messages[0].content
    assert isinstance(content, list)
    assert len(content) == 2  # 1 audio + 1 text
    assert content[0]["type"] == "input_audio"
    assert content[0]["input_audio"]["format"] == "wav"
    assert "data" in content[0]["input_audio"]

    assert item.multimodal_metrics is not None
    assert item.multimodal_metrics["audio_clips"] == 1
    # Output-audio fields are not auto-populated from input-audio config.
    assert item.modalities is None
    assert item.audio is None

    # Check payload shape — no output-audio fields in payload.
    payload = await item.to_payload(effective_model_name="gpt-audio", max_tokens=100, ignore_eos=False, streaming=False)
    assert "modalities" not in payload
    assert "audio" not in payload
