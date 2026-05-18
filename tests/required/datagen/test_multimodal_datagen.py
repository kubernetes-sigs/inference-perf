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

import pytest

from inference_perf.apis.base import LazyLoadInferenceAPIData
from inference_perf.apis.chat import ChatCompletionAPIData
from inference_perf.config import (
    APIConfig,
    APIType,
    AudioDatagenConfig,
    DataConfig,
    DataGenType,
    Distribution,
    DistributionType,
    ImageDatagenConfig,
    SyntheticMultimodalDatagenConfig,
    VideoDatagenConfig,
)
from inference_perf.datagen.multimodal_datagen import MultimodalDataGenerator


def _make_mock_tokenizer() -> MagicMock:
    mock_tokenizer = MagicMock()
    mock_tokenizer.count_tokens.return_value = 10
    hf_tokenizer = MagicMock()
    hf_tokenizer.vocab_size = 1000
    hf_tokenizer.decode.return_value = "decoded prompt text"
    mock_tokenizer.get_tokenizer.return_value = hf_tokenizer
    return mock_tokenizer


def _load_first(gen: MultimodalDataGenerator) -> ChatCompletionAPIData:
    lazy_item = next(gen.get_data())
    assert isinstance(lazy_item, LazyLoadInferenceAPIData)
    item = gen.load_lazy_data(lazy_item)
    assert isinstance(item, ChatCompletionAPIData)
    return item


@pytest.mark.asyncio
async def test_multimodal_datagen_prefix_mode() -> None:
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
    item = _load_first(gen)

    # Datagen produces text-only message + a typed spec; bytes do not yet exist.
    assert isinstance(item.messages[0].content, str)
    assert item.multimodal_spec is not None
    assert len(item.multimodal_spec.images) == 2
    assert all(img.insertion_point == 0.0 for img in item.multimodal_spec.images)
    assert len(item.multimodal_spec.videos) == 0

    # to_request_body materializes bytes and weaves them into the user message.
    payload = await item.to_request_body(effective_model_name="gpt-img", max_tokens=100, ignore_eos=False, streaming=False)
    content = payload["messages"][0]["content"]
    assert isinstance(content, list)
    assert len(content) == 3  # 2 images + 1 text
    assert content[0]["type"] == "image_url"
    assert content[1]["type"] == "image_url"
    assert content[2]["type"] == "text"

    # Realized metrics are recorded back on the API data for process_response.
    assert item.realized_images is not None
    assert item.realized_images.count == 2


@pytest.mark.asyncio
async def test_multimodal_datagen_suffix_mode() -> None:
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
    item = _load_first(gen)

    payload = await item.to_request_body(effective_model_name="gpt-img", max_tokens=100, ignore_eos=False, streaming=False)
    content = payload["messages"][0]["content"]
    assert isinstance(content, list)
    assert len(content) == 2  # 1 text + 1 image
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"


@pytest.mark.asyncio
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
    item = _load_first(gen)

    assert item.multimodal_spec is not None
    assert len(item.multimodal_spec.videos) == 1
    assert item.multimodal_spec.videos[0].frames == 10  # default profile frames

    payload = await item.to_request_body(effective_model_name="gpt-video", max_tokens=100, ignore_eos=False, streaming=False)
    content = payload["messages"][0]["content"]
    assert isinstance(content, list)
    assert len(content) == 2  # video + text
    assert content[0]["type"] == "video_url"
    video_url = content[0]["video_url"]["url"]
    assert video_url.startswith("data:video/mp4;base64,")
    assert len(video_url) > len("data:video/mp4;base64,") + 100

    assert item.realized_videos is not None
    assert item.realized_videos.count == 1
    assert item.realized_videos.instances[0].frames == 10


@pytest.mark.asyncio
async def test_multimodal_datagen_image_jpeg_representation() -> None:
    """``image.representation: jpeg`` emits ``data:image/jpeg`` data URLs on the wire."""
    from inference_perf.payloads import ImageRepresentation

    api_config = APIConfig(type=APIType.Chat)
    data_config = DataConfig(
        type=DataGenType.Synthetic,
        multimodal=SyntheticMultimodalDatagenConfig(
            image=ImageDatagenConfig(
                count=Distribution(type=DistributionType.UNIFORM, min=1, max=1, mean=1),
                insertion_point=0.0,
                representation=ImageRepresentation.JPEG,
            ),
        ),
    )
    gen = MultimodalDataGenerator(api_config, data_config, _make_mock_tokenizer())
    item = _load_first(gen)

    assert item.multimodal_spec is not None
    assert item.multimodal_spec.images[0].representation == ImageRepresentation.JPEG

    payload = await item.to_request_body(effective_model_name="gpt", max_tokens=10, ignore_eos=False, streaming=False)
    image_blocks = [c for c in payload["messages"][0]["content"] if c.get("type") == "image_url"]
    assert len(image_blocks) == 1
    url = image_blocks[0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_multimodal_datagen_video_jpeg_frames() -> None:
    """``representation: jpeg_frames`` emits JPEG-encoded frame blocks."""
    from inference_perf.payloads import VideoRepresentation

    api_config = APIConfig(type=APIType.Chat)
    data_config = DataConfig(
        type=DataGenType.Synthetic,
        multimodal=SyntheticMultimodalDatagenConfig(
            video=VideoDatagenConfig(
                count=Distribution(type=DistributionType.UNIFORM, min=1, max=1, mean=1),
                insertion_point=0.0,
                representation=VideoRepresentation.JPEG_FRAMES,
            ),
        ),
    )
    gen = MultimodalDataGenerator(api_config, data_config, _make_mock_tokenizer())
    item = _load_first(gen)

    payload = await item.to_request_body(effective_model_name="gpt", max_tokens=10, ignore_eos=False, streaming=False)
    image_blocks = [c for c in payload["messages"][0]["content"] if c.get("type") == "image_url"]
    assert len(image_blocks) > 0
    assert all(b["image_url"]["url"].startswith("data:image/jpeg;base64,") for b in image_blocks)


@pytest.mark.asyncio
async def test_multimodal_datagen_video_png_frames() -> None:
    """`representation: png_frames` emits N image_url blocks but reports as one Video."""
    from inference_perf.payloads import VideoRepresentation

    api_config = APIConfig(type=APIType.Chat)
    data_config = DataConfig(
        type=DataGenType.Synthetic,
        multimodal=SyntheticMultimodalDatagenConfig(
            video=VideoDatagenConfig(
                count=Distribution(type=DistributionType.UNIFORM, min=1, max=1, mean=1),
                insertion_point=0.0,
                representation=VideoRepresentation.PNG_FRAMES,
            ),
        ),
    )
    gen = MultimodalDataGenerator(api_config, data_config, _make_mock_tokenizer())
    item = _load_first(gen)

    from inference_perf.payloads import ImageRepresentation, SyntheticFramesVideoSpec

    assert item.multimodal_spec is not None
    assert len(item.multimodal_spec.videos) == 1
    vid = item.multimodal_spec.videos[0]
    assert isinstance(vid, SyntheticFramesVideoSpec)
    expected_frames = vid.frames
    assert vid.frame_representation == ImageRepresentation.PNG

    payload = await item.to_request_body(effective_model_name="gpt-video", max_tokens=100, ignore_eos=False, streaming=False)
    content = payload["messages"][0]["content"]
    assert isinstance(content, list)
    # N image_url frames + 1 trailing text block (insertion_point=0.0 puts media first).
    image_blocks = [c for c in content if c.get("type") == "image_url"]
    video_blocks = [c for c in content if c.get("type") == "video_url"]
    assert len(image_blocks) == expected_frames
    assert len(video_blocks) == 0
    assert image_blocks[0]["image_url"]["url"].startswith("data:image/png;base64,")

    # Realized metric: one logical Video, not N Images. bytes is the sum across frames.
    assert item.realized_videos is not None
    assert item.realized_videos.count == 1
    assert item.realized_videos.instances[0].frames == expected_frames
    assert item.realized_videos.instances[0].bytes > 0
    assert item.realized_images is None


@pytest.mark.asyncio
async def test_multimodal_datagen_interleaved_center() -> None:
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
    item = _load_first(gen)

    payload = await item.to_request_body(effective_model_name="gpt-img", max_tokens=100, ignore_eos=False, streaming=False)
    content = payload["messages"][0]["content"]
    assert isinstance(content, list)
    # With insertion_point=0.5: Text(part 1), Image, Text(part 2)
    assert len(content) == 3
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[2]["type"] == "text"


@pytest.mark.asyncio
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
    item = _load_first(gen)

    payload = await item.to_request_body(effective_model_name="gpt-audio", max_tokens=100, ignore_eos=False, streaming=False)
    content = payload["messages"][0]["content"]
    assert isinstance(content, list)
    assert len(content) == 2  # 1 audio + 1 text
    assert content[0]["type"] == "input_audio"
    assert content[0]["input_audio"]["format"] == "wav"
    assert "data" in content[0]["input_audio"]

    assert item.realized_audios is not None
    assert item.realized_audios.count == 1


def test_multimodal_datagen_returns_text_only_before_request_body() -> None:
    """Datagen must not generate bytes or build a multimodal content list —
    it produces only the typed spec, kept lightweight for the lifecycle metric."""
    api_config = APIConfig(type=APIType.Chat)
    data_config = DataConfig(
        type=DataGenType.Synthetic,
        multimodal=SyntheticMultimodalDatagenConfig(
            image=ImageDatagenConfig(
                count=Distribution(type=DistributionType.UNIFORM, min=3, max=3, mean=3),
                insertion_point=0.5,
            ),
        ),
    )
    gen = MultimodalDataGenerator(api_config, data_config, _make_mock_tokenizer())
    item = _load_first(gen)

    assert isinstance(item.messages[0].content, str)
    # No bytes have been materialized yet — realized_* are still empty.
    assert item.realized_images is None
    assert item.realized_videos is None
    assert item.realized_audios is None

    # Spec is fully populated with concrete sampled values.
    assert item.multimodal_spec is not None
    assert len(item.multimodal_spec.images) == 3
    assert all(isinstance(img.width, int) and img.width > 0 for img in item.multimodal_spec.images)


# Silence unused import warning when running this file standalone.
_ = cast
