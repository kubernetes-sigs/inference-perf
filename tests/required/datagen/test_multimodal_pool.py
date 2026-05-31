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
"""End-to-end checks for the bounded-distinct payload media pool (#498)."""

from typing import Generator, cast
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
    MediaPoolConfig,
    Resolution,
    SharedPrefix,
    SyntheticMultimodalDatagenConfig,
    VideoDatagenConfig,
    VideoProfile,
)
from inference_perf.datagen.multimodal_datagen import MultimodalDataGenerator
from inference_perf.datagen.multimodal_sampling import reset_payload_pools
from inference_perf.datagen.shared_prefix_datagen import SharedPrefixDataGenerator
from inference_perf.payloads.audio import pool as audio_pool_mod
from inference_perf.payloads.image import pool as image_pool_mod
from inference_perf.payloads.video import pool as video_pool_mod
from inference_perf.payloads import SyntheticImageSpec
from inference_perf.utils.custom_tokenizer import CustomTokenizer


@pytest.fixture(autouse=True)
def _reset_pools() -> Generator[None, None, None]:
    reset_payload_pools()
    yield
    reset_payload_pools()


def _make_mock_tokenizer() -> MagicMock:
    """Mock that round-trips exact-length text via "text_N" markers (mirrors
    ``test_shared_prefix_multimodal.py``)."""
    mock_tokenizer = MagicMock(spec=CustomTokenizer)
    hf = MagicMock()
    hf.vocab_size = 1000
    hf.decode = MagicMock(side_effect=lambda ids, **kw: f"text_{len(ids)}")
    hf.batch_decode = MagicMock(side_effect=lambda batch, **kw: [f"text_{len(ids)}" for ids in batch])
    mock_tokenizer.get_tokenizer.return_value = hf

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


def _fixed_count(n: int) -> Distribution:
    return Distribution(type=DistributionType.UNIFORM, min=n, max=n, mean=n, std_dev=0.0)


@pytest.mark.asyncio
async def test_payload_image_pool_bounds_distinct_blobs() -> None:
    """Payload-side image bytes across many requests collapse to exactly ``pool.size``
    distinct values when ``image.pool`` is configured."""
    data_config = DataConfig(
        type=DataGenType.Synthetic,
        multimodal=SyntheticMultimodalDatagenConfig(
            image=ImageDatagenConfig(
                count=_fixed_count(1),
                insertion_point=0.0,
                pool=MediaPoolConfig(size=3),
            ),
        ),
    )
    gen = MultimodalDataGenerator(APIConfig(type=APIType.Chat), data_config, _make_mock_tokenizer())

    pool = image_pool_mod.get_pool()
    assert pool is not None
    assert pool.size == 3

    urls = set()
    data_iter = gen.get_data()
    for _ in range(40):
        item = gen.load_lazy_data(cast(LazyLoadInferenceAPIData, next(data_iter)))
        assert isinstance(item, ChatCompletionAPIData)
        payload = await item.to_request_body(effective_model_name="m", max_tokens=10, ignore_eos=False, streaming=False)
        for block in payload["messages"][0]["content"]:
            if block.get("type") == "image_url":
                urls.add(block["image_url"]["url"])

    assert len(urls) == 3


@pytest.mark.asyncio
async def test_synthetic_image_specs_carry_pool_index() -> None:
    data_config = DataConfig(
        type=DataGenType.Synthetic,
        multimodal=SyntheticMultimodalDatagenConfig(
            image=ImageDatagenConfig(
                count=_fixed_count(2),
                insertion_point=0.5,
                resolutions=Resolution(width=24, height=24),
                pool=MediaPoolConfig(size=2),
            ),
        ),
    )
    gen = MultimodalDataGenerator(APIConfig(type=APIType.Chat), data_config, _make_mock_tokenizer())
    item = gen.load_lazy_data(cast(LazyLoadInferenceAPIData, next(gen.get_data())))
    assert isinstance(item, ChatCompletionAPIData)
    assert item.multimodal_spec is not None
    for img in item.multimodal_spec.images:
        assert isinstance(img, SyntheticImageSpec)
        assert img.pool_index is not None
        assert 0 <= img.pool_index < 2
        # Pool entry dimensions mirror onto the spec so ``get_metrics`` is
        # exact without a pool round-trip.
        assert img.width == 24 and img.height == 24


@pytest.mark.asyncio
async def test_payload_audio_pool_bounds_distinct_durations() -> None:
    data_config = DataConfig(
        type=DataGenType.Synthetic,
        multimodal=SyntheticMultimodalDatagenConfig(
            audio=AudioDatagenConfig(
                count=_fixed_count(1),
                insertion_point=0.0,
                durations=0.1,
                pool=MediaPoolConfig(size=2),
            ),
        ),
    )
    gen = MultimodalDataGenerator(APIConfig(type=APIType.Chat), data_config, _make_mock_tokenizer())
    pool = audio_pool_mod.get_pool()
    assert pool is not None
    assert pool.size == 2

    item = gen.load_lazy_data(cast(LazyLoadInferenceAPIData, next(gen.get_data())))
    assert isinstance(item, ChatCompletionAPIData)
    assert item.multimodal_spec is not None
    for aud in item.multimodal_spec.audios:
        assert aud.pool_index is not None
        assert 0 <= aud.pool_index < 2


@pytest.mark.asyncio
async def test_payload_video_mp4_pool_replaces_legacy_per_profile_bucket() -> None:
    data_config = DataConfig(
        type=DataGenType.Synthetic,
        multimodal=SyntheticMultimodalDatagenConfig(
            video=VideoDatagenConfig(
                count=_fixed_count(1),
                insertion_point=0.0,
                profiles=VideoProfile(resolution=Resolution(width=32, height=32), frames=2),
                pool=MediaPoolConfig(size=2),
            ),
        ),
    )
    gen = MultimodalDataGenerator(APIConfig(type=APIType.Chat), data_config, _make_mock_tokenizer())
    pool = video_pool_mod.get_pool()
    assert pool is not None
    assert pool.size == 2

    urls = set()
    data_iter = gen.get_data()
    for _ in range(20):
        item = gen.load_lazy_data(cast(LazyLoadInferenceAPIData, next(data_iter)))
        assert isinstance(item, ChatCompletionAPIData)
        payload = await item.to_request_body(effective_model_name="m", max_tokens=10, ignore_eos=False, streaming=False)
        for block in payload["messages"][0]["content"]:
            if block.get("type") == "video_url":
                urls.add(block["video_url"]["url"])

    # Bounded to exactly pool.size (not the legacy hidden 4-per-profile bucket).
    assert len(urls) == 2


@pytest.mark.asyncio
async def test_no_pool_config_preserves_legacy_behavior() -> None:
    """Omitting ``pool`` leaves the registry empty so the legacy code path runs:
    fresh image bytes per request (so the distinct-URL set grows with N)."""
    data_config = DataConfig(
        type=DataGenType.Synthetic,
        multimodal=SyntheticMultimodalDatagenConfig(
            image=ImageDatagenConfig(count=_fixed_count(1), insertion_point=0.0),
        ),
    )
    gen = MultimodalDataGenerator(APIConfig(type=APIType.Chat), data_config, _make_mock_tokenizer())
    assert image_pool_mod.get_pool() is None

    urls = set()
    data_iter = gen.get_data()
    for _ in range(8):
        item = gen.load_lazy_data(cast(LazyLoadInferenceAPIData, next(data_iter)))
        assert isinstance(item, ChatCompletionAPIData)
        payload = await item.to_request_body(effective_model_name="m", max_tokens=10, ignore_eos=False, streaming=False)
        for block in payload["messages"][0]["content"]:
            if block.get("type") == "image_url":
                urls.add(block["image_url"]["url"])
    # Fresh bytes per request => 8 distinct URLs (overwhelmingly likely with random colors).
    assert len(urls) == 8


@pytest.mark.asyncio
async def test_shared_prefix_pool_does_not_affect_prefix_bytes() -> None:
    """With ``pool`` configured on the payload, prefix bytes remain deterministic per
    group (prefix-cache benchmarks still work) while payload bytes draw from the pool."""
    multimodal_config = SyntheticMultimodalDatagenConfig(
        image=ImageDatagenConfig(count=_fixed_count(1), insertion_point=0.0, pool=MediaPoolConfig(size=2)),
    )
    prefix_multimodal = SyntheticMultimodalDatagenConfig(
        image=ImageDatagenConfig(count=_fixed_count(1), insertion_point=0.0),
    )
    data_config = DataConfig(
        type=DataGenType.SharedPrefix,
        multimodal=multimodal_config,
        shared_prefix=SharedPrefix(
            num_groups=1,
            num_prompts_per_group=6,
            multimodal=prefix_multimodal,
        ),
    )
    gen = SharedPrefixDataGenerator(APIConfig(type=APIType.Chat), data_config, _make_mock_tokenizer())
    assert image_pool_mod.get_pool() is not None  # payload pool registered
    iter_data = gen.get_data()

    prefix_urls = []
    payload_urls = set()
    for _ in range(6):
        api_data = gen.load_lazy_data(cast(LazyLoadInferenceAPIData, next(iter_data)))
        assert isinstance(api_data, ChatCompletionAPIData)
        # Prefix-side specs always leave pool_index=None (deterministic seeded path).
        assert api_data.prefix_multimodal_spec is not None
        for img in api_data.prefix_multimodal_spec.images:
            assert isinstance(img, SyntheticImageSpec)
            assert img.pool_index is None
        # Payload-side specs carry a pool_index.
        assert api_data.multimodal_spec is not None
        for img in api_data.multimodal_spec.images:
            assert isinstance(img, SyntheticImageSpec)
            assert img.pool_index is not None

        payload = await api_data.to_request_body(
            effective_model_name="m", max_tokens=10, ignore_eos=False, streaming=False
        )
        image_blocks = [c for c in payload["messages"][0]["content"] if c.get("type") == "image_url"]
        # Two image blocks per request: prefix + payload.
        assert len(image_blocks) == 2
        prefix_urls.append(image_blocks[0]["image_url"]["url"])
        payload_urls.add(image_blocks[1]["image_url"]["url"])

    # Prefix bytes are identical across the 6 requests in the single group.
    assert len(set(prefix_urls)) == 1
    # Payload bytes are drawn from a pool of 2.
    assert len(payload_urls) == 2
