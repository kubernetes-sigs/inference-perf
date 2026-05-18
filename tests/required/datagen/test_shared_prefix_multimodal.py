from unittest.mock import MagicMock
import pytest

from inference_perf.datagen.shared_prefix_datagen import SharedPrefixDataGenerator
from inference_perf.config import (
    APIConfig,
    DataConfig,
    Resolution,
    SharedPrefix,
    SyntheticMultimodalDatagenConfig,
    ImageDatagenConfig,
    VideoDatagenConfig,
    VideoProfile,
    APIType,
    DataGenType,
    Distribution,
)
from inference_perf.payloads import VideoRepresentation
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


def _build_generator(num_groups: int = 1, num_prompts_per_group: int = 2) -> SharedPrefixDataGenerator:
    """Generator with one prefix-side image per group + one payload-side image per request."""
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
            num_groups=num_groups,
            num_prompts_per_group=num_prompts_per_group,
            multimodal=shared_prefix_multimodal,  # Prefix
        ),
    )
    return SharedPrefixDataGenerator(APIConfig(type="chat"), data_config, _make_mock_tokenizer())


def test_shared_prefix_multimodal_post_load_shape() -> None:
    """After phase-2 port, post-load API data carries text-only messages plus
    typed prefix/payload specs. Wire content is built at to_request_body time."""
    generator = _build_generator()
    assert APIType.Chat in generator.get_supported_apis()

    lazy_data = next(generator.get_data())
    api_data = generator.load_lazy_data(cast(LazyLoadInferenceAPIData, lazy_data))

    assert isinstance(api_data, ChatCompletionAPIData)
    assert len(api_data.messages) == 1
    assert isinstance(api_data.messages[0].content, str)  # text-only at this stage
    assert api_data.prefix_text is not None
    assert api_data.prefix_multimodal_spec is not None
    assert len(api_data.prefix_multimodal_spec.images) == 1
    assert api_data.multimodal_spec is not None
    assert len(api_data.multimodal_spec.images) == 1


@pytest.mark.asyncio
async def test_shared_prefix_multimodal_request_body_has_both_images() -> None:
    """to_request_body materializes prefix + payload media into the user message."""
    generator = _build_generator()
    api_data = generator.load_lazy_data(cast(LazyLoadInferenceAPIData, next(generator.get_data())))
    assert isinstance(api_data, ChatCompletionAPIData)

    payload = await api_data.to_request_body(effective_model_name="test", max_tokens=10, ignore_eos=False, streaming=False)
    content = payload["messages"][0]["content"]
    assert isinstance(content, list)
    image_blocks = [c for c in content if c.get("type") == "image_url"]
    assert len(image_blocks) == 2  # one from prefix, one from payload

    # Realized metrics aggregate prefix + payload images.
    assert api_data.realized_images is not None
    assert api_data.realized_images.count == 2


@pytest.mark.asyncio
async def test_shared_prefix_multimodal_prefix_bytes_stable_across_requests() -> None:
    """Prefix-side bytes must be identical across requests in the same group
    (server prefix-cache hits depend on this)."""
    generator = _build_generator(num_prompts_per_group=2)
    iter_data = generator.get_data()

    api_data_1 = generator.load_lazy_data(cast(LazyLoadInferenceAPIData, next(iter_data)))
    api_data_2 = generator.load_lazy_data(cast(LazyLoadInferenceAPIData, next(iter_data)))
    assert isinstance(api_data_1, ChatCompletionAPIData) and isinstance(api_data_2, ChatCompletionAPIData)
    # Same group: same prefix spec.
    assert api_data_1.prefix_multimodal_spec == api_data_2.prefix_multimodal_spec

    payload_1 = await api_data_1.to_request_body(effective_model_name="t", max_tokens=10, ignore_eos=False, streaming=False)
    payload_2 = await api_data_2.to_request_body(effective_model_name="t", max_tokens=10, ignore_eos=False, streaming=False)

    # The first image_url block of each request comes from the prefix spec
    # (prefix-side images use insertion_point=0.0). They must match byte-for-byte.
    prefix_block_1 = next(c for c in payload_1["messages"][0]["content"] if c.get("type") == "image_url")
    prefix_block_2 = next(c for c in payload_2["messages"][0]["content"] if c.get("type") == "image_url")
    assert prefix_block_1["image_url"]["url"] == prefix_block_2["image_url"]["url"]


@pytest.mark.asyncio
async def test_shared_prefix_multimodal_prefix_image_sets_partition_by_group() -> None:
    """With 3 groups × 2 prompts/group, the 6 prompts must partition into
    exactly 3 groups of 2, where:
      - prompts within a group share the same prefix image set, and
      - no group's prefix image set matches another group's.

    Regression test for the bug where the deterministic seed lacked a per-group
    component, causing every group to produce byte-identical prefix images.
    """
    num_groups = 3
    num_prompts_per_group = 2
    generator = _build_generator(num_groups=num_groups, num_prompts_per_group=num_prompts_per_group)

    # Drain all 6 prompts. Each request carries one prefix image (deterministic
    # per group, repeats across prompts in the group) and one payload image
    # (sampled fresh per request, unique). Bucket by group_id (=
    # ``prefix_cache_key``) and collect rendered image URLs.
    iter_data = generator.get_data()
    urls_by_group: dict[int, list[list[str]]] = {}
    for _ in range(num_groups * num_prompts_per_group):
        api_data = generator.load_lazy_data(cast(LazyLoadInferenceAPIData, next(iter_data)))
        assert isinstance(api_data, ChatCompletionAPIData)
        group_id = api_data.prefix_cache_key
        assert group_id is not None

        payload = await api_data.to_request_body(effective_model_name="t", max_tokens=10, ignore_eos=False, streaming=False)
        urls = [c["image_url"]["url"] for c in payload["messages"][0]["content"] if c.get("type") == "image_url"]
        urls_by_group.setdefault(group_id, []).append(urls)

    # Partition shape: exactly num_groups groups, each with num_prompts_per_group prompts.
    assert len(urls_by_group) == num_groups, (
        f"expected {num_groups} distinct groups, got {len(urls_by_group)}: {sorted(urls_by_group)}"
    )
    for gid, url_lists in urls_by_group.items():
        assert len(url_lists) == num_prompts_per_group, (
            f"group {gid}: expected {num_prompts_per_group} prompts, got {len(url_lists)}"
        )

    # The prefix images for a group are the URLs that appear in every one of
    # the group's requests (the payload-side image is fresh per request and
    # won't repeat). Intersect URL sets across the group's requests to
    # recover the prefix image set without depending on positional ordering.
    prefix_image_set_by_group: dict[int, frozenset[str]] = {
        gid: frozenset.intersection(*(frozenset(urls) for urls in url_lists)) for gid, url_lists in urls_by_group.items()
    }
    for gid, prefix_set in prefix_image_set_by_group.items():
        assert prefix_set, f"group {gid}: no URL appears in every request — prefix image is not stable within the group"

    # The prefix image set of any group must not match another group's set.
    distinct_sets = set(prefix_image_set_by_group.values())
    assert len(distinct_sets) == num_groups, (
        f"expected {num_groups} distinct prefix image sets across groups, got {len(distinct_sets)}: "
        f"{prefix_image_set_by_group}"
    )


def _build_generator_with_prefix_video(representation: VideoRepresentation) -> SharedPrefixDataGenerator:
    """Generator with one prefix-side video at the configured representation, no payload media."""
    shared_prefix_multimodal = SyntheticMultimodalDatagenConfig(
        video=VideoDatagenConfig(
            count=Distribution(min=1, max=1, mean=1, std_dev=0),
            insertion_point=0.0,
            representation=representation,
            profiles=VideoProfile(resolution=Resolution(width=64, height=64), frames=4),
        )
    )
    data_config = DataConfig(
        type=DataGenType.SharedPrefix,
        shared_prefix=SharedPrefix(num_groups=1, num_prompts_per_group=2, multimodal=shared_prefix_multimodal),
    )
    return SharedPrefixDataGenerator(APIConfig(type="chat"), data_config, _make_mock_tokenizer())


@pytest.mark.asyncio
async def test_shared_prefix_video_mp4_bytes_stable_across_requests() -> None:
    """Prefix-side videos in MP4 mode use deterministic seeding (bypassing the
    pool) so the same spec produces identical MP4 bytes across requests."""
    generator = _build_generator_with_prefix_video(VideoRepresentation.MP4)
    iter_data = generator.get_data()

    api_data_1 = generator.load_lazy_data(cast(LazyLoadInferenceAPIData, next(iter_data)))
    api_data_2 = generator.load_lazy_data(cast(LazyLoadInferenceAPIData, next(iter_data)))
    assert isinstance(api_data_1, ChatCompletionAPIData) and isinstance(api_data_2, ChatCompletionAPIData)

    p1 = await api_data_1.to_request_body(effective_model_name="t", max_tokens=10, ignore_eos=False, streaming=False)
    p2 = await api_data_2.to_request_body(effective_model_name="t", max_tokens=10, ignore_eos=False, streaming=False)

    video_block_1 = next(c for c in p1["messages"][0]["content"] if c.get("type") == "video_url")
    video_block_2 = next(c for c in p2["messages"][0]["content"] if c.get("type") == "video_url")
    assert video_block_1["video_url"]["url"] == video_block_2["video_url"]["url"]


@pytest.mark.asyncio
async def test_shared_prefix_video_png_frames_bytes_stable_across_requests() -> None:
    """Prefix-side videos in PNG_FRAMES mode produce identical PNG bytes per frame
    across requests. Every emitted image_url block must match byte-for-byte."""
    generator = _build_generator_with_prefix_video(VideoRepresentation.PNG_FRAMES)
    iter_data = generator.get_data()

    api_data_1 = generator.load_lazy_data(cast(LazyLoadInferenceAPIData, next(iter_data)))
    api_data_2 = generator.load_lazy_data(cast(LazyLoadInferenceAPIData, next(iter_data)))
    assert isinstance(api_data_1, ChatCompletionAPIData) and isinstance(api_data_2, ChatCompletionAPIData)

    p1 = await api_data_1.to_request_body(effective_model_name="t", max_tokens=10, ignore_eos=False, streaming=False)
    p2 = await api_data_2.to_request_body(effective_model_name="t", max_tokens=10, ignore_eos=False, streaming=False)

    frames_1 = [c["image_url"]["url"] for c in p1["messages"][0]["content"] if c.get("type") == "image_url"]
    frames_2 = [c["image_url"]["url"] for c in p2["messages"][0]["content"] if c.get("type") == "image_url"]
    assert frames_1 == frames_2
    assert len(frames_1) == 4  # matches the configured frame count


@pytest.mark.asyncio
async def test_shared_prefix_video_jpeg_frames_bytes_stable_across_requests() -> None:
    """Byte-stability contract holds for JPEG-encoded frames too."""
    shared_prefix_multimodal = SyntheticMultimodalDatagenConfig(
        video=VideoDatagenConfig(
            count=Distribution(min=1, max=1, mean=1, std_dev=0),
            insertion_point=0.0,
            representation=VideoRepresentation.JPEG_FRAMES,
            profiles=VideoProfile(resolution=Resolution(width=64, height=64), frames=4),
        )
    )
    data_config = DataConfig(
        type=DataGenType.SharedPrefix,
        shared_prefix=SharedPrefix(num_groups=1, num_prompts_per_group=2, multimodal=shared_prefix_multimodal),
    )
    generator = SharedPrefixDataGenerator(APIConfig(type="chat"), data_config, _make_mock_tokenizer())

    iter_data = generator.get_data()
    api_data_1 = generator.load_lazy_data(cast(LazyLoadInferenceAPIData, next(iter_data)))
    api_data_2 = generator.load_lazy_data(cast(LazyLoadInferenceAPIData, next(iter_data)))
    assert isinstance(api_data_1, ChatCompletionAPIData) and isinstance(api_data_2, ChatCompletionAPIData)

    p1 = await api_data_1.to_request_body(effective_model_name="t", max_tokens=10, ignore_eos=False, streaming=False)
    p2 = await api_data_2.to_request_body(effective_model_name="t", max_tokens=10, ignore_eos=False, streaming=False)

    frames_1 = [c["image_url"]["url"] for c in p1["messages"][0]["content"] if c.get("type") == "image_url"]
    frames_2 = [c["image_url"]["url"] for c in p2["messages"][0]["content"] if c.get("type") == "image_url"]
    assert frames_1 == frames_2
    assert all(u.startswith("data:image/jpeg;base64,") for u in frames_1)
    assert len(frames_1) == 4
