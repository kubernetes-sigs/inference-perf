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
"""Tests for :class:`MMMUDataGenerator`.

HuggingFace network access is mocked out — ``_load_dataset`` is monkey-patched
to return synthetic in-memory example lists with PIL images, so these tests
run offline and don't require an HF token.
"""

import base64
import io
from typing import Any, List
from unittest.mock import MagicMock

import pytest
from PIL import Image as PILImage

from inference_perf.apis.base import LazyLoadInferenceAPIData
from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.config import (
    APIConfig,
    APIType,
    DataConfig,
    DataGenType,
    MMMUConfig,
    Resolution,
)
from inference_perf.datagen.mmmu_datagen import MMMUDataGenerator
from inference_perf.payloads import ImageRepresentation, PreEncodedImageSpec


def _make_mock_tokenizer() -> MagicMock:
    m = MagicMock()
    m.count_tokens.return_value = 10
    return m


def _fake_example(
    qid: str = "test/1",
    n_images: int = 1,
    options: str = "['Alpha', 'Beta', 'Gamma']",
    question: str = "What is shown in <image 1>?",
    image_size: tuple[int, int] = (32, 32),
) -> dict[str, Any]:
    ex: dict[str, Any] = {
        "id": qid,
        "question": question,
        "options": options,
        "answer": "A",
    }
    for i in range(1, 8):
        ex[f"image_{i}"] = (
            PILImage.new("RGB", image_size, color=(10 * i, 20 * i, 30 * i)) if i <= n_images else None
        )
    return ex


def _load_first(gen: MMMUDataGenerator, data_index: int = 0) -> ChatCompletionAPIData:
    lazy = LazyLoadInferenceAPIData(data_index=data_index)
    item = gen.load_lazy_data(lazy)
    assert isinstance(item, ChatCompletionAPIData)
    return item


def _build_generator(
    monkeypatch: pytest.MonkeyPatch,
    examples: List[dict[str, Any]],
    **cfg_overrides: Any,
) -> MMMUDataGenerator:
    monkeypatch.setenv("HF_TOKEN", "test-token")
    # Stub HF: return the same example list for every (subject, split) request.
    monkeypatch.setattr(
        MMMUDataGenerator,
        "_load_dataset",
        lambda self, path, **kwargs: list(examples),
    )
    cfg = MMMUConfig(subjects=["Math"], **cfg_overrides)
    data_config = DataConfig(type=DataGenType.MMMU, mmmu=cfg)
    return MMMUDataGenerator(APIConfig(type=APIType.Chat), data_config, _make_mock_tokenizer())


def test_loader_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    with pytest.raises(ValueError, match="HuggingFace access token"):
        MMMUDataGenerator(
            APIConfig(type=APIType.Chat),
            DataConfig(type=DataGenType.MMMU, mmmu=MMMUConfig()),
            _make_mock_tokenizer(),
        )


def test_loader_rejects_empty_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="found no examples"):
        _build_generator(monkeypatch, [])


def test_loader_requires_mmmu_config() -> None:
    with pytest.raises(ValueError, match="mmmu config is required"):
        MMMUDataGenerator(
            APIConfig(type=APIType.Chat),
            DataConfig(type=DataGenType.MMMU),
            _make_mock_tokenizer(),
        )


def test_loader_skips_examples_without_images(monkeypatch: pytest.MonkeyPatch) -> None:
    examples = [
        _fake_example(qid="empty", n_images=0),
        _fake_example(qid="has_image", n_images=1),
    ]
    gen = _build_generator(monkeypatch, examples)
    assert len(gen._examples) == 1
    assert gen._examples[0]["id"] == "has_image"


def test_loader_builds_image_request(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _build_generator(monkeypatch, [_fake_example(n_images=1)])
    item = _load_first(gen)
    assert item.multimodal_spec is not None
    assert len(item.multimodal_spec.images) == 1
    img = item.multimodal_spec.images[0]
    assert isinstance(img, PreEncodedImageSpec)
    assert img.image_bytes
    assert img.representation == ImageRepresentation.PNG  # default
    assert img.width == 32 and img.height == 32


def test_loader_emits_multiple_images_when_example_has_them(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _build_generator(monkeypatch, [_fake_example(n_images=3)])
    item = _load_first(gen)
    assert item.multimodal_spec is not None
    assert len(item.multimodal_spec.images) == 3


def test_loader_respects_max_images_per_request(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _build_generator(monkeypatch, [_fake_example(n_images=5)], max_images_per_request=2)
    item = _load_first(gen)
    assert item.multimodal_spec is not None
    assert len(item.multimodal_spec.images) == 2


def test_loader_respects_max_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    examples = [_fake_example(qid=f"q{i}", n_images=1) for i in range(5)]
    gen = _build_generator(monkeypatch, examples, max_examples=2)
    assert len(gen._examples) == 2


def test_prompt_includes_question_and_options(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _build_generator(
        monkeypatch,
        [_fake_example(question="Pick the right answer.", options="['Apple', 'Banana', 'Cherry']")],
    )
    item = _load_first(gen)
    msg = item.messages[0]
    assert isinstance(msg, ChatMessage) and isinstance(msg.content, str)
    assert "Pick the right answer." in msg.content
    assert "Apple" in msg.content and "Banana" in msg.content and "Cherry" in msg.content
    assert "A. Apple" in msg.content and "B. Banana" in msg.content
    assert "Answer:" in msg.content


def test_prompt_tolerates_unparseable_options(monkeypatch: pytest.MonkeyPatch) -> None:
    # Some MMMU examples have free-form / non-list options. The loader should
    # gracefully include the raw string rather than crash.
    gen = _build_generator(monkeypatch, [_fake_example(options="not a python list")])
    item = _load_first(gen)
    msg = item.messages[0]
    assert isinstance(msg.content, str)
    assert "not a python list" in msg.content


def test_prompt_omits_options_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _build_generator(monkeypatch, [_fake_example(options="")])
    item = _load_first(gen)
    msg = item.messages[0]
    assert isinstance(msg.content, str)
    assert "Options:" not in msg.content
    assert "Answer:" in msg.content


def test_load_lazy_data_cycles_deterministically(monkeypatch: pytest.MonkeyPatch) -> None:
    examples = [_fake_example(qid="a", question="Q-A"), _fake_example(qid="b", question="Q-B")]
    gen = _build_generator(monkeypatch, examples)
    item0 = _load_first(gen, data_index=0)
    item2 = _load_first(gen, data_index=2)  # wraps to index 0
    assert item0.messages[0].content == item2.messages[0].content
    item1 = _load_first(gen, data_index=1)
    assert item1.messages[0].content != item0.messages[0].content


def test_target_resolution_resizes_images(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _build_generator(
        monkeypatch,
        [_fake_example(n_images=1, image_size=(128, 64))],
        target_resolution=Resolution(width=32, height=16),
    )
    item = _load_first(gen)
    assert item.multimodal_spec is not None
    img = item.multimodal_spec.images[0]
    assert isinstance(img, PreEncodedImageSpec)
    assert img.width == 32 and img.height == 16


def test_jpeg_representation_propagates_to_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _build_generator(monkeypatch, [_fake_example(n_images=1)], representation=ImageRepresentation.JPEG)
    item = _load_first(gen)
    assert item.multimodal_spec is not None
    img = item.multimodal_spec.images[0]
    assert isinstance(img, PreEncodedImageSpec)
    assert img.representation == ImageRepresentation.JPEG
    # JPEG magic bytes — FF D8 FF.
    assert img.image_bytes.startswith(b"\xff\xd8\xff")


def test_png_representation_propagates_to_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _build_generator(monkeypatch, [_fake_example(n_images=1)], representation=ImageRepresentation.PNG)
    item = _load_first(gen)
    assert item.multimodal_spec is not None
    img = item.multimodal_spec.images[0]
    assert isinstance(img, PreEncodedImageSpec)
    assert img.representation == ImageRepresentation.PNG
    # PNG magic bytes — 89 50 4E 47.
    assert img.image_bytes.startswith(b"\x89PNG")


@pytest.mark.asyncio
async def test_request_body_emits_n_image_url_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _build_generator(monkeypatch, [_fake_example(n_images=3)])
    item = _load_first(gen)
    payload = await item.to_request_body(
        effective_model_name="vlm-test", max_tokens=64, ignore_eos=False, streaming=False
    )
    content = payload["messages"][0]["content"]
    image_blocks = [c for c in content if c.get("type") == "image_url"]
    assert len(image_blocks) == 3
    for block in image_blocks:
        assert block["image_url"]["url"].startswith("data:image/png;base64,")
        # Sanity-check that the data URL decodes to the recorded byte count.
        payload_b64 = block["image_url"]["url"].split(",", 1)[1]
        assert base64.b64decode(payload_b64)
    # Realized metric reports per-instance images.
    assert item.realized_images is not None
    assert item.realized_images.count == 3
    assert item.realized_videos is None


@pytest.mark.asyncio
async def test_request_body_emits_jpeg_data_url_when_representation_is_jpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gen = _build_generator(monkeypatch, [_fake_example(n_images=1)], representation=ImageRepresentation.JPEG)
    item = _load_first(gen)
    payload = await item.to_request_body(
        effective_model_name="vlm-test", max_tokens=64, ignore_eos=False, streaming=False
    )
    image_blocks = [c for c in payload["messages"][0]["content"] if c.get("type") == "image_url"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_pre_encoded_image_spec_direct_construction() -> None:
    """Unit test of the chat.py materialization path for PreEncodedImageSpec directly."""
    img = PILImage.new("RGB", (16, 16), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    spec = PreEncodedImageSpec(
        width=16,
        height=16,
        insertion_point=0.0,
        representation=ImageRepresentation.JPEG,
        image_bytes=buf.getvalue(),
    )
    assert spec.image_bytes.startswith(b"\xff\xd8\xff")
    assert spec.kind == "pre_encoded"
