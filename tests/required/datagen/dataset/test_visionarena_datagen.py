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
"""Loader behavior for the VisionArena-Chat data generator.

``load_dataset`` is monkeypatched with an in-memory iterable mirroring the
VisionArena schema (``conversation`` turns + undecoded ``images``), so these
tests never touch the network.
"""

import io
from typing import Any, List

import pytest
from PIL import Image as PILImage

from inference_perf.apis.base import LazyLoadInferenceAPIData
from inference_perf.apis.chat import ChatCompletionAPIData
from inference_perf.config import Config, DataGenType
from inference_perf.datagen.dataset import visionarena_datagen
from inference_perf.datagen.dataset.visionarena_datagen import VisionArenaDataGenerator
from inference_perf.payloads import ImageRepresentation, PreEncodedImageSpec


def _png_bytes(w: int = 16, h: int = 8) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (w, h), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(w: int = 16, h: int = 8) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (w, h), (30, 30, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def _webp_bytes(w: int = 16, h: int = 8) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (w, h), (30, 200, 30)).save(buf, format="WEBP")
    return buf.getvalue()


def _gif_bytes(w: int = 10, h: int = 10) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (w, h), (1, 2, 3)).save(buf, format="GIF")
    return buf.getvalue()


def _image_field(raw: bytes) -> dict[str, Any]:
    """Mirror the undecoded HuggingFace Image form (decode=false)."""
    return {"bytes": raw, "path": None}


def _row(prompt: str, image_blobs: List[bytes]) -> dict[str, Any]:
    # VisionArena declares conversation as List[List[{role, content}]] (each
    # turn wrapped in a single-element list); mirror that real shape here.
    return {
        "conversation": [[{"role": "user", "content": prompt}]],
        "images": [_image_field(b) for b in image_blobs],
    }


def _make_generator(monkeypatch: pytest.MonkeyPatch, rows: List[Any], **va_overrides: Any) -> VisionArenaDataGenerator:
    monkeypatch.setattr(visionarena_datagen, "load_dataset", lambda *a, **k: list(rows))
    config = Config.model_validate(
        {"api": {"type": "chat"}, "data": {"type": DataGenType.VisionArena, "visionarena": va_overrides}}
    )
    return VisionArenaDataGenerator(config.api, config.data, None)


def _materialize(gen: VisionArenaDataGenerator, idx: int = 0) -> ChatCompletionAPIData:
    """Realize one request and narrow it to the concrete chat type for mypy."""
    data = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=idx))
    assert isinstance(data, ChatCompletionAPIData)
    return data


def _images(gen: VisionArenaDataGenerator, idx: int = 0) -> List[PreEncodedImageSpec]:
    """Return the realized image specs of one request (all pre-encoded)."""
    spec = _materialize(gen, idx).multimodal_spec
    assert spec is not None
    result: List[PreEncodedImageSpec] = []
    for img in spec.images:
        assert isinstance(img, PreEncodedImageSpec)
        result.append(img)
    return result


def test_pool_built_and_request_carries_pre_encoded_image(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _png_bytes()
    gen = _make_generator(monkeypatch, [_row("describe", [raw]), _row("what is this", [_png_bytes(8, 8)])])
    assert len(gen._pool) == 2

    first = next(gen.get_data())
    assert isinstance(first, LazyLoadInferenceAPIData)

    data = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0))
    assert isinstance(data, ChatCompletionAPIData)
    assert data.messages[0].content == "describe"
    assert data.multimodal_spec is not None
    images = data.multimodal_spec.images
    assert len(images) == 1
    spec = images[0]
    assert isinstance(spec, PreEncodedImageSpec)
    assert spec.image_bytes == raw  # PNG passed through verbatim, not re-encoded
    assert spec.representation == ImageRepresentation.PNG
    assert (spec.width, spec.height) == (16, 8)


def test_cycles_through_pool_by_index(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _make_generator(monkeypatch, [_row("a", [_png_bytes()]), _row("b", [_png_bytes()])])
    assert _materialize(gen, 0).messages[0].content == "a"
    assert _materialize(gen, 1).messages[0].content == "b"
    # Wraps around.
    assert _materialize(gen, 2).messages[0].content == "a"


def test_skips_rows_without_prompt_or_images(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"conversation": [[{"role": "assistant", "content": "no user turn"}]], "images": [_image_field(_png_bytes())]},
        _row("has prompt but no images", []),
        _row("good row", [_png_bytes()]),
    ]
    gen = _make_generator(monkeypatch, rows)
    assert len(gen._pool) == 1
    assert _materialize(gen, 0).messages[0].content == "good row"


def test_extract_prompt_handles_nested_and_flat_conversation(monkeypatch: pytest.MonkeyPatch) -> None:
    # Real schema wraps each turn in a list: List[List[{role, content}]]. A flat
    # List[{role, content}] (mirror/variant) must also resolve.
    nested = {"conversation": [[{"role": "user", "content": "nested"}]], "images": [_image_field(_png_bytes())]}
    flat = {"conversation": [{"role": "user", "content": "flat"}], "images": [_image_field(_png_bytes())]}
    gen = _make_generator(monkeypatch, [nested, flat])
    assert len(gen._pool) == 2
    assert _materialize(gen, 0).messages[0].content == "nested"
    assert _materialize(gen, 1).messages[0].content == "flat"


def test_num_rows_caps_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_row(f"prompt {i}", [_png_bytes()]) for i in range(5)]
    gen = _make_generator(monkeypatch, rows, num_rows=2)
    assert len(gen._pool) == 2


def test_max_images_per_request_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _make_generator(
        monkeypatch,
        [_row("multi", [_png_bytes(), _png_bytes(), _png_bytes()])],
        max_images_per_request=2,
    )
    assert len(_images(gen)) == 2


def test_jpeg_passes_through_as_jpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _jpeg_bytes()
    gen = _make_generator(monkeypatch, [_row("jpeg", [raw])])
    spec = _images(gen)[0]
    assert spec.representation == ImageRepresentation.JPEG
    assert spec.image_bytes == raw


def test_webp_passes_through_as_webp(monkeypatch: pytest.MonkeyPatch) -> None:
    # WEBP is wire-supported, so it must be forwarded verbatim rather than
    # re-encoded to lossless PNG (which would inflate the payload 10-30x).
    raw = _webp_bytes()
    gen = _make_generator(monkeypatch, [_row("webp", [raw])])
    spec = _images(gen)[0]
    assert spec.representation == ImageRepresentation.WEBP
    assert spec.image_bytes == raw


def test_unsupported_format_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    # GIF arrives as encoded bytes in a non-wire format: skip it rather than
    # silently re-encode. A row whose only image is unsupported is dropped;
    # a supported image in the same row still comes through.
    rows = [
        _row("gif only", [_gif_bytes()]),
        _row("mixed", [_gif_bytes(), _png_bytes()]),
    ]
    gen = _make_generator(monkeypatch, rows, max_images_per_request=2)
    assert len(gen._pool) == 1  # "gif only" dropped, "mixed" kept
    assert _materialize(gen).messages[0].content == "mixed"
    images = _images(gen)
    assert len(images) == 1  # only the PNG survived
    assert images[0].representation == ImageRepresentation.PNG


def test_bare_pil_image_is_reencoded_to_png(monkeypatch: pytest.MonkeyPatch) -> None:
    # Some HF configs hand back decoded PIL images instead of undecoded bytes;
    # the loader must re-encode those (no original encoded bytes to pass through).
    row = {"conversation": [[{"role": "user", "content": "decoded"}]], "images": [PILImage.new("RGB", (12, 9), (0, 128, 0))]}
    gen = _make_generator(monkeypatch, [row])
    spec = _images(gen)[0]
    assert spec.representation == ImageRepresentation.PNG
    assert (spec.width, spec.height) == (12, 9)
    # Re-encoded bytes are a valid PNG.
    assert PILImage.open(io.BytesIO(spec.image_bytes)).format == "PNG"


def test_insertion_point_float_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _make_generator(monkeypatch, [_row("a", [_png_bytes()])], insertion_point=0.5)
    spec = _images(gen)[0]
    assert spec.insertion_point == 0.5


def test_insertion_point_distribution_sampled_and_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _make_generator(
        monkeypatch,
        [_row("a", [_png_bytes()])],
        insertion_point={"type": "uniform", "min": 0.0, "max": 1.0},
    )

    def ip(idx: int) -> float:
        return _images(gen, idx)[0].insertion_point

    # Reproducible: the same index always samples the same point (RNG seeded by index).
    assert ip(0) == ip(0)
    points = [ip(i) for i in range(8)]
    assert all(0.0 <= p <= 1.0 for p in points)  # sampled within the configured range
    assert len(set(points)) > 1  # and actually varies per request, not a constant


def test_empty_dataset_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError):
        _make_generator(monkeypatch, [])
