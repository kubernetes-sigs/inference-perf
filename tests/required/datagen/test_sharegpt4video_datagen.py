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
"""Unit tests for the ShareGPT4Video loader and the pre-encoded-frames path."""

from __future__ import annotations

import io
import logging
import os
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
    ShareGPT4VideoConfig,
)
from inference_perf.datagen import sharegpt4video_datagen
from inference_perf.datagen.sharegpt4video_datagen import ShareGPT4VideoDataGenerator
from inference_perf.payloads import (
    ImageRepresentation,
    MultimodalSpec,
    PreEncodedFramesVideoSpec,
    SyntheticMp4VideoSpec,
    VideoRepresentation,
)


def _make_mock_tokenizer() -> MagicMock:
    mock_tokenizer = MagicMock()
    mock_tokenizer.count_tokens.return_value = 10
    hf_tokenizer = MagicMock()
    hf_tokenizer.vocab_size = 1000
    hf_tokenizer.decode.return_value = "decoded prompt text"
    mock_tokenizer.get_tokenizer.return_value = hf_tokenizer
    return mock_tokenizer


def _load_first(gen: ShareGPT4VideoDataGenerator, data_index: int = 0) -> ChatCompletionAPIData:
    """Materialize one request from the generator with isinstance asserts for mypy."""
    lazy_item = next(gen.get_data())
    assert isinstance(lazy_item, LazyLoadInferenceAPIData)
    lazy_item.data_index = data_index
    item = gen.load_lazy_data(lazy_item)
    assert isinstance(item, ChatCompletionAPIData)
    return item


def _fake_row(video_path: str = "pixabay/abc.mp4", n_keyframes: int = 4) -> dict[str, Any]:
    return {
        "video_id": os.path.basename(video_path).split(".")[0],
        "video_path": video_path,
        "keyframe": list(range(n_keyframes)),
        "captions": [
            {"idx": 0, "content": "A scene begins."},
            {"idx": 1, "content": "The action continues."},
        ],
    }


def _build_generator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    rows: List[dict[str, Any]],
    *,
    representation: VideoRepresentation = VideoRepresentation.JPEG_FRAMES,
    max_frames: int = 16,
    available_zips: List[str] | None = None,
    create_videos: bool = True,
) -> ShareGPT4VideoDataGenerator:
    monkeypatch.setenv("HF_TOKEN", "test-token")

    monkeypatch.setattr("inference_perf.datagen.gated_hf_dataset.load_dataset", lambda *a, **kw: list(rows))

    if available_zips is None:
        available_zips = ["zip_folder/pixabay/pixabay_videos_1.zip"]

    def fake_list_remote_zips(self: ShareGPT4VideoDataGenerator) -> List[str]:
        return list(available_zips or [])

    monkeypatch.setattr(ShareGPT4VideoDataGenerator, "_list_remote_zips", fake_list_remote_zips)

    # Stub download/extract: materialize each row's video file under videos/
    # if its source matches this zip. Cheap enough to call inline from the
    # background thread.
    def fake_download_and_extract(self: ShareGPT4VideoDataGenerator, repo_path: str) -> None:
        marker = self._marker_path(repo_path)
        if os.path.exists(marker):
            return
        if create_videos:
            for row in rows:
                vp = row.get("video_path")
                if not vp:
                    continue
                source = vp.split("/", 1)[0] if "/" in vp else ""
                if f"/{source}/" in repo_path:
                    target = os.path.join(self.videos_dir, vp)
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    if not os.path.isfile(target):
                        with open(target, "wb") as f:
                            f.write(b"\x00")
        with open(marker, "w") as f:
            f.write(repo_path)

    monkeypatch.setattr(ShareGPT4VideoDataGenerator, "_download_and_extract_zip", fake_download_and_extract)

    api_config = APIConfig(type=APIType.Chat)
    data_config = DataConfig(
        type=DataGenType.ShareGPT4Video,
        sharegpt4video=ShareGPT4VideoConfig(
            cache_dir=str(tmp_path / "cache"),
            representation=representation,
            target_resolution={"width": 64, "height": 64},
            max_frames_per_request=max_frames,
            insertion_point=0.0,
        ),
    )
    gen = ShareGPT4VideoDataGenerator(api_config, data_config, _make_mock_tokenizer())

    # Wait for the background downloader to finish so tests can reason about
    # the final pool state deterministically. Threads created in __init__ are
    # short-lived since the fake download is in-process.
    gen._downloader_thread.join(timeout=5)
    assert not gen._downloader_thread.is_alive(), "background downloader did not exit"

    def fake_extract(self: ShareGPT4VideoDataGenerator, video_path: str, indices: List[int]) -> List[bytes]:
        img = PILImage.new("RGB", (self._target_w, self._target_h), color=(10, 20, 30))
        buf = io.BytesIO()
        img.save(buf, format=self._pil_format)
        encoded = buf.getvalue()
        return [encoded for _ in indices]

    monkeypatch.setattr(ShareGPT4VideoDataGenerator, "_extract_frames", fake_extract)
    return gen


def test_loader_requires_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    api_config = APIConfig(type=APIType.Chat)
    data_config = DataConfig(
        type=DataGenType.ShareGPT4Video,
        sharegpt4video=ShareGPT4VideoConfig(cache_dir=str(tmp_path)),
    )
    with pytest.raises(ValueError, match="HuggingFace access token"):
        ShareGPT4VideoDataGenerator(api_config, data_config, _make_mock_tokenizer())


def test_loader_rejects_mp4_representation() -> None:
    with pytest.raises(ValueError, match="Frames-only"):
        ShareGPT4VideoConfig(representation=VideoRepresentation.MP4)


def test_mp4_spec_requires_frames() -> None:
    """SyntheticMp4VideoSpec.frames is inherited as required from the base."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SyntheticMp4VideoSpec(width=64, height=64, insertion_point=0.0)  # type: ignore[call-arg]


def test_pre_encoded_spec_derives_frames_from_bytes() -> None:
    """frames is filled in automatically from len(frames_bytes)."""
    spec = PreEncodedFramesVideoSpec(  # type: ignore[call-arg]  # frames auto-derived
        width=64,
        height=64,
        insertion_point=0.0,
        frame_representation=ImageRepresentation.JPEG,
        frames_bytes=[b"\x00", b"\x01", b"\x02"],
    )
    assert spec.frames == 3


def test_cache_dir_defaults_under_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setattr("inference_perf.datagen.gated_hf_dataset.load_dataset", lambda *a, **kw: [_fake_row()])
    monkeypatch.setattr(ShareGPT4VideoDataGenerator, "_list_remote_zips", lambda self: [])

    # No zips -> background loop exits immediately without producing files.
    # Bootstrap should then raise because the pool is empty.
    api_config = APIConfig(type=APIType.Chat)
    data_config = DataConfig(
        type=DataGenType.ShareGPT4Video,
        sharegpt4video=ShareGPT4VideoConfig(target_resolution={"width": 64, "height": 64}),
    )
    with pytest.raises(RuntimeError, match="bootstrap failed"):
        ShareGPT4VideoDataGenerator(api_config, data_config, _make_mock_tokenizer())


def test_loud_log_announces_cache_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=sharegpt4video_datagen.__name__):
        _build_generator(monkeypatch, tmp_path, [_fake_row()])
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "ShareGPT4Video cache directory:" in text
    assert "Pre-populate" in text
    assert "background thread" in text


def test_bootstrap_uses_pre_populated_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """If videos already exist on disk, init returns instantly without downloading."""
    cache_dir = tmp_path / "cache"
    videos_dir = cache_dir / "videos" / "pixabay"
    videos_dir.mkdir(parents=True)
    (videos_dir / "abc.mp4").write_bytes(b"\x00")

    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setattr("inference_perf.datagen.gated_hf_dataset.load_dataset", lambda *a, **kw: [_fake_row()])
    monkeypatch.setattr(ShareGPT4VideoDataGenerator, "_list_remote_zips", lambda self: [])

    download_called: list[str] = []

    def fail_if_called(self: ShareGPT4VideoDataGenerator, repo_path: str) -> None:
        download_called.append(repo_path)

    monkeypatch.setattr(ShareGPT4VideoDataGenerator, "_download_and_extract_zip", fail_if_called)

    api_config = APIConfig(type=APIType.Chat)
    data_config = DataConfig(
        type=DataGenType.ShareGPT4Video,
        sharegpt4video=ShareGPT4VideoConfig(cache_dir=str(cache_dir)),
    )
    gen = ShareGPT4VideoDataGenerator(api_config, data_config, _make_mock_tokenizer())
    gen._downloader_thread.join(timeout=5)

    assert download_called == [], "pre-populated cache should not trigger downloads"
    assert gen._available_paths == ["pixabay/abc.mp4"]


@pytest.mark.asyncio
async def test_loader_builds_frames_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    gen = _build_generator(monkeypatch, tmp_path, [_fake_row(n_keyframes=4)])
    item = _load_first(gen)
    assert item.multimodal_spec is not None
    vid = item.multimodal_spec.videos[0]
    assert isinstance(vid, PreEncodedFramesVideoSpec)
    assert len(vid.frames_bytes) == 4
    assert vid.frames == 4
    assert vid.frame_representation == ImageRepresentation.JPEG
    msg = item.messages[0]
    assert isinstance(msg, ChatMessage) and isinstance(msg.content, str)
    assert "A scene begins." in msg.content


@pytest.mark.asyncio
async def test_request_body_emits_n_image_url_blocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    gen = _build_generator(monkeypatch, tmp_path, [_fake_row(n_keyframes=3)])
    item = _load_first(gen)
    payload = await item.to_request_body(effective_model_name="vlm-test", max_tokens=64, ignore_eos=False, streaming=False)
    content = payload["messages"][0]["content"]
    image_blocks = [c for c in content if c.get("type") == "image_url"]
    video_blocks = [c for c in content if c.get("type") == "video_url"]
    assert len(image_blocks) == 3
    assert len(video_blocks) == 0
    assert image_blocks[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert item.realized_videos is not None
    assert item.realized_videos.count == 1
    assert item.realized_videos.instances[0].frames == 3
    assert item.realized_images is None


@pytest.mark.asyncio
async def test_png_frames_representation_uses_png_mime(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    gen = _build_generator(monkeypatch, tmp_path, [_fake_row(n_keyframes=2)], representation=VideoRepresentation.PNG_FRAMES)
    item = _load_first(gen)
    payload = await item.to_request_body(effective_model_name="vlm-test", max_tokens=64, ignore_eos=False, streaming=False)
    image_blocks = [c for c in payload["messages"][0]["content"] if c.get("type") == "image_url"]
    assert all(b["image_url"]["url"].startswith("data:image/png;base64,") for b in image_blocks)


@pytest.mark.asyncio
async def test_max_frames_truncates(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    gen = _build_generator(monkeypatch, tmp_path, [_fake_row(n_keyframes=32)], max_frames=8)
    item = _load_first(gen)
    assert item.multimodal_spec is not None
    vid = item.multimodal_spec.videos[0]
    assert isinstance(vid, PreEncodedFramesVideoSpec)
    assert len(vid.frames_bytes) == 8


def test_pool_grows_after_background_extracts(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """The background thread should extract each zip and update the pool."""
    rows = [
        _fake_row(video_path="pixabay/a.mp4"),
        _fake_row(video_path="pixabay/b.mp4"),
    ]
    gen = _build_generator(monkeypatch, tmp_path, rows, available_zips=["zip_folder/pixabay/pixabay_videos_1.zip"])
    assert sorted(gen._available_paths) == ["pixabay/a.mp4", "pixabay/b.mp4"]


def test_load_lazy_data_cycles_deterministically(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """data_index should index into the available pool deterministically."""
    rows = [
        _fake_row(video_path="pixabay/a.mp4"),
        _fake_row(video_path="pixabay/b.mp4"),
    ]
    gen = _build_generator(monkeypatch, tmp_path, rows, available_zips=["zip_folder/pixabay/pixabay_videos_1.zip"])
    # Two items in pool; data_index 0 and 2 should produce the same video.
    item0 = _load_first(gen, data_index=0)
    item2 = _load_first(gen, data_index=2)
    assert item0.messages[0].content == item2.messages[0].content


@pytest.mark.asyncio
async def test_pre_encoded_videos_materialization_direct() -> None:
    """Direct unit test of the chat.py materialization branch when frames_bytes is set."""
    img = PILImage.new("RGB", (32, 32), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    frame = buf.getvalue()

    spec = MultimodalSpec(
        videos=[
            PreEncodedFramesVideoSpec(
                width=32,
                height=32,
                frames=3,
                insertion_point=0.0,
                frame_representation=ImageRepresentation.JPEG,
                frames_bytes=[frame, frame, frame],
            )
        ]
    )
    item = ChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="hello")],
        multimodal_spec=spec,
    )
    payload = await item.to_request_body(effective_model_name="vlm-test", max_tokens=64, ignore_eos=False, streaming=False)
    blocks = payload["messages"][0]["content"]
    assert sum(1 for b in blocks if b.get("type") == "image_url") == 3
    assert item.realized_videos is not None
    assert item.realized_videos.instances[0].frames == 3
    assert item.realized_videos.instances[0].bytes == 3 * len(frame)
