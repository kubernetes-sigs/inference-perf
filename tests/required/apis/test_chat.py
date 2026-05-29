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
import base64
import logging
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest

from inference_perf.apis import chat as chat_module
from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.config import APIType
from inference_perf.payloads import (
    ImageRepresentation,
    MultimodalSpec,
    PreEncodedFramesVideoSpec,
    SyntheticAudioSpec,
    SyntheticFramesVideoSpec,
    SyntheticImageSpec,
)


@pytest.mark.asyncio
async def test_chat_completion_api_data() -> None:
    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="Hello, world!")])
    assert data.get_api_type() == APIType.Chat
    assert len(data.messages) == 1
    assert await data.to_request_body("test-model", 100, False, False) == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "max_tokens": 100,
        "ignore_eos": False,
        "stream": False,
    }


def test_count_prompt_tokens_includes_prefix_text() -> None:
    """``_count_prompt_tokens`` sums prefix_text tokens alongside message
    tokens — the total reflects the actual prompt sent to the model."""
    tokenizer = MagicMock()
    # One token per whitespace-separated word.
    tokenizer.count_tokens.side_effect = lambda s: len(s.split())

    data = ChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="three words here")],
        prefix_text="prefix has four tokens",
    )
    # 4 (prefix) + 3 (message) = 7
    assert data._count_prompt_tokens(tokenizer) == 7


def test_count_prompt_tokens_without_prefix_text_unchanged() -> None:
    """Existing behavior holds when prefix_text is unset."""
    tokenizer = MagicMock()
    tokenizer.count_tokens.side_effect = lambda s: len(s.split())

    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="five tokens in this prompt")])
    assert data._count_prompt_tokens(tokenizer) == 5


def _reset_multimodal_progress_state() -> None:
    """Zero the module-level multimodal heartbeat counters between tests."""
    chat_module._multimodal_materialized_requests = 0
    chat_module._multimodal_materialized_images = 0
    chat_module._multimodal_materialized_videos = 0
    chat_module._multimodal_materialized_audios = 0
    chat_module._multimodal_materialized_video_frames = 0
    chat_module._last_multimodal_progress_log_time = None


def _make_multimodal_request(images: int = 1, audios: int = 0, videos: int = 0) -> ChatCompletionAPIData:
    spec = MultimodalSpec(
        images=[SyntheticImageSpec(width=8, height=8, insertion_point=0.0) for _ in range(images)],
        videos=[SyntheticFramesVideoSpec(width=8, height=8, frames=2, insertion_point=0.0) for _ in range(videos)],
        audios=[SyntheticAudioSpec(duration=0.1, insertion_point=0.0) for _ in range(audios)],
    )
    return ChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="hi")],
        multimodal_spec=spec,
    )


@pytest.mark.asyncio
async def test_multimodal_heartbeat_fires_on_interval(caplog: Any) -> None:
    """Each materialized multimodal request advances counters; a heartbeat fires once per interval."""
    _reset_multimodal_progress_state()

    # Drive monotonic forward by the configured interval on every call so each
    # to_request_body crosses the heartbeat boundary.
    fake_time: Iterator[float] = iter((i * chat_module._MULTIMODAL_PROGRESS_LOG_INTERVAL_SEC for i in range(1, 100)))

    caplog.set_level(logging.INFO, logger=chat_module.__name__)
    with patch("inference_perf.apis.chat.time.monotonic", side_effect=lambda: next(fake_time)):
        for _ in range(3):
            await _make_multimodal_request(images=2, audios=1).to_request_body("m", 10, False, False)

    progress = [r.message for r in caplog.records if "Multimodal datagen progress" in r.message]
    assert len(progress) == 3
    assert "materialized 3 requests" in progress[-1]
    assert "images=6" in progress[-1]
    assert "audios=3" in progress[-1]


@pytest.mark.asyncio
async def test_multimodal_heartbeat_skips_within_interval() -> None:
    """Sub-interval materializations advance counters but only log once."""
    _reset_multimodal_progress_state()

    base_time = 1_000_000.0
    fake_time = iter([base_time, base_time + 0.1, base_time + 0.2, base_time + 0.3])

    with (
        patch.object(chat_module, "logger") as mock_logger,
        patch("inference_perf.apis.chat.time.monotonic", side_effect=lambda: next(fake_time)),
    ):
        for _ in range(4):
            await _make_multimodal_request(images=1).to_request_body("m", 10, False, False)

    assert mock_logger.info.call_count == 1
    assert chat_module._multimodal_materialized_requests == 4
    assert chat_module._multimodal_materialized_images == 4


@pytest.mark.asyncio
async def test_materialize_pre_encoded_frames_video() -> None:
    """``PreEncodedFramesVideoSpec`` is the only materializer branch reached
    by dataset-loader provenance (frame bytes supplied, not synthesized).
    Verifies the loader contract: one ``image_url`` block per frame, bytes
    emitted verbatim (base64-wrapped, no re-encoding), mime-typed by
    ``frame_representation``, and the realized ``Video`` metric reports the
    summed input bytes."""
    frame_bytes_list = [b"PNG_FRAME_ONE_BYTES", b"PNG_FRAME_TWO_BYTES", b"PNG_FRAME_THREE_BYTES"]
    video_spec = PreEncodedFramesVideoSpec(
        width=128,
        height=64,
        frames=len(frame_bytes_list),
        insertion_point=0.0,
        frame_representation=ImageRepresentation.PNG,
        frames_bytes=frame_bytes_list,
    )
    data = ChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="describe this video")],
        multimodal_spec=MultimodalSpec(videos=[video_spec]),
    )

    payload = await data.to_request_body(effective_model_name="gpt-vlm", max_tokens=100, ignore_eos=False, streaming=False)
    content = payload["messages"][0]["content"]
    assert isinstance(content, list)

    image_blocks = [c for c in content if c.get("type") == "image_url"]
    assert len(image_blocks) == len(frame_bytes_list)
    for block, raw in zip(image_blocks, frame_bytes_list, strict=True):
        expected = f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"
        assert block["image_url"]["url"] == expected

    assert data.realized_videos is not None and data.realized_videos.count == 1
    metric = data.realized_videos.instances[0]
    assert metric.bytes == sum(len(b) for b in frame_bytes_list)
    assert metric.frames == len(frame_bytes_list)
    assert metric.pixels == 128 * 64


@pytest.mark.asyncio
async def test_materialize_pre_encoded_frames_video_jpeg_mime() -> None:
    """``frame_representation=JPEG`` switches the data-URL mime to ``image/jpeg``."""
    video_spec = PreEncodedFramesVideoSpec(
        width=32,
        height=32,
        frames=1,
        insertion_point=0.0,
        frame_representation=ImageRepresentation.JPEG,
        frames_bytes=[b"JPEG_BYTES"],
    )
    data = ChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="x")],
        multimodal_spec=MultimodalSpec(videos=[video_spec]),
    )

    payload = await data.to_request_body(effective_model_name="gpt-vlm", max_tokens=10, ignore_eos=False, streaming=False)
    image_blocks = [c for c in payload["messages"][0]["content"] if c.get("type") == "image_url"]
    assert image_blocks[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
