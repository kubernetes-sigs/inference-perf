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
from collections.abc import AsyncGenerator
import logging
from typing import Any, Iterator, List, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientResponse

from inference_perf.apis import UnaryResponseMetrics
from inference_perf.apis import chat as chat_module
from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.config import APIConfig, APIType
from inference_perf.payloads import (
    ImageRepresentation,
    MultimodalSpec,
    PreEncodedFramesVideoSpec,
    PreEncodedImageSpec,
    SyntheticAudioSpec,
    SyntheticFramesVideoSpec,
    SyntheticImageSpec,
)


def _make_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.count_tokens = lambda text, **kw: max(1, len((text or "").split()))
    return tok


def _make_config(streaming: bool) -> MagicMock:
    cfg = MagicMock()
    cfg.streaming = streaming
    return cfg


class _FakeStreamingResponse:
    """Minimal aiohttp ClientResponse stand-in that yields preset SSE bytes."""

    def __init__(self, chunks: List[bytes]) -> None:
        self.status = 200
        self.headers = {"content-type": "text/event-stream"}
        self.content = self._make_content(chunks)

    @staticmethod
    def _make_content(chunks: List[bytes]) -> MagicMock:
        content = MagicMock()

        async def iter_any() -> AsyncGenerator[bytes, None]:
            for chunk in chunks:
                yield chunk

        content.iter_any = iter_any
        return content


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
    tokenizer.count_tokens.side_effect = lambda s, **kw: len(s.split())

    data = ChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="three words here")],
        prefix_text="prefix has four tokens",
    )
    # 4 (prefix) + 3 (message) = 7
    assert data._count_prompt_tokens(tokenizer) == 7


def test_count_prompt_tokens_without_prefix_text_unchanged() -> None:
    """Existing behavior holds when prefix_text is unset."""
    tokenizer = MagicMock()
    tokenizer.count_tokens.side_effect = lambda s, **kw: len(s.split())

    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="five tokens in this prompt")])
    assert data._count_prompt_tokens(tokenizer) == 5


def test_count_prompt_tokens_includes_tool_definitions() -> None:
    """``_count_prompt_tokens`` should count tool_definitions tokens too —
    tool schemas are serialized into the prompt the server actually sees,
    so omitting them undercounts prompt length."""
    tokenizer = MagicMock()
    tokenizer.count_tokens.side_effect = lambda s, **kw: len(s.split())

    data = ChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="three words here")],
        tool_definitions=[
            {
                "type": "function",
                "name": "get_weather",
                "description": "five tokens for this tool",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    )
    # 3 (message) + 15 (json.dumps'd tool_definitions, whitespace-tokenized)
    assert data._count_prompt_tokens(tokenizer) == 18


@pytest.mark.asyncio
async def test_process_response_non_streaming_uses_server_prompt_tokens() -> None:
    """When the server reports usage.prompt_tokens, request_metrics.text.input_tokens
    is resolved from it rather than client-side tokenization"""
    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="hi")])
    tokenizer = _make_tokenizer()

    response = MagicMock()
    response.json = AsyncMock(
        return_value={
            "choices": [{"message": {"content": "hello there"}}],
            "usage": {"prompt_tokens": 42, "completion_tokens": 2},
        }
    )

    info = await data.process_response(response, _make_config(streaming=False), tokenizer)

    assert info.request_metrics.text.input_tokens == 42
    assert isinstance(info.response_metrics, UnaryResponseMetrics)
    assert info.response_metrics.server_usage == {"prompt_tokens": 42, "completion_tokens": 2}


@pytest.mark.asyncio
async def test_process_response_non_streaming_falls_back_without_server_usage() -> None:
    """No usage in the response body falls back to client-side tokenization."""
    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="one two three")])
    tokenizer = _make_tokenizer()

    response = MagicMock()
    response.json = AsyncMock(return_value={"choices": [{"message": {"content": "hi"}}]})

    info = await data.process_response(response, _make_config(streaming=False), tokenizer)

    assert info.request_metrics.text.input_tokens == 3


@pytest.mark.asyncio
async def test_process_response_streaming_uses_server_prompt_tokens() -> None:
    """Streaming trailing usage chunk resolves input_tokens the same way as non-streaming."""
    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="hi")])
    tokenizer = _make_tokenizer()

    sse = (
        b'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n'
        b'data: {"choices": [], "usage": {"prompt_tokens": 17, "completion_tokens": 1}}\n\n'
        b"data: [DONE]\n\n"
    )
    response = cast(ClientResponse, _FakeStreamingResponse([sse]))

    info = await data.process_response(response, _make_config(streaming=True), tokenizer)

    assert info.request_metrics.text.input_tokens == 17


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


@pytest.mark.asyncio
async def test_materialize_pre_encoded_image() -> None:
    """``PreEncodedImageSpec`` is the image-side dataset-loader provenance:
    bytes supplied by the loader are emitted verbatim (base64-wrapped, no
    re-encoding) as a single ``image_url`` block, mime-typed by
    ``representation``, and the realized ``Image`` metric reports the input
    byte count and declared geometry."""
    raw = b"PRE_ENCODED_PNG_IMAGE_BYTES"
    image_spec = PreEncodedImageSpec(
        width=200,
        height=100,
        insertion_point=0.0,
        representation=ImageRepresentation.PNG,
        image_bytes=raw,
    )
    data = ChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="describe this image")],
        multimodal_spec=MultimodalSpec(images=[image_spec]),
    )

    payload = await data.to_request_body(effective_model_name="gpt-vlm", max_tokens=100, ignore_eos=False, streaming=False)
    image_blocks = [c for c in payload["messages"][0]["content"] if c.get("type") == "image_url"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["image_url"]["url"] == f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"

    assert data.realized_images is not None and data.realized_images.count == 1
    metric = data.realized_images.instances[0]
    assert metric.bytes == len(raw)
    assert metric.pixels == 200 * 100


@pytest.mark.asyncio
async def test_materialize_pre_encoded_image_jpeg_mime() -> None:
    """``representation=JPEG`` switches the data-URL mime to ``image/jpeg``."""
    image_spec = PreEncodedImageSpec(
        width=32,
        height=32,
        insertion_point=0.0,
        representation=ImageRepresentation.JPEG,
        image_bytes=b"JPEG_BYTES",
    )
    data = ChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="x")],
        multimodal_spec=MultimodalSpec(images=[image_spec]),
    )

    payload = await data.to_request_body(effective_model_name="gpt-vlm", max_tokens=10, ignore_eos=False, streaming=False)
    image_blocks = [c for c in payload["messages"][0]["content"] if c.get("type") == "image_url"]
    assert image_blocks[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_materialize_pre_encoded_image_webp_mime() -> None:
    """``representation=WEBP`` switches the data-URL mime to ``image/webp``."""
    image_spec = PreEncodedImageSpec(
        width=32,
        height=32,
        insertion_point=0.0,
        representation=ImageRepresentation.WEBP,
        image_bytes=b"WEBP_BYTES",
    )
    data = ChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="x")],
        multimodal_spec=MultimodalSpec(images=[image_spec]),
    )

    payload = await data.to_request_body(effective_model_name="gpt-vlm", max_tokens=10, ignore_eos=False, streaming=False)
    image_blocks = [c for c in payload["messages"][0]["content"] if c.get("type") == "image_url"]
    assert image_blocks[0]["image_url"]["url"].startswith("data:image/webp;base64,")


@pytest.mark.asyncio
async def test_chat_completion_process_response_streaming_request_id() -> None:
    """Verifies that streaming chat responses extract server_request_id from SSE chunk payload."""
    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="hello")])
    mock_resp = MagicMock()
    mock_resp.headers = {}
    mock_resp.content = MagicMock()

    async def iter_any() -> AsyncGenerator[bytes, None]:
        yield b'data: {"id": "chatcmpl-stream-789", "choices": [{"delta": {"content": "world"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    mock_resp.content.iter_any = iter_any
    tok = MagicMock()
    tok.count_tokens = lambda s, **kw: len((s or "").split())

    info = await data.process_response(mock_resp, APIConfig(type=APIType.Chat, streaming=True), tok)
    assert info.server_request_id == "chatcmpl-stream-789"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_payload", "response_headers", "expected_id"),
    [
        (
            {"id": "chatcmpl-unary-789", "choices": [{"message": {"content": "world"}}]},
            {},
            "chatcmpl-unary-789",
        ),
        (
            {"choices": [{"message": {"content": "world"}}]},
            {"x-request-id": "req-unary-hdr-789"},
            "req-unary-hdr-789",
        ),
    ],
)
async def test_chat_completion_process_response_unary_request_id(
    response_payload: dict[str, Any], response_headers: dict[str, str], expected_id: str
) -> None:
    """Verifies unary chat response extracts server_request_id from JSON payload or header fallback."""
    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="hello")])
    mock_resp = MagicMock()
    mock_resp.headers = response_headers
    mock_resp.json = AsyncMock(return_value=response_payload)
    tok = MagicMock()
    tok.count_tokens = lambda s, **kw: len((s or "").split())

    info = await data.process_response(mock_resp, APIConfig(type=APIType.Chat, streaming=False), tok)
    assert info.server_request_id == expected_id

