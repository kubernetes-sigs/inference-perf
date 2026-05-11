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
"""Real media generation for multimodal benchmarking.

Produces PNG/WAV/MP4 bytes whose size tracks the configured dimensions, so
the wire payload reflects the benchmark config rather than a fixed
placeholder. Images and audio use the stdlib only; video uses PyAV.
"""

from __future__ import annotations

import io
import struct
import wave
import zlib

import av
import numpy as np
from PIL import Image as PILImage


_DEFAULT_VIDEO_FPS = 10
_AUDIO_SAMPLE_RATE = 16000
_DEFAULT_JPEG_QUALITY = 75


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def generate_png_bytes(width: int, height: int, rng: np.random.Generator) -> bytes:
    """Encode a valid PNG with a random solid RGB color using stdlib only."""
    if width <= 0 or height <= 0:
        raise ValueError(f"PNG dimensions must be positive, got {width}x{height}")

    r, g, b = (int(x) for x in rng.integers(0, 256, size=3))

    signature = b"\x89PNG\r\n\x1a\n"
    # IHDR: bit_depth=8, color_type=2 (RGB), compression=0, filter=0, interlace=0
    ihdr = _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))

    pixel_row = b"\x00" + bytes([r, g, b]) * width  # filter byte 0 (None) + pixel data
    raw = pixel_row * height
    idat = _png_chunk(b"IDAT", zlib.compress(raw, level=1))

    iend = _png_chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


def generate_jpeg_bytes(width: int, height: int, rng: np.random.Generator, quality: int = _DEFAULT_JPEG_QUALITY) -> bytes:
    """Encode a valid JPEG with a random solid RGB color via Pillow.

    Solid color is intentional for size predictability — JPEG of solid
    blocks compresses heavily, but with a per-spec ``quality`` knob the
    output size still tracks ``(width × height)``.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"JPEG dimensions must be positive, got {width}x{height}")

    r, g, b = (int(x) for x in rng.integers(0, 256, size=3))
    img = PILImage.new("RGB", (width, height), (r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def generate_wav_bytes(duration_sec: float) -> bytes:
    """Encode a valid silent 16-bit mono PCM WAV at the requested duration."""
    num_samples = max(1, int(round(duration_sec * _AUDIO_SAMPLE_RATE)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_AUDIO_SAMPLE_RATE)
        w.writeframes(b"\x00\x00" * num_samples)
    return buf.getvalue()


def _round_even(n: int) -> int:
    # H.264 with yuv420p subsampling requires even dimensions.
    if n < 2:
        return 2
    return n - (n % 2)


def generate_mp4_bytes(
    width: int, height: int, num_frames: int, rng: np.random.Generator, fps: int = _DEFAULT_VIDEO_FPS
) -> bytes:
    """Encode a valid H.264/MP4 at the requested resolution and frame count.

    Frames are solid colors with a small per-frame offset so the codec does
    not collapse the stream to a near-zero payload.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    w = _round_even(width)
    h = _round_even(height)

    r, g, b = (int(x) for x in rng.integers(0, 256, size=3))
    base = np.empty((h, w, 3), dtype=np.uint8)
    base[..., 0] = r
    base[..., 1] = g
    base[..., 2] = b

    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    try:
        stream = container.add_stream("h264", rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"

        for i in range(num_frames):
            # Per-frame offset busts trivial codec constant-frame compression.
            offset = (i * 7) % 32
            frame_arr = np.clip(base.astype(np.int16) + offset, 0, 255).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():  # flush encoder
            container.mux(packet)
    finally:
        container.close()

    return buf.getvalue()
