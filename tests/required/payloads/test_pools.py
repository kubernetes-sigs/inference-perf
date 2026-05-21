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
"""Per-modality bounded-distinct payload pools (image/video/audio)."""

from typing import Generator

import numpy as np
import pytest

from inference_perf.payloads.audio import pool as audio_pool_mod
from inference_perf.payloads.audio.pool import AudioPool
from inference_perf.payloads.image import pool as image_pool_mod
from inference_perf.payloads.image.pool import ImagePool
from inference_perf.payloads.image.spec import ImageRepresentation
from inference_perf.payloads.video import pool as video_pool_mod
from inference_perf.payloads.video.pool import VideoPool
from inference_perf.payloads.video.spec import VideoRepresentation


@pytest.fixture(autouse=True)
def _reset_pools() -> Generator[None, None, None]:
    image_pool_mod.reset_pool()
    video_pool_mod.reset_pool()
    audio_pool_mod.reset_pool()
    yield
    image_pool_mod.reset_pool()
    video_pool_mod.reset_pool()
    audio_pool_mod.reset_pool()


def test_image_pool_eagerly_materializes_distinct_blobs() -> None:
    rng = np.random.default_rng(0)
    pool = ImagePool(size=5, resolution_sampler=lambda: (32, 32), representation=ImageRepresentation.PNG, rng=rng)

    assert pool.size == 5
    # Random colors yield distinct PNGs — bounded-distinct contract.
    blobs = {pool.get(i).raw_bytes for i in range(5)}
    assert len(blobs) == 5


def test_image_pool_jpeg_data_urls() -> None:
    rng = np.random.default_rng(0)
    pool = ImagePool(size=2, resolution_sampler=lambda: (16, 16), representation=ImageRepresentation.JPEG, rng=rng)
    for i in range(2):
        assert pool.get(i).data_url.startswith("data:image/jpeg;base64,")


def test_image_pool_size_zero_rejected() -> None:
    with pytest.raises(ValueError):
        ImagePool(
            size=0,
            resolution_sampler=lambda: (16, 16),
            representation=ImageRepresentation.PNG,
            rng=np.random.default_rng(),
        )


def test_video_pool_mp4_one_blob_per_entry() -> None:
    rng = np.random.default_rng(0)
    pool = VideoPool(
        size=2,
        profile_sampler=lambda: (32, 32, 2),
        representation=VideoRepresentation.MP4,
        rng=rng,
    )
    for i in range(2):
        entry = pool.get(i)
        assert entry.representation == VideoRepresentation.MP4
        assert len(entry.frame_blobs) == 1
        assert entry.frame_urls[0].startswith("data:video/mp4;base64,")


def test_video_pool_frames_mode_one_blob_per_frame() -> None:
    rng = np.random.default_rng(0)
    pool = VideoPool(
        size=2,
        profile_sampler=lambda: (16, 16, 3),
        representation=VideoRepresentation.JPEG_FRAMES,
        rng=rng,
    )
    entry = pool.get(0)
    assert len(entry.frame_blobs) == 3
    assert all(url.startswith("data:image/jpeg;base64,") for url in entry.frame_urls)


def test_audio_pool_eagerly_materializes_wav() -> None:
    pool = AudioPool(size=4, duration_sampler=lambda: 0.1)
    assert pool.size == 4
    for i in range(4):
        entry = pool.get(i)
        assert entry.duration == 0.1
        # WAV is deterministic by duration — identical duration → identical bytes.
        assert entry.raw_bytes == pool.get(0).raw_bytes


def test_registry_set_and_get_per_modality() -> None:
    assert image_pool_mod.get_pool() is None
    assert video_pool_mod.get_pool() is None
    assert audio_pool_mod.get_pool() is None

    rng = np.random.default_rng(0)
    img = ImagePool(size=1, resolution_sampler=lambda: (8, 8), representation=ImageRepresentation.PNG, rng=rng)
    image_pool_mod.set_pool(img)
    assert image_pool_mod.get_pool() is img

    # Other modalities' registries are independent.
    assert video_pool_mod.get_pool() is None
    assert audio_pool_mod.get_pool() is None

    image_pool_mod.reset_pool()
    assert image_pool_mod.get_pool() is None
