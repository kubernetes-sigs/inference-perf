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
"""Sampling helpers for synthetic multimodal datagen.

Turns config-level distributions (``ImageDatagenConfig.resolutions``,
``VideoDatagenConfig.profiles``, ``AudioDatagenConfig.durations``) into
concrete per-instance values used to populate ``MultimodalSpec``.
No bytes here — see :mod:`inference_perf.mediagen` for byte synthesis.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from inference_perf.config import (
    AnyResolution,
    AudioDatagenConfig,
    Distribution,
    ImageDatagenConfig,
    Resolution,
    ResolutionPreset,
    SyntheticMultimodalDatagenConfig,
    VideoDatagenConfig,
    VideoProfile,
)
from inference_perf.payloads import (
    ImageRepresentation,
    MultimodalSpec,
    SyntheticAudioSpec,
    SyntheticFramesVideoSpec,
    SyntheticImageSpec,
    SyntheticMp4VideoSpec,
    VideoRepresentation,
    VideoSpecUnion,
)
from inference_perf.payloads.audio import pool as audio_pool_mod
from inference_perf.payloads.image import pool as image_pool_mod
from inference_perf.payloads.video import pool as video_pool_mod
from inference_perf.utils.distribution import sample_from_distribution


_PRESET_TO_WH: dict[ResolutionPreset, Tuple[int, int]] = {
    ResolutionPreset.P4K: (3840, 2160),
    ResolutionPreset.P1080: (1920, 1080),
    ResolutionPreset.P720: (1280, 720),
    ResolutionPreset.P360: (640, 360),
}

_DEFAULT_IMAGE_WH: Tuple[int, int] = (512, 512)
_DEFAULT_VIDEO_WH: Tuple[int, int] = (512, 512)
_DEFAULT_VIDEO_FRAMES = 10
_DEFAULT_AUDIO_SECONDS = 1.0


def resolution_to_wh(res: AnyResolution) -> Tuple[int, int]:
    if isinstance(res, ResolutionPreset):
        return _PRESET_TO_WH[res]
    return res.width, res.height


def _choose_index(weights: list[float], rng: np.random.Generator) -> int:
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total <= 0:
        return 0
    return int(rng.choice(len(weights), p=w / total))


def sample_image_resolution(cfg: ImageDatagenConfig, rng: np.random.Generator) -> Tuple[int, int]:
    res = cfg.resolutions
    if res is None:
        return _DEFAULT_IMAGE_WH
    if isinstance(res, list):
        idx = _choose_index([r.weight for r in res], rng)
        return resolution_to_wh(res[idx].resolution)
    return resolution_to_wh(res)


def sample_video_profile(cfg: VideoDatagenConfig, rng: np.random.Generator) -> VideoProfile:
    profiles = cfg.profiles
    if profiles is None:
        return VideoProfile(
            resolution=Resolution(width=_DEFAULT_VIDEO_WH[0], height=_DEFAULT_VIDEO_WH[1]),
            frames=_DEFAULT_VIDEO_FRAMES,
        )
    if isinstance(profiles, list):
        idx = _choose_index([p.weight for p in profiles], rng)
        return profiles[idx].profile
    return profiles


def sample_audio_duration(cfg: AudioDatagenConfig, rng: np.random.Generator) -> float:
    durations = cfg.durations
    if durations is None:
        return _DEFAULT_AUDIO_SECONDS
    if isinstance(durations, list):
        idx = _choose_index([d.weight for d in durations], rng)
        return float(durations[idx].duration)
    return float(durations)


def sample_insertion_point(
    config_insertion_point: Optional[Union[float, Distribution]],
    rng: np.random.Generator,
) -> float:
    """Resolve a config ``insertion_point`` (None | float | Distribution) to a concrete float in [0, 1].

    ``None`` means "uniform random over the prompt"; a float is taken as-is;
    a Distribution is sampled.
    """
    if config_insertion_point is None:
        return float(rng.uniform(0.0, 1.0))
    if isinstance(config_insertion_point, float):
        return config_insertion_point
    return float(sample_from_distribution(config_insertion_point, 1, rng)[0])


def configure_payload_pools(cfg: SyntheticMultimodalDatagenConfig, rng: np.random.Generator) -> None:
    """Build and register per-modality payload pools from a multimodal config.

    Idempotent within a process: each call replaces any previously registered
    pool for the same modality. Only invoked for the payload-side config —
    prefix-side bytes always bypass the pool to preserve prefix-cache
    determinism per group.
    """
    if cfg.image and cfg.image.pool and cfg.image.count:
        img_cfg = cfg.image

        def _img_sampler() -> Tuple[int, int]:
            return sample_image_resolution(img_cfg, rng)

        image_pool_mod.set_pool(
            image_pool_mod.ImagePool(img_cfg.pool.size, _img_sampler, img_cfg.representation, rng)
        )
    else:
        image_pool_mod.set_pool(None)

    if cfg.video and cfg.video.pool and cfg.video.count:
        vid_cfg = cfg.video

        def _vid_sampler() -> Tuple[int, int, int]:
            profile = sample_video_profile(vid_cfg, rng)
            w, h = resolution_to_wh(profile.resolution)
            return w, h, profile.frames

        video_pool_mod.set_pool(
            video_pool_mod.VideoPool(vid_cfg.pool.size, _vid_sampler, vid_cfg.representation, rng)
        )
    else:
        video_pool_mod.set_pool(None)

    if cfg.audio and cfg.audio.pool and cfg.audio.count:
        aud_cfg = cfg.audio

        def _aud_sampler() -> float:
            return sample_audio_duration(aud_cfg, rng)

        audio_pool_mod.set_pool(audio_pool_mod.AudioPool(aud_cfg.pool.size, _aud_sampler))
    else:
        audio_pool_mod.set_pool(None)


def reset_payload_pools() -> None:
    """Clear every modality's payload pool. Intended for tests."""
    image_pool_mod.reset_pool()
    video_pool_mod.reset_pool()
    audio_pool_mod.reset_pool()


def sample_spec_from_config(
    cfg: SyntheticMultimodalDatagenConfig,
    rng: np.random.Generator,
    use_pool: bool,
) -> MultimodalSpec:
    """Sample a ``MultimodalSpec`` for one request.

    When ``use_pool`` is True and a modality has a registered eager pool,
    items for that modality carry a ``pool_index`` and inherit their
    dimensions / frames / duration from the corresponding pool entry. When
    False (prefix-side) or when no pool is registered, items are drawn from
    the modality's distribution exactly as before — the legacy code path.
    """
    spec = MultimodalSpec()

    img_cfg = cfg.image
    if img_cfg and img_cfg.count:
        count = int(sample_from_distribution(img_cfg.count, 1, rng)[0])
        img_pool = image_pool_mod.get_pool() if use_pool else None
        for _ in range(count):
            ipt = sample_insertion_point(img_cfg.insertion_point, rng)
            if img_pool is not None:
                idx = int(rng.integers(0, img_pool.size))
                entry = img_pool.get(idx)
                spec.images.append(
                    SyntheticImageSpec(
                        width=entry.width,
                        height=entry.height,
                        insertion_point=ipt,
                        representation=img_cfg.representation,
                        pool_index=idx,
                    )
                )
            else:
                w, h = sample_image_resolution(img_cfg, rng)
                spec.images.append(
                    SyntheticImageSpec(
                        width=w, height=h, insertion_point=ipt, representation=img_cfg.representation
                    )
                )

    vid_cfg = cfg.video
    if vid_cfg and vid_cfg.count:
        count = int(sample_from_distribution(vid_cfg.count, 1, rng)[0])
        vid_pool = video_pool_mod.get_pool() if use_pool else None
        for _ in range(count):
            ipt = sample_insertion_point(vid_cfg.insertion_point, rng)
            if vid_pool is not None:
                idx = int(rng.integers(0, vid_pool.size))
                entry = vid_pool.get(idx)
                video_spec_pool: VideoSpecUnion
                if vid_cfg.representation == VideoRepresentation.MP4:
                    video_spec_pool = SyntheticMp4VideoSpec(
                        width=entry.width,
                        height=entry.height,
                        frames=entry.frames,
                        insertion_point=ipt,
                        pool_index=idx,
                    )
                else:
                    video_spec_pool = SyntheticFramesVideoSpec(
                        width=entry.width,
                        height=entry.height,
                        frames=entry.frames,
                        insertion_point=ipt,
                        frame_representation=(
                            ImageRepresentation.JPEG
                            if vid_cfg.representation == VideoRepresentation.JPEG_FRAMES
                            else ImageRepresentation.PNG
                        ),
                        pool_index=idx,
                    )
                spec.videos.append(video_spec_pool)
            else:
                profile = sample_video_profile(vid_cfg, rng)
                w, h = resolution_to_wh(profile.resolution)
                video_spec_fresh: VideoSpecUnion
                if vid_cfg.representation == VideoRepresentation.MP4:
                    video_spec_fresh = SyntheticMp4VideoSpec(
                        width=w, height=h, frames=profile.frames, insertion_point=ipt
                    )
                else:
                    video_spec_fresh = SyntheticFramesVideoSpec(
                        width=w,
                        height=h,
                        frames=profile.frames,
                        insertion_point=ipt,
                        frame_representation=(
                            ImageRepresentation.JPEG
                            if vid_cfg.representation == VideoRepresentation.JPEG_FRAMES
                            else ImageRepresentation.PNG
                        ),
                    )
                spec.videos.append(video_spec_fresh)

    aud_cfg = cfg.audio
    if aud_cfg and aud_cfg.count:
        count = int(sample_from_distribution(aud_cfg.count, 1, rng)[0])
        aud_pool = audio_pool_mod.get_pool() if use_pool else None
        for _ in range(count):
            ipt = sample_insertion_point(aud_cfg.insertion_point, rng)
            if aud_pool is not None:
                idx = int(rng.integers(0, aud_pool.size))
                entry = aud_pool.get(idx)
                spec.audios.append(
                    SyntheticAudioSpec(duration=entry.duration, insertion_point=ipt, pool_index=idx)
                )
            else:
                spec.audios.append(
                    SyntheticAudioSpec(duration=sample_audio_duration(aud_cfg, rng), insertion_point=ipt)
                )

    return spec
