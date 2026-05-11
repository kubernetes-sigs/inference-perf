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
    VideoDatagenConfig,
    VideoProfile,
)
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
