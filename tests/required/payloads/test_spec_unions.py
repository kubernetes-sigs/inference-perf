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
"""Round-trip tests for the per-modality discriminated unions.

Each ``MultimodalSpec`` field is a Pydantic discriminated union (on the
``kind`` literal) over the provenance variants for that modality. These tests
pin two contracts:

1. Construction from a plain dict with ``kind: "<provenance>"`` dispatches to
   the right concrete subclass — guards against typos and union drop-outs.
2. ``model_dump`` → ``model_validate`` round-trips preserve the concrete
   subtype — guards against the discriminator being silently lost during
   serialization (e.g. for trace export / replay).
3. Unknown ``kind`` values are rejected at validation time — guards against
   silent fallback to a default variant.
"""

from typing import Any, Dict, Type

import pytest
from pydantic import BaseModel, ValidationError

from inference_perf.payloads import (
    LocalFileAudioSpec,
    LocalFileImageSpec,
    LocalFileVideoSpec,
    MultimodalSpec,
    PreEncodedAudioSpec,
    PreEncodedFramesVideoSpec,
    PreEncodedImageSpec,
    RemoteAudioSpec,
    RemoteImageSpec,
    RemoteVideoSpec,
    SyntheticAudioSpec,
    SyntheticFramesVideoSpec,
    SyntheticImageSpec,
    SyntheticMp4VideoSpec,
)

_IMAGE_VARIANTS = [
    ({"kind": "synthetic", "width": 16, "height": 16, "insertion_point": 0.0}, SyntheticImageSpec),
    (
        {"kind": "pre_encoded", "width": 16, "height": 16, "insertion_point": 0.0, "image_bytes": b"X"},
        PreEncodedImageSpec,
    ),
    ({"kind": "remote", "width": 16, "height": 16, "insertion_point": 0.0, "url": "https://x"}, RemoteImageSpec),
    ({"kind": "local_file", "width": 16, "height": 16, "insertion_point": 0.0, "path": "/x"}, LocalFileImageSpec),
]

_AUDIO_VARIANTS = [
    ({"kind": "synthetic", "duration": 1.0, "insertion_point": 0.0}, SyntheticAudioSpec),
    (
        {"kind": "pre_encoded", "duration": 1.0, "insertion_point": 0.0, "audio_bytes": b"X"},
        PreEncodedAudioSpec,
    ),
    ({"kind": "remote", "duration": 1.0, "insertion_point": 0.0, "url": "https://x"}, RemoteAudioSpec),
    ({"kind": "local_file", "duration": 1.0, "insertion_point": 0.0, "path": "/x"}, LocalFileAudioSpec),
]

_VIDEO_VARIANTS = [
    (
        {"kind": "synthetic_mp4", "width": 32, "height": 32, "frames": 4, "insertion_point": 0.0},
        SyntheticMp4VideoSpec,
    ),
    (
        {"kind": "synthetic_frames", "width": 32, "height": 32, "frames": 4, "insertion_point": 0.0},
        SyntheticFramesVideoSpec,
    ),
    (
        {
            "kind": "pre_encoded_frames",
            "width": 32,
            "height": 32,
            "insertion_point": 0.0,
            "frames_bytes": [b"a", b"b"],
        },
        PreEncodedFramesVideoSpec,
    ),
    (
        {"kind": "remote", "width": 32, "height": 32, "frames": 4, "insertion_point": 0.0, "url": "https://x"},
        RemoteVideoSpec,
    ),
    (
        {"kind": "local_file", "width": 32, "height": 32, "frames": 4, "insertion_point": 0.0, "path": "/x"},
        LocalFileVideoSpec,
    ),
]


def _assert_round_trips(spec: MultimodalSpec, field: str, expected_cls: Type[BaseModel]) -> None:
    assert isinstance(getattr(spec, field)[0], expected_cls)
    round_tripped = MultimodalSpec.model_validate(spec.model_dump())
    assert isinstance(getattr(round_tripped, field)[0], expected_cls)


@pytest.mark.parametrize("payload, expected_cls", _IMAGE_VARIANTS)
def test_image_union_discriminates_on_kind(payload: Dict[str, Any], expected_cls: Type[BaseModel]) -> None:
    _assert_round_trips(MultimodalSpec.model_validate({"images": [payload]}), "images", expected_cls)


@pytest.mark.parametrize("payload, expected_cls", _AUDIO_VARIANTS)
def test_audio_union_discriminates_on_kind(payload: Dict[str, Any], expected_cls: Type[BaseModel]) -> None:
    _assert_round_trips(MultimodalSpec.model_validate({"audios": [payload]}), "audios", expected_cls)


@pytest.mark.parametrize("payload, expected_cls", _VIDEO_VARIANTS)
def test_video_union_discriminates_on_kind(payload: Dict[str, Any], expected_cls: Type[BaseModel]) -> None:
    _assert_round_trips(MultimodalSpec.model_validate({"videos": [payload]}), "videos", expected_cls)


@pytest.mark.parametrize(
    "field, payload",
    [
        ("images", {"kind": "not_a_real_kind", "width": 1, "height": 1, "insertion_point": 0.0}),
        ("audios", {"kind": "not_a_real_kind", "duration": 1.0, "insertion_point": 0.0}),
        ("videos", {"kind": "not_a_real_kind", "width": 1, "height": 1, "frames": 1, "insertion_point": 0.0}),
    ],
)
def test_union_rejects_unknown_kind(field: str, payload: Dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        MultimodalSpec.model_validate({field: [payload]})
