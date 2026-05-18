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
"""Per-provenance metric semantics + cross-modality contract enforcement.

Two complementary kinds of test:

1. **Semantics**: for each concrete provenance, ``get_metrics(wire_bytes)``
   produces a metric record consistent with what that provenance can
   honestly claim about the bytes on the wire. Synthetic / pre-encoded /
   local-file trust the spec's declared geometry; remote reports zeros
   because we only sent a URL.

2. **Contract**: walks every concrete subclass under each modality's
   discriminated union and asserts ``get_metrics`` is actually overridden.
   Catches the "new provenance variant was added but inherits the abstract
   method by accident" failure mode. The :func:`abstractmethod` machinery
   would also catch this on first instantiation, but this test fires it
   at import time without depending on a caller path.
"""

from typing import Any, Type, get_args

import pytest

from inference_perf.payloads import (
    Audio,
    AudioSpecUnion,
    Image,
    ImageRepresentation,
    ImageSpecUnion,
    LocalFileAudioSpec,
    LocalFileImageSpec,
    LocalFileVideoSpec,
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
    Video,
    VideoSpecUnion,
)
from inference_perf.payloads.media_base import MediaSpec


# --- semantics: image -----------------------------------------------------


def test_synthetic_image_reports_real_geometry_and_bytes() -> None:
    spec = SyntheticImageSpec(width=64, height=32, insertion_point=0.0)
    assert spec.get_metrics(wire_bytes=900) == Image(pixels=64 * 32, bytes=900, aspect_ratio=2.0)


def test_pre_encoded_image_trusts_declared_dims() -> None:
    spec = PreEncodedImageSpec(width=10, height=5, insertion_point=0.0, image_bytes=b"X" * 250)
    assert spec.get_metrics(wire_bytes=250) == Image(pixels=50, bytes=250, aspect_ratio=2.0)


def test_local_file_image_trusts_declared_dims() -> None:
    spec = LocalFileImageSpec(width=16, height=16, insertion_point=0.0, path="/tmp/x.png")
    assert spec.get_metrics(wire_bytes=4096) == Image(pixels=256, bytes=4096, aspect_ratio=1.0)


def test_remote_image_reports_zeros_regardless_of_declared_dims() -> None:
    # ``width``/``height`` populated but considered unverified caller hints —
    # the remote contract makes "we never saw the bytes" visible in reports.
    spec = RemoteImageSpec(width=999, height=999, insertion_point=0.0, url="https://example/x.png")
    assert spec.get_metrics(wire_bytes=42) == Image()


# --- semantics: audio -----------------------------------------------------


def test_synthetic_audio_reports_wire_bytes_and_duration() -> None:
    spec = SyntheticAudioSpec(duration=3.5, insertion_point=0.0)
    assert spec.get_metrics(wire_bytes=12345) == Audio(bytes=12345, seconds=3.5)


def test_pre_encoded_audio_trusts_declared_duration() -> None:
    spec = PreEncodedAudioSpec(duration=2.0, insertion_point=0.0, audio_bytes=b"X" * 500)
    assert spec.get_metrics(wire_bytes=500) == Audio(bytes=500, seconds=2.0)


def test_local_file_audio_trusts_declared_duration() -> None:
    spec = LocalFileAudioSpec(duration=7.5, insertion_point=0.0, path="/tmp/x.wav")
    assert spec.get_metrics(wire_bytes=8192) == Audio(bytes=8192, seconds=7.5)


def test_remote_audio_reports_zeros() -> None:
    spec = RemoteAudioSpec(duration=999.0, insertion_point=0.0, url="https://example/x.mp3")
    assert spec.get_metrics(wire_bytes=42) == Audio()


# --- semantics: video -----------------------------------------------------


def test_synthetic_mp4_video_reports_real_geometry() -> None:
    spec = SyntheticMp4VideoSpec(width=320, height=240, frames=24, insertion_point=0.0)
    assert spec.get_metrics(wire_bytes=10_000) == Video(pixels=320 * 240, bytes=10_000, aspect_ratio=320 / 240, frames=24)


def test_synthetic_frames_video_reports_real_geometry() -> None:
    spec = SyntheticFramesVideoSpec(
        width=64, height=64, frames=4, insertion_point=0.0, frame_representation=ImageRepresentation.JPEG
    )
    assert spec.get_metrics(wire_bytes=5000) == Video(pixels=64 * 64, bytes=5000, aspect_ratio=1.0, frames=4)


def test_pre_encoded_frames_video_trusts_declared_geometry() -> None:
    spec = PreEncodedFramesVideoSpec(
        width=128,
        height=64,
        frames=3,
        insertion_point=0.0,
        frames_bytes=[b"a", b"b", b"c"],
    )
    # ``frames`` auto-derives from len(frames_bytes) — see model_validator on the class.
    assert spec.get_metrics(wire_bytes=3) == Video(pixels=128 * 64, bytes=3, aspect_ratio=2.0, frames=3)


def test_local_file_video_trusts_declared_geometry() -> None:
    spec = LocalFileVideoSpec(width=640, height=480, frames=120, insertion_point=0.0, path="/tmp/x.mp4")
    assert spec.get_metrics(wire_bytes=1_000_000) == Video(
        pixels=640 * 480, bytes=1_000_000, aspect_ratio=640 / 480, frames=120
    )


def test_remote_video_reports_zeros() -> None:
    spec = RemoteVideoSpec(width=999, height=999, frames=999, insertion_point=0.0, url="https://example/x.mp4")
    assert spec.get_metrics(wire_bytes=42) == Video()


# --- contract enforcement -------------------------------------------------


def _concrete_variants(union_type: Any) -> list[Type[MediaSpec[Any]]]:
    """Unwrap an Annotated[Union[...], Field(discriminator=...)] alias into
    its concrete member classes (all of which subclass :class:`MediaSpec`)."""
    annotated_args = get_args(union_type)
    # First arg is the underlying Union; rest are Field discriminators.
    union = annotated_args[0]
    return list(get_args(union))


@pytest.mark.parametrize(
    "union_alias, label",
    [
        (ImageSpecUnion, "ImageSpecUnion"),
        (AudioSpecUnion, "AudioSpecUnion"),
        (VideoSpecUnion, "VideoSpecUnion"),
    ],
)
def test_every_provenance_overrides_get_metrics(union_alias: Any, label: str) -> None:
    """Walk each modality's union and assert every concrete variant
    overrides ``get_metrics`` (not inherited from the abstract base).
    Adding a new variant without implementing the method would fail here
    even before someone tries to instantiate it."""
    variants = _concrete_variants(union_alias)
    assert variants, f"{label} is empty — did the union annotation change?"

    bad = [cls.__name__ for cls in variants if cls.get_metrics is MediaSpec.get_metrics]
    assert not bad, (
        f"{label} member(s) inherit the abstract get_metrics — every "
        f"provenance must implement its own measurement rule. Offenders: {bad}"
    )

    # And the abstract-method machinery actually marks subclasses concrete:
    for cls in variants:
        assert not getattr(cls, "__abstractmethods__", ()), (
            f"{cls.__name__} still has abstract methods {cls.__abstractmethods__}; it cannot be instantiated."
        )


def test_modality_base_classes_remain_abstract() -> None:
    """The intermediate modality bases (ImageSpec / AudioSpec / VideoSpec)
    must stay abstract — they exist to be subclassed by concrete provenance
    classes, never instantiated directly."""
    from inference_perf.payloads.audio.spec.base import AudioSpec
    from inference_perf.payloads.image.spec.base import ImageSpec
    from inference_perf.payloads.video.spec.base import VideoSpec

    for cls in (ImageSpec, AudioSpec, VideoSpec):
        assert "get_metrics" in getattr(cls, "__abstractmethods__", ()), (
            f"{cls.__name__} is no longer abstract — concrete provenance "
            "subclasses would silently inherit a no-op get_metrics."
        )
