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
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from inference_perf.config.datagen.multimodal import AnyResolution, ResolutionPreset
from inference_perf.payloads import VideoRepresentation


class GatedHFDatasetConfig(BaseModel):
    """Shared config base for loaders backed by gated HuggingFace datasets."""

    token: Optional[str] = Field(
        default=None,
        description=(
            "HuggingFace access token used to download the gated dataset. "
            "Falls back to ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` env vars when omitted."
        ),
    )


class ShareGPT4VideoConfig(GatedHFDatasetConfig):
    """Configuration for the ShareGPT4Video dataset loader.

    The loader streams captions + keyframe indices from the gated HuggingFace
    dataset and fetches the source MP4 zips (hosted on the same dataset repo
    under ``zip_folder/<source>/``) into a local cache directory via a
    background thread. The hot request path never blocks on the network — each
    request samples from videos that are already on disk, and the on-disk pool
    grows over time as background downloads complete. Init blocks only until
    the first zip has been extracted so there is always at least one usable
    video when load generation begins.

    Each request decodes the configured keyframes via PyAV, resizes to
    ``target_resolution``, re-encodes as PNG or JPEG, and emits the frames in
    the Frames wire format (one ``image_url`` block per frame at a single
    insertion point) — compatible with VLMs that don't accept ``video_url``
    natively.

    .. warning::
       The HF repo is ~1.5 TB total; a single source zip is 15-21 GB. The
       background downloader will keep pulling zips as the benchmark runs, so
       monitor disk usage. Pre-populate ``cache_dir`` to skip downloads
       entirely. Set ``HF_HUB_ENABLE_HF_TRANSFER=1`` for faster parallel
       downloads if you want more variety sooner.
    """

    cache_dir: Optional[str] = Field(
        default=None,
        description=(
            "Directory used for both the HuggingFace datasets cache and the extracted "
            "video files. Defaults to ``./sharegpt4video_cache`` under the process's "
            "current working directory. The loader logs the resolved absolute path at "
            "startup so users can pre-populate it."
        ),
    )
    representation: VideoRepresentation = Field(
        default=VideoRepresentation.JPEG_FRAMES,
        description=(
            "Frame wire encoding: ``png_frames`` (lossless) or ``jpeg_frames`` (smaller payload). "
            "``mp4`` is not supported here — this loader is Frames-only for v1."
        ),
    )
    target_resolution: AnyResolution = Field(
        default=ResolutionPreset.P720,
        description="Frames are resized to this resolution before encoding.",
    )
    max_frames_per_request: int = Field(
        default=16, gt=0, description="Cap on frames emitted per request; truncates the dataset's keyframe list."
    )
    insertion_point: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Placement of the frame block within the caption text (0.0=start, 1.0=end).",
    )
    hf_dataset_name: str = Field(
        default="ShareGPT4Video/ShareGPT4Video",
        description="HuggingFace dataset identifier; override only when mirroring the dataset elsewhere.",
    )
    hf_data_files: Optional[str] = Field(
        default=None,
        description="Optional ``data_files`` glob forwarded to ``load_dataset``.",
    )
    hf_split: str = Field(
        default="train",
        description="HuggingFace split to stream.",
    )

    @model_validator(mode="after")
    def validate_representation(self) -> "ShareGPT4VideoConfig":
        if self.representation == VideoRepresentation.MP4:
            raise ValueError(
                "ShareGPT4Video loader is Frames-only for v1; set representation to 'png_frames' or 'jpeg_frames'."
            )
        return self
