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
"""Configuration for the VisionArena-Chat dataset loader."""

from typing import Optional, Union

from pydantic import BaseModel, Field

from inference_perf.config.common import Distribution


class VisionArenaConfig(BaseModel):
    """Configuration for the VisionArena-Chat dataset loader.

    Streams real image+prompt pairs from the public
    ``lmarena-ai/VisionArena-Chat`` HuggingFace dataset. The dataset stores
    images undecoded (raw encoded bytes), so each request attaches them as
    pre-encoded image blocks without a re-encode round trip, and uses the first
    user turn of the row's conversation as the prompt text.

    The loader streams up to ``num_rows`` usable rows into an in-memory pool at
    startup (bounding memory, since the images are the heavy part) and cycles
    deterministically through that pool on the hot request path. The dataset is
    public, so no HuggingFace token is required.
    """

    hf_dataset_name: str = Field(
        default="lmarena-ai/VisionArena-Chat",
        description="HuggingFace dataset identifier; override only when mirroring the dataset elsewhere.",
    )
    hf_split: str = Field(
        default="train",
        description="HuggingFace split to stream.",
    )
    hf_data_files: Optional[str] = Field(
        default=None,
        description="Optional ``data_files`` glob forwarded to ``load_dataset``.",
    )
    num_rows: int = Field(
        default=1000,
        gt=0,
        description=(
            "Number of usable rows to stream into the in-memory request pool at startup. "
            "Caps memory use; the benchmark cycles through this pool."
        ),
    )
    max_images_per_request: int = Field(
        default=1,
        gt=0,
        description="Cap on images attached per request; truncates a row's image list.",
    )
    insertion_point: Optional[Union[float, Distribution]] = Field(
        default=0.0,
        description=(
            "Placement of the image block(s) within the prompt text. Float in [0.0, 1.0] "
            "(0=start, 1=end), or a Distribution to sample per request."
        ),
    )
