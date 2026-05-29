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
"""VisionArena-Chat dataset loader.

Streams real image+prompt pairs from the public ``lmarena-ai/VisionArena-Chat``
HuggingFace dataset. The dataset stores images undecoded (raw encoded bytes),
so each request attaches them as ``PreEncodedImageSpec`` blocks without a
re-encode round trip, and uses the first user turn of the row's conversation as
the prompt. A bounded pool of rows is streamed into memory at startup and the
hot request path cycles through it deterministically.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Generator, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from PIL import Image as PILImage

from inference_perf.apis.base import InferenceAPIData, LazyLoadInferenceAPIData
from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.config import APIConfig, APIType, DataConfig, VisionArenaConfig
from inference_perf.payloads import ImageRepresentation, MultimodalSpec, PreEncodedImageSpec
from inference_perf.utils.custom_tokenizer import CustomTokenizer

from .base import DataGenerator, LazyLoadDataMixin
from .multimodal_sampling import sample_insertion_point

logger = logging.getLogger(__name__)

# A processed image ready to attach: (encoded_bytes, width, height, representation).
ProcessedImage = Tuple[bytes, int, int, ImageRepresentation]

# PIL format name -> wire representation, for verbatim passthrough of encoded
# source bytes. Encoded images in any other format are skipped rather than
# silently re-encoded; extend this map (and ImageRepresentation) to add support.
_WIRE_FORMATS = {
    "JPEG": ImageRepresentation.JPEG,
    "PNG": ImageRepresentation.PNG,
    "WEBP": ImageRepresentation.WEBP,
}


class VisionArenaDataGenerator(DataGenerator, LazyLoadDataMixin):
    """Emits VisionArena-Chat requests with real images as pre-encoded blocks.

    The dataset is public, so no HuggingFace token is required. Rows without a
    usable user prompt or without at least one decodable image are skipped.
    """

    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: Optional[CustomTokenizer]) -> None:
        DataGenerator.__init__(self, api_config, config, tokenizer)

        if config.visionarena is None:
            raise ValueError("visionarena config is required for VisionArenaDataGenerator")
        self.va_config: VisionArenaConfig = config.visionarena

        self._pool: List[dict[str, Any]] = self._build_pool()
        if not self._pool:
            raise RuntimeError(
                f"VisionArena pool is empty: no usable rows found in "
                f"'{self.va_config.hf_dataset_name}' split '{self.va_config.hf_split}'. "
                "Each usable row needs a user-turn prompt and at least one decodable image."
            )
        logger.info("Loaded %d VisionArena rows into the request pool", len(self._pool))

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Chat]

    def is_io_distribution_supported(self) -> bool:
        return False

    def is_shared_prefix_supported(self) -> bool:
        return False

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        i = 0
        while True:
            yield LazyLoadInferenceAPIData(data_index=i)
            i += 1

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> InferenceAPIData:
        # Deterministic cycle through the pool for reproducibility across workers
        # that see the same data_index sequence. The RNG is seeded by data_index
        # so insertion-point sampling (when configured as a Distribution) is also
        # reproducible: a given index always yields the same request.
        entry = self._pool[data.data_index % len(self._pool)]
        rng = np.random.default_rng(data.data_index)
        images = [
            PreEncodedImageSpec(
                width=width,
                height=height,
                representation=representation,
                image_bytes=image_bytes,
                insertion_point=sample_insertion_point(self.va_config.insertion_point, rng),
            )
            for (image_bytes, width, height, representation) in entry["images"]
        ]
        # Single-turn only for now: each request is just the row's first user
        # turn. Source rows can be multi-turn (num_turns up to ~6), but the
        # follow-up turns and assistant replies are dropped here. Multi-turn
        # replay (growing context, cross-turn cache reuse) is a likely follow-up;
        # it would emit the full conversation instead of one message.
        return ChatCompletionAPIData(
            messages=[ChatMessage(role="user", content=entry["prompt"])],
            multimodal_spec=MultimodalSpec(images=images),
        )

    # -------------------------- pool construction --------------------------

    def _build_pool(self) -> List[dict[str, Any]]:
        load_kwargs: dict[str, Any] = {"streaming": True, "split": self.va_config.hf_split}
        if self.va_config.hf_data_files is not None:
            load_kwargs["data_files"] = self.va_config.hf_data_files

        logger.info("Streaming VisionArena dataset '%s' ...", self.va_config.hf_dataset_name)
        pool: List[dict[str, Any]] = []
        for row in load_dataset(self.va_config.hf_dataset_name, **load_kwargs):
            if len(pool) >= self.va_config.num_rows:
                break
            entry = self._row_to_entry(row)
            if entry is not None:
                pool.append(entry)
        return pool

    def _row_to_entry(self, row: Any) -> Optional[dict[str, Any]]:
        if not isinstance(row, dict):
            return None
        prompt = self._extract_prompt(row.get("conversation"))
        if not prompt:
            return None
        raw_images = row.get("images") or []
        if not isinstance(raw_images, list):
            return None
        images: List[ProcessedImage] = []
        for img in raw_images[: self.va_config.max_images_per_request]:
            processed = self._process_image(img)
            if processed is not None:
                images.append(processed)
        if not images:
            return None
        return {"prompt": prompt, "images": images}

    @staticmethod
    def _extract_prompt(conversation: Any) -> Optional[str]:
        """Return the first user-turn content from a VisionArena conversation.

        The dataset declares ``conversation`` as ``List[List[{role, content}]]``
        -- each turn is wrapped in a single-element list -- so we flatten one
        level. A flat ``List[{role, content}]`` (e.g. a mirror or variant) is
        tolerated too.
        """
        if not isinstance(conversation, list):
            return None
        for turn in conversation:
            messages = turn if isinstance(turn, list) else [turn]
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "user":
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content
        return None

    def _process_image(self, img: Any) -> Optional[ProcessedImage]:
        """Normalize one dataset image into (bytes, width, height, representation).

        Handles both the undecoded HuggingFace Image form (``{"bytes": ...}``)
        and an already-decoded PIL image. Encoded JPEG/PNG/WEBP bytes pass
        through verbatim (the wire accepts all three), so the request carries
        the real dataset payload at its native resolution. Encoded bytes in any
        other format are skipped (return ``None``) rather than silently
        re-encoded -- add formats to ``_WIRE_FORMATS`` explicitly as needed. An
        already-decoded PIL image has no encoded bytes, so it is encoded once to
        PNG. Per-model resizing is left to the model server, not done here.
        """
        pil: Optional[PILImage.Image] = None
        raw: Optional[bytes] = None
        if isinstance(img, dict) and isinstance(img.get("bytes"), (bytes, bytearray)):
            raw = bytes(img["bytes"])
            try:
                pil = PILImage.open(io.BytesIO(raw))
                pil.load()
            except Exception:
                return None
        elif isinstance(img, PILImage.Image):
            pil = img
        else:
            return None

        if raw is not None:
            # Encoded source bytes: forward verbatim if the wire supports the
            # format, otherwise skip rather than silently re-encode it.
            representation = _WIRE_FORMATS.get((pil.format or "").upper())
            if representation is None:
                return None
        else:
            # A decoded PIL image carries no encoded bytes, so it must be
            # encoded once; PNG is lossless and universally wire-supported.
            representation = ImageRepresentation.PNG
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            raw = buf.getvalue()

        return (raw, pil.width, pil.height, representation)
