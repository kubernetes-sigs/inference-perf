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
"""MMMU real-dataset loader for multimodal benchmarking.

MMMU (Massive Multi-discipline Multimodal Understanding,
https://mmmu-benchmark.github.io/) is a college-level VLM evaluation
benchmark spanning 30 subjects across 6 disciplines. The loader streams
examples from the gated HuggingFace mirror at ``MMMU/MMMU``, extracts the
embedded PIL images, re-encodes them, and emits chat-completion requests
with the question + options as text and the images attached as
:class:`PreEncodedImageSpec` blocks.

Unlike :class:`ShareGPT4VideoDataGenerator`, MMMU is small enough to fit in
memory — init loads all configured subjects synchronously and
``load_lazy_data`` does an O(1) lookup. No background download thread.
"""

from __future__ import annotations

import ast
import io
import logging
from typing import Any, Generator, List, Optional, Tuple

from PIL import Image as PILImage

from inference_perf.apis.base import InferenceAPIData, LazyLoadInferenceAPIData
from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.config import APIConfig, APIType, DataConfig, MMMUConfig
from inference_perf.datagen.multimodal_sampling import resolution_to_wh
from inference_perf.payloads import (
    ImageRepresentation,
    MultimodalSpec,
    PreEncodedImageSpec,
)
from inference_perf.utils.custom_tokenizer import CustomTokenizer

from .base import DataGenerator, LazyLoadDataMixin
from .gated_hf_dataset import GatedHFDataset

logger = logging.getLogger(__name__)


# MMMU canonical subject configs. The dataset is partitioned per-subject on
# HuggingFace, so loading "all" means iterating these and concatenating. List
# sourced from the dataset card at https://huggingface.co/datasets/MMMU/MMMU;
# override via ``MMMUConfig.subjects`` if upstream changes.
_MMMU_SUBJECTS: Tuple[str, ...] = (
    "Accounting",
    "Agriculture",
    "Architecture_and_Engineering",
    "Art",
    "Art_Theory",
    "Basic_Medical_Science",
    "Biology",
    "Chemistry",
    "Clinical_Medicine",
    "Computer_Science",
    "Design",
    "Diagnostics_and_Laboratory_Medicine",
    "Economics",
    "Electronics",
    "Energy_and_Power",
    "Finance",
    "Geography",
    "History",
    "Literature",
    "Manufacturing",
    "Marketing",
    "Materials",
    "Math",
    "Mechanical_Engineering",
    "Music",
    "Pharmacy",
    "Physics",
    "Psychology",
    "Public_Health",
    "Sociology",
)

_MAX_IMAGES_PER_EXAMPLE = 7


class MMMUDataGenerator(DataGenerator, LazyLoadDataMixin, GatedHFDataset):
    """Emits MMMU benchmark requests as chat-completions with image_url blocks."""

    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: Optional[CustomTokenizer]) -> None:
        DataGenerator.__init__(self, api_config, config, tokenizer)
        if config.mmmu is None:
            raise ValueError("mmmu config is required for MMMUDataGenerator")
        self.cfg: MMMUConfig = config.mmmu
        GatedHFDataset.__init__(self, token=self.cfg.token)

        self._pil_format = "JPEG" if self.cfg.representation == ImageRepresentation.JPEG else "PNG"
        self._target_wh: Optional[Tuple[int, int]] = (
            resolution_to_wh(self.cfg.target_resolution) if self.cfg.target_resolution is not None else None
        )

        # Load examples from all configured subjects. MMMU is small enough that
        # this fits in memory.
        self._examples: List[dict[str, Any]] = self._load_examples()
        if not self._examples:
            raise RuntimeError(
                "MMMU loader found no examples with images for the configured subjects/split. "
                f"subjects={self.cfg.subjects or 'all'} split={self.cfg.hf_split}"
            )
        logger.info(
            "MMMU loader ready: %d examples across %d subjects, split=%s",
            len(self._examples),
            len(self.cfg.subjects or _MMMU_SUBJECTS),
            self.cfg.hf_split,
        )

    # ---- HF loading -------------------------------------------------------

    def _load_examples(self) -> List[dict[str, Any]]:
        subjects = list(self.cfg.subjects or _MMMU_SUBJECTS)
        out: List[dict[str, Any]] = []
        for subj in subjects:
            ds = self._load_dataset(
                self.cfg.hf_dataset_name,
                name=subj,
                split=self.cfg.hf_split,
                cache_dir=self.cfg.cache_dir,
            )
            for ex in ds:
                if self._example_has_images(ex):
                    out.append(ex)
                    if self.cfg.max_examples is not None and len(out) >= self.cfg.max_examples:
                        return out
        return out

    @staticmethod
    def _example_has_images(ex: dict[str, Any]) -> bool:
        for i in range(1, _MAX_IMAGES_PER_EXAMPLE + 1):
            if ex.get(f"image_{i}") is not None:
                return True
        return False

    # ---- DataGenerator interface ------------------------------------------

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
        idx = data.data_index % len(self._examples)
        ex = self._examples[idx]
        image_specs = self._build_image_specs(ex)
        prompt = self._build_prompt(ex)
        spec = MultimodalSpec(images=image_specs)
        return ChatCompletionAPIData(
            messages=[ChatMessage(role="user", content=prompt)],
            multimodal_spec=spec,
        )

    # ---- Per-request materialization helpers -----------------------------

    def _build_image_specs(self, ex: dict[str, Any]) -> List[PreEncodedImageSpec]:
        specs: List[PreEncodedImageSpec] = []
        cap = min(self.cfg.max_images_per_request, _MAX_IMAGES_PER_EXAMPLE)
        for i in range(1, cap + 1):
            raw = ex.get(f"image_{i}")
            if raw is None:
                continue
            img = raw if isinstance(raw, PILImage.Image) else None
            if img is None:
                # Tolerate dicts (some HF schemas wrap images) by skipping; we
                # only handle PIL.Image in v1.
                continue
            if self._target_wh is not None and (img.width, img.height) != self._target_wh:
                img = img.resize(self._target_wh)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format=self._pil_format)
            encoded = buf.getvalue()
            specs.append(
                PreEncodedImageSpec(
                    width=img.width,
                    height=img.height,
                    insertion_point=self.cfg.insertion_point,
                    representation=self.cfg.representation,
                    image_bytes=encoded,
                )
            )
        return specs

    @staticmethod
    def _build_prompt(ex: dict[str, Any]) -> str:
        parts: List[str] = []
        question = ex.get("question") or ""
        parts.append(str(question))
        opts = ex.get("options")
        if opts:
            opts_list = opts if isinstance(opts, list) else _parse_options(opts)
            if opts_list:
                parts.append("")
                parts.append("Options:")
                letters = "ABCDEFGHIJ"
                for letter, opt in zip(letters, opts_list):
                    parts.append(f"{letter}. {opt}")
        parts.append("")
        parts.append("Answer:")
        return "\n".join(parts)


def _parse_options(opts_str: str) -> List[str]:
    """MMMU stores ``options`` as a Python-literal string list. Best-effort parse."""
    try:
        parsed = ast.literal_eval(opts_str)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return [opts_str] if opts_str else []
