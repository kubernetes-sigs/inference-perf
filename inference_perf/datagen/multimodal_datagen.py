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
import base64
from typing import Any, Generator, List, Optional, Tuple, Union
import numpy as np
from inference_perf.apis.base import InferenceAPIData, LazyLoadInferenceAPIData
from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.config import APIConfig, APIType, DataConfig, Distribution
from inference_perf.datagen.multimodal_assets import (
    generate_mp4_bytes,
    generate_png_bytes,
    generate_mp3_bytes,
    resolution_to_wh,
    sample_audio_duration,
    sample_image_resolution,
    sample_video_profile,
)
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.utils.distribution import sample_from_distribution
from .base import DataGenerator, LazyLoadDataMixin


# Size of the rotating pool used to amortize per-profile video encoding cost
# while still producing distinct byte payloads (so server-side mm caches
# don't collapse all requests to a single hit).
_VIDEO_POOL_SIZE = 4


class MultimodalDataGenerator(DataGenerator, LazyLoadDataMixin):
    """Generates synthetic multimodal data (text, images, video, audio) for benchmarking."""

    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: Optional[CustomTokenizer]) -> None:
        super().__init__(api_config, config, tokenizer)

        if config.multimodal is None:
            raise ValueError("Multimodal config is required for MultimodalDataGenerator")

        self.multimodal_config = config.multimodal
        self.rng: np.random.Generator = np.random.default_rng()

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for MultimodalDataGenerator")

        # Cache vocab_size so dummy prompts generated from random token IDs
        # tokenize back to approximately the requested token length (rather
        # than the compressible "word word word" placeholder used before).
        hf_tokenizer = self.tokenizer.get_tokenizer()
        if hasattr(hf_tokenizer, "vocab_size") and hf_tokenizer.vocab_size:
            self.vocab_size: int = int(hf_tokenizer.vocab_size)
        elif hasattr(hf_tokenizer, "get_vocab") and callable(hf_tokenizer.get_vocab):
            self.vocab_size = len(hf_tokenizer.get_vocab())
        else:
            try:
                self.vocab_size = len(hf_tokenizer)
            except TypeError as e:
                raise ValueError(
                    "Tokenizer does not expose vocab_size, get_vocab(), or len(). "
                    "Cannot generate prompts at configured token lengths."
                ) from e
        if self.vocab_size <= 0:
            raise ValueError(f"Tokenizer vocabulary size must be positive, got {self.vocab_size}.")

        # (width, height, frames) -> list of encoded MP4 byte blobs. Filled lazily.
        self._video_pool: dict[Tuple[int, int, int], List[bytes]] = {}

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Chat]

    def is_io_distribution_supported(self) -> bool:
        return True

    def is_shared_prefix_supported(self) -> bool:
        return False

    def _generate_dummy_text(self, length: int) -> str:
        """Generate dummy text whose re-tokenization lands near *length* tokens.

        Decoding random token IDs from the configured tokenizer's vocabulary
        gives a prompt that actually tokenizes back to ~length tokens, so
        prefill cost tracks the configured ``input_distribution``.
        """
        if length <= 0 or self.tokenizer is None:
            return ""
        token_ids = self.rng.integers(0, self.vocab_size, size=length, dtype=np.int64).tolist()
        return str(self.tokenizer.get_tokenizer().decode(token_ids, skip_special_tokens=True))

    def _get_insertion_point(self, config_insertion_point: Optional[Union[float, Distribution]]) -> float:
        if config_insertion_point is None:
            return float(self.rng.uniform(0.0, 1.0))
        if isinstance(config_insertion_point, float):
            return config_insertion_point
        return float(sample_from_distribution(config_insertion_point, 1, self.rng)[0])

    def _assemble_content(self, text_str: str, media_items: list[tuple[dict[str, Any], float]]) -> list[dict[str, Any]]:
        if not media_items:
            return [{"type": "text", "text": text_str}]

        media_items.sort(key=lambda x: x[1])
        content = []
        last_idx = 0
        text_len = len(text_str)

        for block, point in media_items:
            point = max(0.0, min(1.0, point))
            idx = int(point * text_len)

            if idx > last_idx:
                content.append({"type": "text", "text": text_str[last_idx:idx]})
                last_idx = idx

            content.append(block)

        if last_idx < text_len:
            content.append({"type": "text", "text": text_str[last_idx:]})

        return content

    def _image_block(self, data_url: str) -> dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": data_url}}

    def _audio_block(self, data_url: str) -> dict[str, Any]:
        return {"type": "audio_url", "audio_url": {"url": data_url}}

    def _get_video_bytes(self, width: int, height: int, frames: int) -> bytes:
        """Return an MP4 encoding for the requested profile, lazily populating a pool."""
        key = (width, height, frames)
        pool = self._video_pool.get(key)
        if pool is None:
            pool = []
            self._video_pool[key] = pool
        if len(pool) < _VIDEO_POOL_SIZE:
            pool.append(generate_mp4_bytes(width, height, frames, self.rng))
        return pool[int(self.rng.integers(0, len(pool)))]

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> InferenceAPIData:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for MultimodalDataGenerator")

        media_items: list[tuple[dict[str, Any], float]] = []
        image_count = 0
        video_count = 0
        frame_count = 0
        image_instances: list[dict[str, Any]] = []
        video_instances: list[dict[str, Any]] = []
        audio_instances: list[dict[str, Any]] = []

        if self.multimodal_config.image:
            img_cfg = self.multimodal_config.image
            if img_cfg.count:
                image_count = int(sample_from_distribution(img_cfg.count, 1, self.rng)[0])

            for _ in range(image_count):
                w, h = sample_image_resolution(img_cfg, self.rng)
                png_bytes = generate_png_bytes(w, h, self.rng)
                data_url = f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}"
                point = self._get_insertion_point(img_cfg.insertion_point)
                media_items.append((self._image_block(data_url), point))
                image_instances.append({"pixels": w * h, "bytes": len(png_bytes), "aspect_ratio": w / h if h > 0 else 0.0})

        if self.multimodal_config.video:
            vid_cfg = self.multimodal_config.video
            if vid_cfg.count:
                video_count = int(sample_from_distribution(vid_cfg.count, 1, self.rng)[0])

            for _ in range(video_count):
                profile = sample_video_profile(vid_cfg, self.rng)
                w, h = resolution_to_wh(profile.resolution)
                mp4_bytes = self._get_video_bytes(w, h, profile.frames)
                data_url = f"data:video/mp4;base64,{base64.b64encode(mp4_bytes).decode('ascii')}"
                point = self._get_insertion_point(vid_cfg.insertion_point)
                media_items.append(
                    (
                        {"type": "video_url", "video_url": {"url": data_url}},
                        point,
                    )
                )
                frame_count += profile.frames
                video_instances.append(
                    {
                        "pixels": w * h,
                        "bytes": len(mp4_bytes),
                        "aspect_ratio": w / h if h > 0 else 0.0,
                        "frames": profile.frames,
                    }
                )

        audio_count = 0
        audio_seconds = 0.0
        if self.multimodal_config.audio:
            aud_cfg = self.multimodal_config.audio
            if aud_cfg.count:
                audio_count = int(sample_from_distribution(aud_cfg.count, 1, self.rng)[0])

            for _ in range(audio_count):
                duration = sample_audio_duration(aud_cfg, self.rng)
                mp3_bytes = generate_mp3_bytes(duration, self.rng)
                data_url = f"data:audio/mp3;base64,{base64.b64encode(mp3_bytes).decode('ascii')}"
                point = self._get_insertion_point(aud_cfg.insertion_point)
                media_items.append((self._audio_block(data_url), point))
                audio_seconds += duration
                audio_instances.append({"bytes": len(mp3_bytes), "seconds": duration})

        text_len = 100
        if self.input_distribution:
            text_len = int(sample_from_distribution(self.input_distribution, 1, self.rng)[0])

        text_str = self._generate_dummy_text(text_len)
        content = self._assemble_content(text_str, media_items)

        messages = [ChatMessage(role="user", content=content)]

        max_tokens = 0
        if self.output_distribution:
            max_tokens = int(sample_from_distribution(self.output_distribution, 1, self.rng)[0])

        multimodal_metrics = {
            "images": image_count,
            "videos": video_count,
            "frames": frame_count,
            "image_instances": image_instances,
            "video_instances": video_instances,
            "audio_clips": audio_count,
            "audio_seconds": audio_seconds,
            "audio_instances": audio_instances,
        }

        return ChatCompletionAPIData(messages=messages, max_tokens=max_tokens, multimodal_metrics=multimodal_metrics)

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        i = 0
        while True:
            yield LazyLoadInferenceAPIData(data_index=i)
            i += 1
