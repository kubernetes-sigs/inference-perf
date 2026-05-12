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
from typing import Generator, List, Optional

import numpy as np

from inference_perf.apis.base import InferenceAPIData, LazyLoadInferenceAPIData
from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
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
from inference_perf.config import APIConfig, APIType, DataConfig
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.utils.distribution import sample_from_distribution
from inference_perf.datagen.multimodal_sampling import (
    resolution_to_wh,
    sample_audio_duration,
    sample_image_resolution,
    sample_insertion_point,
    sample_video_profile,
)

from .base import DataGenerator, LazyLoadDataMixin


class MultimodalDataGenerator(DataGenerator, LazyLoadDataMixin):
    """Samples synthetic multimodal request specs for benchmarking.

    Datagen produces only a ``MultimodalSpec`` (resolutions, profiles,
    durations, insertion points) plus a text prompt; byte materialization
    and OpenAI multimodal content assembly happen in
    :meth:`ChatCompletionAPIData.to_request_body` so the heavy bytes never
    live on the request lifecycle metric.
    """

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

    def _build_spec(self) -> MultimodalSpec:
        spec = MultimodalSpec()

        img_cfg = self.multimodal_config.image
        if img_cfg and img_cfg.count:
            img_count_dist = img_cfg.count
            count = int(sample_from_distribution(img_count_dist, 1, self.rng)[0])
            for _ in range(count):
                w, h = sample_image_resolution(img_cfg, self.rng)
                spec.images.append(
                    SyntheticImageSpec(
                        width=w,
                        height=h,
                        insertion_point=sample_insertion_point(img_cfg.insertion_point, self.rng),
                        representation=img_cfg.representation,
                    )
                )

        vid_cfg = self.multimodal_config.video
        if vid_cfg and vid_cfg.count:
            vid_count_dist = vid_cfg.count
            count = int(sample_from_distribution(vid_count_dist, 1, self.rng)[0])
            for _ in range(count):
                profile = sample_video_profile(vid_cfg, self.rng)
                w, h = resolution_to_wh(profile.resolution)
                insertion_point = sample_insertion_point(vid_cfg.insertion_point, self.rng)
                video_spec: VideoSpecUnion
                if vid_cfg.representation == VideoRepresentation.MP4:
                    video_spec = SyntheticMp4VideoSpec(
                        width=w, height=h, frames=profile.frames, insertion_point=insertion_point
                    )
                else:
                    video_spec = SyntheticFramesVideoSpec(
                        width=w,
                        height=h,
                        frames=profile.frames,
                        insertion_point=insertion_point,
                        frame_representation=(
                            ImageRepresentation.JPEG
                            if vid_cfg.representation == VideoRepresentation.JPEG_FRAMES
                            else ImageRepresentation.PNG
                        ),
                    )
                spec.videos.append(video_spec)

        aud_cfg = self.multimodal_config.audio
        if aud_cfg and aud_cfg.count:
            aud_count_dist = aud_cfg.count
            count = int(sample_from_distribution(aud_count_dist, 1, self.rng)[0])
            for _ in range(count):
                spec.audios.append(
                    SyntheticAudioSpec(
                        duration=sample_audio_duration(aud_cfg, self.rng),
                        insertion_point=sample_insertion_point(aud_cfg.insertion_point, self.rng),
                    )
                )

        return spec

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> InferenceAPIData:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for MultimodalDataGenerator")

        spec = self._build_spec()

        text_len = 100
        if self.input_distribution:
            text_len = int(sample_from_distribution(self.input_distribution, 1, self.rng)[0])
        text_str = self._generate_dummy_text(text_len)

        return ChatCompletionAPIData(
            messages=[ChatMessage(role="user", content=text_str)],
            multimodal_spec=spec,
        )

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        i = 0
        while True:
            yield LazyLoadInferenceAPIData(data_index=i)
            i += 1
