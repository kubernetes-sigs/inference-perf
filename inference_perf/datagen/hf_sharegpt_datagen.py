# Copyright 2025 The Kubernetes Authors.
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
from .base import DataGenerator, InferenceData, ChatCompletionData, ChatMessage
from inference_perf.config import APIType
from typing import Generator
from datasets import load_dataset


class HFShareGPTDataGenerator(DataGenerator):
    def __init__(self) -> None:
        self.sharegpt_dataset = iter(
            load_dataset(
                "anon8231489123/ShareGPT_Vicuna_unfiltered",
                data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
                streaming=True,
                split="train",
            )
        )
        self.min_num_turns = 2
        self.data_key = "conversations"
        self.role_key = "from"
        self.content_key = "value"
        # initialize data collection
        next(self.sharegpt_dataset)

    def get_api(self) -> APIType:
        return APIType.Chat

    def get_data(self) -> Generator[InferenceData, None, None]:
        if self.sharegpt_dataset is not None:
            while True:
                data = next(self.sharegpt_dataset)
                if (
                    data is None
                    or data[self.data_key] is None
                    or len(data[self.data_key]) < self.min_num_turns
                    or len(data[self.data_key]) == 0
                ):
                    continue
                else:
                    yield InferenceData(
                        type=APIType.Chat,
                        chat=ChatCompletionData(
                            messages=[
                                ChatMessage(role=conversation[self.role_key], content=conversation[self.content_key])
                                for conversation in data[self.data_key]
                            ]
                        ),
                    )
