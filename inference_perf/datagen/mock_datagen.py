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
from .base import DataGenerator, InferenceData, CompletionData
from typing import Generator


class MockDataGenerator(DataGenerator):
    def __init__(self) -> None:
        pass

    def get_data(self) -> Generator[InferenceData, None, None]:
        i = 0
        while True:
            i += 1
            yield InferenceData(data=CompletionData(prompt="text" + str(i)))
