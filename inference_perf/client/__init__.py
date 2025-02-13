# Copyright 2025
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
from abc import ABC, abstractmethod
from dataset import InferenceData
import time


class Client(ABC):
    @abstractmethod
    def __init__(self, *args) -> None:
        pass

    @abstractmethod
    def process_request(selff, data: InferenceData) -> None:
        raise NotImplementedError


class TGIClient(Client):
    def __init__(self, uri: str) -> None:
        self.uri = uri

    def process_request(self, data: InferenceData) -> None:
        time.sleep(5)
        print(data.system_prompt)
