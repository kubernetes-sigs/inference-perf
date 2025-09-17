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
from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel
from .config import CircuitBreakerConfig
from .triggers.base import Trigger, build_trigger


class CircuitBreaker(ABC):
    """
    Responsible for determine if the benchmark should short-circuit or not.
    """

    def __init__(self, config: CircuitBreakerConfig) -> None:
        self.name = config.name
        self._triggers: List[Trigger] = [build_trigger(t) for t in config.triggers]
        self._open = False

    @abstractmethod
    def feed(self, metric: BaseModel) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_open(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
