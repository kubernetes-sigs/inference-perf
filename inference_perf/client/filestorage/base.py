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

import logging
from abc import ABC, abstractmethod
from typing import List
from inference_perf.config import StorageConfigBase
from inference_perf.utils import ReportFile

logger = logging.getLogger(__name__)


class StorageClient(ABC):
    def __init__(self, config: StorageConfigBase) -> None:
        self.config = config
        logger.info(f"Report files will be stored at: {self.config.path}")

    @abstractmethod
    def save_report(self, reports: List[ReportFile]) -> None:
        raise NotImplementedError()
