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
from typing import List
from inference_perf.client.filestorage import StorageClient
from inference_perf.config import StorageConfigBase
from inference_perf.utils import ReportFile
import json
import os

logger = logging.getLogger(__name__)


class LocalStorageClient(StorageClient):
    def __init__(self, config: StorageConfigBase) -> None:
        self.config = config
        logger.info(f"Report files will be stored at: {self.config.path}")

    def save_report(self, reports: List[ReportFile]) -> None:
        for report in reports:
            filename = report.get_filename()
            report_path = f"{self.config.path if self.config.path else ''}/{self.config.report_file_prefix if self.config.report_file_prefix else ''}{filename}"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(report.get_contents(), indent=2))
            logger.info(f"Report saved to: {report_path}")
