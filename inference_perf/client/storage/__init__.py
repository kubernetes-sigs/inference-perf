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
<<<<<<<< HEAD:inference_perf/client/storage/__init__.py
from .base import StorageClient
from .gcs import GoogleCloudStorageClient
========
>>>>>>>> 9afd930bf7a97b1014ba1a425fabf71705ddfe4a:inference_perf/metrics/observed.py

from typing import List
from inference_perf.metrics.base import MetricsSource
from inference_perf.config import MetricsConfig, RequestMetric

<<<<<<<< HEAD:inference_perf/client/storage/__init__.py
__all__ = ["StorageClient", "GoogleCloudStorageClient"]
========
class ObservedMetricsCollector(MetricsSource):
    def __init__(self, config: MetricsConfig) -> None:
        self.config = config
        self.metrics: List[RequestMetric] = []
        pass
>>>>>>>> 9afd930bf7a97b1014ba1a425fabf71705ddfe4a:inference_perf/metrics/observed.py
