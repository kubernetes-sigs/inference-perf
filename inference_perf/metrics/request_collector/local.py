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

from inference_perf.metrics.request_collector import RequestMetricCollector
from inference_perf.apis import RequestLifecycleMetric


class LocalRequestMetricCollector(RequestMetricCollector):
    """Responsible for accumulating client request metrics"""

    def record_metric(self, metric: RequestLifecycleMetric) -> None:
        self._collect(metric)
