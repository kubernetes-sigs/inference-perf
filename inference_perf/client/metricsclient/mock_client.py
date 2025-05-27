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
from .base import MetricsClient, PerfRuntimeParameters, ModelServerMetrics


class MockMetricsClient(MetricsClient):
    def __init__(self) -> None:
        pass

    def collect_metrics_summary(self, runtime_parameters: PerfRuntimeParameters) -> ModelServerMetrics | None:
        return None

    def collect_metrics_for_stage(self, runtime_parameters: PerfRuntimeParameters, stage_id: int) -> ModelServerMetrics | None:
        return None

    def wait(self) -> None:
        pass
