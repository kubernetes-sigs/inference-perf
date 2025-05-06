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
import json
from typing import Any, List
from inference_perf.config import ReportConfig, RequestMetric
from inference_perf.metrics.observed import ObservedMetricsCollector


class ReportFile:
    name: str
    contents: dict[str, Any]

    def __init__(self, name: str, contents: dict[str, Any]):
        self.name = f"{name}.json"
        self.contents = contents
        self._store_locally()

    def _store_locally(self) -> None:
        with open(self.get_filename(), "w", encoding="utf-8") as f:
            f.write(json.dumps(self.get_contents(), indent=2))

    def get_filename(self) -> str:
        return self.name

    def get_contents(self) -> dict[str, Any]:
        return self.contents


class ReportGenerator:
    def __init__(self, config: ReportConfig, observed_metrics_collector: ObservedMetricsCollector) -> None:
        self.config = config
        self.metrics_collector = observed_metrics_collector

    def collect_request_metric(self, metric: RequestMetric) -> None:
        self.metrics_collector.record_metric(metric)

    async def generate_reports(self) -> List[ReportFile]:
        print("\n\nGenerating Report ..")
        if self.config is not None:
            if self.config is not None:
                report = self.config.get_report(self.metrics_collector.get_metrics())
                return [ReportFile(name="report", contents=report)] if report else []

        else:
            print("Report generation failed - no metrics collected")
            return []
