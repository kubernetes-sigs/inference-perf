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

from inference_perf.client.metrics import ClientRequestMetric, ClientRequestMetricsCollector
from inference_perf.config import ReportConfig


class ReportFile:
    name: str
    contents: dict[str, Any]

    def __init__(self, name: str, contents: dict[str, Any]):
        self.name = f"{name}.json"
        self.contents = contents
        self._store_locally()

    def _store_locally(self) -> None:
        filename = self.get_filename()
        contents = self.get_contents()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(json.dumps(contents, indent=2))

    def get_filename(self) -> str:
        return self.name

    def get_contents(self) -> dict[str, Any]:
        return self.contents


class ReportGenerator:
    def __init__(self, config: ReportConfig, client_request_metrics_collector: ClientRequestMetricsCollector) -> None:
        self.config = config
        self.client_request_metrics_collector = client_request_metrics_collector

    def collect_request_metric(self, metric: ClientRequestMetric) -> None:
        self.client_request_metrics_collector.record_metric(metric)

    async def generate_reports(self) -> List[ReportFile]:
        if len(self.client_request_metrics_collector.get_metrics()) == 0:
            print("Report generation failed - no metrics collected")
            return []
        elif self.config is None:
            print("Report generation disabled, skipping report generation")
            return []
        else:
            print("\n\nGenerating Report ..")
            report: dict[str, Any] = {}
            if self.config.observed:
                observed_report: dict[str, Any] = {}
                if self.config.observed.summary:
                    observed_report["summary"] = self.client_request_metrics_collector.get_summary_report()
                if self.config.observed.per_request:
                    observed_report["per_request"] = self.client_request_metrics_collector.get_per_request_report()
                report["observed"] = observed_report
            if self.config.prometheus:
                prometheus_report: dict[str, Any] = {}
                if prometheus_report is not None:
                    report["prometheus"] = prometheus_report
            return [ReportFile(name="report", contents=report)] if report else []
