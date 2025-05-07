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

from inference_perf.client.base import ClientRequestMetric, ClientRequestMetricsCollector
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
    def __init__(self, client_request_metrics_collector: ClientRequestMetricsCollector) -> None:
        self.client_request_metrics_collector = client_request_metrics_collector

    async def generate_reports(self, config: ReportConfig) -> List[ReportFile]:
        print(f"Generating report according to config {config.model_dump_json()}")
        if len(self.client_request_metrics_collector.get_metrics()) == 0:
            print("Report generation failed, no metrics collected")
            return []
        elif not hasattr(config, "observed") and not hasattr(config, "prometheus"):
            print("Report generation disabled, skipping report generation")
            return []
        else:
            print("\n\nGenerating Report ..")
            report: dict[str, Any] = {}
            if config.observed:
                report["observed"] = self.client_request_metrics_collector.get_report(config=config.observed)
            if config.prometheus:
                prometheus_report: dict[str, Any] = {}
                if prometheus_report is not None:
                    report["prometheus"] = prometheus_report
            return [ReportFile(name="report", contents=report)] if report else []
