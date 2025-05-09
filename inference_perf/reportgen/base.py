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
import statistics
from typing import Any, List, Optional
from inference_perf.client.base import ClientRequestMetricsCollector
from inference_perf.client.client_interfaces.prometheus.prometheus_client import PrometheusMetricsCollector
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
    def __init__(
        self,
        client_request_metrics_collector: ClientRequestMetricsCollector,
        prometheus_metrics_collector: Optional[PrometheusMetricsCollector],
    ) -> None:
        self.client_request_metrics_collector = client_request_metrics_collector
        self.prometheus_metrics_collector = prometheus_metrics_collector

    async def generate_reports(self, config: ReportConfig, duration: float) -> List[ReportFile]:
        print(f"Generating report according to config {config.model_dump_json()}")
        if len(self.client_request_metrics_collector.list_metrics()) == 0:
            print("Report generation failed, no metrics collected")
            return []
        elif not hasattr(config, "observed") and not hasattr(config, "prometheus"):
            print("Report generation disabled, skipping report generation")
            return []
        else:
            print("\n\nGenerating Report ..")
            report: dict[str, Any] = {}
            if self.client_request_metrics_collector is not None and config.observed:
                report["observed"] = self.client_request_metrics_collector.get_report(config=config.observed)
            if self.prometheus_client_collector is not None and config.prometheus:
                prometheus_report: dict[str, Any] = {}
                if prometheus_report is not None:
                    report["prometheus"] = await self.prometheus_metrics_collector.to_report(duration=duration)
            return [ReportFile(name="report", contents=report)] if report else []
