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
from typing import Any, List, Optional

from pydantic import BaseModel
from inference_perf.client.client_interfaces.prometheus.base import PrometheusMetricsCollector
from inference_perf.config import ReportConfig
from inference_perf.datagen.base import PromptMetricsCollector


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


class ReportGenerator(BaseModel):
    client_request_metrics_collector: PromptMetricsCollector
    prometheus_metrics_collector: Optional[PrometheusMetricsCollector]

    async def generate_reports(self, config: ReportConfig, duration: float) -> List[ReportFile]:
        print(f"Generating report according to config {config.model_dump_json()}")
        if not hasattr(config, "observed") and not hasattr(config, "prometheus"):
            print("Report generation disabled, skipping report generation")
            return []
        else:
            report: dict[str, Any] = {}

            if (
                self.client_request_metrics_collector is not None
                and config.prompt is not None
                and len(self.client_request_metrics_collector.list_metrics()) != 0
            ):
                print("Reporting prompt metrics")
                report["observed"] = await self.client_request_metrics_collector.to_report(
                    report_config=config.prompt, duration=duration
                )
            else:
                print("Not reporting prompt metrics")

            print("HERE", self.prometheus_metrics_collector, "AND", config.prometheus)
            if self.prometheus_metrics_collector is not None and config.prometheus is not None:
                print("Reporting prometheus metrics")
                prometheus_report: dict[str, Any] = {}
                if prometheus_report is not None:
                    report["prometheus"] = await self.prometheus_metrics_collector.to_report(
                        report_config=config.prometheus, duration=duration
                    )
            else:
                print("Not reporting prometheus metrics")

            return [ReportFile(name="report", contents=report)] if report else []
