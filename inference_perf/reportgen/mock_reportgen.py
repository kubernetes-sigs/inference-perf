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
from .base import ReportGenerator, Metric
from typing import List


class MockReportGenerator(ReportGenerator):
    def __init__(self) -> None:
        self.metrics: List[Metric] = []

    def collect_metrics(self, metric: Metric) -> None:
        self.metrics.append(metric)

    def generate_report(self) -> None:
        print("\n\nGenerating Report ..")
        print("Report: Total Requests = " + str(len(self.metrics)))
