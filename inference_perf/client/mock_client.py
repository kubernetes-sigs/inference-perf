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
from inference_perf.datagen import InferenceData
from inference_perf.reportgen import ReportGenerator, Metric
from .base import ModelServerClient


class MockModelServerClient(ModelServerClient):
    def __init__(self, uri: str) -> None:
        self.uri = uri

    def set_report_generator(self, reportgen: ReportGenerator) -> None:
        self.reportgen = reportgen

    def process_request(self, data: InferenceData) -> None:
        print("Processing request - " + data.system_prompt)
        self.reportgen.collect_metrics(Metric(name=data.system_prompt))
