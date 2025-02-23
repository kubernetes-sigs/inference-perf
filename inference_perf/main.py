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
from inference_perf.loadgen import LoadGenerator, LoadType
from inference_perf.datagen import MockDataGenerator
from inference_perf.client import ModelServerClient, vLLMModelServerClient
from inference_perf.reportgen import ReportGenerator, MockReportGenerator
import asyncio


class InferencePerfRunner:
    def __init__(self, client: ModelServerClient, loadgen: LoadGenerator, reportgen: ReportGenerator) -> None:
        self.client = client
        self.loadgen = loadgen
        self.reportgen = reportgen
        self.client.set_report_generator(self.reportgen)

    def run(self) -> None:
        asyncio.run(self.loadgen.run(self.client))

    def generate_report(self) -> None:
        asyncio.run(self.reportgen.generate_report())


def main_cli() -> None:
    # Define Model Server Client
    client = vLLMModelServerClient(uri="http://0.0.0.0:8000", model_name="openai-community/gpt2")

    # Define LoadGenerator
    loadgen = LoadGenerator(MockDataGenerator(), LoadType.CONSTANT, rate=2, duration=5)

    # Define ReportGenerator
    reportgen = MockReportGenerator()

    # Setup Perf Test Runner
    perfrunner = InferencePerfRunner(client, loadgen, reportgen)

    # Run Perf Test
    perfrunner.run()

    # Generate Report
    perfrunner.generate_report()


if __name__ == "__main__":
    main_cli()
