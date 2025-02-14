# Copyright 2025
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
from loadgen import LoadGenerator, LoadType
from dataset import MockDataGenerator
from client import Client, MockModelServerClient
from reportgen import ReportGenerator, MockReportGenerator


class InferencePerfRunner:
    def __init__(self, client: Client, loadgen: LoadGenerator, reportgen: ReportGenerator) -> None:
        self.client = client
        self.loadgen = loadgen
        self.reportgen = reportgen
        self.client.set_report_generator(self.reportgen)

    def run(self) -> None:
        self.loadgen.run(self.client)

    def generate_report(self) -> None:
        self.reportgen.generate_report()


def main():
    # Define Model Server Client
    client = MockModelServerClient(uri="0.0.0.0:0")

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
    main()
