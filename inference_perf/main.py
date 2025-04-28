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
from inference_perf.loadgen import LoadGenerator
from inference_perf.config import DataGenType
from inference_perf.datagen import MockDataGenerator, HFShareGPTDataGenerator
from inference_perf.client import ModelServerClient, vLLMModelServerClient
from inference_perf.reportgen import ReportGenerator, MockReportGenerator
from inference_perf.metrics import MockMetricsClient
from inference_perf.config import read_config
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
    config = read_config()

    # Define Model Server Client
    if config.vllm:
        client = vLLMModelServerClient(
            uri=config.vllm.url, model_name=config.vllm.model_name, tokenizer=config.tokenizer, api_type=config.vllm.api
        )
    else:
        raise Exception("vLLM client config missing")

    # Define DataGenerator
    if config.data:
        datagen = MockDataGenerator(config.vllm.api)
        if config.data.type == DataGenType.ShareGPT:
            datagen = HFShareGPTDataGenerator(config.vllm.api)
            
        if not datagen:
            raise Exception("dataset incompatible with model server api type")
    else:
        raise Exception("data config missing")

    # Define LoadGenerator
    if config.load:
        loadgen = LoadGenerator(datagen, config.load)
    else:
        raise Exception("load config missing")

    # Define Metrics Client
    if config.metrics:
        metricsclient = MockMetricsClient(uri=config.metrics.url)
    else:
        raise Exception("metrics config missing")

    # Define Report Generator
    if config.report:
        reportgen = MockReportGenerator(metricsclient)
    else:
        raise Exception("report config missing")

    # Setup Perf Test Runner
    perfrunner = InferencePerfRunner(client, loadgen, reportgen)

    # Run Perf Test
    perfrunner.run()

    # Generate Report
    perfrunner.generate_report()


if __name__ == "__main__":
    main_cli()
