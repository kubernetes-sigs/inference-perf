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
import time
from inference_perf.loadgen import LoadGenerator
from inference_perf.config import DataGenType, MetricsClientType
from inference_perf.datagen import MockDataGenerator, HFShareGPTDataGenerator
from inference_perf.client import ModelServerClient, vLLMModelServerClient
from inference_perf.metrics.base import MetricsClient, PerfRuntimeParameters
from inference_perf.metrics.prometheus_client import PrometheusMetricsClient
from inference_perf.reportgen import ReportGenerator, MockReportGenerator
from inference_perf.config import read_config
import asyncio


class InferencePerfRunner:
    def __init__(self, client: ModelServerClient, loadgen: LoadGenerator, reportgen: ReportGenerator) -> None:
        self.client = client
        self.loadgen = loadgen
        self.reportgen = reportgen

    def run(self) -> None:
        asyncio.run(self.loadgen.run(self.client))

    def generate_report(self, runtime_parameters: PerfRuntimeParameters) -> None:
        asyncio.run(self.reportgen.generate_report(runtime_parameters=runtime_parameters))


def main_cli() -> None:
    config = read_config()

    # Define Model Server Client
    if config.vllm:
        model_server_client = vLLMModelServerClient(
            uri=config.vllm.url, model_name=config.vllm.model_name, tokenizer=config.tokenizer, api_type=config.vllm.api
        )
    else:
        raise Exception("vLLM client config missing")

    # Define DataGenerator
    if config.data:
        datagen = MockDataGenerator(config.vllm.api)
        if config.data.type == DataGenType.ShareGPT:
            datagen = HFShareGPTDataGenerator(config.vllm.api)
    else:
        raise Exception("data config missing")

    # Define LoadGenerator
    if config.load:
        loadgen = LoadGenerator(datagen, config.load)
    else:
        raise Exception("load config missing")

    # Define Metrics Client
    metrics_client: MetricsClient | None = None
    if config.metrics_client:
        if config.metrics_client.type == MetricsClientType.PROMETHEUS:
            metrics_client = PrometheusMetricsClient(metrics_client_config=config.metrics_client)

    # Define Report Generator
    if config.report:
        reportgen = MockReportGenerator(metrics_client)
    else:
        raise Exception("report config missing")

    # Setup Perf Test Runner
    perfrunner = InferencePerfRunner(model_server_client, loadgen, reportgen)

    start_time = time.time()

    # Run Perf Test
    perfrunner.run()

    # Wait for metrics collection
    if metrics_client is not None:
        # Wait for the metrics to be ready
        metrics_client.wait()

    end_time = time.time()
    duration = end_time - start_time  # Calculate the duration of the test

    # Generate Report after the test
    perfrunner.generate_report(PerfRuntimeParameters(end_time, duration, model_server_client))


if __name__ == "__main__":
    main_cli()
