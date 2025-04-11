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
from inference_perf.metrics.base import PerfRuntimeParameters
from inference_perf.metrics.prometheus_client import PrometheusMetricsClient
from inference_perf.reportgen import ReportGenerator, MockReportGenerator
from inference_perf.metrics import MockMetricsClient
from inference_perf.config import read_config
import asyncio

PROMETHEUS_SCRAPE_BUFFER_SEC = 5


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
            uri=config.vllm.url, model_name=config.vllm.model_name, api_type=config.vllm.api
        )
    else:
        raise Exception("vLLM client config missing")

    # Define DataGenerator
    if config.data:
        datagen = HFShareGPTDataGenerator() if config.data.type == DataGenType.ShareGPT else MockDataGenerator()
    else:
        raise Exception("data config missing")

    # Define LoadGenerator
    if config.load:
        loadgen = LoadGenerator(datagen, config.load.type, rate=config.load.rate, duration=config.load.duration)
    else:
        raise Exception("load config missing")

    # Define Metrics Client
    if config.metrics_client:
        if config.metrics_client.type == MetricsClientType.PROMETHEUS:
            if config.metrics_client.prometheus:
                url = config.metrics_client.prometheus.url
                if not url:
                    raise Exception("prometheus url missing")
                scrape_interval = config.metrics_client.prometheus.scrape_interval or 30
            else:
                raise Exception("prometheus config missing")
            metrics_client = PrometheusMetricsClient(base_url=url)
        else:
            metrics_client = MockMetricsClient()
    else:
        raise Exception("metrics config missing")

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
    if config.metrics_client is not None and config.metrics_client.type == MetricsClientType.PROMETHEUS:
        # Wait for the metrics to be scraped, added a buffer to even the last request's metrics are collected
        wait_time = scrape_interval + PROMETHEUS_SCRAPE_BUFFER_SEC
        print(f"Waiting for {wait_time} seconds for Prometheus to collect metrics...")
        time.sleep(wait_time)
    end_time = time.time()
    duration = end_time - start_time  # Calculate the duration of the test

    # Generate Report after the test
    perfrunner.generate_report(
        PerfRuntimeParameters(end_time, duration, model_server_client)
    )  # TODO pass start_time and sleep if the metrics server need it, e.g. Prometheus


if __name__ == "__main__":
    main_cli()
