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
from inference_perf.config import DataGenType, MetricsServerType
from inference_perf.datagen import MockDataGenerator, HFShareGPTDataGenerator
from inference_perf.client import ModelServerClient, vLLMModelServerClient
from inference_perf.metrics.prometheus_client import PrometheusMetricsClient
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

    def generate_report(self, duration = None) -> None:
        asyncio.run(self.reportgen.generate_report(duration, self.client))


def main_cli() -> None:
    scrape_interval = 0
    config = read_config()

    # Define Model Server Client
    if config.vllm:
        client = vLLMModelServerClient(uri=config.vllm.url, model_name=config.vllm.model_name, api_type=config.vllm.api)
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
    if config.metrics:
        if config.metrics.server_type == MetricsServerType.Prometheus and config.metrics.prometheus_server_config:
            server_url = config.metrics.prometheus_server_config.url
            scrape_interval = config.metrics.prometheus_server_config.scrape_interval
            metricsclient = PrometheusMetricsClient(base_url=server_url)
        else:
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

    start_time = time.time()

    # Run Perf Test
    perfrunner.run()
    
    if scrape_interval > 0:
        # Wait for the metrics to be collected
        print(f"Waiting for {2*scrape_interval} seconds for metrics to be collected...")
        time.sleep(2*scrape_interval)
    duration = time.time() - start_time  # Calculate the duration of the test

    # Generate Report after the test
    # engine is passed for PrometheusMetricsClient to collect metrics for the specific engine
    perfrunner.generate_report(duration) # TODO pass start_time and sleep if the metrics server need it, e.g. Prometheus


if __name__ == "__main__":
    main_cli()
