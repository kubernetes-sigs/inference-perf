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
from typing import List, Optional
from inference_perf.loadgen import LoadGenerator
from inference_perf.config import DataGenType
from inference_perf.datagen import DataGenerator, MockDataGenerator, HFShareGPTDataGenerator
from inference_perf.client import ModelServerClient, vLLMModelServerClient
from inference_perf.metrics.prometheus_client import PerfRuntimeParameters, PrometheusMetricsCollector
from inference_perf.client.storage import StorageClient, GoogleCloudStorageClient
from inference_perf.reportgen import ReportGenerator, ReportFile
from inference_perf.config import read_config
import asyncio


class InferencePerfRunner:
    def __init__(
        self,
        client: ModelServerClient,
        loadgen: LoadGenerator,
        reportgen: ReportGenerator,
        storage_clients: List[StorageClient],
    ) -> None:
        self.client = client
        self.loadgen = loadgen
        self.reportgen = reportgen
        self.storage_clients = storage_clients

    def run(self) -> None:
        asyncio.run(self.loadgen.run(self.client))

    def generate_reports(self, runtime_parameters: PerfRuntimeParameters) -> List[ReportFile]:
        return asyncio.run(self.reportgen.generate_reports(runtime_parameters=runtime_parameters))

    def save_reports(self, reports: List[ReportFile]) -> None:
        for storage_client in self.storage_clients:
            storage_client.save_report(reports)


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
        datagen: DataGenerator
        if config.data.type == DataGenType.ShareGPT:
            datagen = HFShareGPTDataGenerator(config.vllm.api)
        else:
            datagen = MockDataGenerator(config.vllm.api)
    else:
        raise Exception("data config missing")

    # Define LoadGenerator
    if config.load:
        loadgen = LoadGenerator(datagen, config.load)
    else:
        raise Exception("load config missing")

    # Define Metrics Clients
    prometheus_metrics_client: Optional[PrometheusMetricsCollector] = (
        PrometheusMetricsCollector(config=config.metrics_client.prometheus)
        if config.metrics_clietn is not None and config.metrics_client.prometheus
        else None
    )

    # Define Report Generator
    reportgen = ReportGenerator(
        client_request_metrics_collector=model_server_client.collector, prometheus_metrics_collector=prometheus_metrics_client
    )

    # Define Storage Clients
    storage_clients: List[StorageClient] = []
    if config.storage:
        if config.storage.google_cloud_storage:
            storage_clients.append(GoogleCloudStorageClient(config=config.storage.google_cloud_storage))

    # Setup Perf Test Runner
    perfrunner = InferencePerfRunner(model_server_client, loadgen, reportgen, storage_clients)

    start_time = time.time()

    # Run Perf Test
    perfrunner.run()

    end_time = time.time()
    duration = end_time - start_time  # Calculate the duration of the test

    # Generate Report after the tests
    reports = perfrunner.generate_reports(PerfRuntimeParameters(start_time, duration, model_server_client))

    # Save Reports
    perfrunner.save_reports(reports=reports)


if __name__ == "__main__":
    main_cli()
