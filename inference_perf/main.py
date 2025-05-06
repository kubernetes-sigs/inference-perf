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
from typing import List
from inference_perf.loadgen import LoadGenerator
from inference_perf.config import DataGenType, read_config
from inference_perf.datagen import MockDataGenerator, HFShareGPTDataGenerator
from inference_perf.client import ModelServerClient, vLLMModelServerClient
from inference_perf.reportgen import ReportGenerator, ReportFile
from inference_perf.metrics import ObservedMetricsCollector
from inference_perf.client.storage import StorageClient, GoogleCloudStorageClient
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
        self.client.set_report_generator(self.reportgen)

    def run(self) -> None:
        asyncio.run(self.loadgen.run(self.client))

    def generate_reports(self) -> List[ReportFile]:
        return asyncio.run(self.reportgen.generate_reports())

    def save_reports(self, reports: List[ReportFile]):
        for storage_client in self.storage_clients:
            storage_client.save_report(reports)


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
    else:
        raise Exception("data config missing")

    # Define LoadGenerator
    if config.load:
        loadgen = LoadGenerator(datagen, config.load)
    else:
        raise Exception("load config missing")

    # Define collector for observed metrics
    observed_metrics_client = ObservedMetricsCollector(config.metrics)

    # Define Report Generator
    reportgen = ReportGenerator(config=config.report, observed_metrics_collector=observed_metrics_client)

    # Define Storage Clients
    storage_clients: List[StorageClient] = []
    if config.storage:
        if config.storage.google_cloud_storage:
            storage_clients.append(GoogleCloudStorageClient(config=config.storage.google_cloud_storage))

    # Setup Perf Test Runner
    perfrunner = InferencePerfRunner(client, loadgen, reportgen, storage_clients)

    # Run Perf Test
    perfrunner.run()

    # Generate Reports
    reports = perfrunner.generate_reports()

    # Save Reports
    perfrunner.save_reports(reports=reports)


if __name__ == "__main__":
    main_cli()
