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
from inference_perf.client.base import ClientRequestMetricsCollector
from inference_perf.datagen.base import DataGenerator
from inference_perf.loadgen import LoadGenerator
from inference_perf.config import DataGenType, read_config
from inference_perf.datagen import MockDataGenerator, HFShareGPTDataGenerator
from inference_perf.client import ModelServerClient, vLLMModelServerClient
from inference_perf.reportgen import ReportGenerator
from inference_perf.client.storage import StorageClient, GoogleCloudStorageClient
import asyncio


class InferencePerfRunner:
    def __init__(
        self,
        client: ModelServerClient,
        loadgen: LoadGenerator,
    ) -> None:
        self.client = client
        self.loadgen = loadgen

    def run(self) -> None:
        asyncio.run(self.loadgen.run(self.client))


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

    # Define Storage Clients
    storage_clients: List[StorageClient] = []
    if config.storage:
        if config.storage.google_cloud_storage:
            storage_clients.append(GoogleCloudStorageClient(config=config.storage.google_cloud_storage))

    # Setup Perf Test Runner
    perfrunner = InferencePerfRunner(client, loadgen)

    # Run Perf Test
    perfrunner.run()

    # Define Report Generator
    reportgen = ReportGenerator(client_request_metrics_collector=client.collector)

    # Generate Reports
    reports = asyncio.run(reportgen.generate_reports(config=config.report))

    # Save Reports
    for storage_client in storage_clients:
        storage_client.save_report(reports)


if __name__ == "__main__":
    main_cli()
