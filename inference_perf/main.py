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
from inference_perf.datagen.base import IODistribution
from inference_perf.loadgen import LoadGenerator
from inference_perf.config import DataGenType, MetricsClientType, ModelServerType
from inference_perf.datagen import DataGenerator, MockDataGenerator, HFShareGPTDataGenerator, SyntheticDataGenerator
from inference_perf.client.modelserver import ModelServerClient, vLLMModelServerClient
from inference_perf.metrics.base import MetricsClient, PerfRuntimeParameters
from inference_perf.metrics.prometheus_client import PrometheusMetricsClient
from inference_perf.client.storage import StorageClient, GoogleCloudStorageClient
from inference_perf.reportgen import ReportGenerator, ReportFile
from inference_perf.config import read_config
from inference_perf.utils.custom_tokenizer import CustomTokenizer
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

    # Create tokenizer based on tokenizer config
    tokenizer: Optional[CustomTokenizer] = None
    if config.tokenizer and config.tokenizer.pretrained_model_name_or_path:
        try:
            tokenizer = CustomTokenizer(config.tokenizer)
        except Exception as e:
            raise Exception("Tokenizer initialization failed") from e

    # Define Model Server Client
    model_server_client: ModelServerClient
    if config.server:
        if config.server.type == ModelServerType.VLLM:
            # The type error for vLLMModelServerClient's tokenizer argument indicates it expects CustomTokenizer, not Optional.
            if tokenizer is None:
                raise Exception(
                    "vLLM client is configured, but it requires a custom tokenizer which was not provided or initialized successfully. "
                    "Please ensure a valid tokenizer is configured in the 'tokenizer' section of your config file."
                )
            model_server_client = vLLMModelServerClient(
                api_type=config.api,
                uri=config.server.base_url,
                model_name=config.server.model_name,
                tokenizer=tokenizer,
            )
    else:
        raise Exception("model server client config missing")

    # Define DataGenerator
    datagen: DataGenerator
    if config.data:
        # Common checks for generators that require a tokenizer
        if config.data.type in [DataGenType.ShareGPT, DataGenType.Synthetic]:
            if tokenizer is None:
                raise Exception(
                    f"{config.data.type.value} data generator requires a configured tokenizer. "
                    "Please ensure a valid tokenizer is configured in the 'tokenizer' section of your config file."
                )

        if config.data.type == DataGenType.ShareGPT:
            datagen = HFShareGPTDataGenerator(config.api, None, tokenizer)

        elif config.data.type == DataGenType.Synthetic:
            if config.data.input_distribution is None:
                raise Exception("SyntheticDataGenerator requires 'input_distribution' to be configured")

            io_distribution = IODistribution(input=config.data.input_distribution)
            datagen = SyntheticDataGenerator(config.api, ioDistribution=io_distribution, tokenizer=tokenizer)
        else:
            datagen = MockDataGenerator(config.api)
    else:
        raise Exception("data config missing")

    # Define LoadGenerator
    if config.load:
        loadgen = LoadGenerator(datagen, config.load)
    else:
        raise Exception("load config missing")

    # Define Metrics Client
    metrics_client: Optional[MetricsClient] = None
    if config.metrics:
        if config.metrics.type == MetricsClientType.PROMETHEUS and config.metrics.prometheus:
            metrics_client = PrometheusMetricsClient(config=config.metrics.prometheus)

    # Define Storage Clients
    storage_clients: List[StorageClient] = []
    if config.storage:
        if config.storage.google_cloud_storage:
            storage_clients.append(GoogleCloudStorageClient(config=config.storage.google_cloud_storage))

    # Define Report Generator
    reportgen = ReportGenerator(metrics_client)

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
