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
from argparse import ArgumentParser
from inference_perf.analysis.analyze import analyze_reports
from typing import List, Optional
from inference_perf.loadgen import LoadGenerator
from inference_perf.config import (
    DataGenType,
    MetricsClientType,
    ModelServerType,
    ReportConfig,
    read_config,
)
from inference_perf.datagen import (
    DataGenerator,
    MockDataGenerator,
    HFShareGPTDataGenerator,
    SyntheticDataGenerator,
    RandomDataGenerator,
    SharedPrefixDataGenerator,
)
from inference_perf.client.modelserver import ModelServerClient, vLLMModelServerClient
from inference_perf.client.metricsclient.base import MetricsClient, PerfRuntimeParameters
from inference_perf.client.metricsclient.prometheus_client import PrometheusMetricsClient, GoogleManagedPrometheusMetricsClient
from inference_perf.client.filestorage import StorageClient, GoogleCloudStorageClient, LocalStorageClient, SimpleStorageServiceClient
from inference_perf.client.requestdatacollector import (
    RequestDataCollector,
    LocalRequestDataCollector,
    MultiprocessRequestDataCollector,
)
from inference_perf.reportgen import ReportGenerator
from inference_perf.utils import CustomTokenizer, ReportFile
from inference_perf.logger import setup_logging
import asyncio
import time


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
        async def _run() -> None:
            collector = self.reportgen.get_metrics_collector()
            if isinstance(collector, MultiprocessRequestDataCollector):
                collector.start()
            await self.loadgen.run(self.client)
            if isinstance(collector, MultiprocessRequestDataCollector):
                await collector.stop()

        asyncio.run(_run())

    def generate_reports(self, report_config: ReportConfig, runtime_parameters: PerfRuntimeParameters) -> List[ReportFile]:
        return asyncio.run(self.reportgen.generate_reports(report_config=report_config, runtime_parameters=runtime_parameters))

    def save_reports(self, reports: List[ReportFile]) -> None:
        for storage_client in self.storage_clients:
            storage_client.save_report(reports)

    def stop(self) -> None:
        asyncio.run(self.loadgen.stop())


def main_cli() -> None:
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_file", help="Config File", required=False)
    parser.add_argument("-a", "--analyze", help="Path to a report directory to analyze.", required=False)
    parser.add_argument(
        "--log-level", help="Logging level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.analyze:
        analyze_reports(args.analyze)
        return

    if not args.config_file:
        parser.error("argument -c/--config_file is required when not using --analyze")

    config = read_config(args.config_file)

    # Define Metrics Client
    metrics_client: Optional[MetricsClient] = None
    if config.metrics:
        if config.metrics.type == MetricsClientType.PROMETHEUS and config.metrics.prometheus:
            if config.metrics.prometheus.google_managed:
                metrics_client = GoogleManagedPrometheusMetricsClient(config.metrics.prometheus)
            else:
                metrics_client = PrometheusMetricsClient(config=config.metrics.prometheus)

    # Define Storage Clients
    storage_clients: List[StorageClient] = []
    if config.storage:
        if config.storage.local_storage:
            storage_clients.append(LocalStorageClient(config=config.storage.local_storage))
        if config.storage.google_cloud_storage:
            storage_clients.append(GoogleCloudStorageClient(config=config.storage.google_cloud_storage))
        if config.storage.simple_storage_service:
            storage_clients.append(SimpleStorageServiceClient(config=config.storage.simple_storage_service))

    # Define Report Generator
    collector: RequestDataCollector
    if config.load.num_workers > 0:
        collector = MultiprocessRequestDataCollector()
    else:
        collector = LocalRequestDataCollector()
    reportgen = ReportGenerator(metrics_client, collector)

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
            model_server_client = vLLMModelServerClient(
                reportgen.get_metrics_collector(),
                api_config=config.api,
                uri=config.server.base_url,
                model_name=config.server.model_name,
                tokenizer_config=config.tokenizer,
                ignore_eos=config.server.ignore_eos,
                max_tcp_connections=config.load.worker_max_tcp_connections,
                additional_filters=config.metrics.prometheus.filters if config.metrics and config.metrics.prometheus else [],
                api_key=config.server.api_key,
            )
            # vllm_client supports inferring the tokenizer
            tokenizer = model_server_client.tokenizer
    else:
        raise Exception("model server client config missing")

    # Check load exists so datagen can derive total_count from the
    # stage configurations.
    if config.load is None:
        raise Exception("load config missing")

    # Define DataGenerator
    datagen: DataGenerator
    if config.data:
        # Common checks for generators that require a tokenizer / distribution
        if config.data.type in [DataGenType.ShareGPT, DataGenType.Synthetic, DataGenType.Random]:
            if tokenizer is None:
                raise Exception(
                    f"{config.data.type.value} data generator requires a configured tokenizer. "
                    "Please ensure a valid tokenizer is configured in the 'tokenizer' section of your config file."
                )
        if config.data.type in [DataGenType.Synthetic, DataGenType.Random]:
            if config.data.input_distribution is None:
                raise Exception(f"{config.data.type.value} data generator requires 'input_distribution' to be configured")
            if config.data.output_distribution is None:
                raise Exception(f"{config.data.type.value} data generator requires 'output_distribution' to be configured")

            total_count = int(max([stage.rate * stage.duration for stage in config.load.stages])) + 1
            if config.data.input_distribution.total_count is None:
                config.data.input_distribution.total_count = total_count
            if config.data.output_distribution.total_count is None:
                config.data.output_distribution.total_count = total_count

        if config.data.type == DataGenType.SharedPrefix and config.data.shared_prefix is None:
            raise Exception(f"{config.data.type.value} data generator requires 'shared_prefix' to be configured")

        if config.data.type == DataGenType.ShareGPT:
            datagen = HFShareGPTDataGenerator(config.api, config.data, tokenizer)
        elif config.data.type == DataGenType.Synthetic:
            datagen = SyntheticDataGenerator(config.api, config.data, tokenizer)
        elif config.data.type == DataGenType.Random:
            datagen = RandomDataGenerator(config.api, config.data, tokenizer)
        elif config.data.type == DataGenType.SharedPrefix:
            datagen = SharedPrefixDataGenerator(config.api, config.data, tokenizer)
        else:
            datagen = MockDataGenerator(config.api, config.data, tokenizer)
    else:
        raise Exception("data config missing")

    # Define LoadGenerator
    if config.load:
        if (
            isinstance(metrics_client, PrometheusMetricsClient)
            and config.report.prometheus
            and config.report.prometheus.per_stage
        ):
            config.load.interval = max(config.load.interval, metrics_client.scrape_interval)
        loadgen = LoadGenerator(datagen, config.load)
    else:
        raise Exception("load config missing")

    # Setup Perf Test Runner
    perfrunner = InferencePerfRunner(model_server_client, loadgen, reportgen, storage_clients)

    start_time = time.time()

    # Run Perf Test
    perfrunner.run()

    end_time = time.time()
    duration = end_time - start_time  # Calculate the duration of the test

    # Generate Reports after the tests
    reports = perfrunner.generate_reports(
        report_config=config.report,
        runtime_parameters=PerfRuntimeParameters(
            start_time=start_time,
            duration=duration,
            model_server_client=model_server_client,
            stages=loadgen.stage_runtime_info,
        ),
    )

    # Save Reports
    perfrunner.save_reports(reports=reports)

    perfrunner.stop()


if __name__ == "__main__":
    main_cli()
