# Copyright 2026 The Kubernetes Authors.
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
import logging
import time
from typing import Any, Optional
import requests
from inference_perf.client.modelserver.base import BaseMetrics
from inference_perf.config import PrometheusClientConfig
from ..base import MetricsClient, PerfRuntimeParameters, ModelServerMetrics

PROMETHEUS_SCRAPE_BUFFER_SEC = 2

logger = logging.getLogger(__name__)


class PrometheusMetricsClient(MetricsClient):
    def __init__(self, config: PrometheusClientConfig) -> None:
        if config:
            if not config.url:
                raise Exception("prometheus url missing")
            self.query_url = config.url.unicode_string().rstrip("/") + "/api/v1/query"
            logger.debug(f"Prometheus metrics client configured, querying metrics from '{self.query_url}'")
            self.scrape_interval = config.scrape_interval or 30
        else:
            raise Exception("prometheus config missing")

    def wait(self) -> None:
        """
        Waits for the Prometheus server to scrape the metrics.
        We have added a buffer of 5 seconds to the scrape interval to ensure that metrics for even the last request are collected.
        """
        wait_time = self.scrape_interval + PROMETHEUS_SCRAPE_BUFFER_SEC
        time.sleep(wait_time)

    def collect_metrics_summary(self, runtime_parameters: PerfRuntimeParameters) -> Optional[ModelServerMetrics]:
        """
        Collects the summary metrics for the given Perf Benchmark run.

        Args:
        runtime_parameters: The runtime parameters containing details about the Perf Benchmark like the duration and model server client

        Returns:
        A ModelServerMetrics object containing the summary metrics.
        """
        if runtime_parameters is None:
            logger.warning("Perf Runtime parameters are not set, skipping metrics collection")
            return None

        # Get the duration and model server client from the runtime parameters
        query_eval_time = time.time()
        query_duration = query_eval_time - runtime_parameters.start_time

        return self.get_model_server_metrics(runtime_parameters.model_server_metrics, query_duration, query_eval_time)

    def collect_metrics_for_stage(
        self, runtime_parameters: PerfRuntimeParameters, stage_id: int
    ) -> Optional[ModelServerMetrics]:
        """
        Collects the summary metrics for a specific stage.

        Args:
        runtime_parameters: The runtime parameters containing details about the Perf Benchmark like the duration and model server client
        stage_id: The ID of the stage for which to collect metrics

        Returns:
        A ModelServerMetrics object containing the summary metrics for the specified stage.
        """
        if runtime_parameters is None:
            logger.warning("Perf Runtime parameters are not set, skipping metrics collection")
            return None

        if runtime_parameters.stages is None or stage_id not in runtime_parameters.stages:
            logger.warning(
                f"Stage ID {stage_id} is not present in the runtime parameters, skipping metrics collection for this stage"
            )
            return None

        # Get the query evaluation time and duration for the stage
        # The query evaluation time is the end time of the stage plus the scrape interval and a buffer to ensure metrics are collected
        # Duration is calculated as the difference between the eval time and start time of the stage
        logger.debug(f"runtime parameters for stage {stage_id}: {runtime_parameters}")
        query_eval_time = runtime_parameters.stages[stage_id].end_time + self.scrape_interval + PROMETHEUS_SCRAPE_BUFFER_SEC
        query_duration = query_eval_time - runtime_parameters.stages[stage_id].start_time
        return self.get_model_server_metrics(runtime_parameters.model_server_metrics, query_duration, query_eval_time)

    def get_model_server_metrics(
        self, metrics_metadata: Any, query_duration: float, query_eval_time: float
    ) -> Optional[ModelServerMetrics]:
        """
        Collects the summary metrics for the given Model Server Client and query duration.

        Args:
        metrics_metadata: The object containing the relevant model server metrics (dict or list)
        query_duration: The duration for which to collect metrics
        query_eval_time: The time at which the query is evaluated, used to ensure we are querying the correct time range

        Returns:
        A ModelServerMetrics object containing the summary metrics.
        """
        model_server_metrics: ModelServerMetrics = ModelServerMetrics()

        if not metrics_metadata:
            logger.warning("Metrics metadata is not present for the runtime")
            return None

        if isinstance(metrics_metadata, list):
            metrics_to_process = metrics_metadata
        elif isinstance(metrics_metadata, BaseMetrics):
            if hasattr(metrics_metadata, "get_all_metrics"):
                metrics_to_process = metrics_metadata.get_all_metrics()
            else:
                metrics_to_process = metrics_metadata.custom_metrics
        else:
            logger.warning(f"Unknown metrics_metadata type: {type(metrics_metadata)}")
            return model_server_metrics

        for metric in metrics_to_process:
            for target_attr, query in metric.get_queries(query_duration):
                if not hasattr(model_server_metrics, target_attr):
                    logger.debug(f"Attribute {target_attr} not found in ModelServerMetrics, skipping.")
                    continue

                result = self.execute_query(query, str(query_eval_time))
                if result is not None:
                    attr = getattr(model_server_metrics, target_attr)
                    target_type = type(attr)
                    setattr(model_server_metrics, target_attr, target_type(result))

        return model_server_metrics

    def execute_query(self, query: str, eval_time: str) -> float:
        """
        Executes the given query on the Prometheus server and returns the result.

        Args:
        query: the PromQL query to execute
        eval_time: the time at which the query is evaluated, used to ensure we are querying the correct time range

        Returns:
        The result of the query.
        """
        query_result = 0.0
        try:
            logger.debug(f"making PromQL query: '{query}'")
            response = requests.get(self.query_url, headers=self.get_headers(), params={"query": query, "time": eval_time})
            if response is None:
                logger.error("error executing query: %s" % (query))
                return query_result

            response.raise_for_status()
        except Exception as e:
            logger.error("error executing query: %s" % (e))
            return query_result

        # Check if the response is valid
        # Sample response:
        # {
        #     "status": "success",
        #     "data": {
        #         "resultType": "vector",
        #         "result": [
        #             {
        #                 "metric": {},
        #                 "value": [
        #                     1632741820.781,
        #                     "0.0000000000000000"
        #                 ]
        #             }
        #         ]
        #     }
        # }

        response_obj = response.json()
        logger.debug(f"got result for query '{query}': {response_obj}")
        if response_obj.get("status") != "success":
            logger.error("error executing query: %s" % (response_obj))
            return query_result

        data = response_obj.get("data", {})
        result = data.get("result", [])
        if len(result) > 0 and "value" in result[0]:
            if isinstance(result[0]["value"], list) and len(result[0]["value"]) > 1:
                # Return the value of the first result
                # The value is in the second element of the list
                # e.g. [1632741820.781, "0.0000000000000000"]
                # We need to convert it to float
                # and return it
                # Convert the value to float
                try:
                    query_result = round(float(result[0]["value"][1]), 6)
                except ValueError:
                    logger.error("error converting value to float: %s" % (result[0]["value"][1]))
                    return query_result
        logger.debug(f"inferred result from query '{query}': {query_result}")
        return query_result

    def get_headers(self) -> dict[str, Any]:
        return {}
