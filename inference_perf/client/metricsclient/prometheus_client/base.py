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
import logging
import time
from typing import cast, Any
import requests
from inference_perf.client.modelserver.base import ModelServerClient, ModelServerPrometheusMetric
from inference_perf.config import PrometheusClientConfig
from ..base import MetricsClient, PerfRuntimeParameters, ModelServerMetrics

PROMETHEUS_SCRAPE_BUFFER_SEC = 2

logger = logging.getLogger(__name__)


class PrometheusQueryBuilder:
    def __init__(self, model_server_metric: ModelServerPrometheusMetric, duration: float):
        self.model_server_metric = model_server_metric
        self.duration = duration

    def get_queries(self) -> dict[str, dict[str, str]]:
        """
        Returns a dictionary of queries for each metric type.
        """
        metric_name = self.model_server_metric.name
        filter = self.model_server_metric.filters
        return {
            "gauge": {
                "mean": "avg_over_time(%s{%s}[%.0fs])" % (metric_name, filter, self.duration),
                "median": "quantile_over_time(0.5, %s{%s}[%.0fs])" % (metric_name, filter, self.duration),
                "sd": "stddev_over_time(%s{%s}[%.0fs])" % (metric_name, filter, self.duration),
                "min": "min_over_time(%s{%s}[%.0fs])" % (metric_name, filter, self.duration),
                "max": "max_over_time(%s{%s}[%.0fs])" % (metric_name, filter, self.duration),
                "p90": "quantile_over_time(0.9, %s{%s}[%.0fs])" % (metric_name, filter, self.duration),
                "p99": "quantile_over_time(0.99, %s{%s}[%.0fs])" % (metric_name, filter, self.duration),
            },
            "histogram": {
                "mean": "sum(rate(%s_sum{%s}[%.0fs])) / (sum(rate(%s_count{%s}[%.0fs])) > 0)"
                % (metric_name, filter, self.duration, metric_name, filter, self.duration),
                "median": "histogram_quantile(0.5, sum(rate(%s_bucket{%s}[%.0fs])) by (le))"
                % (metric_name, filter, self.duration),
                "min": "histogram_quantile(0, sum(rate(%s_bucket{%s}[%.0fs])) by (le))" % (metric_name, filter, self.duration),
                "max": "histogram_quantile(1, sum(rate(%s_bucket{%s}[%.0fs])) by (le))" % (metric_name, filter, self.duration),
                "p90": "histogram_quantile(0.9, sum(rate(%s_bucket{%s}[%.0fs])) by (le))"
                % (metric_name, filter, self.duration),
                "p99": "histogram_quantile(0.99, sum(rate(%s_bucket{%s}[%.0fs])) by (le))"
                % (metric_name, filter, self.duration),
            },
            "counter": {
                "rate": "sum(rate(%s{%s}[%.0fs]))" % (metric_name, filter, self.duration),
                "increase": "sum(increase(%s{%s}[%.0fs]))" % (metric_name, filter, self.duration),
                "mean": "avg_over_time(rate(%s{%s}[%.0fs])[%.0fs:%.0fs])"
                % (metric_name, filter, self.duration, self.duration, self.duration),
                "max": "max_over_time(rate(%s{%s}[%.0fs])[%.0fs:%.0fs])"
                % (metric_name, filter, self.duration, self.duration, self.duration),
                "min": "min_over_time(rate(%s{%s}[%.0fs])[%.0fs:%.0fs])"
                % (metric_name, filter, self.duration, self.duration, self.duration),
                "p90": "quantile_over_time(0.9, rate(%s{%s}[%.0fs])[%.0fs:%.0fs])"
                % (metric_name, filter, self.duration, self.duration, self.duration),
                "p99": "quantile_over_time(0.99, rate(%s{%s}[%.0fs])[%.0fs:%.0fs])"
                % (metric_name, filter, self.duration, self.duration, self.duration),
            },
        }

    def build_query(self) -> str:
        """
        Builds the PromQL query for the given metric type and query operation.

        Returns:
        The PromQL query.
        """
        metric_type = self.model_server_metric.type
        query_op = self.model_server_metric.op

        queries = self.get_queries()
        if metric_type not in queries:
            logger.warning("Invalid metric type: %s" % (metric_type))
            return ""
        if query_op not in queries[metric_type]:
            logger.warning("Invalid query operation: %s" % (query_op))
            return ""
        return queries[metric_type][query_op]


class PrometheusMetricsClient(MetricsClient):
    def __init__(self, config: PrometheusClientConfig) -> None:
        if config:
            if not config.url:
                raise Exception("prometheus url missing")
            self.query_url = config.url.unicode_string() + "api/v1/query"
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

    def collect_metrics_summary(self, runtime_parameters: PerfRuntimeParameters) -> ModelServerMetrics | None:
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

        return self.get_model_server_metrics(runtime_parameters.model_server_client, query_duration, query_eval_time)

    def collect_metrics_for_stage(self, runtime_parameters: PerfRuntimeParameters, stage_id: int) -> ModelServerMetrics | None:
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
        query_eval_time = runtime_parameters.stages[stage_id].end_time + self.scrape_interval + PROMETHEUS_SCRAPE_BUFFER_SEC
        query_duration = query_eval_time - runtime_parameters.stages[stage_id].start_time
        return self.get_model_server_metrics(runtime_parameters.model_server_client, query_duration, query_eval_time)

    def get_model_server_metrics(
        self, model_server_client: ModelServerClient, query_duration: float, query_eval_time: float
    ) -> ModelServerMetrics | None:
        """
        Collects the summary metrics for the given Model Server Client and query duration.

        Args:
        model_server_client: The model server client to use for collecting metrics
        query_duration: The duration for which to collect metrics
        query_eval_time: The time at which the query is evaluated, used to ensure we are querying the correct time range

        Returns:
        A ModelServerMetrics object containing the summary metrics.
        """
        model_server_metrics: ModelServerMetrics = ModelServerMetrics()

        # Get the engine and model from the model server client
        if not model_server_client:
            logger.warning("Model server client is not set")
            return None

        metrics_metadata = model_server_client.get_prometheus_metric_metadata()
        if not metrics_metadata:
            logger.warning("Metrics metadata is not present for the runtime")
            return None
        for summary_metric_name in metrics_metadata:
            summary_metric_metadata = metrics_metadata.get(summary_metric_name)
            if summary_metric_metadata is None:
                logger.warning("Metric metadata is not present for metric: %s. Skipping this metric." % (summary_metric_name))
                continue
            summary_metric_metadata = cast(ModelServerPrometheusMetric, summary_metric_metadata)
            if summary_metric_metadata is None:
                logger.warning(
                    "Metric metadata for %s is missing or has an incorrect format. Skipping this metric."
                    % (summary_metric_name)
                )
                continue

            query_builder = PrometheusQueryBuilder(summary_metric_metadata, query_duration)
            query = query_builder.build_query()
            if not query:
                logger.warning("No query found for metric: %s. Skipping metric." % (summary_metric_name))
                continue

            # Execute the query and get the result
            result = self.execute_query(query, str(query_eval_time))
            if result is None:
                logger.error("Error executing query: %s" % (query))
                continue
            # Set the result in metrics summary
            attr = getattr(model_server_metrics, summary_metric_name)
            if attr is not None:
                target_type = type(attr)
                setattr(model_server_metrics, summary_metric_name, target_type(result))

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
            logger.debug(f"Making PromQL query: '{query}'")
            response = requests.get(self.query_url, headers=self.get_headers(), params={"query": query, "time": eval_time})
            if response is None:
                logger.error("Error executing query: %s" % (query))
                return query_result

            response.raise_for_status()
        except Exception as e:
            logger.error("Error executing query: %s" % (e))
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
        if response_obj.get("status") != "success":
            logger.error("Error executing query: %s" % (response_obj))
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
                    logger.error("Error converting value to float: %s" % (result[0]["value"][1]))
                    return query_result
        return query_result

    def get_headers(self) -> dict[str, Any]:
        return {}
