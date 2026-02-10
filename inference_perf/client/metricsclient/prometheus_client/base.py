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
from abc import abstractmethod
import logging
import time
from typing import List, cast, Any, Optional
import requests
from inference_perf.client.modelserver.base import ModelServerPrometheusMetric
from inference_perf.config import PrometheusClientConfig
from ..base import MetricsClient, MetricsMetadata, PerfRuntimeParameters, ModelServerMetrics

PROMETHEUS_SCRAPE_BUFFER_SEC = 2

logger = logging.getLogger(__name__)


# When evaluated, returns a summary of the metric as a map, summary contents depends on the metric type
class PrometheusVectorMetric:
    def __init__(self, name: str, filters: List[str]) -> None:
        self.name = name
        self.filters = ",".join(filters)

    @abstractmethod
    def get_queries(self, duration: float) -> dict[str, str]:
        raise NotImplementedError


class PrometheusGaugeMetric(PrometheusVectorMetric):
    def __init__(self, name: str, filters: List[str]) -> None:
        super().__init__(name, filters)

    def get_queries(self, duration: float) -> dict[str, str]:
        return {
            "mean": "avg_over_time(%s{%s}[%.0fs])" % (self.name, self.filters, duration),
            "median": "quantile_over_time(0.5, %s{%s}[%.0fs])" % (self.name, self.filters, duration),
            "sd": "stddev_over_time(%s{%s}[%.0fs])" % (self.name, self.filters, duration),
            "min": "min_over_time(%s{%s}[%.0fs])" % (self.name, self.filters, duration),
            "max": "max_over_time(%s{%s}[%.0fs])" % (self.name, self.filters, duration),
            "p90": "quantile_over_time(0.9, %s{%s}[%.0fs])" % (self.name, self.filters, duration),
            "p99": "quantile_over_time(0.99, %s{%s}[%.0fs])" % (self.name, self.filters, duration),
        }


class PrometheusCounterMetric(PrometheusVectorMetric):
    def __init__(self, name: str, filters: List[str]) -> None:
        super().__init__(name, filters)

    def get_queries(self, duration: float) -> dict[str, str]:
        return {
            "rate": "sum(rate(%s{%s}[%.0fs]))" % (self.name, self.filters, duration),
            "mean": "avg_over_time(rate(%s{%s}[%.0fs])[%.0fs:%.0fs])"
            % (self.name, self.filters, duration, duration, duration),
            "increase": "sum(increase(%s{%s}[%.0fs]))" % (self.name, self.filters, duration),
        }


class PrometheusHistogramMetric(PrometheusVectorMetric):
    def __init__(self, name: str, filters: List[str]) -> None:
        super().__init__(name, filters)

    def get_queries(self, duration: float) -> dict[str, str]:
        return {
            "mean": "sum(rate(%s_sum{%s}[%.0fs])) / (sum(rate(%s_count{%s}[%.0fs])) > 0)"
            % (self.name, self.filters, duration, self.name, self.filters, duration),
            "median": "histogram_quantile(0.5, sum(rate(%s_bucket{%s}[%.0fs])) by (le))" % (self.name, self.filters, duration),
            "min": "histogram_quantile(0, sum(rate(%s_bucket{%s}[%.0fs])) by (le))" % (self.name, self.filters, duration),
            "max": "histogram_quantile(1, sum(rate(%s_bucket{%s}[%.0fs])) by (le))" % (self.name, self.filters, duration),
            "p90": "histogram_quantile(0.9, sum(rate(%s_bucket{%s}[%.0fs])) by (le))" % (self.name, self.filters, duration),
            "p99": "histogram_quantile(0.99, sum(rate(%s_bucket{%s}[%.0fs])) by (le))" % (self.name, self.filters, duration),
        }


# When evaluated, returns a single value
class PrometheusScalarMetric:
    def __init__(self, op: str, metric: PrometheusVectorMetric) -> None:
        self.op = op
        self.metric = metric

    def get_query(self, duration: float) -> str:
        query = self.metric.get_queries(duration)
        if self.op in query:
            return query[self.op]
        raise Exception(f"query of type {type(self.metric).__name__}, does not contain the operation {self.op}")


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
                "increase": "sum(increase(%s_count{%s}[%.0fs]))" % (metric_name, filter, self.duration),
                "rate": "sum(rate(%s_count{%s}[%.0fs]))" % (metric_name, filter, self.duration),
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
            self.query_url = config.url.unicode_string().rstrip("/") + "/api/v1/query"
            self.query_range_url = config.url.unicode_string().rstrip("/") + "/api/v1/query_range"
            self.federate_url = config.url.unicode_string().rstrip("/") + "/federate"
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

    def collect_raw_metrics(
        self,
        filters: List[str],
        metrics_metadata: Optional[MetricsMetadata] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        interval: int = 5,
    ) -> dict[str, str] | None:
        """
        Collects the raw metrics from the Prometheus federate endpoint or query endpoint as fallback.

        Args:
        filters: The filters to apply to the metrics collection
        metrics_metadata: Optional metadata for specific metrics to fetch if general query is restricted
        start_time: Start time for range query
        end_time: End time for range query
        interval: Step interval for range query

        Returns:
        A dictionary mapping metric names to their raw metrics in text format.
        """
        match_param = "{" + ",".join(filters) + "}"
        # If google_managed is true, we use the query endpoint as /federate is not supported
        is_google_managed = "monitoring.googleapis.com" in self.query_url
        is_range_query = start_time is not None and end_time is not None

        if not is_range_query and not is_google_managed:
            try:
                logger.debug(f"making PromQL federate query: '{self.federate_url}' with match[]='{match_param}'")
                response = requests.get(self.federate_url, headers=self.get_headers(), params={"match[]": match_param})
                if response is not None and response.status_code == 200:
                    metrics: dict[str, list[str]] = {}
                    for line in response.text.splitlines():
                        if not line:
                            continue
                        if line.startswith("#"):
                            parts = line.split(" ")
                            if len(parts) >= 3:
                                metric_name = parts[2]
                                if metric_name not in metrics:
                                    metrics[metric_name] = []
                                metrics[metric_name].append(line)
                        else:
                            metric_name = line.split("{")[0].split(" ")[0]
                            if metric_name not in metrics:
                                metrics[metric_name] = []
                            metrics[metric_name].append(line)
                    return {k: "\n".join(v) + "\n" for k, v in metrics.items()}
                if response is not None:
                    logger.debug(f"federate query failed with status {response.status_code}, falling back to query endpoint")
            except Exception as e:
                logger.debug(f"federate query failed: {e}, falling back to query endpoint")

        # Fallback to query/query_range endpoint and convert JSON to Prometheus text format
        prom_metrics: dict[str, list[str]] = {}
        url = self.query_range_url if is_range_query else self.query_url

        # If we have metrics_metadata, we should try to fetch those specifically if the general query fails
        # This is especially important for GMP which doesn't allow bare vector selectors
        queries_to_try = [match_param]
        if is_google_managed and metrics_metadata:
            # For GMP, we can't use the match_param alone, so we collect the names of all metrics we track
            tracked_metric_names = set()
            for metric_key in metrics_metadata:
                metadata = metrics_metadata.get(metric_key)
                if metadata:
                    tracked_metric_names.add(metadata.name)

            queries_to_try += [f"{name}{match_param}" for name in tracked_metric_names]
            logger.debug(
                f"GMP detected, will try querying {len(queries_to_try)} queries for {'range' if is_range_query else 'instant'} query: {tracked_metric_names}"
            )

        for query in queries_to_try:
            try:
                params: dict[str, Any] = {"query": query}
                if is_range_query:
                    params["start"] = str(start_time)
                    params["end"] = str(end_time)
                    params["step"] = f"{interval}s"

                logger.debug(f"making PromQL query: '{url}' with params={params}")
                response = requests.get(url, headers=self.get_headers(), params=params)
                if response is None:
                    continue

                if response.status_code != 200:
                    logger.debug(f"query for '{query}' failed with status {response.status_code}")
                    continue

                response_obj = response.json()
                if response_obj.get("status") != "success":
                    continue

                data = response_obj.get("data", {})
                results = data.get("result", [])
                for res in results:
                    metric = res.get("metric", {})
                    if not metric:
                        continue

                    metric_name = metric.get("__name__", "unknown")
                    labels = [f'{k}="{v}"' for k, v in metric.items() if k != "__name__"]
                    label_str = "{" + ",".join(labels) + "}" if labels else ""

                    if metric_name not in prom_metrics:
                        prom_metrics[metric_name] = []

                    if is_range_query:
                        values = res.get("values", [])  # List of [timestamp, value]
                        for val_pair in values:
                            if len(val_pair) < 2:
                                continue
                            val = val_pair[1]
                            timestamp_ms = int(float(val_pair[0]) * 1000)
                            prom_metrics[metric_name].append(f"{metric_name}{label_str} {val} {timestamp_ms}")
                    else:
                        value = res.get("value", [])
                        if len(value) < 2:
                            continue
                        val = value[1]
                        prom_metrics[metric_name].append(f"{metric_name}{label_str} {val}")
            except Exception as e:
                logger.debug(f"error querying '{query}': {e}")

        if prom_metrics:
            return {k: "\n".join(v) + "\n" for k, v in prom_metrics.items()}

        return None

    def get_model_server_metrics(
        self, metrics_metadata: MetricsMetadata, query_duration: float, query_eval_time: float
    ) -> Optional[ModelServerMetrics]:
        """
        Collects the summary metrics for the given Model Server Client and query duration.

        Args:
        model_server_metrics: The object containing the relevent model server metrics
        query_duration: The duration for which to collect metrics
        query_eval_time: The time at which the query is evaluated, used to ensure we are querying the correct time range

        Returns:
        A ModelServerMetrics object containing the summary metrics.
        """
        model_server_metrics: ModelServerMetrics = ModelServerMetrics()

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
