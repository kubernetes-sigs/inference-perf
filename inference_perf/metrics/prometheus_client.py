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
import requests
from inference_perf.client.base import ModelServerPrometheusMetric
from .base import MetricsClient, MetricsSummary, PerfRuntimeParameters


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
            print("Invalid metric type: %s" % (metric_type))
            return ""
        if query_op not in queries[metric_type]:
            print("Invalid query operation: %s" % (query_op))
            return ""
        return queries[metric_type][query_op]


class PrometheusMetricsClient(MetricsClient):
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def collect_metrics_summary(self, runtime_parameters: PerfRuntimeParameters) -> MetricsSummary | None:
        """
        Collects the summary metrics for the given Perf Benchmark Runtime Parameters.

        Args:
        runtime_parameters: The runtime parameters containing details about the Perf Benchmark like the duration and model server client

        Returns:
        A MetricsSummary object containing the summary metrics.
        """
        metrics_summary: MetricsSummary = MetricsSummary()
        if runtime_parameters is None:
            print("Perf Runtime parameters are not set, skipping metrics collection")
            return None

        # Get the duration and model server client from the runtime parameters
        eval_time = runtime_parameters.eval_time
        duration = runtime_parameters.duration
        model_server_client = runtime_parameters.model_server_client

        # Get the engine and model from the model server client
        if not model_server_client:
            print("Model server client is not set")
            return None

        metrics_metadata = model_server_client.get_prometheus_metric_metadata()
        if not metrics_metadata:
            print("Metrics metadata is not present for the runtime")
            return None

        # Populate query builders for each metric
        query_builders = {}
        for summary_metric_name in metrics_metadata:
            if metrics_metadata[summary_metric_name] is None:
                print("Metric metadata is not present for metric: %s" % (summary_metric_name))
                continue
            query_builders[summary_metric_name] = PrometheusQueryBuilder(metrics_metadata[summary_metric_name], duration)

        for summary_metric_name in query_builders:
            query = query_builders[summary_metric_name].build_query()
            if not query:
                print("No query found for metric: %s. Skipping metric." % (summary_metric_name))
                continue

            # Execute the query and get the result
            result = self.execute_query(query, str(eval_time))
            if result is None:
                print("Error executing query: %s" % (query))
                continue
            # Set the result in the metrics summary
            attr = getattr(metrics_summary, summary_metric_name)
            if attr is not None:
                target_type = type(attr)
                setattr(metrics_summary, summary_metric_name, target_type(result))

        return metrics_summary

    def execute_query(self, query: str, eval_time: str) -> float:
        """
        Executes the given query on the Prometheus server and returns the result.

        Args:
        query: the PromQL query to execute

        Returns:
        The result of the query.
        """
        query_result = 0.0
        response = requests.get(f"{self.base_url}/api/v1/query", params={"query": query, "time": eval_time})
        if response is None:
            print("Error executing query: %s" % (query))
            return query_result

        response.raise_for_status()

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
            print("Error executing query: %s" % (response_obj))
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
                    query_result = float(result[0]["value"][1])
                except ValueError:
                    print("Error converting value to float: %s" % (result[0]["value"][1]))
                    return query_result
        return query_result
