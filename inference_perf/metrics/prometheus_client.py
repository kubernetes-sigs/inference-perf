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
import string
import requests
from .base import MetricsClient, MetricsSummary

class PrometheusQueryBuilder:
    def __init__(self, metric_name: str, query_op: str, metric_type: str, filter: str, duration: int):
        self.metric_name = metric_name
        self.query_op = query_op
        self.metric_type = metric_type
        self.filter = filter
        self.duration = duration

    def get_queries(self) -> dict:
        """
        Returns a dictionary of queries for each metric type.
        """
        
        return {
        "gauge": {
            "mean": "avg_over_time(%s{%s}[%.0fs])" % (self.metric_name, self.filter, self.duration),
            "Median": "quantile_over_time(0.5, %s{%s}[%.0fs])" % (self.metric_name, self.filter, self.duration),
            "Sd": "stddev_over_time(%s{%s}[%.0fs])" % (self.metric_name, self.filter, self.duration),
            "Min": "min_over_time(%s{%s}[%.0fs])" % (self.metric_name, self.filter, self.duration),
            "Max": "max_over_time(%s{%s}[%.0fs])" % (self.metric_name, self.filter, self.duration),
            "P90": "quantile_over_time(0.9, %s{%s}[%.0fs])" % (self.metric_name, self.filter, self.duration),
            "P95": "quantile_over_time(0.95, %s{%s}[%.0fs])" % (self.metric_name, self.filter, self.duration),
            "P99": "quantile_over_time(0.99, %s{%s}[%.0fs])" % (self.metric_name, self.filter, self.duration),
        },
        "histogram": {
            "mean": "sum(rate(%s_sum{%s}[%.0fs])) / sum(rate(%s_count{%s}[%.0fs]))" % (self.metric_name, self.filter, self.duration, self.metric_name, self.filter, self.duration),
            "Median": "histogram_quantile(0.5, sum(rate(%s_bucket{%s}[%.0fs])) by (le))" % (self.metric_name, self.filter, self.duration),
            "Min": "histogram_quantile(0, sum(rate(%s_bucket{%s}[%.0fs])) by (le))" % (self.metric_name, self.filter, self.duration),
            "Max": "histogram_quantile(1, sum(rate(%s_bucket{%s}[%.0fs])) by (le))" % (self.metric_name, self.filter, self.duration),
            "P90": "histogram_quantile(0.9, sum(rate(%s_bucket{%s}[%.0fs])) by (le))" % (self.metric_name, self.filter, self.duration),
            "P95": "histogram_quantile(0.95, sum(rate(%s_bucket{%s}[%.0fs])) by (le))" % (self.metric_name, self.filter, self.duration),
            "P99": "histogram_quantile(0.99, sum(rate(%s_bucket{%s}[%.0fs])) by (le))" % (self.metric_name, self.filter, self.duration),
        },
        "counter": {
            "Rate": "rate(%s{%s}[%.0fs])" % (self.metric_name, self.filter, self.duration),
            "increase": "increase(%s{%s}[%.0fs])" % (self.metric_name, self.filter, self.duration),
            "mean": "avg_over_time(rate(%s{%s}[%.0fs])[%.0fs:%.0fs])" % (self.metric_name, self.filter, self.duration, self.duration, self.duration),
            "Max": "max_over_time(rate(%s{%s}[%.0fs])[%.0fs:%.0fs])" % (self.metric_name, self.filter, self.duration, self.duration, self.duration),
            "Min": "min_over_time(rate(%s{%s}[%.0fs])[%.0fs:%.0fs])" % (self.metric_name, self.filter, self.duration, self.duration, self.duration),
            "P90": "quantile_over_time(0.9, rate(%s{%s}[%.0fs])[%.0fs:%.0fs])" % (self.metric_name, self.filter, self.duration, self.duration, self.duration),
            "P95": "quantile_over_time(0.5, rate(%s{%s}[%.0fs])[%.0fs:%.0fs])" % (self.metric_name, self.filter, self.duration, self.duration, self.duration),
            "P99": "quantile_over_time(0.99, rate(%s{%s}[%.0fs])[%.0fs:%.0fs])" % (self.metric_name, self.filter, self.duration, self.duration, self.duration),
        },
    }

    def build_query(self) -> string:
        """
        Builds the PromQL query for the given metric type and query operation.

        Returns:
        The PromQL query.
        """
        queries = self.get_queries()
        if self.metric_type not in queries:
            print("Invalid metric type: %s" % (self.metric_type))
            return None
        if self.query_op not in queries[self.metric_type]:
            print("Invalid query operation: %s" % (self.query_op))
            return None
        return queries[self.metric_type][self.query_op]

class PrometheusMetricsClient(MetricsClient):
    def __init__(self, base_url: string) -> None:
        # add scrape_interval
        self.base_url = base_url
    
    def collect_metrics_summary(self, duration, model_server_client) -> MetricsSummary | None:
        """
        Collects the summary metrics for the given duration and engine.

        Args:
        duration: The duration for which to collect metrics. This is equal to the duration for which the perf ran with some buffer for metrics collection.
        engine: The engine for which to collect metrics.
        model: The model for which to collect metrics. This can be used to filter the metrics by model if the engine supports this metric label.

        Returns:
        A MetricsSummary object containing the summary metrics.
        """
        metrics_summary: MetricsSummary = MetricsSummary(
            total_requests=0,
            avg_queue_length=0.0,
            avg_time_to_first_token=0.0,
            avg_time_per_output_token=0.0,
            avg_prompt_tokens=0.0,
            avg_output_tokens=0.0,
            avg_request_latency=0.0,
        )
        # Get the engine and model from the model server client
        if not model_server_client:
            print("Model server client is not set")
            return None
        model = model_server_client.get_model_name()
        engine = model_server_client.get_engine()

        query_builders = self.get_summary_metrics(duration, model)
        # Construct the query for each metric
        for metric in query_builders[engine]:
            metric_name = query_builders[engine][metric].metric_name
            query_op = query_builders[engine][metric].query_op

            query = query_builders[engine][metric].build_query()
            if not query:
                print("No query found for metric: %s, operation: %s" % (metric_name, query_op))
                print("Skipping metric: %s" % (metric_name))
                continue
            # Print the query for debugging purposes
            print("Collecting metric: %s, operation: %s using query: %s" % (metric_name, query_op, query))
            
            # Execute the query and get the result
            result = self.execute_query(query)
            if not result:
                print("No result found for metric: %s, operation: %s using query: %s" % (metric_name, query_op, query))
                print("Skipping metric: %s" % (metric_name))
                continue
        
            setattr(metrics_summary, metric, result)

        return metrics_summary
    
    def get_summary_metrics(self, duration, model = "") -> dict:
        """
        Returns a dictionary of query builders for summary metrics.

        Args:
        duration: The duration for which to collect metrics.
        model: The model is used to filter the metrics by model if the engine supports this metric label.

        Returns:
        A dictionary of query builders for summary metrics.
        """
        return {
            "vllm": {
                "avg_queue_length": PrometheusQueryBuilder("vllm:num_requests_waiting", "mean", "gauge", "model_name='%s'" % model, duration),
                "avg_time_to_first_token": PrometheusQueryBuilder("vllm:time_to_first_token_seconds", "mean", "histogram", "model_name='%s'" % model, duration),
                "avg_time_per_output_token": PrometheusQueryBuilder("vllm:time_per_output_token_seconds", "mean", "histogram", "model_name='%s'" % model, duration),
                "avg_prompt_tokens": PrometheusQueryBuilder("vllm:prompt_tokens_total", "mean", "counter", "model_name='%s'" % model, duration),
                "avg_output_tokens": PrometheusQueryBuilder("vllm:generation_tokens_total", "mean", "counter", "model_name='%s'" % model, duration),
                "total_requests": PrometheusQueryBuilder("vllm:request_success_total", "increase", "counter", "model_name='%s'" % model, duration),
                "avg_request_latency": PrometheusQueryBuilder("vllm:e2e_request_latency_seconds", "mean", "histogram", "model_name='%s'" % model, duration),
            }
        }

    def execute_query(self, query: str) -> float:
        """
        Executes the given query on the Prometheus server and returns the result.

        Args:
        query: the PromQL query to execute

        Returns:
        The result of the query.
        """
        response = requests.get(f"{self.base_url}/api/v1/query", params={"query": query})
        response.raise_for_status()
        response = response.json()
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
        if response.get("status") != "success":
            print("Error executing query: %s" % (response))
            return None

        data = response.get('data', {})
        result = data.get('result', [])
        if result and 'value' in result[0]:
            return result[0]['value'][1]
        return None