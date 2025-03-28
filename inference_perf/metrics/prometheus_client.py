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

class PrometheusMetricsClient(MetricsClient):
    def __init__(self, base_url: string) -> None:
        self.base_url = base_url
    
    def collect_metrics_summary(self, job: string, engine: string, duration_in_seconds: float) -> MetricsSummary | None:
        """
        Collects metrics summary from the Prometheus server.
        
        Returns:
            MetricsSummary: An object containing the collected metrics summary.
        """
        metrics = self.get_metrics_to_scrape_for_engine(engine)

        # find metric types for each metric in the dictionary from Prometheus
        all_metrics_metadata = self.get_metrics_metadata()
        if not all_metrics_metadata:
            print("Failed to fetch metrics metadata")
            return None
       
        # construct the query for each metric
        for metric in metrics[engine]:
            metrics_result = []
            queries = self.get_queries(metric, job, duration_in_seconds)

            metric_type = all_metrics_metadata['data'][metric]
            if all_metrics_metadata['data'][metric] is None:
                print("No metric found for: %s" % metric)
                return
            metric_type = metric_type[0]['type']

            for op, query in queries[metric_type].items():
                query = query % (metric, job, 60)
                print("Executing query: %s" % query)

                result = self.execute_query(query)
                if not result:
                    return
            
                # Append the result to the metrics_result list
                metrics_result.append((op, result))
        
            metrics_summary[metric] = metrics_result

        metrics_summary = MetricsSummary(result)
        return metrics_summary

    def get_metrics_metadata(self) -> dict:
        """
        Fetches the metadata for all metrics from the Prometheus server.

        Returns:
            dict: A dictionary containing the metadata for each metric.
        """

        # find metric types for each metric in the dictionary
        response = requests.post(f"{self.base_url}/api/v1/metadata")
        all_metrics_metadata = response.json()

        if response.ok is not True:
            print("HTTP Error: %s" % (all_metrics_metadata))
            return
        if all_metrics_metadata["status"] != "success":
            print("Metadata error response: %s" % all_metrics_metadata["error"])
            return
        return all_metrics_metadata


    def execute_query(self, query: str) -> float:
        response = requests.get(f"{self.base_url}/api/v1/query", params={"query": query})
        response.raise_for_status()
        response = response.json()

        if response.get("status") != "success":
            print("Error executing query: %s" % (response))
            return None
        # Check if the response contains data and result
        # If the query returns no data, return None
        if 'data' in response and 'result' in response['data']:
            return float(response['data']['result'][0]['value']) if response['data']['result'] else None
        return None

    def get_metrics_to_scrape_for_engine(self, engine) -> dict:
        metrics_to_scrape = {
            "vllm": {
                "avg_queue_length": "vllm:num_requests_waiting",
                "avg_time_to_first_token": "vllm:time_to_first_token_seconds",
                "avg_time_per_output_token": "vllm:time_per_output_token_seconds",
                "avg_prompt_tokens": "vllm:prompt_tokens_total",
                "avg_output_tokens": "vllm:generation_tokens_total",
                "total_requests": "vllm:request_success_total",
                "avg_time_per_request": "vllm:e2e_request_latency_seconds",
            }
        }
        return metrics_to_scrape[engine]

    def get_queries(self, metric, job, duration) -> dict:
        return {
        "gauge": {
            "Mean": "avg_over_time(%s{job='%s'}[%.0fs])" % (metric, job, duration),
            "Median": "quantile_over_time(0.5, %s{job='%s'}[%.0fs])" % (metric, job, duration),
            "Sd": "stddev_over_time(%s{job='%s'}[%.0fs])" % (metric, job, duration),
            "Min": "min_over_time(%s{job='%s'}[%.0fs])" % (metric, job, duration),
            "Max": "max_over_time(%s{job='%s'}[%.0fs])" % (metric, job, duration),
            "P90": "quantile_over_time(0.9, %s{job='%s'}[%.0fs])" % (metric, job, duration),
            "P95": "quantile_over_time(0.95, %s{job='%s'}[%.0fs])" % (metric, job, duration),
            "P99": "quantile_over_time(0.99, %s{job='%s'}[%.0fs])" % (metric, job, duration),
        },
        "histogram": {
            "Mean": "sum(rate(%s_sum{job='%s'}[%.0fs])) / sum(rate(%s_count{job='%s'}[%.0fs]))" % (metric, job, duration, metric, job, duration),
            "Median": "histogram_quantile(0.5, sum(rate(%s_bucket{job='%s'}[%.0fs])) by (le))" % (metric, job, duration),
            "Min": "histogram_quantile(0, sum(rate(%s_bucket{job='%s'}[%.0fs])) by (le))" % (metric, job, duration),
            "Max": "histogram_quantile(1, sum(rate(%s_bucket{job='%s'}[%.0fs])) by (le))" % (metric, job, duration),
            "P90": "histogram_quantile(0.9, sum(rate(%s_bucket{job='%s'}[%.0fs])) by (le))" % (metric, job, duration),
            "P95": "histogram_quantile(0.95, sum(rate(%s_bucket{job='%s'}[%.0fs])) by (le))" % (metric, job, duration),
            "P99": "histogram_quantile(0.99, sum(rate(%s_bucket{job='%s'}[%.0fs])) by (le))" % (metric, job, duration),
        },
        "counter": {
            "Rate": "rate(%s{job='%s'}[%.0fs])" % (metric, job, duration),
            "Increase": "increase(%s{job='%s'}[%.0fs])" % (metric, job, duration),
            "Mean": "avg_over_time(rate(%s{job='%s'}[%.0fs])[%.0fs:%.0fs])" % (metric, job, duration, duration, duration),
            "Max": "max_over_time(rate(%s{job='%s'}[%.0fs])[%.0fs:%.0fs])" % (metric, job, duration, duration, duration),
            "Min": "min_over_time(rate(%s{job='%s'}[%.0fs])[%.0fs:%.0fs])" % (metric, job, duration, duration, duration),
            "P90": "quantile_over_time(0.9, rate(%s{job='%s'}[%.0fs])[%.0fs:%.0fs])" % (metric, job, duration, duration, duration),
            "P95": "quantile_over_time(0.5, rate(%s{job='%s'}[%.0fs])[%.0fs:%.0fs])" % (metric, job, duration, duration, duration),
            "P99": "quantile_over_time(0.99, rate(%s{job='%s'}[%.0fs])[%.0fs:%.0fs])" % (metric, job, duration, duration, duration),
        },
    }