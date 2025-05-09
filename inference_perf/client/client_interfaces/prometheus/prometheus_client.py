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
from typing import List, Optional
from pydantic import HttpUrl
import requests
from inference_perf.client.client_interfaces.prometheus.prometheus import PrometheusEnabledModelServerClient
from inference_perf.config import PrometheusCollectorConfig
from inference_perf.metrics.base import Metric, MetricCollector

PROMETHEUS_SCRAPE_BUFFER_SEC = 5


class PerfRuntimeParameters:
    def __init__(self, start_time: float, duration: float, model_server_client: PrometheusEnabledModelServerClient) -> None:
        self.start_time = start_time
        self.duration = duration
        self.model_server_client = model_server_client


class PrometheusMetric(Metric):
    name: str
    metric: str
    filter = Optional[str] = ""

    def get_query_set(self, duration: str) -> dict[str, str]:
        raise NotImplementedError

    async def query_summary(self, url: HttpUrl, duration: float) -> dict[str, str]:
        report = {}
        queries = self.get_query_set(duration=str(duration))
        for query_name, query in queries.items():
            report[query_name] = await self.query(url, query)
        return report

    async def query(self, url: HttpUrl, query: str, eval_time: str) -> float:
        """
        Executes the given query on the Prometheus server and returns the result.

        Args:
        query: the PromQL query to execute

        Returns:
        The result of the query.
        """
        query_result = 0.0
        try:
            response = requests.get(f"{url}/api/v1/query", params={"query": query, "time": eval_time})
            if response is None:
                print("Error executing query: %s" % (query))
                return query_result

            response.raise_for_status()
        except Exception as e:
            print("Error executing query: %s" % (e))
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


class PrometheusHistogramMetric(PrometheusMetric):
    def get_query_set(self, duration: str) -> dict[str, str]:
        return {
            "mean": "sum(rate(%s_sum{%s}[%.0fs])) / (sum(rate(%s_count{%s}[%.0fs])) > 0)" % (self.name, filter, duration, self.name, filter, duration),
            "median": "histogram_quantile(0.5, sum(rate(%s_bucket{%s}[%.0fs])) by (le))" % (self.name, filter, duration),
            "min": "histogram_quantile(0, sum(rate(%s_bucket{%s}[%.0fs])) by (le))" % (self.name, filter, duration),
            "max": "histogram_quantile(1, sum(rate(%s_bucket{%s}[%.0fs])) by (le))" % (self.name, filter, duration),
            "p90": "histogram_quantile(0.9, sum(rate(%s_bucket{%s}[%.0fs])) by (le))" % (self.name, filter, duration),
            "p99": "histogram_quantile(0.99, sum(rate(%s_bucket{%s}[%.0fs])) by (le))" % (self.name, filter, duration),
        }


class PrometheusGaugeMetric(PrometheusMetric):
    def get_query_set(self, duration: str) -> dict[str, str]:
        return {
            "mean": "avg_over_time(%s{%s}[%.0fs])" % (self.name, filter, duration),
            "median": "quantile_over_time(0.5, %s{%s}[%.0fs])" % (self.name, filter, duration),
            "sd": "stddev_over_time(%s{%s}[%.0fs])" % (self.name, filter, duration),
            "min": "min_over_time(%s{%s}[%.0fs])" % (self.name, filter, duration),
            "max": "max_over_time(%s{%s}[%.0fs])" % (self.name, filter, duration),
            "p90": "quantile_over_time(0.9, %s{%s}[%.0fs])" % (self.name, filter, duration),
            "p99": "quantile_over_time(0.99, %s{%s}[%.0fs])" % (self.name, filter, duration),
        }


class PrometheusCounterMetric(PrometheusMetric):
    def get_query_set(self, duration: str) -> dict[str, str]:
        return {
            "rate": "sum(rate(%s{%s}[%.0fs]))" % (self.name, filter, duration),
            "increase": "sum(increase(%s{%s}[%.0fs]))" % (self.name, filter, duration),
            "mean": "avg_over_time(rate(%s{%s}[%.0fs])[%.0fs:%.0fs])" % (self.name, filter, duration, duration, duration),
            "max": "max_over_time(rate(%s{%s}[%.0fs])[%.0fs:%.0fs])" % (self.name, filter, duration, duration, duration),
            "min": "min_over_time(rate(%s{%s}[%.0fs])[%.0fs:%.0fs])" % (self.name, filter, duration, duration, duration),
            "p90": "quantile_over_time(0.9, rate(%s{%s}[%.0fs])[%.0fs:%.0fs])"
            % (self.name, filter, duration, duration, duration),
            "p99": "quantile_over_time(0.99, rate(%s{%s}[%.0fs])[%.0fs:%.0fs])"
            % (self.name, filter, duration, duration, duration),
        }


class PrometheusMetricsCollector(MetricCollector[PrometheusMetric]):
    def __init__(self, config: PrometheusCollectorConfig, metrics: List[PrometheusMetric]) -> None:
        self.metrics = metrics
        self.config = config

    async def get_report(self, duration: float) -> dict[str, float]:
        total_report = {}
        for metric in self.metrics:
            total_report[metric.name] = await metric.query_summary(url=self.config.url, duration=duration)
        return total_report
