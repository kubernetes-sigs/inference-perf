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

from typing import Optional
from pydantic import HttpUrl
from inference_perf.metrics.base import Metric


class PrometheusMetric(Metric):
    name: str
    metric: str
    filter: Optional[str] = ""
    url: Optional[HttpUrl] = None

    def set_target_url(self, url: HttpUrl) -> None:
        self.url = url

    def get_query_set(self, duration: float) -> dict[str, str]:
        raise NotImplementedError


class PrometheusHistogramMetric(PrometheusMetric):
    def get_query_set(self, duration: float) -> dict[str, str]:
        return {
            "mean": "sum(rate(%s_sum{%s}[%.0fs])) / (sum(rate(%s_count{%s}[%.0fs])) > 0)"
            % (self.name, filter, duration, self.name, filter, duration),
            "median": "histogram_quantile(0.5, sum(rate(%s_bucket{%s}[%.0fs])) by (le))" % (self.name, filter, duration),
            "min": "histogram_quantile(0, sum(rate(%s_bucket{%s}[%.0fs])) by (le))" % (self.name, filter, duration),
            "max": "histogram_quantile(1, sum(rate(%s_bucket{%s}[%.0fs])) by (le))" % (self.name, filter, duration),
            "p90": "histogram_quantile(0.9, sum(rate(%s_bucket{%s}[%.0fs])) by (le))" % (self.name, filter, duration),
            "p99": "histogram_quantile(0.99, sum(rate(%s_bucket{%s}[%.0fs])) by (le))" % (self.name, filter, duration),
        }


class PrometheusGaugeMetric(PrometheusMetric):
    def get_query_set(self, duration: float) -> dict[str, str]:
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
    def get_query_set(self, duration: float) -> dict[str, str]:
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
