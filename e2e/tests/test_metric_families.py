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
"""Type rules ``resolves`` applies to histogram-derived series (#669)."""

from typing import Any, Dict

import pytest
from utils.metric_families import resolves

from inference_perf.client.modelserver.metrics import CounterMetric, GaugeMetric, HistogramMetric
from inference_perf.client.modelserver.metrics.base import Metric


# HistogramMetric("x") over {"x": "histogram"} resolves. CounterMetric("x_count") resolves over a
# histogram or a summary family, which is how SGLang's request count is declared.
@pytest.mark.parametrize(
    "metric, families",
    [
        (HistogramMetric("x"), {"x": "histogram"}),
        (CounterMetric("x_count"), {"x": "histogram"}),
        (CounterMetric("x_sum"), {"x": "summary"}),
        (CounterMetric("x_bucket"), {"x": "histogram"}),
    ],
    ids=[
        "histogram-over-histogram",
        "counter-count-over-histogram",
        "counter-sum-over-summary",
        "counter-bucket-over-histogram",
    ],
)
def test_derived_series_resolve_from_a_matching_family(metric: Metric[Any], families: Dict[str, str]) -> None:
    assert resolves(metric, families)


# HistogramMetric("x") over {"x": "summary"} does not resolve: a summary has no x_bucket series, so
# histogram_quantile would return nothing. A gauge over a cumulative _count series does not resolve.
@pytest.mark.parametrize(
    "metric, families",
    [
        (HistogramMetric("x"), {"x": "summary"}),
        (CounterMetric("x_bucket"), {"x": "summary"}),
        (GaugeMetric("x_count"), {"x": "histogram"}),
        (GaugeMetric("x_sum"), {"x": "summary"}),
    ],
    ids=["histogram-over-summary", "counter-bucket-over-summary", "gauge-count-over-histogram", "gauge-sum-over-summary"],
)
def test_derived_series_do_not_resolve_from_a_mismatched_family(metric: Metric[Any], families: Dict[str, str]) -> None:
    assert not resolves(metric, families)
