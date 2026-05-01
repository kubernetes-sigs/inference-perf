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

"""Metrics accumulators.

This package houses the runtime metric collectors (per-request and per-session).
They are pure storage — they accumulate observations during a benchmark run so
reportgen can read them back afterward. Aggregation and summary statistics live
in ``inference_perf.reportgen``, not here.
"""

from .session_collector import SessionMetricsCollector
from .request_collector import (
    RequestMetricCollector,
    LocalRequestMetricCollector,
    MultiprocessRequestMetricCollector,
)

__all__ = [
    "SessionMetricsCollector",
    "RequestMetricCollector",
    "LocalRequestMetricCollector",
    "MultiprocessRequestMetricCollector",
]
