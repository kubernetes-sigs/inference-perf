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

"""Aggregation point for the exported metric sets.

``core.py`` holds the run/stage/request specs exported on every run and
``latency.py`` the per-request histograms (TTFT and TPOT gated on streaming).
Further config-conditional sets should live in sibling modules and be
appended to ``ALL_SPECS`` here as they are added.
"""

from typing import Any, Tuple

from inference_perf.observability.metrics.registry import MetricSpec

from .core import CORE_SPECS
from .latency import LATENCY_SPECS

ALL_SPECS: Tuple[MetricSpec[Any], ...] = (*CORE_SPECS, *LATENCY_SPECS)

__all__ = ["ALL_SPECS", "CORE_SPECS", "LATENCY_SPECS"]
