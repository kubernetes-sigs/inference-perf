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

# Public surface for the BR0.2 pydantic models.
#
# The models themselves are vendored from llm-d/llm-d-benchmark in
# `base.py`, `schema_v0_2.py`, and `schema_v0_2_components.py` (see headers in
# those files for the upstream commit SHA). Import from this module rather than
# the vendored files directly so a future schema bump only touches the vendored
# files.
from .base import (
    UNITS_GEN_LATENCY,
    UNITS_GEN_THROUGHPUT,
    UNITS_MEMORY,
    UNITS_PORTION,
    UNITS_POWER,
    UNITS_QUANTITY,
    UNITS_REQUEST_THROUGHPUT,
    UNITS_TIME,
    BenchmarkReport,
    Units,
    WorkloadGenerator,
)
from .schema_v0_2 import (
    VERSION,
    AggregateLatency,
    AggregateRequestPerformance,
    AggregateRequests,
    AggregateThroughput,
    BenchmarkReportV02,
    Component,
    ComponentHealth,
    ComponentMetadata,
    ComponentNative,
    ComponentObservability,
    ControllerReplicaStatus,
    Distribution,
    Load,
    LoadMetadata,
    LoadNative,
    LoadPrefix,
    LoadSource,
    LoadStandardized,
    MultiTurn,
    Observability,
    PodStartupInfo,
    PodStartupTimes,
    ReplicaHealth,
    ReplicaStatus,
    ReplicaStatusSnapshot,
    RequestPerformance,
    ResourceMetrics,
    Results,
    Run,
    RunTime,
    Scenario,
    SequenceLength,
    Statistics,
    TimeSeriesData,
    TimeSeriesLatency,
    TimeSeriesPoint,
    TimeSeriesRequestPerformance,
    TimeSeriesResourceMetrics,
    TimeSeriesThroughput,
)
from .schema_v0_2_components import COMPONENTS

__all__ = [
    "AggregateLatency",
    "AggregateRequestPerformance",
    "AggregateRequests",
    "AggregateThroughput",
    "BenchmarkReport",
    "BenchmarkReportV02",
    "COMPONENTS",
    "Component",
    "ComponentHealth",
    "ComponentMetadata",
    "ComponentNative",
    "ComponentObservability",
    "ControllerReplicaStatus",
    "Distribution",
    "Load",
    "LoadMetadata",
    "LoadNative",
    "LoadPrefix",
    "LoadSource",
    "LoadStandardized",
    "MultiTurn",
    "Observability",
    "PodStartupInfo",
    "PodStartupTimes",
    "ReplicaHealth",
    "ReplicaStatus",
    "ReplicaStatusSnapshot",
    "RequestPerformance",
    "ResourceMetrics",
    "Results",
    "Run",
    "RunTime",
    "Scenario",
    "SequenceLength",
    "Statistics",
    "TimeSeriesData",
    "TimeSeriesLatency",
    "TimeSeriesPoint",
    "TimeSeriesRequestPerformance",
    "TimeSeriesResourceMetrics",
    "TimeSeriesThroughput",
    "UNITS_GEN_LATENCY",
    "UNITS_GEN_THROUGHPUT",
    "UNITS_MEMORY",
    "UNITS_PORTION",
    "UNITS_POWER",
    "UNITS_QUANTITY",
    "UNITS_REQUEST_THROUGHPUT",
    "UNITS_TIME",
    "Units",
    "VERSION",
    "WorkloadGenerator",
]
