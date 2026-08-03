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

# Public surface for the BR0.2.1 pydantic models.
#
# The models themselves are vendored from llm-d/llm-d-benchmark in
# `base.py`, `schema_v0_2.py`, `schema_v0_2_components.py`, and
# `schema_v0_2_1.py` (see headers in those files for the upstream commit SHA).
# v0.2.1 extends v0.2 in place, so class names overridden there (the request
# aggregates and the report root's containment chain) are re-exported from
# `schema_v0_2_1` and everything else from `schema_v0_2`. Import from this
# module rather than the vendored files directly so a future schema bump only
# touches the vendored files.
from .base import (
    UNITS_BANDWIDTH,
    UNITS_GEN_LATENCY,
    UNITS_GEN_THROUGHPUT,
    UNITS_MEDIA_THROUGHPUT,
    UNITS_MEMORY,
    UNITS_PORTION,
    UNITS_POWER,
    UNITS_QUANTITY,
    UNITS_RATIO,
    UNITS_REQUEST_THROUGHPUT,
    UNITS_TIME,
    BenchmarkReport,
    Units,
    WorkloadGenerator,
)
from .schema_v0_2 import (
    AggregateLatency,
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
    ResourceMetrics,
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
from .schema_v0_2_1 import (
    VERSION,
    AggregateRequestPerformance,
    AggregateRequests,
    AggregateThroughput,
    AudioPayloadStats,
    BenchmarkReportV021,
    ImagePayloadStats,
    MediaPayloadStats,
    MultiModalRequests,
    RequestPerformance,
    Results,
    VideoPayloadStats,
    VisualPayloadStats,
)
from .schema_v0_2_components import COMPONENTS

__all__ = [
    "AggregateLatency",
    "AggregateRequestPerformance",
    "AggregateRequests",
    "AggregateThroughput",
    "AudioPayloadStats",
    "BenchmarkReport",
    "BenchmarkReportV021",
    "COMPONENTS",
    "Component",
    "ComponentHealth",
    "ComponentMetadata",
    "ComponentNative",
    "ComponentObservability",
    "ControllerReplicaStatus",
    "Distribution",
    "ImagePayloadStats",
    "Load",
    "LoadMetadata",
    "LoadNative",
    "LoadPrefix",
    "LoadSource",
    "LoadStandardized",
    "MediaPayloadStats",
    "MultiModalRequests",
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
    "UNITS_BANDWIDTH",
    "UNITS_GEN_LATENCY",
    "UNITS_GEN_THROUGHPUT",
    "UNITS_MEDIA_THROUGHPUT",
    "UNITS_MEMORY",
    "UNITS_PORTION",
    "UNITS_POWER",
    "UNITS_QUANTITY",
    "UNITS_RATIO",
    "UNITS_REQUEST_THROUGHPUT",
    "UNITS_TIME",
    "Units",
    "VERSION",
    "VideoPayloadStats",
    "VisualPayloadStats",
    "WorkloadGenerator",
]
