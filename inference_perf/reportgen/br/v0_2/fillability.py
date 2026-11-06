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
"""Which BR0.2 fields a base report may set, and which ones inference-perf owns.

Every leaf of the BR0.2 schema has an entry in ``FILLABILITY``. A
``Statistics`` or ``TimeSeriesData`` block counts as one leaf, since
inference-perf writes each one whole. A test fails when the installed schema
has a leaf the table does not, so a schema bump forces a decision per new
field. A second test holds every leaf that has ever been WRITTEN or RESERVED
to stay that way: a field consumers have read as measured by inference-perf
never becomes one a base report can set.
"""

from __future__ import annotations

import types
import typing
from enum import Enum
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Type

from pydantic import BaseModel

from .schema import BenchmarkReportV021, Statistics, TimeSeriesData


class Fillability(str, Enum):
    # A base report may set it. inference-perf never writes it.
    FILLABLE = "fillable"
    # A base report may set it. inference-perf writes it when the base does not.
    DEFAULTED = "defaulted"
    # inference-perf writes it. A base report may not set it.
    WRITTEN = "written"
    # A measurement inference-perf owns but does not write today, either not yet
    # or not anymore. A base report may not set it.
    RESERVED = "reserved"

    @property
    def base_may_set(self) -> bool:
        return self in (Fillability.FILLABLE, Fillability.DEFAULTED)


_LEAF_MODELS = (Statistics, TimeSeriesData)

FILLABILITY: Dict[str, Fillability] = {
    "results.component_health.component_label": Fillability.FILLABLE,
    "results.component_health.failed_replicas": Fillability.FILLABLE,
    "results.component_health.replica_health.healthy": Fillability.FILLABLE,
    "results.component_health.replica_health.logs": Fillability.FILLABLE,
    "results.component_health.replica_health.replica_id": Fillability.FILLABLE,
    "results.component_health.replica_health.restarts": Fillability.FILLABLE,
    "results.component_health.total_restarts": Fillability.FILLABLE,
    "results.observability.components.aggregate.cache_hit_rate": Fillability.FILLABLE,
    "results.observability.components.aggregate.cpu_cache_usage": Fillability.FILLABLE,
    "results.observability.components.aggregate.cpu_memory_usage": Fillability.FILLABLE,
    "results.observability.components.aggregate.cpu_utilization": Fillability.FILLABLE,
    "results.observability.components.aggregate.gpu_cache_usage": Fillability.FILLABLE,
    "results.observability.components.aggregate.gpu_memory_usage": Fillability.FILLABLE,
    "results.observability.components.aggregate.gpu_utilization": Fillability.FILLABLE,
    "results.observability.components.aggregate.kv_cache_usage": Fillability.FILLABLE,
    "results.observability.components.aggregate.power_consumption": Fillability.FILLABLE,
    "results.observability.components.aggregate.preemptions": Fillability.FILLABLE,
    "results.observability.components.aggregate.running_requests": Fillability.FILLABLE,
    "results.observability.components.aggregate.storage_usage": Fillability.FILLABLE,
    "results.observability.components.aggregate.swapped_requests": Fillability.FILLABLE,
    "results.observability.components.aggregate.waiting_requests": Fillability.FILLABLE,
    "results.observability.components.component_label": Fillability.FILLABLE,
    "results.observability.components.graph_path": Fillability.FILLABLE,
    "results.observability.components.raw_data_path": Fillability.FILLABLE,
    "results.observability.components.replica_id": Fillability.FILLABLE,
    "results.observability.components.time_series.cpu_cache_usage": Fillability.FILLABLE,
    "results.observability.components.time_series.cpu_memory_usage": Fillability.FILLABLE,
    "results.observability.components.time_series.cpu_utilization": Fillability.FILLABLE,
    "results.observability.components.time_series.external_prefix_cache_hit_rate": Fillability.FILLABLE,
    "results.observability.components.time_series.external_prefix_cache_hits": Fillability.FILLABLE,
    "results.observability.components.time_series.external_prefix_cache_queries": Fillability.FILLABLE,
    "results.observability.components.time_series.generation_tokens": Fillability.FILLABLE,
    "results.observability.components.time_series.gpu_cache_usage": Fillability.FILLABLE,
    "results.observability.components.time_series.gpu_memory_usage": Fillability.FILLABLE,
    "results.observability.components.time_series.gpu_utilization": Fillability.FILLABLE,
    "results.observability.components.time_series.kv_cache_usage": Fillability.FILLABLE,
    "results.observability.components.time_series.kv_offload_load_bytes": Fillability.FILLABLE,
    "results.observability.components.time_series.kv_offload_load_time": Fillability.FILLABLE,
    "results.observability.components.time_series.kv_offload_store_bytes": Fillability.FILLABLE,
    "results.observability.components.time_series.kv_offload_store_time": Fillability.FILLABLE,
    "results.observability.components.time_series.num_preemptions": Fillability.FILLABLE,
    "results.observability.components.time_series.num_requests_running": Fillability.FILLABLE,
    "results.observability.components.time_series.num_requests_waiting": Fillability.FILLABLE,
    "results.observability.components.time_series.pool_avg_kv_cache_utilization": Fillability.FILLABLE,
    "results.observability.components.time_series.pool_avg_queue_size": Fillability.FILLABLE,
    "results.observability.components.time_series.pool_avg_running_requests": Fillability.FILLABLE,
    "results.observability.components.time_series.pool_ready_pods": Fillability.FILLABLE,
    "results.observability.components.time_series.power_consumption": Fillability.FILLABLE,
    "results.observability.components.time_series.prefix_cache_hit_rate": Fillability.FILLABLE,
    "results.observability.components.time_series.prefix_cache_hits": Fillability.FILLABLE,
    "results.observability.components.time_series.prefix_cache_queries": Fillability.FILLABLE,
    "results.observability.components.time_series.prompt_tokens": Fillability.FILLABLE,
    "results.observability.components.time_series.storage_usage": Fillability.FILLABLE,
    # The stage's dropped_requests as a rate (#799).
    "results.observability.drop_rate": Fillability.RESERVED,
    "results.observability.pod_startup_times.aggregate": Fillability.FILLABLE,
    "results.observability.pod_startup_times.collected_at": Fillability.FILLABLE,
    "results.observability.pod_startup_times.graph_path": Fillability.FILLABLE,
    "results.observability.pod_startup_times.pods.creation_timestamp": Fillability.FILLABLE,
    "results.observability.pod_startup_times.pods.model": Fillability.FILLABLE,
    "results.observability.pod_startup_times.pods.name": Fillability.FILLABLE,
    "results.observability.pod_startup_times.pods.node": Fillability.FILLABLE,
    "results.observability.pod_startup_times.pods.ready_timestamp": Fillability.FILLABLE,
    "results.observability.pod_startup_times.pods.role": Fillability.FILLABLE,
    "results.observability.pod_startup_times.pods.startup_seconds": Fillability.FILLABLE,
    "results.observability.replica_status.aggregate_ready_replicas": Fillability.FILLABLE,
    "results.observability.replica_status.controllers.available_replicas": Fillability.FILLABLE,
    "results.observability.replica_status.controllers.desired_replicas": Fillability.FILLABLE,
    "results.observability.replica_status.controllers.kind": Fillability.FILLABLE,
    "results.observability.replica_status.controllers.model": Fillability.FILLABLE,
    "results.observability.replica_status.controllers.name": Fillability.FILLABLE,
    "results.observability.replica_status.controllers.ready_replicas": Fillability.FILLABLE,
    "results.observability.replica_status.controllers.role": Fillability.FILLABLE,
    "results.observability.replica_status.controllers.updated_replicas": Fillability.FILLABLE,
    "results.observability.replica_status.graph_path": Fillability.FILLABLE,
    "results.observability.replica_status.namespace": Fillability.FILLABLE,
    "results.observability.replica_status.time_series.controllers.available_replicas": Fillability.FILLABLE,
    "results.observability.replica_status.time_series.controllers.desired_replicas": Fillability.FILLABLE,
    "results.observability.replica_status.time_series.controllers.kind": Fillability.FILLABLE,
    "results.observability.replica_status.time_series.controllers.model": Fillability.FILLABLE,
    "results.observability.replica_status.time_series.controllers.name": Fillability.FILLABLE,
    "results.observability.replica_status.time_series.controllers.ready_replicas": Fillability.FILLABLE,
    "results.observability.replica_status.time_series.controllers.role": Fillability.FILLABLE,
    "results.observability.replica_status.time_series.controllers.updated_replicas": Fillability.FILLABLE,
    "results.observability.replica_status.time_series.namespace": Fillability.FILLABLE,
    "results.observability.replica_status.time_series.timestamp": Fillability.FILLABLE,
    "results.observability.replica_status.timestamp": Fillability.FILLABLE,
    "results.profiling": Fillability.FILLABLE,
    "results.request_performance.aggregate.latency.inter_token_latency": Fillability.WRITTEN,
    "results.request_performance.aggregate.latency.normalized_time_per_output_token": Fillability.WRITTEN,
    "results.request_performance.aggregate.latency.request_latency": Fillability.WRITTEN,
    "results.request_performance.aggregate.latency.time_per_output_token": Fillability.WRITTEN,
    "results.request_performance.aggregate.latency.time_to_first_token": Fillability.WRITTEN,
    "results.request_performance.aggregate.requests.failures": Fillability.WRITTEN,
    # Requests cancelled at stage teardown (#858).
    "results.request_performance.aggregate.requests.incomplete": Fillability.RESERVED,
    "results.request_performance.aggregate.requests.input_length": Fillability.WRITTEN,
    "results.request_performance.aggregate.requests.multimodal.audio.count": Fillability.WRITTEN,
    "results.request_performance.aggregate.requests.multimodal.audio.duration": Fillability.WRITTEN,
    "results.request_performance.aggregate.requests.multimodal.audio.filesize": Fillability.WRITTEN,
    "results.request_performance.aggregate.requests.multimodal.image.aspect_ratio": Fillability.WRITTEN,
    "results.request_performance.aggregate.requests.multimodal.image.count": Fillability.WRITTEN,
    "results.request_performance.aggregate.requests.multimodal.image.filesize": Fillability.WRITTEN,
    "results.request_performance.aggregate.requests.multimodal.image.pixels": Fillability.WRITTEN,
    "results.request_performance.aggregate.requests.multimodal.video.aspect_ratio": Fillability.WRITTEN,
    "results.request_performance.aggregate.requests.multimodal.video.count": Fillability.WRITTEN,
    "results.request_performance.aggregate.requests.multimodal.video.filesize": Fillability.WRITTEN,
    "results.request_performance.aggregate.requests.multimodal.video.frames": Fillability.WRITTEN,
    "results.request_performance.aggregate.requests.multimodal.video.pixels": Fillability.WRITTEN,
    "results.request_performance.aggregate.requests.output_length": Fillability.WRITTEN,
    "results.request_performance.aggregate.requests.request_size": Fillability.WRITTEN,
    "results.request_performance.aggregate.requests.total": Fillability.WRITTEN,
    "results.request_performance.aggregate.throughput.audio_rate": Fillability.WRITTEN,
    "results.request_performance.aggregate.throughput.image_rate": Fillability.WRITTEN,
    "results.request_performance.aggregate.throughput.input_token_rate": Fillability.WRITTEN,
    "results.request_performance.aggregate.throughput.output_token_rate": Fillability.WRITTEN,
    "results.request_performance.aggregate.throughput.request_rate": Fillability.WRITTEN,
    "results.request_performance.aggregate.throughput.total_token_rate": Fillability.WRITTEN,
    "results.request_performance.aggregate.throughput.video_rate": Fillability.WRITTEN,
    # Per-interval request metrics (#810).
    "results.request_performance.time_series.latency.inter_token_latency": Fillability.RESERVED,
    "results.request_performance.time_series.latency.normalized_time_per_output_token": Fillability.RESERVED,
    "results.request_performance.time_series.latency.request_latency": Fillability.RESERVED,
    "results.request_performance.time_series.latency.time_per_output_token": Fillability.RESERVED,
    "results.request_performance.time_series.latency.time_to_first_token": Fillability.RESERVED,
    "results.request_performance.time_series.throughput.input_token_rate": Fillability.RESERVED,
    "results.request_performance.time_series.throughput.output_token_rate": Fillability.RESERVED,
    "results.request_performance.time_series.throughput.request_rate": Fillability.RESERVED,
    "results.request_performance.time_series.throughput.total_token_rate": Fillability.RESERVED,
    "results.request_performance.time_series.throughput.units": Fillability.RESERVED,
    "results.session_performance.sessions.events_cancelled_per_session": Fillability.WRITTEN,
    "results.session_performance.sessions.events_per_session": Fillability.WRITTEN,
    "results.session_performance.sessions.failed": Fillability.WRITTEN,
    "results.session_performance.sessions.input_tokens_per_session": Fillability.WRITTEN,
    "results.session_performance.sessions.output_tokens_per_session": Fillability.WRITTEN,
    "results.session_performance.sessions.session_duration": Fillability.WRITTEN,
    "results.session_performance.sessions.session_rate": Fillability.WRITTEN,
    "results.session_performance.sessions.succeeded": Fillability.WRITTEN,
    "results.session_performance.sessions.total": Fillability.WRITTEN,
    "results.session_performance.sessions.total_events": Fillability.WRITTEN,
    "results.session_performance.sessions.total_events_cancelled": Fillability.WRITTEN,
    "results.session_performance.sessions.total_events_completed": Fillability.WRITTEN,
    "run.cid": Fillability.FILLABLE,
    "run.description": Fillability.FILLABLE,
    # A sweep spread over several invocations sets one eid for all of them.
    "run.eid": Fillability.DEFAULTED,
    "run.keywords": Fillability.FILLABLE,
    "run.pid": Fillability.FILLABLE,
    "run.time.duration": Fillability.WRITTEN,
    "run.time.end": Fillability.WRITTEN,
    "run.time.start": Fillability.WRITTEN,
    "run.uid": Fillability.WRITTEN,
    "run.user": Fillability.FILLABLE,
    "scenario.load.metadata.cfg_id": Fillability.FILLABLE,
    "scenario.load.metadata.concurrent_with": Fillability.FILLABLE,
    "scenario.load.metadata.description": Fillability.FILLABLE,
    "scenario.load.metadata.schema_version": Fillability.FILLABLE,
    "scenario.load.metadata.treatment": Fillability.FILLABLE,
    "scenario.load.metadata.treatment_group": Fillability.FILLABLE,
    "scenario.load.native.args": Fillability.FILLABLE,
    "scenario.load.native.config": Fillability.FILLABLE,
    "scenario.load.native.envars": Fillability.FILLABLE,
    "scenario.load.standardized.concurrency": Fillability.FILLABLE,
    "scenario.load.standardized.input_seq_len.distribution": Fillability.FILLABLE,
    "scenario.load.standardized.input_seq_len.max": Fillability.FILLABLE,
    "scenario.load.standardized.input_seq_len.min": Fillability.FILLABLE,
    "scenario.load.standardized.input_seq_len.std_dev": Fillability.FILLABLE,
    "scenario.load.standardized.input_seq_len.value": Fillability.FILLABLE,
    "scenario.load.standardized.multi_turn.enabled": Fillability.FILLABLE,
    "scenario.load.standardized.multi_turn.max_turns.distribution": Fillability.FILLABLE,
    "scenario.load.standardized.multi_turn.max_turns.max": Fillability.FILLABLE,
    "scenario.load.standardized.multi_turn.max_turns.min": Fillability.FILLABLE,
    "scenario.load.standardized.multi_turn.max_turns.std_dev": Fillability.FILLABLE,
    "scenario.load.standardized.multi_turn.max_turns.value": Fillability.FILLABLE,
    "scenario.load.standardized.output_seq_len.distribution": Fillability.FILLABLE,
    "scenario.load.standardized.output_seq_len.max": Fillability.FILLABLE,
    "scenario.load.standardized.output_seq_len.min": Fillability.FILLABLE,
    "scenario.load.standardized.output_seq_len.std_dev": Fillability.FILLABLE,
    "scenario.load.standardized.output_seq_len.value": Fillability.FILLABLE,
    "scenario.load.standardized.parallelism": Fillability.FILLABLE,
    "scenario.load.standardized.prefix.num_groups": Fillability.FILLABLE,
    "scenario.load.standardized.prefix.num_prefixes": Fillability.FILLABLE,
    "scenario.load.standardized.prefix.num_users_per_group": Fillability.FILLABLE,
    "scenario.load.standardized.prefix.prefix_len.distribution": Fillability.FILLABLE,
    "scenario.load.standardized.prefix.prefix_len.max": Fillability.FILLABLE,
    "scenario.load.standardized.prefix.prefix_len.min": Fillability.FILLABLE,
    "scenario.load.standardized.prefix.prefix_len.std_dev": Fillability.FILLABLE,
    "scenario.load.standardized.prefix.prefix_len.value": Fillability.FILLABLE,
    "scenario.load.standardized.rate_qps": Fillability.FILLABLE,
    "scenario.load.standardized.source": Fillability.FILLABLE,
    "scenario.load.standardized.stage": Fillability.FILLABLE,
    "scenario.load.standardized.tool": Fillability.FILLABLE,
    "scenario.load.standardized.tool_version": Fillability.FILLABLE,
    "scenario.stack.metadata.cfg_id": Fillability.FILLABLE,
    "scenario.stack.metadata.description": Fillability.FILLABLE,
    "scenario.stack.metadata.label": Fillability.FILLABLE,
    "scenario.stack.metadata.schema_version": Fillability.FILLABLE,
    "scenario.stack.native.args": Fillability.FILLABLE,
    "scenario.stack.native.config": Fillability.FILLABLE,
    "scenario.stack.native.envars": Fillability.FILLABLE,
    "scenario.stack.standardized.accelerator.count": Fillability.FILLABLE,
    "scenario.stack.standardized.accelerator.model": Fillability.FILLABLE,
    "scenario.stack.standardized.accelerator.parallelism.dp": Fillability.FILLABLE,
    "scenario.stack.standardized.accelerator.parallelism.dp_local": Fillability.FILLABLE,
    "scenario.stack.standardized.accelerator.parallelism.ep": Fillability.FILLABLE,
    "scenario.stack.standardized.accelerator.parallelism.pp": Fillability.FILLABLE,
    "scenario.stack.standardized.accelerator.parallelism.tp": Fillability.FILLABLE,
    "scenario.stack.standardized.accelerator.parallelism.workers": Fillability.FILLABLE,
    "scenario.stack.standardized.kind": Fillability.FILLABLE,
    "scenario.stack.standardized.model.name": Fillability.FILLABLE,
    "scenario.stack.standardized.replicas": Fillability.FILLABLE,
    "scenario.stack.standardized.role": Fillability.FILLABLE,
    "scenario.stack.standardized.tool": Fillability.FILLABLE,
    "scenario.stack.standardized.tool_version": Fillability.FILLABLE,
    "version": Fillability.WRITTEN,
}


def fillability(path: str) -> Fillability:
    """The class of a dotted leaf path such as ``run.description``."""
    # A leaf from a newer 0.2.x schema than this table knows is one
    # inference-perf has no code to write, so a base report may set it.
    return FILLABILITY.get(path, Fillability.FILLABLE)


def schema_leaves() -> FrozenSet[str]:
    """Every leaf path of the installed BR0.2 schema, across all union arms."""
    leaves: Set[str] = set()

    def walk(models: Sequence[Type[BaseModel]], prefix: str) -> None:
        for key, children in _children(models).items():
            if children is None:
                leaves.add(prefix + key)
            else:
                walk(children, f"{prefix}{key}.")

    walk([BenchmarkReportV021], "")
    return frozenset(leaves)


def document_leaves(document: Mapping[str, object]) -> List[str]:
    """The leaf paths a BR0.2 document sets, in document order.

    Raises ValueError naming every key the schema does not have. Keys under a
    model that takes extra fields (``results.observability``) count as leaves.
    """
    leaves: List[str] = []
    unknown: List[str] = []

    def walk(node: Mapping[str, object], models: Sequence[Type[BaseModel]], prefix: str) -> None:
        fields = _children(models)
        for key, value in node.items():
            path = f"{prefix}{key}"
            if key not in fields:
                (leaves if _takes_extra(models) else unknown).append(path)
                continue
            children = fields[key]
            if children is None:
                leaves.append(path)
                continue
            for item in value if isinstance(value, list) else [value]:
                if isinstance(item, Mapping):
                    walk(item, children, f"{path}.")
                else:
                    # Wrong shape for a model field. Schema validation reports it.
                    leaves.append(path)

    walk(document, [BenchmarkReportV021], "")
    if unknown:
        raise ValueError(f"not BR0.2 fields: {', '.join(unknown)}")
    return list(dict.fromkeys(leaves))


# Field key -> the models it can hold, merged across union arms. None marks a leaf.
def _children(models: Sequence[Type[BaseModel]]) -> Dict[str, Optional[List[Type[BaseModel]]]]:
    children: Dict[str, Optional[List[Type[BaseModel]]]] = {}
    for model in models:
        for name, field in model.model_fields.items():
            key = field.alias or name
            arms = _model_arms(field.annotation)
            known = children.get(key, [])
            if known is None or not arms or any(arm in _LEAF_MODELS for arm in arms):
                children[key] = None
            else:
                children[key] = list(dict.fromkeys([*known, *arms]))
    return children


def _model_arms(annotation: object) -> List[Type[BaseModel]]:
    return [arm for arm in _arms(annotation) if isinstance(arm, type) and issubclass(arm, BaseModel)]


# Unwraps Optional, unions, lists and Annotated down to the member types.
def _arms(annotation: object) -> List[object]:
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        return [arm for arg in typing.get_args(annotation) if arg is not type(None) for arm in _arms(arg)]
    if origin in (list, typing.Annotated):
        return _arms(typing.get_args(annotation)[0])
    return [annotation]


def _takes_extra(models: Sequence[Type[BaseModel]]) -> bool:
    return any(model.model_config.get("extra") == "allow" for model in models)
