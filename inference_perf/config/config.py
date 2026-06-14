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
import logging
from datetime import datetime
from typing import Any, List, Optional

import yaml
from pydantic import BaseModel, model_validator

from inference_perf.circuit_breaker import CircuitBreakerConfig
from inference_perf.config.apis import APIConfig
from inference_perf.config.client.filestorage import StorageConfig
from inference_perf.config.client.modelserver import ModelServerClientConfig
from inference_perf.config.datagen import DataConfig, DataGenType
from inference_perf.config.loadgen import (
    ConcurrentLoadStage,
    LoadConfig,
    LoadType,
    StandardLoadStage,
    TraceSessionReplayLoadStage,
)
from inference_perf.config.metrics import MetricsClientConfig
from inference_perf.config.reportgen import ReportConfig
from inference_perf.config.utils import CustomTokenizerConfig


class Config(BaseModel):
    api: APIConfig = APIConfig()
    data: DataConfig = DataConfig()
    load: LoadConfig = LoadConfig()
    metrics: Optional[MetricsClientConfig] = None
    report: ReportConfig = ReportConfig()
    storage: Optional[StorageConfig] = StorageConfig()
    server: Optional[ModelServerClientConfig] = None
    tokenizer: Optional[CustomTokenizerConfig] = None
    circuit_breakers: Optional[List[CircuitBreakerConfig]] = None

    @model_validator(mode="after")
    def validate_trace_replay_load_type(self) -> "Config":
        """Validate that trace replay data types use trace_session_replay load type."""
        if self.data.type in (DataGenType.OTelTraceReplay, DataGenType.WekaTraceReplay):
            if self.load.type != LoadType.TRACE_SESSION_REPLAY:
                raise ValueError(
                    f"data.type '{self.data.type.value}' requires load.type 'trace_session_replay', "
                    f"but got '{self.load.type.value}'. Trace replay with dependencies requires "
                    f"session-based load dispatch to properly handle event dependencies and timing."
                )
        return self


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def read_config(config_file: Optional[str] = None, cli_overrides: Optional[dict[str, Any]] = None) -> Config:
    logger = logging.getLogger(__name__)
    cfg: dict[str, Any] = {}
    if config_file:
        logger.info("Using configuration from: %s", config_file)
        with open(config_file, "r") as stream:
            cfg = yaml.safe_load(stream) or {}

    default_cfg = Config().model_dump(mode="json")
    merged_cfg = deep_merge(default_cfg, cfg)

    if cli_overrides:
        merged_cfg = deep_merge(merged_cfg, cli_overrides)

    # Handle timestamp substitution in storage paths
    if "storage" in merged_cfg and merged_cfg["storage"]:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for storage_type in ["local_storage", "google_cloud_storage", "simple_storage_service"]:
            if (
                storage_type in merged_cfg["storage"]
                and merged_cfg["storage"][storage_type]
                and "path" in merged_cfg["storage"][storage_type]
            ):
                path = merged_cfg["storage"][storage_type]["path"]
                if path and "{timestamp}" in path:
                    merged_cfg["storage"][storage_type]["path"] = path.replace("{timestamp}", timestamp)

    # Handle stage type conversion based on load type
    if "load" in merged_cfg and "stages" in merged_cfg["load"] and merged_cfg["load"]["stages"]:
        load_type = merged_cfg["load"].get("type", "constant")
        stages = merged_cfg["load"]["stages"]

        if load_type == "concurrent":
            # Convert to ConcurrentLoadStage objects
            concurrent_stages = []
            for stage in stages:
                concurrent_stages.append(ConcurrentLoadStage(**stage))
            merged_cfg["load"]["stages"] = concurrent_stages
        elif load_type == "trace_session_replay":
            # Convert to TraceSessionReplayLoadStage objects
            trace_session_stages = []
            for stage in stages:
                trace_session_stages.append(TraceSessionReplayLoadStage(**stage))
            merged_cfg["load"]["stages"] = trace_session_stages
        else:
            # Convert to StandardLoadStage objects for constant/poisson/trace_replay
            standard_stages = []
            for stage in stages:
                standard_stages.append(StandardLoadStage(**stage))
            merged_cfg["load"]["stages"] = standard_stages

    logger.info(
        "Benchmarking with the following config:\n\n%s\n", yaml.dump(merged_cfg, sort_keys=False, default_flow_style=False)
    )
    return Config(**merged_cfg)
