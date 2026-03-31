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
import os
from datetime import datetime
from os import cpu_count
import time
from typing import Any

import yaml
import json
from google.protobuf import json_format  # type: ignore[import-untyped, unused-ignore]
import protovalidate

from inference_perf.circuit_breaker import CircuitBreakerConfig
from api.generated.python.api.config.v1 import config_pb2

# Directly re-export all generated types to avoid importing from api.generated.python.config.config_p2p
from api.generated.python.api.config.v1.config_p2p import (
    APIType as APIType,
    ResponseFormatType as ResponseFormatType,
    TraceFormat as TraceFormat,
    DataGenType as DataGenType,
    ModelServerType as ModelServerType,
    LoadType as LoadType,
    StageGenType as StageGenType,
    MetricsClientType as MetricsClientType,
    ResponseFormat as ResponseFormat,
    APIConfig as APIConfig,
    TraceConfig as TraceConfig,
    Distribution as Distribution,
    SharedPrefix as SharedPrefix,
    DataConfig as DataConfig,
    StandardLoadStage as StandardLoadStage,
    ConcurrentLoadStage as ConcurrentLoadStage,
    SweepConfig as SweepConfig,
    MultiLoRAConfig as MultiLoRAConfig,
    LoadConfig as LoadConfig,
    StorageConfigBase as StorageConfigBase,
    GoogleCloudStorageConfig as GoogleCloudStorageConfig,
    SimpleStorageServiceConfig as SimpleStorageServiceConfig,
    StorageConfig as StorageConfig,
    RequestLifecycleMetricsReportConfig as RequestLifecycleMetricsReportConfig,
    PrometheusMetricsReportConfig as PrometheusMetricsReportConfig,
    ReportConfig as ReportConfig,
    PrometheusClientConfig as PrometheusClientConfig,
    MetricsClientConfig as MetricsClientConfig,
    ModelServerClientConfig as ModelServerClientConfig,
    CustomTokenizerConfig as CustomTokenizerConfig,
    Config as Config,
)


def to_api_format(fmt: ResponseFormat) -> dict[str, Any]:
    """Convert ResponseFormat to the format expected by vLLM/OpenAI API."""
    if fmt.type == ResponseFormatType.JSON_OBJECT:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": fmt.name,
            "schema": fmt.json_schema,
        },
    }


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


CONVERSIONS = [
    (["api", "type"], APIType),
    (["data", "type"], DataGenType),
    (["load", "type"], LoadType),
    (["metrics", "type"], MetricsClientType),
    (["server", "type"], ModelServerType),
    (["api", "response_format", "type"], ResponseFormatType),
    (["load", "sweep", "type"], StageGenType),
    (["data", "trace", "format"], TraceFormat),
    (["load", "trace", "format"], TraceFormat),
]


def sanitize_config(cfg: dict[str, Any]) -> dict[str, Any]:

    if cfg.get("server", {}).get("type") in ["mock", "MOCK"]:
        cfg["server"]["type"] = "MOCK_SERVER"
    if cfg.get("data", {}).get("type") == "shareGPT":
        cfg["data"]["type"] = "SHAREGPT"
    for s in ["data", "load"]:
        if cfg.get(s, {}).get("trace", {}).get("format") in ["AzurePublicDataset", "azure_public_dataset"]:
            cfg[s]["trace"]["format"] = "AZURE_PUBLIC_DATASET"
    if "shared_prefix" in cfg.get("data", {}):
        sp = cfg["data"]["shared_prefix"]
        if "num_unique_system_prompts" in sp:
            sp["num_groups"] = sp.pop("num_unique_system_prompts")
        if "num_users_per_system_prompt" in sp:
            sp["num_prompts_per_group"] = sp.pop("num_users_per_system_prompt")
    if "load" in cfg and "stages" in cfg["load"]:
        cfg["load"]["concurrent_stages" if cfg["load"].get("type") == "concurrent" else "standard_stages"] = cfg["load"].pop(
            "stages"
        )
    if isinstance(cfg.get("api", {}).get("response_format", {}).get("json_schema"), dict):
        import json

        cfg["api"]["response_format"]["json_schema"] = json.dumps(cfg["api"]["response_format"]["json_schema"])
    for path, e in CONVERSIONS:
        curr = cfg
        for k in path[:-1]:
            curr = curr.get(k, {})
        val = curr.get(path[-1])
        if isinstance(val, str):
            try:
                curr[path[-1]] = e[val.upper()].value
            except KeyError:
                pass
    return cfg


def read_config(config_file: str) -> Config:
    logger = logging.getLogger(__name__)
    logger.info("Using configuration from: %s", config_file)

    # 1. Load language-agnostic defaults
    defaults_path = os.path.join(os.path.dirname(__file__), "..", "api", "config", "defaults.yaml")
    with open(defaults_path, "r") as f:
        defaults = yaml.safe_load(f)

    # 2. Process dynamic defaults (placeholders)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")

    def process_dynamic(d: Any) -> None:
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, str) and "{{ timestamp }}" in v:
                    d[k] = v.replace("{{ timestamp }}", timestamp)
                else:
                    process_dynamic(v)
        elif isinstance(d, list):
            for i in d:
                process_dynamic(i)

    process_dynamic(defaults)

    # Handle num_workers and base_seed which are dynamic
    if defaults["load"]["num_workers"] == 0:
        defaults["load"]["num_workers"] = max(1, cpu_count() or 1)
    if defaults["load"].get("base_seed") == 0:
        defaults["load"]["base_seed"] = int(time.time() * 1000)

    # 3. Load user config
    with open(config_file, "r") as stream:
        user_cfg = yaml.safe_load(stream)

    # 4. Deep merge
    merged_cfg = deep_merge(defaults, user_cfg)

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
            concurrent_stages = []
            for stage in stages:
                concurrent_stages.append(ConcurrentLoadStage(**stage))
            merged_cfg["load"]["stages"] = concurrent_stages

    # Extract and store circuit breakers if they are dicts
    cb_dicts = merged_cfg.pop("circuit_breakers", [])
    cb_objects = []
    for d in cb_dicts:
        if isinstance(d, dict):
            cb_objects.append(CircuitBreakerConfig.model_validate(d))
        else:
            cb_objects.append(d)  # Keep as is if already string

    # Sanitize legacy formats
    merged_cfg = sanitize_config(merged_cfg)

    # Validate using protovalidate
    proto_cfg = config_pb2.Config()
    try:
        json_format.Parse(json.dumps(merged_cfg), proto_cfg)
        validator = protovalidate.Validator()  # type: ignore[no-untyped-call]
        validator.validate(proto_cfg)
    except Exception as e:
        logger.error("Configuration validation failed!")
        raise e

    logger.info("Final merged configuration ready.")
    config = Config(**merged_cfg)
    if cb_objects:
        config.circuit_breakers = cb_objects  # type: ignore[assignment]
    return config
