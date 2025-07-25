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
from datetime import datetime
from pydantic import BaseModel, HttpUrl, model_validator
from typing import Any, Optional, List
from enum import Enum
from os import cpu_count
import yaml
import logging


class APIType(Enum):
    Completion = "completion"
    Chat = "chat"


class APIConfig(BaseModel):
    type: APIType = APIType.Completion
    streaming: bool = False


class DataGenType(Enum):
    Mock = "mock"
    ShareGPT = "shareGPT"
    Synthetic = "synthetic"
    Random = "random"
    SharedPrefix = "shared_prefix"


# Represents the distribution for input prompts and output generations.
class Distribution(BaseModel):
    min: int = 10
    max: int = 1024
    mean: float = 512
    std_dev: float = 200
    total_count: Optional[int] = None


# Configuration for shared prefix datagen which allows users to specify shared prefixes.
class SharedPrefix(BaseModel):
    num_groups: int = 10
    num_prompts_per_group: int = 10
    system_prompt_len: int = 100
    question_len: int = 50
    output_len: int = 50


class DataConfig(BaseModel):
    type: DataGenType = DataGenType.Mock
    # Distributions are only supported for synthetic/random dataset at this moment
    input_distribution: Optional[Distribution] = None
    output_distribution: Optional[Distribution] = None
    shared_prefix: Optional[SharedPrefix] = None


class ModelServerType(Enum):
    VLLM = "vllm"


class LoadType(Enum):
    CONSTANT = "constant"
    POISSON = "poisson"


class MetricsClientType(Enum):
    PROMETHEUS = "prometheus"
    DEFAULT = "default"


class LoadStage(BaseModel):
    rate: float
    duration: int


class LoadConfig(BaseModel):
    type: LoadType = LoadType.CONSTANT
    interval: float = 1.0
    stages: List[LoadStage] = []
    num_workers: int = max(1, cpu_count() // 2)  # type: ignore
    worker_max_concurrency: int = 10
    worker_max_tcp_connections: int = 2500


class StorageConfigBase(BaseModel):
    path: str = f"reports-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    report_file_prefix: Optional[str] = None


class GoogleCloudStorageConfig(StorageConfigBase):
    bucket_name: str

class SimpleStorageServiceConfig(StorageConfigBase):
    bucket_name: str

class StorageConfig(BaseModel):
    local_storage: StorageConfigBase = StorageConfigBase()
    google_cloud_storage: Optional[GoogleCloudStorageConfig] = None
    simple_storage_service: Optional[SimpleStorageServiceConfig] = None


class RequestLifecycleMetricsReportConfig(BaseModel):
    summary: Optional[bool] = True
    per_stage: Optional[bool] = True
    per_request: Optional[bool] = False


class PrometheusMetricsReportConfig(BaseModel):
    summary: Optional[bool] = True
    per_stage: Optional[bool] = False


class ReportConfig(BaseModel):
    request_lifecycle: RequestLifecycleMetricsReportConfig = RequestLifecycleMetricsReportConfig()
    prometheus: Optional[PrometheusMetricsReportConfig] = PrometheusMetricsReportConfig()


class PrometheusClientConfig(BaseModel):
    scrape_interval: int = 15
    url: Optional[HttpUrl] = None
    filters: List[str] = []
    google_managed: bool = False

    @model_validator(mode="after")
    def check_exclusive_fields(self) -> "PrometheusClientConfig":
        if bool(self.url) == bool(self.google_managed):
            raise ValueError("Exactly one of 'url' or 'google_managed' must be set.")
        return self


class MetricsClientConfig(BaseModel):
    type: MetricsClientType
    prometheus: Optional[PrometheusClientConfig] = None


class ModelServerClientConfig(BaseModel):
    type: ModelServerType = ModelServerType.VLLM
    model_name: Optional[str] = None
    base_url: str
    ignore_eos: bool = True
    api_key: Optional[str] = None


class CustomTokenizerConfig(BaseModel):
    pretrained_model_name_or_path: Optional[str] = None
    trust_remote_code: Optional[bool] = None
    token: Optional[str] = None


class Config(BaseModel):
    api: APIConfig = APIConfig()
    data: DataConfig = DataConfig()
    load: LoadConfig = LoadConfig()
    metrics: Optional[MetricsClientConfig] = None
    report: ReportConfig = ReportConfig()
    storage: Optional[StorageConfig] = StorageConfig()
    server: Optional[ModelServerClientConfig] = None
    tokenizer: Optional[CustomTokenizerConfig] = None


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def read_config(config_file: str) -> Config:
    logger = logging.getLogger(__name__)
    logger.info("Using configuration from: %s", config_file)
    with open(config_file, "r") as stream:
        cfg = yaml.safe_load(stream)

    default_cfg = Config().model_dump(mode="json")
    merged_cfg = deep_merge(default_cfg, cfg)

    logger.info(
        "Benchmarking with the following config:\n\n%s\n", yaml.dump(merged_cfg, sort_keys=False, default_flow_style=False)
    )
    return Config(**merged_cfg)
