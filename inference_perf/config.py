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
from pydantic import BaseModel, HttpUrl, model_validator, Field
from inference_perf.circuit_breaker import CircuitBreakerConfig
from typing import Any, Optional, List, Union
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
    headers: Optional[dict[str, str]] = None


class TraceFormat(Enum):
    AZURE_PUBLIC_DATASET = "AzurePublicDataset"


class TraceConfig(BaseModel):
    file: str
    format: TraceFormat = TraceFormat.AZURE_PUBLIC_DATASET


class DataGenType(Enum):
    Mock = "mock"
    ShareGPT = "shareGPT"
    Synthetic = "synthetic"
    Random = "random"
    SharedPrefix = "shared_prefix"
    CNNDailyMail = "cnn_dailymail"
    InfinityInstruct = "infinity_instruct"
    BillsumConversations = "billsum_conversations"


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
    question_len_std: float = 0  # Variation in question length within a group
    question_len_min: Optional[int] = None
    question_len_max: Optional[int] = None
    output_len_std: float = 0  # Variation in output length within a group
    output_len_min: Optional[int] = None
    output_len_max: Optional[int] = None

    output_len: int = 50
    enable_multi_turn_chat: bool = False


class DataConfig(BaseModel):
    type: DataGenType = DataGenType.Mock

    # Valid only for shareGPT type at this moment
    path: Optional[str] = None  # path to the downloaded shareGPT dataset

    # Distributions are only supported for synthetic/random dataset at this moment
    input_distribution: Optional[Distribution] = None
    output_distribution: Optional[Distribution] = None
    shared_prefix: Optional[SharedPrefix] = None

    # Trace file is only supported for random dataset at this moment
    trace: Optional[TraceConfig] = None


class ModelServerType(Enum):
    VLLM = "vllm"
    SGLANG = "sglang"
    TGI = "tgi"
    MOCK = "mock"


class LoadType(Enum):
    CONSTANT = "constant"
    POISSON = "poisson"
    TRACE_REPLAY = "trace_replay"
    CONCURRENT = "concurrent"


class MetricsClientType(Enum):
    PROMETHEUS = "prometheus"
    DEFAULT = "default"


class LoadStage(BaseModel):
    """Base class for load stages. Use specific subclasses for different load types."""

    pass


class StandardLoadStage(LoadStage):
    """Load stage for CONSTANT and POISSON load types."""

    rate: float = Field(..., gt=0, description="Request rate (QPS)")
    duration: int = Field(..., gt=0, description="Duration in seconds")

    # These fields should not be set for standard load types
    num_requests: Optional[int] = Field(default=None, description="Not used for standard load types")
    concurrency_level: Optional[int] = Field(default=None, description="Not used for standard load types")

    @model_validator(mode="after")
    def validate_standard_fields(self) -> "StandardLoadStage":
        if self.num_requests is not None:
            raise ValueError("num_requests should not be set for CONSTANT/POISSON load types")
        if self.concurrency_level is not None:
            raise ValueError("concurrency_level should not be set for CONSTANT/POISSON load types")
        return self


class ConcurrentLoadStage(LoadStage):
    """Load stage for CONCURRENT load type."""

    num_requests: int = Field(..., gt=0, description="Number of requests to send")
    concurrency_level: int = Field(..., gt=0, description="Concurrency level")

    # These fields are set at runtime for load generation but should not be configured
    rate: Optional[float] = Field(None, description="Set at runtime for load generation")
    duration: Optional[int] = Field(None, description="Set at runtime for load generation")

    @model_validator(mode="after")
    def validate_concurrent_fields(self) -> "ConcurrentLoadStage":
        # Allow rate and duration to be set at runtime, but they should start as None
        # No validation needed here since they're set dynamically
        return self


class StageGenType(Enum):
    GEOM = "geometric"
    LINEAR = "linear"


class SweepConfig(BaseModel):
    type: StageGenType
    num_requests: int = 2000
    timeout: float = 60
    num_stages: int = 5
    stage_duration: int = 180
    saturation_percentile: float = 95


class LoadConfig(BaseModel):
    type: LoadType = LoadType.CONSTANT
    interval: float = 1.0
    stages: Union[List[StandardLoadStage], List[ConcurrentLoadStage]] = []
    sweep: Optional[SweepConfig] = None
    num_workers: int = max(1, cpu_count())  # type: ignore
    worker_max_concurrency: int = 100
    worker_max_tcp_connections: int = 2500
    trace: Optional[TraceConfig] = None
    circuit_breakers: List[str] = []
    request_timeout: Optional[float] = None

    @model_validator(mode="after")
    def validate_load_config(self) -> "LoadConfig":
        # Validate that sweep is not used with concurrent load type
        if self.type == LoadType.CONCURRENT and self.sweep is not None:
            raise ValueError("Cannot have sweep config with CONCURRENT load type")

        # Validate stage types match load type
        if self.type == LoadType.CONCURRENT:
            for i, stage in enumerate(self.stages):
                if not isinstance(stage, ConcurrentLoadStage):
                    raise ValueError(
                        f"Stage {i}: CONCURRENT load type requires ConcurrentLoadStage, got {type(stage).__name__}"
                    )
        else:  # CONSTANT or POISSON
            for i, stage in enumerate(self.stages):
                if not isinstance(stage, StandardLoadStage):
                    raise ValueError(
                        f"Stage {i}: {self.type.value.upper()} load type requires StandardLoadStage, got {type(stage).__name__}"
                    )

        return self


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
    circuit_breakers: Optional[List[CircuitBreakerConfig]] = None


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
        else:
            # Convert to StandardLoadStage objects for constant/poisson
            standard_stages = []
            for stage in stages:
                standard_stages.append(StandardLoadStage(**stage))
            merged_cfg["load"]["stages"] = standard_stages

    logger.info(
        "Benchmarking with the following config:\n\n%s\n", yaml.dump(merged_cfg, sort_keys=False, default_flow_style=False)
    )
    return Config(**merged_cfg)
