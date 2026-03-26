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
from enum import Enum
from os import cpu_count
import time
from typing import Any, List, Optional, Union, Dict

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, HttpUrl, model_validator

from inference_perf.circuit_breaker import CircuitBreakerConfig


class APIType(Enum):
    Completion = "completion"
    Chat = "chat"


class ResponseFormatType(Enum):
    JSON_SCHEMA = "json_schema"
    JSON_OBJECT = "json_object"


class ResponseFormat(BaseModel):
    """Configuration for structured output via response_format parameter.

    See vLLM docs: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
    """

    type: ResponseFormatType = ResponseFormatType.JSON_SCHEMA
    name: str = "structured_output"  # Name for the json_schema
    json_schema: Optional[dict[str, Any]] = None

    def to_api_format(self) -> dict[str, Any]:
        """Convert to the format expected by vLLM/OpenAI API."""
        if self.type == ResponseFormatType.JSON_OBJECT:
            return {"type": "json_object"}
        # json_schema type
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.name,
                "schema": self.json_schema,
            },
        }


class APIConfig(BaseModel):
    type: APIType = APIType.Completion
    streaming: bool = False
    headers: Optional[dict[str, str]] = None
    slo_unit: Optional[str] = None
    slo_tpot_header: Optional[str] = None
    slo_ttft_header: Optional[str] = None
    response_format: Optional[ResponseFormat] = None


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
    OTelTraceReplay = "otel_trace_replay"


# Represents the distribution for input prompts and output generations.
class Distribution(BaseModel):
    min: int = 10
    max: int = 1024
    mean: float = 512
    std_dev: float = 200
    total_count: Optional[int] = None


# Configuration for shared prefix datagen which allows users to specify shared prefixes.
class SharedPrefix(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    num_groups: int = Field(
        10,
        validation_alias=AliasChoices("num_unique_system_prompts", "num_groups"),
        serialization_alias="num_unique_system_prompts",
    )

    num_prompts_per_group: int = Field(
        10,
        validation_alias=AliasChoices("num_users_per_system_prompt", "num_prompts_per_group"),
        serialization_alias="num_users_per_system_prompt",
    )

    system_prompt_len: int = 100
    question_len: int = 50
    output_len: int = 50
    question_distribution: Optional[Distribution] = None
    output_distribution: Optional[Distribution] = None
    enable_multi_turn_chat: bool = False


class OTelTraceReplayConfig(BaseModel):
    """Configuration for OTel trace replay data generator."""

    trace_directory: Optional[str] = Field(None, description="Directory containing OTel JSON trace files")
    trace_files: Optional[List[str]] = Field(None, description="List of paths to specific OTel JSON trace files")

    # Model configuration
    use_static_model: bool = Field(False, description="Use a single static model for all requests")
    static_model_name: str = Field("", description="Static model name (required if use_static_model=True)")
    model_mapping: Optional[Dict[str, str]] = Field(None, description="Map recorded model names to target models")

    # Request configuration
    default_max_tokens: int = Field(1000, gt=0, description="Default max_tokens if not specified in trace")

    # Error handling
    include_errors: bool = Field(True, description="Include spans with error status")
    skip_invalid_files: bool = Field(False, description="Skip invalid trace files instead of failing")

    @model_validator(mode="after")
    def validate_static_model(self) -> "OTelTraceReplayConfig":
        # Validate that exactly one of trace_directory or trace_files is provided
        sources_provided = sum(
            [
                self.trace_directory is not None,
                self.trace_files is not None,
            ]
        )

        if sources_provided == 0:
            raise ValueError("Either trace_directory or trace_files must be provided")
        if sources_provided > 1:
            raise ValueError("Cannot specify both trace_directory and trace_files; choose one")

        # Validate static model configuration
        if self.use_static_model and not self.static_model_name:
            raise ValueError("static_model_name is required when use_static_model=True")
        if not self.use_static_model and self.static_model_name and not self.model_mapping:
            raise ValueError("Either use_static_model must be True or model_mapping must be provided")
        return self


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

    # OTel trace replay configuration
    otel_trace_replay: Optional[OTelTraceReplayConfig] = None


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
    TRACE_SESSION_REPLAY = "trace_session_replay"


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


logger = logging.getLogger(__name__)


class TraceSessionReplayLoadStage(LoadStage):
    """Load stage for TRACE_SESSION_REPLAY load type.

    A stage runs exactly ``num_sessions`` sessions (a slice of the corpus) at
    ``concurrent_sessions`` concurrency.  A session cursor on ``LoadGenerator``
    advances across stages so each stage draws the next N sessions — mirroring
    how ``get_data()`` advances through data across Standard/Concurrent stages.

    Modes:
    1. Simple concurrency control: set concurrent_sessions (and optionally num_sessions)
    2. Rate-based with concurrency: set concurrent_sessions + session_rate (+ num_sessions)
    """

    # Session concurrency control (REQUIRED)
    concurrent_sessions: int = Field(
        ...,  # Required field
        ge=0,
        description=(
            "Maximum number of sessions active simultaneously. "
            "0 = all sessions active at once (stress test mode). "
            "N > 0 = at most N sessions active; when one completes, next is activated."
        ),
    )

    # Optional rate limiting
    session_rate: Optional[float] = Field(
        None,
        gt=0,
        description="Sessions to start per second (optional, omit for no rate limit)",
    )
    num_sessions: Optional[int] = Field(
        None,
        gt=0,
        description=(
            "Number of sessions to run in this stage. "
            "Draws the next N sessions from the corpus. "
            "None = all remaining sessions."
        ),
    )
    timeout: Optional[float] = Field(
        None,
        gt=0,
        description=(
            "Wall-clock safety limit in seconds. If exceeded, in-flight sessions are "
            "cancelled and stage exits as FAILED. Optional."
        ),
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_trace_session_fields(self) -> "TraceSessionReplayLoadStage":
        # Validate session_rate vs concurrent_sessions
        if self.session_rate is not None and self.concurrent_sessions > 0:
            if self.session_rate > self.concurrent_sessions:
                raise ValueError(
                    f"session_rate ({self.session_rate}) cannot exceed "
                    f"concurrent_sessions ({self.concurrent_sessions}). "
                    f"You can't start sessions faster than the concurrency limit allows."
                )

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


class MultiLoRAConfig(BaseModel):
    name: str
    split: float


class LoadConfig(BaseModel):
    type: LoadType = LoadType.CONSTANT
    interval: float = 1.0
    stages: Union[List[StandardLoadStage], List[ConcurrentLoadStage], List[TraceSessionReplayLoadStage]] = []
    sweep: Optional[SweepConfig] = None
    num_workers: int = max(1, cpu_count())  # type: ignore
    worker_max_concurrency: int = 100
    worker_max_tcp_connections: int = 2500
    trace: Optional[TraceConfig] = None
    circuit_breakers: List[str] = []
    request_timeout: Optional[float] = None
    lora_traffic_split: Optional[List[MultiLoRAConfig]] = None
    base_seed: int = Field(default_factory=lambda: int(time.time() * 1000))

    @model_validator(mode="after")
    def validate_load_config(self) -> "LoadConfig":
        # Validate that sweep is not used with concurrent or trace session replay load types
        if self.type in (LoadType.CONCURRENT, LoadType.TRACE_SESSION_REPLAY) and self.sweep is not None:
            raise ValueError(f"Cannot have sweep config with {self.type.value.upper()} load type")

        # Validate stage types match load type
        if self.type == LoadType.CONCURRENT:
            for i, stage in enumerate(self.stages):
                if not isinstance(stage, ConcurrentLoadStage):
                    raise ValueError(
                        f"Stage {i}: CONCURRENT load type requires ConcurrentLoadStage, got {type(stage).__name__}"
                    )
        elif self.type == LoadType.TRACE_SESSION_REPLAY:
            for i, stage in enumerate(self.stages):
                if not isinstance(stage, TraceSessionReplayLoadStage):
                    raise ValueError(
                        f"Stage {i}: TRACE_SESSION_REPLAY load type requires TraceSessionReplayLoadStage, got {type(stage).__name__}"
                    )
        else:  # CONSTANT, POISSON, or TRACE_REPLAY
            for i, stage in enumerate(self.stages):
                if not isinstance(stage, StandardLoadStage):
                    raise ValueError(
                        f"Stage {i}: {self.type.value.upper()} load type requires StandardLoadStage, got {type(stage).__name__}"
                    )

        # Validate multilora traffic split adds up to 1.0 if present
        if self.lora_traffic_split is not None:
            total = sum(config.split for config in self.lora_traffic_split)
            if total != 1.0:
                raise ValueError("MultiLoRA traffic split in load config does not add up to 1.0")

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
    per_adapter: Optional[bool] = True
    per_adapter_stage: Optional[bool] = False
    percentiles: List[float] = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]


class PrometheusMetricsReportConfig(BaseModel):
    summary: Optional[bool] = True
    per_stage: Optional[bool] = False


class SessionLifecycleReportConfig(BaseModel):
    summary: Optional[bool] = True
    per_stage: Optional[bool] = True
    per_session: Optional[bool] = False


class ReportConfig(BaseModel):
    request_lifecycle: RequestLifecycleMetricsReportConfig = RequestLifecycleMetricsReportConfig()
    prometheus: Optional[PrometheusMetricsReportConfig] = PrometheusMetricsReportConfig()
    session_lifecycle: SessionLifecycleReportConfig = SessionLifecycleReportConfig()


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
    cert_path: Optional[str] = None
    key_path: Optional[str] = None


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

    @model_validator(mode="after")
    def validate_otel_trace_replay_load_type(self) -> "Config":
        """Validate that otel_trace_replay data type uses trace_session_replay load type."""
        if self.data.type == DataGenType.OTelTraceReplay:
            if self.load.type != LoadType.TRACE_SESSION_REPLAY:
                raise ValueError(
                    f"data.type 'otel_trace_replay' requires load.type 'trace_session_replay', "
                    f"but got '{self.load.type.value}'. OTel trace replay with dependencies requires "
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


def read_config(config_file: str) -> Config:
    logger = logging.getLogger(__name__)
    logger.info("Using configuration from: %s", config_file)
    with open(config_file, "r") as stream:
        cfg = yaml.safe_load(stream)

    default_cfg = Config().model_dump(mode="json")
    merged_cfg = deep_merge(default_cfg, cfg)

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
