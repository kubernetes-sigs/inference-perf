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
from math import sqrt
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, HttpUrl, model_validator

from inference_perf.circuit_breaker import CircuitBreakerConfig
from inference_perf.payloads.multimodal_spec import ImageRepresentation, VideoRepresentation


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

    type: ResponseFormatType = Field(ResponseFormatType.JSON_SCHEMA, description="Response-format variant to request.")
    name: str = Field("structured_output", description="Schema name embedded in the json_schema payload.")
    json_schema: Optional[dict[str, Any]] = Field(
        None, description="JSON Schema describing the required output shape (for json_schema type)."
    )

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
    type: APIType = Field(APIType.Completion, description="API type to exercise on the model server.")
    streaming: bool = Field(False, description="Enable streaming so TTFT, ITL, and TPOT can be measured.")
    headers: Optional[dict[str, str]] = Field(None, description="Custom HTTP headers attached to every request.")
    slo_unit: Optional[str] = Field(None, description="Unit for SLO header values (e.g. 'ms', 's'). Defaults to 'ms'.")
    slo_tpot_header: Optional[str] = Field(None, description="Header name carrying the per-request TPOT SLO.")
    slo_ttft_header: Optional[str] = Field(None, description="Header name carrying the per-request TTFT SLO.")
    response_format: Optional[ResponseFormat] = Field(
        None, description="Structured-output schema sent via the response_format API parameter."
    )


class TraceFormat(Enum):
    AZURE_PUBLIC_DATASET = "AzurePublicDataset"


class TraceConfig(BaseModel):
    file: str = Field(..., description="Path to the trace file to replay.")
    format: TraceFormat = Field(TraceFormat.AZURE_PUBLIC_DATASET, description="On-disk format of the trace file.")


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
    ConversationReplay = "conversation_replay"


class DistributionType(str, Enum):
    NORMAL = "normal"
    SKEW_NORMAL = "skew_normal"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    POISSON = "poisson"
    FIXED = "fixed"


# Represents the distribution for input prompts and output generations.
class Distribution(BaseModel):
    min: int = Field(10, description="Minimum sampled value (tokens, items, etc.).")
    max: int = Field(1024, description="Maximum sampled value.")
    mean: float = Field(512, description="Distribution mean.")
    std_dev: float = Field(200, description="Standard deviation. Mutually exclusive with `variance`.")
    total_count: Optional[int] = Field(
        None, description="Total number of samples to draw, when the distribution is materialized eagerly."
    )
    type: DistributionType = Field(DistributionType.NORMAL, description="Distribution family used to sample values.")
    variance: Optional[float] = Field(
        None, description="Variance. Mutually exclusive with `std_dev`; converted to std_dev when set."
    )
    skew: float = Field(0.0, description="Skew parameter; only used when type=skew_normal.")

    @model_validator(mode="after")
    def validate_distribution(self) -> "Distribution":
        if self.variance is not None and self.std_dev > 0:
            raise ValueError("Specify either 'std_dev' or 'variance', not both.")
        if self.variance is not None:
            if self.variance < 0:
                raise ValueError("Variance cannot be negative.")
            self.std_dev = sqrt(self.variance)
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) cannot be greater than max ({self.max}).")
        if self.std_dev < 0:
            raise ValueError("std_dev cannot be negative.")
        return self


# --- Base Utility Types ---
class ResolutionPreset(str, Enum):
    """Standard resolution shortcuts for image and video specs.

    Mappings:
      - ``"4k"``    → 3840 × 2160
      - ``"1080p"`` → 1920 × 1080
      - ``"720p"``  → 1280 × 720
      - ``"360p"``  → 640 × 360
    """

    P4K = "4k"
    P1080 = "1080p"
    P720 = "720p"
    P360 = "360p"


class Resolution(BaseModel):
    height: int = Field(description="Pixel height (e.g. 1080).")
    width: int = Field(description="Pixel width (e.g. 1920).")


AnyResolution = Union[ResolutionPreset, Resolution]


class WeightedResolution(BaseModel):
    resolution: AnyResolution = Field(description='A ``ResolutionPreset`` (e.g. ``"1080p"``) or explicit ``Resolution``.')
    weight: float = Field(default=1.0, description="Relative frequency of this resolution being selected from the list.")


class VideoProfile(BaseModel):
    resolution: AnyResolution = Field(description="Frame resolution. Preset string or explicit ``Resolution``.")
    frames: int = Field(description="Number of frames in the video. Required.")


class WeightedVideoProfile(BaseModel):
    profile: VideoProfile = Field(description="The video profile (resolution + frames) to sample.")
    weight: float = Field(default=1.0, description="Relative frequency of this exact video profile being selected.")


class WeightedDuration(BaseModel):
    duration: float = Field(description="The length of the audio clip in seconds.")
    weight: float = Field(default=1.0, description="Relative frequency of this duration being selected.")


# --- Modality-Specific Request Configs ---
class MediaDatagenConfig(BaseModel):
    count: Optional[Distribution] = Field(
        default=None, description="Distribution of the number of media items to generate per request."
    )
    insertion_point: Optional[Union[float, Distribution]] = Field(
        default=None,
        description="Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from.",
    )


class ImageDatagenConfig(MediaDatagenConfig):
    resolutions: Optional[Union[AnyResolution, List[WeightedResolution]]] = Field(
        default=None, description="Resolution or list of weighted resolutions for generated images."
    )
    representation: ImageRepresentation = Field(
        default=ImageRepresentation.PNG,
        description=(
            "Wire encoding for emitted image bytes: ``png`` (default, lossless) or ``jpeg`` "
            "(lossy, smaller payload). Some VLMs prefer one or the other; consult the model's spec sheet."
        ),
    )


class VideoDatagenConfig(MediaDatagenConfig):
    profiles: Optional[Union[VideoProfile, List[WeightedVideoProfile]]] = Field(
        default=None, description="Video profile or list of weighted video profiles for generated videos."
    )
    representation: VideoRepresentation = Field(
        default=VideoRepresentation.MP4,
        description=(
            "Wire-format strategy. ``mp4`` sends one ``video_url`` block carrying an MP4 blob "
            "(measures full pipeline including server-side decode). ``png_frames`` and "
            "``jpeg_frames`` send ``frames`` × ``image_url`` blocks at one insertion point in "
            "the named encoding (no decode dependency, useful for prefix-cache benchmarks "
            "and servers that don't accept ``video_url``)."
        ),
    )


class AudioDatagenConfig(MediaDatagenConfig):
    durations: Optional[Union[float, List[WeightedDuration]]] = Field(
        default=None, description="Duration or list of weighted durations for generated audio clips."
    )


# --- Main Wrapper Models ---
class SyntheticMultimodalDatagenConfig(BaseModel):
    """Configuration for standard multimodal data generation.

    Caveat: resolutions, video profiles, and audio durations specified here are
    sent to the model server as-is. There is no model-aware validation — real
    VLMs each impose their own limits (per-request media count via
    --limit-mm-per-prompt, vision-encoder pixel caps, video frame budgets,
    audio duration caps, max-model-len). Out-of-range payloads typically fail
    at the wire (counted in `failures`) or get silently downsized server-side,
    which makes per-modality byte/pixel throughput numbers reflect what was
    sent rather than what the model processed. Consult your model's spec sheet
    when picking values. See docs/config.md ("Multimodal Data Generation").
    """

    image: Optional[ImageDatagenConfig] = Field(None, description="Image generation settings (omit to disable images).")
    video: Optional[VideoDatagenConfig] = Field(None, description="Video generation settings (omit to disable video).")
    audio: Optional[AudioDatagenConfig] = Field(None, description="Audio generation settings (omit to disable audio).")


# Configuration for shared prefix datagen which allows users to specify shared prefixes.
class SharedPrefix(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    num_groups: int = Field(
        10,
        validation_alias=AliasChoices("num_unique_system_prompts", "num_groups"),
        serialization_alias="num_unique_system_prompts",
        description="Number of distinct shared-prefix groups (one shared system prompt per group).",
    )

    num_prompts_per_group: int = Field(
        10,
        validation_alias=AliasChoices("num_users_per_system_prompt", "num_prompts_per_group"),
        serialization_alias="num_users_per_system_prompt",
        description="Number of unique user prompts generated per shared-prefix group.",
    )

    system_prompt_len: Union[int, Distribution] = Field(
        100, description="Shared system-prompt length, as a fixed int or a Distribution."
    )
    question_len: Union[int, Distribution] = Field(50, description="Per-question length, as a fixed int or a Distribution.")
    output_len: Union[int, Distribution] = Field(
        50, description="Per-response output length, as a fixed int or a Distribution."
    )
    seed: Optional[int] = Field(None, description="Random seed for deterministic prompt generation.")

    question_distribution: Optional[Distribution] = Field(
        None,
        description="Legacy: distribution for question lengths. Prefer the inline distribution form on `question_len`.",
    )
    output_distribution: Optional[Distribution] = Field(
        None,
        description="Legacy: distribution for output lengths. Prefer the inline distribution form on `output_len`.",
    )

    enable_multi_turn_chat: bool = Field(False, description="When true, prompts within a group form multi-turn chat sessions.")
    multimodal: Optional[SyntheticMultimodalDatagenConfig] = Field(
        None, description="Optional multimodal payload generation alongside text prompts."
    )

    @model_validator(mode="after")
    def validate_no_ambiguous_distributions(self) -> "SharedPrefix":
        if isinstance(self.question_len, Distribution) and self.question_distribution is not None:
            raise ValueError(
                "Cannot specify both inline distribution on 'question_len' and legacy 'question_distribution'."
                " Use one or the other."
            )
        if isinstance(self.output_len, Distribution) and self.output_distribution is not None:
            raise ValueError(
                "Cannot specify both inline distribution on 'output_len' and legacy 'output_distribution'."
                " Use one or the other."
            )
        return self


class ConversationReplayConfig(BaseModel):
    """Configuration for conversation replay data generator.

    Generates synthetic multi-turn conversations in-memory from configurable
    distributions. Each conversation has a two-part system prompt (shared prefix
    + dynamic per-conversation suffix) and a sequence of user/assistant turns
    with independently sampled input/output token lengths.
    """

    seed: int = Field(42, description="Random seed for deterministic generation")
    num_conversations: int = Field(200, gt=0, description="Number of conversation blueprints to generate")
    shared_system_prompt_len: int = Field(8359, ge=0, description="Fixed shared system prompt length in tokens")
    dynamic_system_prompt_len: Optional[Distribution] = Field(
        None, description="Per-conversation dynamic system prompt length distribution"
    )
    turns_per_conversation: Optional[Distribution] = Field(None, description="Number of turns per conversation distribution")
    input_tokens_per_turn: Optional[Distribution] = Field(None, description="Input tokens per turn distribution")
    output_tokens_per_turn: Optional[Distribution] = Field(None, description="Output tokens per turn distribution")
    tool_call_latency_sec: Optional[Distribution] = Field(
        None,
        description=(
            "Per-turn tool execution latency distribution in seconds. "
            "When set, each turn sleeps for the sampled duration after model "
            "inference completes and before the next turn begins, simulating "
            "tool call round-trips. The sleep holds the session lock so the "
            "GPU is free to serve other concurrent conversations — correctly "
            "modelling offline agentic workloads. Omit for pure GPU throughput "
            "measurement. Values are in seconds; min/max are whole seconds, "
            "mean/std_dev may be fractional."
        ),
    )


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
    type: DataGenType = Field(
        DataGenType.Mock, description="Which data generator to use; the sibling fields below apply only to specific types."
    )

    path: Optional[str] = Field(
        None,
        description="Filesystem path to a dataset. Required for shareGPT, cnn_dailymail, billsum_conversations, and infinity_instruct.",
    )

    input_distribution: Optional[Distribution] = Field(
        None, description="Input prompt length distribution. Used by synthetic/random datagens."
    )
    output_distribution: Optional[Distribution] = Field(
        None, description="Output length distribution. Used by synthetic/random datagens."
    )
    shared_prefix: Optional[SharedPrefix] = Field(
        None, description="Shared-prefix datagen configuration (when type=shared_prefix)."
    )
    multimodal: Optional[SyntheticMultimodalDatagenConfig] = Field(
        None, description="Multimodal generation block paired with synthetic/shared_prefix text generation."
    )

    trace: Optional[TraceConfig] = Field(None, description="Trace file replayed by the random datagen (when type=random).")

    otel_trace_replay: Optional[OTelTraceReplayConfig] = Field(
        None, description="OTel trace replay configuration (when type=otel_trace_replay)."
    )

    conversation_replay: Optional[ConversationReplayConfig] = Field(
        None, description="Conversation replay configuration (when type=conversation_replay)."
    )


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
    type: StageGenType = Field(..., description="How rate steps between stages are spaced.")
    num_requests: int = Field(2000, description="Total requests to issue across the sweep.")
    timeout: float = Field(60, description="Per-stage timeout in seconds before saturation is declared.")
    num_stages: int = Field(5, description="Number of stages in the sweep.")
    stage_duration: int = Field(180, description="Duration of each stage in seconds.")
    saturation_percentile: float = Field(95, description="Percentile latency used to detect saturation.")


class MultiLoRAConfig(BaseModel):
    name: str = Field(..., description="LoRA adapter name as registered with the model server.")
    split: float = Field(..., description="Fraction of traffic routed to this adapter; all splits must sum to 1.0.")


class LoadConfig(BaseModel):
    type: LoadType = Field(LoadType.CONSTANT, description="Traffic-generation strategy.")
    interval: float = Field(1.0, description="Inter-request interval in seconds for load types that pace by interval.")
    stages: Union[List[StandardLoadStage], List[ConcurrentLoadStage], List[TraceSessionReplayLoadStage]] = Field(
        default=[], description="Ordered list of load stages; stage subclass must match the chosen load type."
    )
    sweep: Optional[SweepConfig] = Field(
        None, description="Auto-generated rate sweep (mutually exclusive with concurrent/trace_session_replay)."
    )
    num_workers: int = Field(default=max(1, cpu_count()), description="Number of worker processes used to drive load.")  # type: ignore
    worker_max_concurrency: int = Field(100, description="Maximum concurrent in-flight requests per worker.")
    worker_max_tcp_connections: int = Field(2500, description="TCP connection pool size per worker.")
    trace: Optional[TraceConfig] = Field(None, description="Trace file replayed for trace-based load types.")
    circuit_breakers: List[str] = Field(
        default=[], description="Names of circuit breakers (defined under `circuit_breakers`) to enable."
    )
    request_timeout: Optional[float] = Field(None, description="Per-request timeout in seconds; None disables the timeout.")
    lora_traffic_split: Optional[List[MultiLoRAConfig]] = Field(
        None, description="Optional traffic split across LoRA adapters; splits must sum to 1.0."
    )
    base_seed: int = Field(
        default_factory=lambda: int(time.time() * 1000),
        description="Base seed shared across workers; defaults to wall-clock time on startup.",
    )

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
    path: str = Field(
        default_factory=lambda: f"reports-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        description="Directory or object-store path where reports are written. Supports `{timestamp}` substitution.",
    )
    report_file_prefix: Optional[str] = Field(
        None, description="Optional filename prefix for report files written to this destination."
    )


class GoogleCloudStorageConfig(StorageConfigBase):
    bucket_name: str = Field(..., description="Target GCS bucket.")


class SimpleStorageServiceConfig(StorageConfigBase):
    bucket_name: str = Field(..., description="Target S3-compatible bucket.")
    endpoint_url: Optional[str] = Field(None, description="Override endpoint URL for S3-compatible services (e.g. MinIO).")
    region_name: Optional[str] = Field(None, description="Region for the S3-compatible service.")
    addressing_style: Optional[Literal["auto", "virtual", "path"]] = Field(
        None, description="S3 addressing style (auto, virtual-hosted, or path-style)."
    )


class StorageConfig(BaseModel):
    local_storage: StorageConfigBase = Field(
        default_factory=StorageConfigBase, description="Local filesystem destination for reports."
    )
    google_cloud_storage: Optional[GoogleCloudStorageConfig] = Field(
        None, description="Optional Google Cloud Storage destination."
    )
    simple_storage_service: Optional[SimpleStorageServiceConfig] = Field(
        None, description="Optional S3-compatible storage destination."
    )


class RequestLifecycleMetricsReportConfig(BaseModel):
    summary: Optional[bool] = Field(True, description="Emit an aggregate summary across all stages.")
    per_stage: Optional[bool] = Field(True, description="Emit one report per load stage.")
    per_request: Optional[bool] = Field(False, description="Emit one row per individual request (large output).")
    per_adapter: Optional[bool] = Field(True, description="Emit one report per LoRA adapter.")
    per_adapter_stage: Optional[bool] = Field(False, description="Emit one report per (adapter, stage) pair.")
    percentiles: List[float] = Field(
        default=[0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9],
        description="Latency percentiles to compute and emit.",
    )


class PrometheusMetricsReportConfig(BaseModel):
    summary: Optional[bool] = Field(True, description="Emit an aggregate Prometheus summary across all stages.")
    per_stage: Optional[bool] = Field(False, description="Emit per-stage Prometheus reports.")


class SessionLifecycleReportConfig(BaseModel):
    summary: Optional[bool] = Field(True, description="Emit aggregate session metrics across all stages.")
    per_stage: Optional[bool] = Field(True, description="Emit per-stage session metrics.")
    per_session: Optional[bool] = Field(False, description="Emit one row per individual session (large output).")


class GoodputConfig(BaseModel):
    constraints: Dict[str, float] = Field(
        default={}, description="Mapping of SLO metric name to threshold; requests meeting all thresholds count as good."
    )


class ReportConfig(BaseModel):
    request_lifecycle: RequestLifecycleMetricsReportConfig = Field(
        default_factory=RequestLifecycleMetricsReportConfig,
        description="Request-lifecycle (latency/throughput) reporting options.",
    )
    prometheus: Optional[PrometheusMetricsReportConfig] = Field(
        default_factory=PrometheusMetricsReportConfig, description="Prometheus-scraped metrics reporting options."
    )
    session_lifecycle: SessionLifecycleReportConfig = Field(
        default_factory=SessionLifecycleReportConfig,
        description="Session-lifecycle reporting options (for session-based load types).",
    )
    goodput: Optional[GoodputConfig] = Field(
        None, description="Optional goodput evaluation; counts requests meeting all SLO constraints."
    )


class PrometheusClientConfig(BaseModel):
    scrape_interval: int = Field(15, description="Prometheus scrape interval in seconds; used to align query windows.")
    url: Optional[HttpUrl] = Field(None, description="Prometheus HTTP endpoint. Mutually exclusive with `google_managed`.")
    filters: List[str] = Field(default=[], description="PromQL label filters applied to every query.")
    google_managed: bool = Field(
        False, description="Use Google Managed Prometheus instead of `url`. Mutually exclusive with `url`."
    )

    @model_validator(mode="after")
    def check_exclusive_fields(self) -> "PrometheusClientConfig":
        if bool(self.url) == bool(self.google_managed):
            raise ValueError("Exactly one of 'url' or 'google_managed' must be set.")
        return self


class MetricsClientConfig(BaseModel):
    type: MetricsClientType = Field(..., description="Metrics backend used to collect server-side metrics.")
    prometheus: Optional[PrometheusClientConfig] = Field(
        None, description="Prometheus client configuration (when type=prometheus)."
    )


class ModelServerClientConfig(BaseModel):
    type: ModelServerType = Field(
        ModelServerType.VLLM, description="Model server flavor; controls server-specific metric mappings."
    )
    model_name: Optional[str] = Field(
        None, description="Model identifier sent on each request. May be auto-detected via /v1/models."
    )
    base_url: str = Field(..., description="Base URL of the model server, e.g. http://0.0.0.0:8000.")
    ignore_eos: bool = Field(True, description="Ask the server to ignore EOS so output length is governed by max_tokens.")
    api_key: Optional[str] = Field(None, description="Bearer token sent as Authorization header.")
    cert_path: Optional[str] = Field(None, description="Path to a TLS client certificate (PEM) for mTLS.")
    key_path: Optional[str] = Field(None, description="Path to a TLS client private key (PEM) for mTLS.")


class CustomTokenizerConfig(BaseModel):
    pretrained_model_name_or_path: Optional[str] = Field(None, description="HuggingFace tokenizer name or local path.")
    trust_remote_code: Optional[bool] = Field(
        None, description="Forwarded to HuggingFace tokenizer loading; required by some custom tokenizers."
    )
    token: Optional[str] = Field(None, description="HuggingFace auth token used to download gated tokenizers.")


class Config(BaseModel):
    api: APIConfig = Field(default_factory=APIConfig, description="API-level request behavior (type, streaming, SLO headers).")
    data: DataConfig = Field(default_factory=DataConfig, description="Workload data generation configuration.")
    load: LoadConfig = Field(default_factory=LoadConfig, description="Load shaping and worker configuration.")
    metrics: Optional[MetricsClientConfig] = Field(
        None, description="Server-side metrics collection (e.g. Prometheus scrape)."
    )
    report: ReportConfig = Field(default_factory=ReportConfig, description="Reporting outputs to emit at the end of a run.")
    storage: Optional[StorageConfig] = Field(default_factory=StorageConfig, description="Destinations for written reports.")
    server: Optional[ModelServerClientConfig] = Field(None, description="Model server connection details.")
    tokenizer: Optional[CustomTokenizerConfig] = Field(
        None, description="Tokenizer used to compute token-level metrics offline."
    )
    circuit_breakers: Optional[List[CircuitBreakerConfig]] = Field(
        None, description="Declarative circuit breakers that can stop the run on metric thresholds."
    )

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
