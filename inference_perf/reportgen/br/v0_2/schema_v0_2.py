# Vendored from llm-d/llm-d-benchmark @ e0b95468e27eff59e09d70275044d6c8a17890b7
# Source: llmdbenchmark/analysis/benchmark_report/schema_v0_2.py
# Do not edit; resync from upstream when bumping the BR0.2 schema.
"""
Benchmark report v0.2
"""

import datetime
from typing import Any, Annotated
from enum import StrEnum, auto

from pydantic import BaseModel, ConfigDict, Discriminator, Field, model_validator

from .base import (
    BenchmarkReport,
    Units,
    UNITS_QUANTITY,
    UNITS_PORTION,
    UNITS_TIME,
    UNITS_MEMORY,
    UNITS_GEN_LATENCY,
    UNITS_GEN_THROUGHPUT,
    UNITS_REQUEST_THROUGHPUT,
    UNITS_POWER,
)
from .schema_v0_2_components import COMPONENTS

# BenchmarkReport schema version
VERSION = "0.2"

# Default model_config to apply to Pydantic classes
MODEL_CONFIG = ConfigDict(
    extra="forbid",  # Do not allow fields that are not part of this schema
    use_attribute_docstrings=True,  # Use docstrings for JSON schema
    populate_by_name=False,  # Must use alias name, not internal field name
    validate_assignment=True,  # Validate field assignment after init
)

###############################################################################
# Stack details
###############################################################################


class ComponentMetadata(BaseModel):
    """Component metadata."""

    model_config = MODEL_CONFIG.copy()

    schema_version: str = "0.0.1"
    """Schema version for the component."""
    label: str
    """Unique name for this particular component."""
    cfg_id: str
    """Configuration ID, a hash of this component's configuration."""
    description: str | None = None
    """Description of this component."""


class ComponentNative(BaseModel):
    """Component configuration in native format."""

    model_config = MODEL_CONFIG.copy()

    args: dict[str, Any] | None = None
    """Command line arguments."""
    envars: dict[str, Any] | None = None
    """Environment variables."""
    config: Any | None = None
    """Configuration file details."""


class Component(BaseModel):
    """Component details."""

    model_config = MODEL_CONFIG.copy()

    metadata: ComponentMetadata
    """Component metadata."""
    standardized: Annotated[COMPONENTS, Discriminator("kind")]
    """Component configuration details in standardized format."""
    native: ComponentNative
    """Component configuration in native format."""


###############################################################################
# Experimental workload
###############################################################################


class LoadMetadata(BaseModel):
    """Workload metadata."""

    model_config = MODEL_CONFIG.copy()

    schema_version: str = "0.0.1"
    """Version of workload description schema."""
    cfg_id: str | None = None
    """Configuration ID, a hash of the workload configuration."""
    description: str | None = None
    """Descriptin of workload."""


class Distribution(StrEnum):
    """Distribution type.

    Attributes
        FIXED: str
            Length is a fixed value.
        GAUSSIAN: str
            Gaussian distribution, with a mean and standard deviation.
        UNIFORM: str
            Uniform distribution between a minimum and maximum value.
        OTHER: str
            An otherwise undefined distribution.
    """

    FIXED = auto()
    GAUSSIAN = auto()
    UNIFORM = auto()
    OTHER = auto()


class SequenceLength(BaseModel):
    """Sequence length."""

    model_config = MODEL_CONFIG.copy()

    distribution: Distribution
    """Sequence length distribution type."""
    value: int | float = Field(..., ge=0)
    """Primary value."""
    std_dev: float | None = Field(None, ge=0)
    """Standard deviation (if Gaussian)."""
    min: int | None = Field(None, ge=0)
    """Minimum value."""
    max: int | None = Field(None, ge=1)
    """Maximum value."""


class LoadPrefix(BaseModel):
    """Input sequence prefix details."""

    model_config = MODEL_CONFIG.copy()

    prefix_len: SequenceLength
    """Length of common prefix."""
    num_groups: int = Field(..., ge=1)
    """Number of groups of "users" that share common prefixes."""
    num_users_per_group: int = Field(..., ge=1)
    """Number of users per group."""
    num_prefixes: int = Field(..., ge=1)
    """Number of common prefixes within a group."""


class MultiTurn(BaseModel):
    """Multi-turn request configuration."""

    model_config = MODEL_CONFIG.copy()

    enabled: bool = True
    """Multi-turn requests are enabled."""
    max_turns: SequenceLength | None = None
    """Maximum number of requests per session."""


class LoadSource(StrEnum):
    """How input tokens are generated.

    Attributes
        RANDOM: str
            Tokens are randomly generated from vocabulary.
        SAMPLED: str
            Tokens are sampled from some data.
        UNKNOWN: str
            The source of tokens used is unknown.
    """

    RANDOM = auto()
    SAMPLED = auto()
    UNKNOWN = auto()


class LoadStandardized(BaseModel):
    """Workload generator configuration details in standardized format."""

    model_config = MODEL_CONFIG.copy()

    tool: str
    """Particular tool used for this component."""
    tool_version: str
    """Version of tool."""
    parallelism: int = Field(1, ge=1)
    """Number of parallel workload generators."""
    source: LoadSource
    """How input tokens are generated."""
    stage: int = Field(0, ge=0)
    """Workload stage number (if multi-stage)."""
    input_seq_len: SequenceLength
    """Input sequence length."""
    output_seq_len: SequenceLength | None = None
    """Output sequence length (if enforced)."""
    prefix: LoadPrefix | None = None
    """Input sequence prefix details."""
    multi_turn: MultiTurn | None = None
    """Multi-turn request configuration."""
    rate_qps: float | None = Field(None, gt=0)
    """Request rate, in queries per second."""
    concurrency: int | float | None = Field(None, ge=1)
    """Request concurrency."""

    @model_validator(mode="after")
    def check_concurrency(self):
        """Concurrency must be an integer, unless value is infinite."""
        if isinstance(self.concurrency, float):
            if self.concurrency != float("inf"):
                raise ValueError("concurrency must be integer or .inf")
        return self


class LoadNative(BaseModel):
    """Workload generator configuration in native format."""

    model_config = MODEL_CONFIG.copy()

    args: dict[str, Any] | None = None
    """Command line arguments."""
    envars: dict[str, Any] | None = None
    """Environment variables."""
    config: Any | None = None
    """Configuration file details."""


# ------------------------------------------------------------------------------
# Root for load
# ------------------------------------------------------------------------------


class Load(BaseModel):
    """Experimental workload details."""

    model_config = MODEL_CONFIG.copy()

    metadata: LoadMetadata
    """Workload metadata."""
    standardized: LoadStandardized
    """Workload generator configuration details in standardized format."""
    native: LoadNative
    """Workload generator configuration in native format."""


###############################################################################
# Request-level metrics
###############################################################################

# ------------------------------------------------------------------------------
# Aggregate request performance
# ------------------------------------------------------------------------------


class Statistics(BaseModel):
    """Statistical information about a property."""

    units: Units
    mean: float
    mode: float | int | None = None
    stddev: float | None = Field(None, ge=0)
    min: float | int | None = None
    p0p1: float | int | None = None
    p1: float | int | None = None
    p5: float | int | None = None
    p10: float | int | None = None
    p25: float | int | None = None
    p50: float | int | None = None  # This is the same as median
    p75: float | int | None = None
    p90: float | int | None = None
    p95: float | int | None = None
    p99: float | int | None = None
    p99p9: float | int | None = None
    max: float | int | None = None


class AggregateRequests(BaseModel):
    """Request statistics."""

    model_config = MODEL_CONFIG.copy()

    total: int = Field(..., ge=0)
    """Total number of requests sent."""
    failures: int | None = Field(None, ge=0)
    """Number of requests which responded with an error."""
    incomplete: int | None = Field(None, ge=0)
    """Number of requests which were not completed."""
    input_length: Statistics | None = None
    """Input sequence length."""
    output_length: Statistics | None = None
    """Output sequence length."""

    @model_validator(mode="after")
    def check_units(self):
        if self.input_length and self.input_length.units not in UNITS_QUANTITY:
            raise ValueError(f'Invalid units "{self.input_length.units}", must be one of: {" ".join(UNITS_QUANTITY)}')
        if self.output_length and self.output_length.units not in UNITS_QUANTITY:
            raise ValueError(f'Invalid units "{self.output_length.units}", must be one of: {" ".join(UNITS_QUANTITY)}')
        return self


class AggregateLatency(BaseModel):
    """Aggregate response latency performance metrics."""

    model_config = MODEL_CONFIG.copy()

    time_to_first_token: Statistics | None = None
    """Time to generate the first token (TTFT)."""
    normalized_time_per_output_token: Statistics | None = None
    """Typical time to generate an output token, including first (NTPOT)."""
    # NOTE: TPOT and ITL can be terms for the same quantity, but can also have
    # different meanings within a tool. Care must be taken when choosing which
    # quantity to use, especially when comparing results across different tools.
    #
    # From GKE
    # https://cloud.google.com/kubernetes-engine/docs/concepts/machine-learning/inference
    # TPOT is calculated across the entire request
    # TPOT = (request_latency - time_to_first_token) / (total_output_tokens - 1)
    # ITL is measured between consecutive output tokens, and those results
    # aggregated to produce statistics.
    #
    # vLLM's benchmarking tools
    # https://github.com/vllm-project/vllm/issues/6531#issuecomment-2684695288
    # Obtaining TPOT statistics appears consistent with GKE definition, but
    # ITL is calculated across multiple requests.
    time_per_output_token: Statistics | None = None
    """Time to generate an output token, excluding first (TPOT, may differ from ITL depending on tool)."""
    inter_token_latency: Statistics | None = None
    """Latency between generated tokens, excluding first (ITL, may differ from TPOT depending on tool)."""
    request_latency: Statistics | None = None
    """End-to-end request latency."""

    @model_validator(mode="after")
    def check_units(self):
        if self.time_to_first_token and self.time_to_first_token.units not in UNITS_TIME:
            raise ValueError(f'Invalid units "{self.time_to_first_token.units}", must be one of: {" ".join(UNITS_TIME)}')
        if self.normalized_time_per_output_token and self.normalized_time_per_output_token.units not in UNITS_GEN_LATENCY:
            raise ValueError(
                f'Invalid units "{self.normalized_time_per_output_token.units}", must be one of: {" ".join(UNITS_GEN_LATENCY)}'
            )
        if self.time_per_output_token and self.time_per_output_token.units not in UNITS_GEN_LATENCY:
            raise ValueError(
                f'Invalid units "{self.time_per_output_token.units}", must be one of: {" ".join(UNITS_GEN_LATENCY)}'
            )
        if self.inter_token_latency and self.inter_token_latency.units not in UNITS_GEN_LATENCY:
            raise ValueError(
                f'Invalid units "{self.inter_token_latency.units}", must be one of: {" ".join(UNITS_GEN_LATENCY)}'
            )
        if self.request_latency and self.request_latency.units not in UNITS_TIME:
            raise ValueError(f'Invalid units "{self.request_latency.units}", must be one of: {" ".join(UNITS_TIME)}')
        return self


class AggregateThroughput(BaseModel):
    """Aggregate response throughput performance metrics."""

    model_config = MODEL_CONFIG.copy()

    input_token_rate: Statistics | None = None
    """Input token rate."""
    output_token_rate: Statistics | None = None
    """Output token rate."""
    total_token_rate: Statistics | None = None
    """Total token rate (input + output)."""
    request_rate: Statistics | None = None
    """Request (query) processing rate."""

    @model_validator(mode="after")
    def check_units(self):
        if self.input_token_rate and self.input_token_rate.units not in UNITS_GEN_THROUGHPUT:
            raise ValueError(
                f'Invalid units "{self.input_token_rate.units}", must be one of: {" ".join(UNITS_GEN_THROUGHPUT)}'
            )
        if self.output_token_rate and self.output_token_rate.units not in UNITS_GEN_THROUGHPUT:
            raise ValueError(
                f'Invalid units "{self.output_token_rate.units}", must be one of: {" ".join(UNITS_GEN_THROUGHPUT)}'
            )
        if self.total_token_rate and self.total_token_rate.units not in UNITS_GEN_THROUGHPUT:
            raise ValueError(
                f'Invalid units "{self.total_token_rate.units}", must be one of: {" ".join(UNITS_GEN_THROUGHPUT)}'
            )
        if self.request_rate and self.request_rate.units not in UNITS_REQUEST_THROUGHPUT:
            raise ValueError(
                f'Invalid units "{self.request_rate.units}", must be one of: {" ".join(UNITS_REQUEST_THROUGHPUT)}'
            )
        return self


class AggregateRequestPerformance(BaseModel):
    """Aggregate performance metrics."""

    model_config = MODEL_CONFIG.copy()

    requests: AggregateRequests | None = None
    """Aggregate request details."""
    latency: AggregateLatency | None = None
    """Aggregate response latency performance metrics."""
    throughput: AggregateThroughput | None = None
    """Aggregate response throughput performance metrics."""


# ------------------------------------------------------------------------------
# Time series request performance
# ------------------------------------------------------------------------------


class TimeSeriesPoint(BaseModel):
    """Time series data point."""

    model_config = MODEL_CONFIG.copy()

    ts: datetime.datetime
    """ISO-8601 timestamp."""
    value: str | float | int | bool | None = None
    """Value for datapoint."""
    mean: float | None = None
    mode: float | int | None = None
    stddev: float | None = Field(None, ge=0)
    min: float | int | None = None
    p0p1: float | int | None = None
    p1: float | int | None = None
    p5: float | int | None = None
    p10: float | int | None = None
    p25: float | int | None = None
    p50: float | int | None = None  # This is the same as median
    p75: float | int | None = None
    p90: float | int | None = None
    p95: float | int | None = None
    p99: float | int | None = None
    p99p9: float | int | None = None
    max: float | int | None = None


class TimeSeriesData(BaseModel):
    """Time series data."""

    model_config = MODEL_CONFIG.copy()

    units: Units
    """Units for time series."""
    series: list[TimeSeriesPoint]
    """Time series data points."""


class TimeSeriesLatency(BaseModel):
    """Time series latency metrics."""

    model_config = MODEL_CONFIG.copy()

    time_to_first_token: TimeSeriesData | None = None
    """Time to generate the first token (TTFT)."""
    normalized_time_per_output_token: TimeSeriesData | None = None
    """Typical time to generate an output token, including first (NTPOT)."""
    time_per_output_token: TimeSeriesData | None = None
    """Time to generate an output token, excluding first (TPOT, may differ from ITL depending on tool)."""
    inter_token_latency: TimeSeriesData | None = None
    """Latency between generated tokens, excluding first (ITL, may differ from TPOT depending on tool)."""
    request_latency: TimeSeriesData | None = None
    """End-to-end request latency."""

    @model_validator(mode="after")
    def check_units(self):
        if self.time_to_first_token and self.time_to_first_token.units not in UNITS_TIME:
            raise ValueError(f'Invalid units "{self.time_to_first_token.units}", must be one of: {" ".join(UNITS_TIME)}')
        if self.normalized_time_per_output_token and self.normalized_time_per_output_token.units not in UNITS_GEN_LATENCY:
            raise ValueError(
                f'Invalid units "{self.normalized_time_per_output_token.units}", must be one of: {" ".join(UNITS_GEN_LATENCY)}'
            )
        if self.time_per_output_token and self.time_per_output_token.units not in UNITS_GEN_LATENCY:
            raise ValueError(
                f'Invalid units "{self.time_per_output_token.units}", must be one of: {" ".join(UNITS_GEN_LATENCY)}'
            )
        if self.inter_token_latency and self.inter_token_latency.units not in UNITS_GEN_LATENCY:
            raise ValueError(
                f'Invalid units "{self.inter_token_latency.units}", must be one of: {" ".join(UNITS_GEN_LATENCY)}'
            )
        if self.request_latency and self.request_latency.units not in UNITS_TIME:
            raise ValueError(f'Invalid units "{self.request_latency.units}", must be one of: {" ".join(UNITS_TIME)}')
        return self


class TimeSeriesThroughput(BaseModel):
    """Time series throughput metrics."""

    model_config = MODEL_CONFIG.copy()

    units: Units = Units.TOKEN_PER_S

    input_token_rate: TimeSeriesData | None = None
    """Input token rate."""
    output_token_rate: TimeSeriesData | None = None
    """Output token rate."""
    total_token_rate: TimeSeriesData | None = None
    """Total token rate (input + output)."""
    request_rate: TimeSeriesData | None = None
    """Request (query) processing rate."""

    @model_validator(mode="after")
    def check_units(self):
        if self.input_token_rate and self.input_token_rate.units not in UNITS_GEN_THROUGHPUT:
            raise ValueError(
                f'Invalid units "{self.input_token_rate.units}", must be one of: {" ".join(UNITS_GEN_THROUGHPUT)}'
            )
        if self.output_token_rate and self.output_token_rate.units not in UNITS_GEN_THROUGHPUT:
            raise ValueError(
                f'Invalid units "{self.output_token_rate.units}", must be one of: {" ".join(UNITS_GEN_THROUGHPUT)}'
            )
        if self.total_token_rate and self.total_token_rate.units not in UNITS_GEN_THROUGHPUT:
            raise ValueError(
                f'Invalid units "{self.total_token_rate.units}", must be one of: {" ".join(UNITS_GEN_THROUGHPUT)}'
            )
        if self.request_rate and self.request_rate.units not in UNITS_REQUEST_THROUGHPUT:
            raise ValueError(
                f'Invalid units "{self.request_rate.units}", must be one of: {" ".join(UNITS_REQUEST_THROUGHPUT)}'
            )
        return self


class TimeSeriesRequestPerformance(BaseModel):
    """Time series performance metrics."""

    model_config = MODEL_CONFIG.copy()

    latency: TimeSeriesLatency | None = None
    """Time series latency metrics."""
    throughput: TimeSeriesThroughput | None = None
    """Time series throughput metrics."""


# ------------------------------------------------------------------------------
# Root for request performance
# ------------------------------------------------------------------------------


class RequestPerformance(BaseModel):
    """Request-level performance metrics."""

    model_config = MODEL_CONFIG.copy()

    aggregate: AggregateRequestPerformance | None = None
    """Aggregate performance metrics."""
    time_series: TimeSeriesRequestPerformance | None = None
    """Time series metrics."""


###############################################################################
# Observability metrics
###############################################################################


class ResourceMetrics(BaseModel):
    """Resource utilization metrics for a component."""

    model_config = MODEL_CONFIG.copy()

    kv_cache_usage: Statistics | None = None
    """KV cache usage percentage."""
    cache_hit_rate: Statistics | None = None
    """Prefix cache hit rate percentage."""
    gpu_cache_usage: Statistics | None = None
    """GPU cache usage percentage."""
    cpu_cache_usage: Statistics | None = None
    """CPU cache usage percentage."""
    gpu_memory_usage: Statistics | None = None
    """GPU memory usage."""
    cpu_memory_usage: Statistics | None = None
    """CPU/RAM memory usage."""
    storage_usage: Statistics | None = None
    """Storage usage."""
    gpu_utilization: Statistics | None = None
    """GPU compute utilization percentage."""
    cpu_utilization: Statistics | None = None
    """CPU utilization percentage."""
    power_consumption: Statistics | None = None
    """Power consumption."""
    running_requests: Statistics | None = None
    """Number of currently running requests."""
    waiting_requests: Statistics | None = None
    """Number of requests waiting in queue."""
    swapped_requests: Statistics | None = None
    """Number of swapped out requests."""
    preemptions: Statistics | None = None
    """Number of request preemptions due to memory pressure."""

    @model_validator(mode="after")
    def check_units(self):
        if self.kv_cache_usage and self.kv_cache_usage.units not in UNITS_PORTION:
            raise ValueError(f'Invalid units "{self.kv_cache_usage.units}", must be one of: {" ".join(UNITS_PORTION)}')
        if self.cache_hit_rate and self.cache_hit_rate.units not in UNITS_PORTION:
            raise ValueError(f'Invalid units "{self.cache_hit_rate.units}", must be one of: {" ".join(UNITS_PORTION)}')
        if self.gpu_cache_usage and self.gpu_cache_usage.units not in UNITS_PORTION:
            raise ValueError(f'Invalid units "{self.gpu_cache_usage.units}", must be one of: {" ".join(UNITS_PORTION)}')
        if self.cpu_cache_usage and self.cpu_cache_usage.units not in UNITS_PORTION:
            raise ValueError(f'Invalid units "{self.cpu_cache_usage.units}", must be one of: {" ".join(UNITS_PORTION)}')
        if self.gpu_memory_usage and self.gpu_memory_usage.units not in UNITS_MEMORY:
            raise ValueError(f'Invalid units "{self.gpu_memory_usage.units}", must be one of: {" ".join(UNITS_MEMORY)}')
        if self.cpu_memory_usage and self.cpu_memory_usage.units not in UNITS_MEMORY:
            raise ValueError(f'Invalid units "{self.cpu_memory_usage.units}", must be one of: {" ".join(UNITS_MEMORY)}')
        if self.storage_usage and self.storage_usage.units not in UNITS_MEMORY:
            raise ValueError(f'Invalid units "{self.storage_usage.units}", must be one of: {" ".join(UNITS_MEMORY)}')
        if self.gpu_utilization and self.gpu_utilization.units not in UNITS_PORTION:
            raise ValueError(f'Invalid units "{self.gpu_utilization.units}", must be one of: {" ".join(UNITS_PORTION)}')
        if self.cpu_utilization and self.cpu_utilization.units not in UNITS_PORTION:
            raise ValueError(f'Invalid units "{self.cpu_utilization.units}", must be one of: {" ".join(UNITS_PORTION)}')
        if self.power_consumption and self.power_consumption.units not in UNITS_POWER:
            raise ValueError(f'Invalid units "{self.power_consumption.units}", must be one of: {" ".join(UNITS_POWER)}')
        if self.running_requests and self.running_requests.units not in UNITS_QUANTITY:
            raise ValueError(f'Invalid units "{self.running_requests.units}", must be one of: {" ".join(UNITS_QUANTITY)}')
        if self.waiting_requests and self.waiting_requests.units not in UNITS_QUANTITY:
            raise ValueError(f'Invalid units "{self.waiting_requests.units}", must be one of: {" ".join(UNITS_QUANTITY)}')
        if self.swapped_requests and self.swapped_requests.units not in UNITS_QUANTITY:
            raise ValueError(f'Invalid units "{self.swapped_requests.units}", must be one of: {" ".join(UNITS_QUANTITY)}')
        if self.preemptions and self.preemptions.units not in UNITS_QUANTITY:
            raise ValueError(f'Invalid units "{self.preemptions.units}", must be one of: {" ".join(UNITS_QUANTITY)}')
        return self


class TimeSeriesResourceMetrics(BaseModel):
    """Time series resource utilization metrics."""

    model_config = MODEL_CONFIG.copy()

    kv_cache_usage: TimeSeriesData | None = None
    """KV cache usage percentage over time."""
    gpu_cache_usage: TimeSeriesData | None = None
    """GPU cache usage percentage over time."""
    cpu_cache_usage: TimeSeriesData | None = None
    """CPU cache usage percentage over time."""
    gpu_memory_usage: TimeSeriesData | None = None
    """GPU memory usage over time."""
    cpu_memory_usage: TimeSeriesData | None = None
    """CPU/RAM memory usage over time."""
    storage_usage: TimeSeriesData | None = None
    """Storage usage over time."""
    gpu_utilization: TimeSeriesData | None = None
    """GPU compute utilization percentage over time."""
    cpu_utilization: TimeSeriesData | None = None
    """CPU utilization percentage over time."""
    power_consumption: TimeSeriesData | None = None
    """Power consumption over time."""

    @model_validator(mode="after")
    def check_units(self):
        if self.kv_cache_usage and self.kv_cache_usage.units not in UNITS_PORTION:
            raise ValueError(f'Invalid units "{self.kv_cache_usage.units}", must be one of: {" ".join(UNITS_PORTION)}')
        if self.gpu_cache_usage and self.gpu_cache_usage.units not in UNITS_PORTION:
            raise ValueError(f'Invalid units "{self.gpu_cache_usage.units}", must be one of: {" ".join(UNITS_PORTION)}')
        if self.cpu_cache_usage and self.cpu_cache_usage.units not in UNITS_PORTION:
            raise ValueError(f'Invalid units "{self.cpu_cache_usage.units}", must be one of: {" ".join(UNITS_PORTION)}')
        if self.gpu_memory_usage and self.gpu_memory_usage.units not in UNITS_MEMORY:
            raise ValueError(f'Invalid units "{self.gpu_memory_usage.units}", must be one of: {" ".join(UNITS_MEMORY)}')
        if self.cpu_memory_usage and self.cpu_memory_usage.units not in UNITS_MEMORY:
            raise ValueError(f'Invalid units "{self.cpu_memory_usage.units}", must be one of: {" ".join(UNITS_MEMORY)}')
        if self.storage_usage and self.storage_usage.units not in UNITS_MEMORY:
            raise ValueError(f'Invalid units "{self.storage_usage.units}", must be one of: {" ".join(UNITS_MEMORY)}')
        if self.gpu_utilization and self.gpu_utilization.units not in UNITS_PORTION:
            raise ValueError(f'Invalid units "{self.gpu_utilization.units}", must be one of: {" ".join(UNITS_PORTION)}')
        if self.cpu_utilization and self.cpu_utilization.units not in UNITS_PORTION:
            raise ValueError(f'Invalid units "{self.cpu_utilization.units}", must be one of: {" ".join(UNITS_PORTION)}')
        if self.power_consumption and self.power_consumption.units not in UNITS_POWER:
            raise ValueError(f'Invalid units "{self.power_consumption.units}", must be one of: {" ".join(UNITS_POWER)}')
        return self


class ComponentObservability(BaseModel):
    """Observability metrics for a specific component."""

    model_config = MODEL_CONFIG.copy()

    component_label: str
    """References the component's label from scenario.stack[].metadata.label"""
    replica_id: str | None = None
    """Specific replica/pod identifier (optional, for per-replica metrics)."""
    aggregate: ResourceMetrics | None = None
    """Aggregate resource metrics."""
    time_series: TimeSeriesResourceMetrics | None = None
    """Time series resource metrics."""
    raw_data_path: str | None = None
    """Path to raw metrics data files."""
    graph_path: str | None = None
    """Path to visualization/graph of metrics."""


# ------------------------------------------------------------------------------
# Pod startup times
# ------------------------------------------------------------------------------


class PodStartupInfo(BaseModel):
    """Startup timing information for a single pod."""

    model_config = MODEL_CONFIG.copy()

    name: str
    """Pod name."""
    model: str | None = None
    """Model identifier."""
    role: str | None = None
    """Pod role (e.g., prefill, decode, aggregate)."""
    node: str | None = None
    """Node the pod was scheduled on."""
    creation_timestamp: datetime.datetime | None = None
    """Timestamp when the pod was created."""
    ready_timestamp: datetime.datetime | None = None
    """Timestamp when the pod became ready."""
    startup_seconds: float | None = Field(None, ge=0)
    """Time in seconds from creation to ready."""


class PodStartupTimes(BaseModel):
    """Pod startup times collected during or before the benchmark."""

    model_config = MODEL_CONFIG.copy()

    collected_at: datetime.datetime | None = None
    """Timestamp when startup times were collected."""
    pods: list[PodStartupInfo] | None = None
    """Per-pod startup information."""
    aggregate: Statistics | None = None
    """Aggregate statistics (mean, p50, p99, etc.) across all pod startup times."""
    graph_path: str | None = None
    """Path to pod startup times visualization."""


# ------------------------------------------------------------------------------
# Replica status
# ------------------------------------------------------------------------------


class ControllerReplicaStatus(BaseModel):
    """Replica status for a single controller (Deployment or StatefulSet)."""

    model_config = MODEL_CONFIG.copy()

    kind: str
    """Controller kind (e.g., Deployment, StatefulSet)."""
    name: str
    """Controller name."""
    model: str | None = None
    """Model identifier."""
    role: str | None = None
    """Role (e.g., prefill, decode)."""
    desired_replicas: int = Field(..., ge=0)
    """Number of desired replicas."""
    available_replicas: int = Field(..., ge=0)
    """Number of available replicas."""
    ready_replicas: int = Field(..., ge=0)
    """Number of ready replicas."""
    updated_replicas: int | None = Field(None, ge=0)
    """Number of updated replicas."""


class ReplicaStatusSnapshot(BaseModel):
    """A single point-in-time replica status snapshot."""

    model_config = MODEL_CONFIG.copy()

    timestamp: datetime.datetime | None = None
    """Timestamp when this snapshot was taken."""
    namespace: str | None = None
    """Kubernetes namespace."""
    controllers: list[ControllerReplicaStatus] | None = None
    """Per-controller replica status at this point in time."""


class ReplicaStatus(BaseModel):
    """Replica status across controllers, with optional time series and aggregate."""

    model_config = MODEL_CONFIG.copy()

    namespace: str | None = None
    """Kubernetes namespace."""
    timestamp: datetime.datetime | None = None
    """Timestamp of the latest snapshot."""
    controllers: list[ControllerReplicaStatus] | None = None
    """Per-controller replica status (latest snapshot)."""
    time_series: list[ReplicaStatusSnapshot] | None = None
    """Time series of replica status snapshots collected during the benchmark."""
    aggregate_ready_replicas: Statistics | None = None
    """Aggregate statistics (min, max, mean, etc.) of total ready replicas over time."""
    graph_path: str | None = None
    """Path to replica status visualization."""


# ------------------------------------------------------------------------------
# Root for observability
# ------------------------------------------------------------------------------


class Observability(BaseModel):
    """Observability metrics."""

    model_config = MODEL_CONFIG.copy()
    # Keep permissive — real reports include ad-hoc metric keys
    # (e.g. vllm_kv_cache_usage_perc, epp_dispatch_latency) that are
    # not yet formalized in the schema.
    model_config["extra"] = "allow"
    components: list[ComponentObservability] | None = None
    """Per-component observability metrics."""
    drop_rate: Statistics | None = None
    """Request drop rate."""
    pod_startup_times: PodStartupTimes | None = None
    """Pod startup times collected during or before the benchmark."""
    replica_status: ReplicaStatus | None = None
    """Replica status across controllers at a point in time."""

    @model_validator(mode="after")
    def check_units(self):
        if self.drop_rate and self.drop_rate.units not in UNITS_PORTION:
            raise ValueError(f'Invalid units "{self.drop_rate.units}", must be one of: {" ".join(UNITS_PORTION)}')
        return self


###############################################################################
# Component health
###############################################################################


class ReplicaHealth(BaseModel):
    """Health information for a specific replica."""

    model_config = MODEL_CONFIG.copy()

    replica_id: str
    """Unique identifier for this replica (e.g., pod name)."""
    restarts: int | None = Field(None, ge=0)
    """Number of times this replica restarted during the benchmark."""
    healthy: bool | None = None
    """Healthy status at completion of benchmark."""
    logs: str | None = None
    """Reference to logs for this specific replica."""


class ComponentHealth(BaseModel):
    """Health and reliability metrics for a component during the benchmark."""

    model_config = MODEL_CONFIG.copy()

    component_label: str
    """References the component's label from scenario.stack[].metadata.label"""
    total_restarts: int | None = Field(None, ge=0)
    """Total restarts across all replicas during benchmark."""
    failed_replicas: int | None = Field(None, ge=0)
    """Number of replicas that hand one or more failures during benchmark."""
    replica_health: list[ReplicaHealth] | None = None
    """Per-replica health details."""


###############################################################################
# Benchmark Report top-level classes
###############################################################################


class RunTime(BaseModel):
    """Time details of experiment."""

    model_config = MODEL_CONFIG.copy()

    start: datetime.datetime | None = None
    """ISO-8601 timestamp for experiment start."""
    end: datetime.datetime | None = None
    """ISO-8601 timestamp for experiment end."""
    duration: str | None = None
    """ISO-8601 duration for experiment."""


class Run(BaseModel):
    """Benchmark run details."""

    model_config = MODEL_CONFIG.copy()

    uid: str
    """Unique ID for this specific benchmark report."""
    eid: str | None = None
    """Experiment ID, common across benchmark reports from a particular experiment."""
    cid: str | None = None
    """Cluster ID, unique to a particular cluster."""
    pid: str | None = None
    """Pod ID, unique to a workload generating and/or data collecting pod."""
    time: RunTime | None = None
    """Time details of experiment."""
    user: str | None = None
    """Username that executed experiment."""
    description: str | None = None
    """User-provided description of the experiment."""
    keywords: list[str] | None = None
    """User-provided keywords/tags for the experiment."""


class Scenario(BaseModel):
    """Benchmark run details."""

    model_config = MODEL_CONFIG.copy()

    stack: list[Component] | None = None
    """List of components used to build the stack."""
    load: Load | None = None
    """Experimental workload details."""


class Results(BaseModel):
    """Benchmark run details."""

    model_config = MODEL_CONFIG.copy()

    request_performance: RequestPerformance | None = None
    """Request-level performance metrics."""

    observability: Observability | None = None
    """Observability metrics."""

    profiling: Any | None = None
    """Profiling results."""

    component_health: list[ComponentHealth] | None = None
    """Component health and reliability metrics during benchmark."""


# ------------------------------------------------------------------------------
# Root class for benchmark report
# ------------------------------------------------------------------------------


class BenchmarkReportV02(BenchmarkReport):
    """Base class for a benchmark report."""

    model_config = MODEL_CONFIG.copy()
    model_config["title"] = "Benchmark Report v0.2"

    version: str = VERSION
    """Version of the schema."""
    run: Run
    """Benchmark run details."""
    scenario: Scenario | None = None
    """Stack configuration and workload details of experiment"""
    results: Results
    """Experiment results."""
