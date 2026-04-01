# This is an automatically generated file, please do not change
# gen by protobuf_to_pydantic[v0.3.3.1](https://github.com/so1n/protobuf_to_pydantic)
# Protobuf Version: 6.33.2
# Pydantic Version: 2.12.5
from google.protobuf.message import Message  # type: ignore
from pydantic import BaseModel
from pydantic import Field
import typing


class SummaryStats(BaseModel):
    """
    Summary statistics for a metric.
    """

    # Mean value.
    mean: float = Field(default=0.0)
    # Minimum value.
    min: float = Field(default=0.0)
    # Maximum value.
    max: float = Field(default=0.0)
    # Percentiles (e.g., "p50", "p90").
    percentiles: "typing.Dict[str, float]" = Field(default_factory=dict)
    # Optional rate, used in some metrics.
    rate: float = Field(default=0.0)


class LatencyMetrics(BaseModel):
    """
    Latency metrics for successful requests.
    """

    # Request latency (End - Start).
    request_latency: SummaryStats = Field(default_factory=SummaryStats)
    # Normalized time per output token.
    normalized_time_per_output_token: SummaryStats = Field(default_factory=SummaryStats)
    # Time per output token.
    time_per_output_token: SummaryStats = Field(default_factory=SummaryStats)
    # Time to first token.
    time_to_first_token: SummaryStats = Field(default_factory=SummaryStats)
    # Inter-token latency.
    inter_token_latency: SummaryStats = Field(default_factory=SummaryStats)


class ThroughputMetrics(BaseModel):
    """
    Throughput metrics for successful requests.
    """

    # Input tokens per second.
    input_tokens_per_sec: float = Field(default=0.0)
    # Output tokens per second.
    output_tokens_per_sec: float = Field(default=0.0)
    # Total tokens per second.
    total_tokens_per_sec: float = Field(default=0.0)
    # Requests per second.
    requests_per_sec: float = Field(default=0.0)


class SloAttainment(BaseModel):
    """
    SLO attainment details.
    """

    # Attainment percentage.
    attainment_pct: float = Field(default=0.0)
    # Number of requests meeting SLO.
    requests_met: int = Field(default=0)
    # Number of requests failing SLO.
    requests_failed: int = Field(default=0)
    # Total number of requests.
    total_requests: int = Field(default=0)
    # SLO threshold value.
    slo: float = Field(default=0.0)


class CombinedSloAttainment(BaseModel):
    """
    Combined SLO attainment details.
    """

    # Attainment percentage.
    attainment_pct: float = Field(default=0.0)
    # Number of requests meeting SLO.
    requests_met: int = Field(default=0)
    # Number of requests failing SLO.
    requests_failed: int = Field(default=0)
    # Total number of requests.
    total_requests: int = Field(default=0)
    # TTFT SLO threshold.
    ttft_slo: float = Field(default=0.0)
    # TPOT SLO threshold.
    tpot_slo: float = Field(default=0.0)
    # Goodput rate.
    goodput_rate: float = Field(default=0.0)


class SloMetrics(BaseModel):
    """
    SLO metrics summary.
    """

    # TTFT SLO attainment.
    ttft_slo: SloAttainment = Field(default_factory=SloAttainment)
    # TPOT SLO attainment.
    tpot_slo: SloAttainment = Field(default_factory=SloAttainment)
    # Combined SLO attainment.
    combined_slo: CombinedSloAttainment = Field(default_factory=CombinedSloAttainment)


class SuccessesMetrics(BaseModel):
    """
    Metrics for successful requests.
    """

    # Total count of successful requests.
    count: int = Field(default=0)
    # Latency metrics.
    latency: LatencyMetrics = Field(default_factory=LatencyMetrics)
    # Throughput metrics.
    throughput: ThroughputMetrics = Field(default_factory=ThroughputMetrics)
    # Prompt length statistics.
    prompt_len: SummaryStats = Field(default_factory=SummaryStats)
    # Output length statistics.
    output_len: SummaryStats = Field(default_factory=SummaryStats)
    # SLO metrics.
    slo_metrics: SloMetrics = Field(default_factory=SloMetrics)
    # Prometheus specific or shared metrics
    # Overall rate.
    rate: float = Field(default=0.0)
    # Queue length statistics.
    queue_len: SummaryStats = Field(default_factory=SummaryStats)
    # KV cache usage percentage.
    kv_cache_usage_percentage: SummaryStats = Field(default_factory=SummaryStats)
    # Number of requests swapped.
    num_requests_swapped: SummaryStats = Field(default_factory=SummaryStats)
    # Number of preemptions total.
    num_preemptions_total: SummaryStats = Field(default_factory=SummaryStats)
    # Prefix cache hit percentage.
    prefix_cache_hit_percent: SummaryStats = Field(default_factory=SummaryStats)
    # Inter-token latency.
    inter_token_latency: SummaryStats = Field(default_factory=SummaryStats)
    # Number of requests running.
    num_requests_running: SummaryStats = Field(default_factory=SummaryStats)
    # Request queue time.
    request_queue_time: SummaryStats = Field(default_factory=SummaryStats)
    # Request inference time.
    request_inference_time: SummaryStats = Field(default_factory=SummaryStats)
    # Request prefill time.
    request_prefill_time: SummaryStats = Field(default_factory=SummaryStats)
    # Request decode time.
    request_decode_time: SummaryStats = Field(default_factory=SummaryStats)
    # Request prompt tokens.
    request_prompt_tokens: SummaryStats = Field(default_factory=SummaryStats)
    # Request generation tokens.
    request_generation_tokens: SummaryStats = Field(default_factory=SummaryStats)
    # Request max num generation tokens.
    request_max_num_generation_tokens: SummaryStats = Field(
        default_factory=SummaryStats
    )
    # Request params n.
    request_params_n: SummaryStats = Field(default_factory=SummaryStats)
    # Request params max tokens.
    request_params_max_tokens: SummaryStats = Field(default_factory=SummaryStats)
    # Request success count.
    request_success_count: float = Field(default=0.0)
    # Iteration tokens.
    iteration_tokens: SummaryStats = Field(default_factory=SummaryStats)
    # Prompt tokens cached.
    prompt_tokens_cached: float = Field(default=0.0)
    # Prompt tokens recomputed.
    prompt_tokens_recomputed: float = Field(default=0.0)
    # External prefix cache hit percentage.
    external_prefix_cache_hit_percent: SummaryStats = Field(
        default_factory=SummaryStats
    )
    # MM cache hit percentage.
    mm_cache_hit_percent: SummaryStats = Field(default_factory=SummaryStats)
    # Corrupted requests.
    corrupted_requests: float = Field(default=0.0)
    # Request prefill KV computed tokens.
    request_prefill_kv_computed_tokens: SummaryStats = Field(
        default_factory=SummaryStats
    )
    # KV block idle before evict.
    kv_block_idle_before_evict: SummaryStats = Field(default_factory=SummaryStats)
    # KV block lifetime.
    kv_block_lifetime: SummaryStats = Field(default_factory=SummaryStats)
    # KV block reuse gap.
    kv_block_reuse_gap: SummaryStats = Field(default_factory=SummaryStats)


class FailuresMetrics(BaseModel):
    """
    Metrics for failed requests.
    """

    # Total count of failed requests.
    count: int = Field(default=0)
    # Request latency statistics.
    request_latency: SummaryStats = Field(default_factory=SummaryStats)
    # Prompt length statistics.
    prompt_len: SummaryStats = Field(default_factory=SummaryStats)


class LoadSummary(BaseModel):
    """
    Summary of the load applied.
    """

    # Total count of requests.
    count: int = Field(default=0)
    # Schedule delay statistics.
    schedule_delay: SummaryStats = Field(default_factory=SummaryStats)
    # Send duration.
    send_duration: float = Field(default=0.0)
    # Requested rate.
    requested_rate: float = Field(default=0.0)
    # Achieved rate.
    achieved_rate: float = Field(default=0.0)
    # Concurrency level.
    concurrency: int = Field(default=0)


class NativeReport(BaseModel):
    """
    Native report containing all metrics.
    """

    # Load summary.
    load_summary: LoadSummary = Field(default_factory=LoadSummary)
    # Metrics for successful requests.
    successes: SuccessesMetrics = Field(default_factory=SuccessesMetrics)
    # Metrics for failed requests.
    failures: FailuresMetrics = Field(default_factory=FailuresMetrics)
