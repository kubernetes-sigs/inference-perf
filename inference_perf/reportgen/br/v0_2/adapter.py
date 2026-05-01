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
import datetime
import hashlib
import importlib.metadata
import json
import os
import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from inference_perf.apis import RequestLifecycleMetric, StreamedInferenceResponseInfo
from inference_perf.client.metricsclient import PerfRuntimeParameters
from inference_perf.client.metricsclient.base import ModelServerMetrics

from .collectors.base import CollectedStackObservability
from .schema import (
    AggregateLatency,
    AggregateRequestPerformance,
    AggregateRequests,
    AggregateThroughput,
    BenchmarkReportV02,
    Distribution,
    Load,
    LoadMetadata,
    LoadNative,
    LoadPrefix,
    LoadSource,
    LoadStandardized,
    MultiTurn,
    RequestPerformance,
    Results,
    Run,
    RunTime,
    Scenario,
    SequenceLength,
    Statistics,
    Units,
    WorkloadGenerator,
)

if TYPE_CHECKING:
    from inference_perf.config import Config
    from inference_perf.utils.custom_tokenizer import CustomTokenizer


_PERCENTILES = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
_PERCENTILE_KEYS = ["p0p1", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "p99p9"]


def generate_br_v0_2(
    config: "Config",
    request_metrics: List[RequestLifecycleMetric],
    runtime_parameters: PerfRuntimeParameters,
    server_metrics: Optional[ModelServerMetrics],
    collected: Optional[CollectedStackObservability],
    tokenizer: Optional["CustomTokenizer"] = None,
    stage_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Project inference-perf state into a BR0.2 report dict.

    Pure shape adapter: no I/O. All deployment-side data (stack, observability,
    component_health) is gathered upstream by a StackObservabilityCollector and
    passed in as `collected`. When `stage_id` is provided, the report describes
    that one stage; otherwise it spans all `request_metrics`.
    """
    collected = collected or CollectedStackObservability()

    report = BenchmarkReportV02(
        run=_build_run(config, request_metrics),
        scenario=_build_scenario(config, request_metrics, runtime_parameters, collected, stage_id),
        results=_build_results(request_metrics, collected, tokenizer),
    )
    return report.dump()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _build_run(config: "Config", metrics: List[RequestLifecycleMetric]) -> Run:
    cfg = config.report.br_v0_2
    return Run(
        uid=uuid.uuid4().hex,
        eid=cfg.experiment_id if cfg else None,
        cid=None,
        pid=os.environ.get("POD_UID"),
        time=_build_run_time(metrics),
        user=os.environ.get("USER"),
        description=cfg.description if cfg else None,
        keywords=list(cfg.keywords) if cfg and cfg.keywords else None,
    )


def _build_run_time(metrics: List[RequestLifecycleMetric]) -> Optional[RunTime]:
    if not metrics:
        return None
    start_epoch = min(m.start_time for m in metrics)
    end_epoch = max(m.end_time for m in metrics)
    start = datetime.datetime.fromtimestamp(start_epoch, tz=datetime.timezone.utc)
    end = datetime.datetime.fromtimestamp(end_epoch, tz=datetime.timezone.utc)
    return RunTime(start=start, end=end, duration=f"PT{end_epoch - start_epoch}S")


# ---------------------------------------------------------------------------
# scenario
# ---------------------------------------------------------------------------


def _build_scenario(
    config: "Config",
    metrics: List[RequestLifecycleMetric],
    runtime_parameters: PerfRuntimeParameters,
    collected: CollectedStackObservability,
    stage_id: Optional[int],
) -> Scenario:
    return Scenario(
        stack=list(collected.stack) if collected.stack else None,
        load=_build_load(config, metrics, runtime_parameters, stage_id),
    )


def _build_load(
    config: "Config",
    metrics: List[RequestLifecycleMetric],
    runtime_parameters: PerfRuntimeParameters,
    stage_id: Optional[int],
) -> Load:
    native_config = config.model_dump(mode="json", by_alias=True)
    return Load(
        metadata=LoadMetadata(cfg_id=_config_hash(native_config)),
        standardized=_build_load_standardized(config, metrics, runtime_parameters, stage_id),
        native=LoadNative(config=native_config),
    )


def _build_load_standardized(
    config: "Config",
    metrics: List[RequestLifecycleMetric],
    runtime_parameters: PerfRuntimeParameters,
    stage_id: Optional[int],
) -> LoadStandardized:
    rate_qps, concurrency = _stage_rate_concurrency(runtime_parameters, stage_id)
    return LoadStandardized(
        tool=str(WorkloadGenerator.INFERENCE_PERF),
        tool_version=_inference_perf_version(),
        parallelism=max(1, config.load.num_workers or 1),
        source=_load_source(config),
        stage=stage_id if stage_id is not None else 0,
        input_seq_len=_input_seq_len(config, metrics),
        output_seq_len=_output_seq_len(config, metrics),
        prefix=_load_prefix(config),
        multi_turn=_multi_turn(config),
        rate_qps=rate_qps,
        concurrency=concurrency,
    )


def _inference_perf_version() -> str:
    try:
        return importlib.metadata.version("inference-perf")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _load_source(config: "Config") -> LoadSource:
    from inference_perf.config import DataGenType

    random_types = {DataGenType.Random, DataGenType.SharedPrefix, DataGenType.Mock, DataGenType.Synthetic}
    if config.data.type in random_types:
        return LoadSource.RANDOM
    return LoadSource.SAMPLED


def _input_seq_len(config: "Config", metrics: List[RequestLifecycleMetric]) -> SequenceLength:
    dist = config.data.input_distribution
    if dist is not None:
        return SequenceLength(
            distribution=Distribution.GAUSSIAN,
            value=dist.mean,
            std_dev=dist.std_dev or None,
            min=dist.min,
            max=dist.max,
        )
    values = [m.info.input_tokens for m in metrics if m.info.input_tokens]
    if not values:
        return SequenceLength(distribution=Distribution.OTHER, value=0)
    return SequenceLength(
        distribution=Distribution.OTHER,
        value=float(np.mean(values)),
        min=int(np.min(values)),
        max=int(np.max(values)),
    )


def _output_seq_len(config: "Config", metrics: List[RequestLifecycleMetric]) -> Optional[SequenceLength]:
    dist = config.data.output_distribution
    if dist is not None:
        return SequenceLength(
            distribution=Distribution.GAUSSIAN,
            value=dist.mean,
            std_dev=dist.std_dev or None,
            min=dist.min,
            max=dist.max,
        )
    values = [
        m.info.response_info.output_tokens for m in metrics if m.info.response_info and m.info.response_info.output_tokens
    ]
    if not values:
        return None
    return SequenceLength(
        distribution=Distribution.OTHER,
        value=float(np.mean(values)),
        min=int(np.min(values)),
        max=int(np.max(values)),
    )


def _load_prefix(config: "Config") -> Optional[LoadPrefix]:
    sp = config.data.shared_prefix
    if sp is None:
        return None
    return LoadPrefix(
        prefix_len=SequenceLength(distribution=Distribution.FIXED, value=sp.system_prompt_len),
        num_groups=sp.num_groups,
        num_users_per_group=sp.num_prompts_per_group,
        num_prefixes=1,
    )


def _multi_turn(config: "Config") -> Optional[MultiTurn]:
    sp = config.data.shared_prefix
    if sp is not None and getattr(sp, "enable_multi_turn_chat", False):
        return MultiTurn(enabled=True)
    return None


def _stage_rate_concurrency(
    rp: PerfRuntimeParameters,
    stage_id: Optional[int],
) -> Tuple[Optional[float], Optional[int]]:
    if not rp.stages:
        return None, None
    stage = rp.stages.get(stage_id) if stage_id is not None else next(iter(rp.stages.values()))
    if stage is None:
        return None, None
    rate = stage.rate if stage.rate and stage.rate > 0 else None
    return rate, stage.concurrency_level


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


def _build_results(
    metrics: List[RequestLifecycleMetric],
    collected: CollectedStackObservability,
    tokenizer: Optional["CustomTokenizer"],
) -> Results:
    return Results(
        request_performance=_build_request_performance(metrics, tokenizer),
        observability=collected.observability,
        component_health=list(collected.component_health) if collected.component_health else None,
    )


def _build_request_performance(
    metrics: List[RequestLifecycleMetric],
    tokenizer: Optional["CustomTokenizer"],
) -> Optional[RequestPerformance]:
    if not metrics:
        return None
    return RequestPerformance(aggregate=_build_aggregate(metrics, tokenizer))


def _build_aggregate(
    metrics: List[RequestLifecycleMetric],
    tokenizer: Optional["CustomTokenizer"],
) -> AggregateRequestPerformance:
    successful = [m for m in metrics if m.error is None]
    failed = [m for m in metrics if m.error is not None]

    input_lengths = [float(m.info.input_tokens) for m in successful]
    output_lengths = [
        float(m.info.response_info.output_tokens)
        for m in successful
        if m.info.response_info and m.info.response_info.output_tokens
    ]

    requests = AggregateRequests(
        total=len(metrics),
        failures=len(failed),
        input_length=_statistics(input_lengths, Units.COUNT),
        output_length=_statistics(output_lengths, Units.COUNT),
    )

    request_latencies, ttft, tpot, itl, ntpot = _per_request_latencies(successful, tokenizer)
    latency = AggregateLatency(
        request_latency=_statistics(request_latencies, Units.S),
        time_to_first_token=_statistics([v for v in ttft if v is not None], Units.S),
        time_per_output_token=_statistics([v for v in tpot if v is not None], Units.S_PER_TOKEN),
        inter_token_latency=_statistics(itl, Units.S_PER_TOKEN),
        normalized_time_per_output_token=_statistics(ntpot, Units.S_PER_TOKEN),
    )

    total_time = _benchmark_window(metrics)
    throughput = _build_throughput(successful, total_time) if total_time > 0 else None

    return AggregateRequestPerformance(requests=requests, latency=latency, throughput=throughput)


def _benchmark_window(metrics: List[RequestLifecycleMetric]) -> float:
    if not metrics:
        return 0.0
    return max(m.end_time for m in metrics) - min(m.start_time for m in metrics)


def _build_throughput(successful: List[RequestLifecycleMetric], total_time: float) -> AggregateThroughput:
    input_tokens = sum(m.info.input_tokens for m in successful)
    output_tokens = sum(
        m.info.response_info.output_tokens for m in successful if m.info.response_info and m.info.response_info.output_tokens
    )
    return AggregateThroughput(
        input_token_rate=_scalar_statistics(input_tokens / total_time, Units.TOKEN_PER_S),
        output_token_rate=_scalar_statistics(output_tokens / total_time, Units.TOKEN_PER_S),
        total_token_rate=_scalar_statistics((input_tokens + output_tokens) / total_time, Units.TOKEN_PER_S),
        request_rate=_scalar_statistics(len(successful) / total_time, Units.QUERY_PER_S),
    )


# ---------------------------------------------------------------------------
# Per-request latency derivation.
# Mirrors the chunk-parsing + latency math in
# reportgen/base.py:summarize_requests but operates on a copy so input metrics
# are not mutated. TODO: extract into a single shared helper once both paths
# are stable.
# ---------------------------------------------------------------------------


def _per_request_latencies(
    metrics: List[RequestLifecycleMetric],
    tokenizer: Optional["CustomTokenizer"],
) -> Tuple[List[float], List[Optional[float]], List[Optional[float]], List[float], List[float]]:
    request_latency: List[float] = []
    ttft: List[Optional[float]] = []
    tpot: List[Optional[float]] = []
    itl: List[float] = []
    ntpot: List[float] = []

    for m in metrics:
        request_latency.append(m.end_time - m.start_time)
        token_times, output_tokens = _resolve_token_stream(m, tokenizer)

        if output_tokens > 0:
            ntpot.append((m.end_time - m.start_time) / output_tokens)
        else:
            ntpot.append(0.0)

        if len(token_times) > 1:
            ttft.append(token_times[0] - m.start_time)
            duration = token_times[-1] - token_times[0]
            tpot.append(duration / (output_tokens - 1) if output_tokens > 1 else None)
            for a, b in zip(token_times, token_times[1:], strict=False):
                itl.append(b - a)
        else:
            ttft.append(None)
            tpot.append(None)

    return request_latency, ttft, tpot, itl, ntpot


def _resolve_token_stream(
    metric: RequestLifecycleMetric,
    tokenizer: Optional["CustomTokenizer"],
) -> Tuple[List[float], int]:
    info = metric.info.response_info
    if not isinstance(info, StreamedInferenceResponseInfo):
        return [], 0

    if info.output_token_times and info.output_tokens:
        return list(info.output_token_times), info.output_tokens

    if not info.response_chunks or tokenizer is None:
        return list(info.output_token_times), info.output_tokens or 0

    token_times: List[float] = []
    total_tokens = 0
    for chunk_str, chunk_time in zip(info.response_chunks, info.chunk_times, strict=False):
        try:
            data = json.loads(chunk_str)
        except json.JSONDecodeError:
            continue
        choices = data.get("choices") or []
        if not choices:
            continue
        delta = choices[0]
        text = delta.get("text") or delta.get("delta", {}).get("content")
        if not text:
            continue
        n = tokenizer.count_tokens(text)
        if n <= 0:
            continue
        token_times.extend([chunk_time] * n)
        total_tokens += n

    return token_times, total_tokens


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _statistics(values: Iterable[float], units: Units) -> Optional[Statistics]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return None
    pcts = np.percentile(arr, _PERCENTILES)
    return Statistics(
        units=units,
        mean=float(arr.mean()),
        stddev=float(arr.std(ddof=1)) if arr.size > 1 else None,
        min=float(arr.min()),
        max=float(arr.max()),
        **{key: float(p) for key, p in zip(_PERCENTILE_KEYS, pcts, strict=True)},
    )


def _scalar_statistics(value: float, units: Units) -> Statistics:
    return Statistics(units=units, mean=float(value))


def _config_hash(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(payload).hexdigest()
