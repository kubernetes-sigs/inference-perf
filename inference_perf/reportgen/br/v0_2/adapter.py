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
"""Project inference-perf request metrics into a BR0.2 ``Results`` object.

Inference-perf is strictly responsible for the ``results`` section of the
BR0.2 report (performance measurements derived from the actual run).
Everything else — stack configuration, run/scenario metadata — is supplied
by the user via the partial report; see ``partial_report.py``.
"""

import json
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

import numpy as np

from inference_perf.apis import RequestLifecycleMetric, StreamedResponseMetrics
from inference_perf.reportgen.base import effective_output_tokens

from .schema import (
    AggregateLatency,
    AggregateRequestPerformance,
    AggregateRequests,
    AggregateThroughput,
    RequestPerformance,
    Results,
    Statistics,
    Units,
)

if TYPE_CHECKING:
    from inference_perf.utils.custom_tokenizer import CustomTokenizer


_PERCENTILES = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
_PERCENTILE_KEYS = ["p0p1", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "p99p9"]


def build_results(
    request_metrics: List[RequestLifecycleMetric],
    tokenizer: Optional["CustomTokenizer"] = None,
    use_server_output_tokens: bool = False,
) -> Results:
    """Build a BR0.2 ``Results`` from inference-perf request metrics.

    Pure function: no I/O, no config dependency. Empty input yields an empty
    ``Results`` (all sub-fields ``None``) — the merge step still produces a
    valid BR0.2 report in that case.

    ``use_server_output_tokens`` must match the value used for the native
    lifecycle reports so both reports of the same run agree on token counts.
    """
    return Results(
        request_performance=_build_request_performance(request_metrics, tokenizer, use_server_output_tokens),
    )


# ---------------------------------------------------------------------------
# Request performance aggregation
# ---------------------------------------------------------------------------


def _build_request_performance(
    metrics: List[RequestLifecycleMetric],
    tokenizer: Optional["CustomTokenizer"],
    use_server_output_tokens: bool,
) -> Optional[RequestPerformance]:
    if not metrics:
        return None
    return RequestPerformance(aggregate=_build_aggregate(metrics, tokenizer, use_server_output_tokens))


def _build_aggregate(
    metrics: List[RequestLifecycleMetric],
    tokenizer: Optional["CustomTokenizer"],
    use_server_output_tokens: bool,
) -> AggregateRequestPerformance:
    successful = [m for m in metrics if m.error is None]
    failed = [m for m in metrics if m.error is not None]

    input_lengths = [float(m.info.input_tokens) for m in successful]
    # Zero is a real output length (immediate EOS, stop sequence) and stays in
    # the distribution, matching the native lifecycle report; only requests
    # with no response metrics at all (count unknown) are excluded.
    output_lengths = [
        float(effective_output_tokens(m.info.response_metrics, use_server_output_tokens))
        for m in successful
        if m.info.response_metrics is not None
    ]

    requests = AggregateRequests(
        total=len(metrics),
        failures=len(failed),
        input_length=_statistics(input_lengths, Units.COUNT),
        output_length=_statistics(output_lengths, Units.COUNT),
    )

    request_latencies, ttft, tpot, itl, ntpot = _per_request_latencies(successful, tokenizer, use_server_output_tokens)
    latency = AggregateLatency(
        request_latency=_statistics(request_latencies, Units.S),
        time_to_first_token=_statistics([v for v in ttft if v is not None], Units.S),
        time_per_output_token=_statistics([v for v in tpot if v is not None], Units.S_PER_TOKEN),
        inter_token_latency=_statistics(itl, Units.S_PER_TOKEN),
        normalized_time_per_output_token=_statistics([v for v in ntpot if v is not None], Units.S_PER_TOKEN),
    )

    # No successes means no measured throughput: emit absence, not zero rates,
    # so a composer can tell an unmeasured window from a measured 0.
    total_time = _benchmark_window(metrics)
    throughput = _build_throughput(successful, total_time, use_server_output_tokens) if successful and total_time > 0 else None

    return AggregateRequestPerformance(requests=requests, latency=latency, throughput=throughput)


def _benchmark_window(metrics: List[RequestLifecycleMetric]) -> float:
    if not metrics:
        return 0.0
    return max(m.end_time for m in metrics) - min(m.start_time for m in metrics)


def _build_throughput(
    successful: List[RequestLifecycleMetric], total_time: float, use_server_output_tokens: bool
) -> AggregateThroughput:
    input_tokens = sum(m.info.input_tokens for m in successful)
    output_tokens = sum(effective_output_tokens(m.info.response_metrics, use_server_output_tokens) for m in successful)
    return AggregateThroughput(
        input_token_rate=_scalar_statistics(input_tokens / total_time, Units.TOKEN_PER_S),
        output_token_rate=_scalar_statistics(output_tokens / total_time, Units.TOKEN_PER_S),
        total_token_rate=_scalar_statistics((input_tokens + output_tokens) / total_time, Units.TOKEN_PER_S),
        request_rate=_scalar_statistics(len(successful) / total_time, Units.QUERY_PER_S),
    )


# ---------------------------------------------------------------------------
# Per-request latency derivation.
# Mirrors the chunk-parsing + latency math in
# reportgen/base.py:summarize_requests without mutating the input metrics:
# timestamps come from output_token_times (or a chunk re-tokenization
# fallback), token counts from effective_output_tokens, matching the native
# lifecycle reports. TODO: extract into a single shared helper once both
# paths are stable.
# ---------------------------------------------------------------------------


def _per_request_latencies(
    metrics: List[RequestLifecycleMetric],
    tokenizer: Optional["CustomTokenizer"],
    use_server_output_tokens: bool,
) -> Tuple[List[float], List[Optional[float]], List[Optional[float]], List[float], List[Optional[float]]]:
    request_latency: List[float] = []
    ttft: List[Optional[float]] = []
    tpot: List[Optional[float]] = []
    itl: List[float] = []
    ntpot: List[Optional[float]] = []

    for m in metrics:
        request_latency.append(m.end_time - m.start_time)
        token_times = _resolve_token_times(m, tokenizer)
        output_tokens = effective_output_tokens(m.info.response_metrics, use_server_output_tokens)

        # NTPOT is computed for every successful request, streaming or not,
        # but is not measurable without output tokens; skip those requests
        # rather than folding in a fabricated 0.0.
        ntpot.append((m.end_time - m.start_time) / output_tokens if output_tokens > 0 else None)

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


def _resolve_token_times(
    metric: RequestLifecycleMetric,
    tokenizer: Optional["CustomTokenizer"],
) -> List[float]:
    info = metric.info.response_metrics
    if not isinstance(info, StreamedResponseMetrics):
        return []

    if info.output_token_times:
        return list(info.output_token_times)

    if not info.response_chunks or tokenizer is None:
        return []

    token_times: List[float] = []
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
        # Count each chunk as a sequence fragment (add_special_tokens=False):
        # re-tokenizing with special tokens prepends a BOS per chunk, which
        # inflates counts and deflates ITL. See #564.
        n = tokenizer.count_tokens(text, add_special_tokens=False)
        if n <= 0:
            continue
        # Every token in a chunk gets the chunk's arrival time, matching
        # summarize_requests: intra-chunk ITL is 0, inter-chunk ITL absorbs
        # the full gap.
        token_times.extend([chunk_time] * n)

    return token_times


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
