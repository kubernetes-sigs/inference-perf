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

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

import numpy as np

from inference_perf.apis import RequestLifecycleMetric, SessionLifecycleMetric, StreamedResponseMetrics
from inference_perf.reportgen.base import effective_output_tokens, retokenize_output_token_times
from inference_perf.payloads import Audios, Images, Videos

from .schema import (
    AggregateLatency,
    AggregateRequestPerformance,
    AggregateRequests,
    AggregateThroughput,
    AudioPayloadStats,
    ImagePayloadStats,
    MultiModalRequests,
    VideoPayloadStats,
    RequestPerformance,
    Results,
    SessionPerformance,
    SessionRequests,
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
    *,
    session_metrics: Optional[List[SessionLifecycleMetric]] = None,
    sessions_not_completed: int = 0,
) -> Results:
    """Build a BR0.2 ``Results`` from inference-perf request metrics.

    Pure function: no I/O, no config dependency. Empty input yields an empty
    ``Results`` (all sub-fields ``None``) — the merge step still produces a
    valid BR0.2 report in that case.

    ``use_server_output_tokens`` must match the value used for the native
    lifecycle reports so both reports of the same run agree on token counts.

    ``session_metrics`` carries the session lifecycle metrics of the same stage,
    already enriched with per-session token totals. When empty or omitted —
    every non-session workload — ``session_performance`` stays ``None`` so the
    emitted partial is unchanged.

    ``sessions_not_completed`` is the stage's count of sessions that were still
    active or never dispatched when the stage ended (the
    ``sessions_not_completed_active + sessions_not_completed_pending`` of the
    stage's ``StageRuntimeInfo``). Those sessions leave no lifecycle metric
    behind, so they are counted here rather than through ``session_metrics``:
    without them a stage cut short by ``max_stage_duration`` would report only
    the sessions that happened to finish, which reads as a clean run. Like the
    native session report, they count towards ``total`` and nothing else.
    """
    return Results(
        request_performance=_build_request_performance(request_metrics, tokenizer, use_server_output_tokens),
        session_performance=_build_session_performance(session_metrics or [], sessions_not_completed),
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


# ---------------------------------------------------------------------------
# Session performance aggregation
# ---------------------------------------------------------------------------


def _build_session_performance(
    metrics: List[SessionLifecycleMetric], sessions_not_completed: int = 0
) -> Optional[SessionPerformance]:
    sessions = _build_session_requests(metrics, sessions_not_completed)
    if sessions is None:
        return None
    return SessionPerformance(sessions=sessions)


def _build_session_requests(
    metrics: List[SessionLifecycleMetric], sessions_not_completed: int = 0
) -> Optional[SessionRequests]:
    """Aggregate per-session metrics into BR0.2 ``SessionRequests``.

    Mirrors ``reportgen/base.py:summarize_sessions``, which renders the same
    figures in the native session lifecycle report, so both reports of a run
    agree on session counts, rates and per-session distributions.

    The metrics are expected to be enriched (see
    ``SessionMetricsCollector.enrich_metrics``) by the time they reach here:
    per-session token totals come from that step, not from the session metrics
    themselves.

    ``sessions_not_completed`` counts the stage's sessions that produced no
    metric, so ``total`` matches the native ``num_sessions``. The rest of the
    block describes the sessions that did report: ``succeeded`` and ``failed``
    mirror ``num_sessions_succeeded`` / ``num_sessions_failed``, and the
    distributions are computed over recorded sessions only — an unrecorded
    session has no duration or event data to contribute. Returns ``None`` when
    the stage ran no sessions at all, so the block is omitted rather than
    emitted as zeros.
    """
    total = len(metrics) + sessions_not_completed
    if total == 0:
        return None

    # Same span convention as the native summary: first session start to last
    # session end, over the sessions that reported. A zero-length span omits the
    # rate rather than dividing by it.
    session_rate: Optional[Statistics] = None
    total_span = max(m.end_time for m in metrics) - min(m.start_time for m in metrics) if metrics else 0.0
    if total_span > 0:
        session_rate = _scalar_statistics(len(metrics) / total_span, Units.QUERY_PER_S)

    return SessionRequests(
        total=total,
        succeeded=sum(1 for m in metrics if m.success is True),
        failed=sum(1 for m in metrics if m.success is False),
        total_events=sum(m.num_events for m in metrics),
        total_events_completed=sum(m.num_events_completed for m in metrics),
        total_events_cancelled=sum(m.num_events_cancelled for m in metrics if m.num_events_cancelled is not None),
        session_rate=session_rate,
        session_duration=_statistics((m.duration_sec for m in metrics), Units.S),
        events_per_session=_statistics((m.num_events for m in metrics), Units.COUNT),
        events_cancelled_per_session=_statistics(
            (m.num_events_cancelled for m in metrics if m.num_events_cancelled is not None), Units.COUNT
        ),
        input_tokens_per_session=_statistics(
            (m.total_input_tokens for m in metrics if m.total_input_tokens is not None), Units.COUNT
        ),
        output_tokens_per_session=_statistics(
            (m.total_output_tokens for m in metrics if m.total_output_tokens is not None), Units.COUNT
        ),
    )


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
        request_size=_statistics(
            (len(m.request_data.encode("utf-8")) for m in successful),
            Units.BYTES,
        ),
        multimodal=_build_multimodal(successful),
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
        image_rate=_media_rate(
            (m.info.request_metrics.image for m in successful),
            total_time,
            Units.IMAGE_PER_S,
        ),
        video_rate=_media_rate(
            (m.info.request_metrics.video for m in successful),
            total_time,
            Units.VIDEO_PER_S,
        ),
        audio_rate=_media_rate(
            (m.info.request_metrics.audio for m in successful),
            total_time,
            Units.AUDIO_PER_S,
        ),
    )


def _media_rate(
    payloads: Iterable[Images | Videos | Audios | None],
    total_time: float,
    units: Units,
) -> Optional[Statistics]:
    recorded = [payload for payload in payloads if payload is not None]
    if not recorded:
        return None
    return _scalar_statistics(sum(payload.count for payload in recorded) / total_time, units)


def _build_multimodal(successful: List[RequestLifecycleMetric]) -> Optional[MultiModalRequests]:
    images = [m.info.request_metrics.image for m in successful]
    videos = [m.info.request_metrics.video for m in successful]
    audios = [m.info.request_metrics.audio for m in successful]
    if not any(images) and not any(videos) and not any(audios):
        return None

    all_images = [instance for group in images if group is not None for instance in group.instances]
    all_videos = [instance for group in videos if group is not None for instance in group.instances]
    all_audios = [instance for group in audios if group is not None for instance in group.instances]
    # Counts describe requests, including text-only requests in a mixed workload.
    # Other distributions describe individual media instances, not request averages.
    image = None
    if any(images):
        image = ImagePayloadStats(
            count=_statistics((group.count if group else 0 for group in images), Units.COUNT),
            filesize=_statistics((instance.bytes for instance in all_images), Units.BYTES),
            pixels=_statistics((instance.pixels for instance in all_images), Units.PIXELS),
            aspect_ratio=_statistics(
                (instance.aspect_ratio for instance in all_images),
                Units.RATIO,
            ),
        )
    video = None
    if any(videos):
        video = VideoPayloadStats(
            count=_statistics((group.count if group else 0 for group in videos), Units.COUNT),
            filesize=_statistics((instance.bytes for instance in all_videos), Units.BYTES),
            # BR counts pixels across frames; native Video.pixels is per-frame.
            pixels=_statistics((instance.pixels * instance.frames for instance in all_videos), Units.PIXELS),
            aspect_ratio=_statistics(
                (instance.aspect_ratio for instance in all_videos),
                Units.RATIO,
            ),
            frames=_statistics((instance.frames for instance in all_videos), Units.COUNT),
        )
    audio = None
    if any(audios):
        audio = AudioPayloadStats(
            count=_statistics((group.count if group else 0 for group in audios), Units.COUNT),
            filesize=_statistics((instance.bytes for instance in all_audios), Units.BYTES),
            duration=_statistics((instance.seconds for instance in all_audios), Units.S),
        )
    return MultiModalRequests(image=image, video=video, audio=audio)


# ---------------------------------------------------------------------------
# Per-request latency derivation.
# Mirrors the latency math in reportgen/base.py:summarize_requests without
# mutating the input metrics: timestamps come from output_token_times (or the
# same chunk re-tokenization the native reports use), token counts from
# effective_output_tokens. TODO: share the latency math too once both paths
# are stable.
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

        ttft.append(token_times[0] - m.start_time if token_times else None)
        if len(token_times) > 1:
            duration = token_times[-1] - token_times[0]
            tpot.append(duration / (output_tokens - 1) if output_tokens > 1 else None)
            for a, b in zip(token_times, token_times[1:], strict=False):
                itl.append(b - a)
        else:
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

    if not (info.response_chunks or info.reasoning_chunks) or tokenizer is None:
        return []

    return retokenize_output_token_times(info, tokenizer)


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
