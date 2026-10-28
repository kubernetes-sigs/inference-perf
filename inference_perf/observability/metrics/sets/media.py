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
"""Media sent and received per modality, observed once per successful request.

The runtime view of the report's image, video and audio blocks, read from the
same RequestLifecycleMetric: input from request_metrics, output from
response_metrics. A field the client never measured (None, e.g. media sent or
returned as a URL) is not observed, matching the report, which leaves it out
of the distribution rather than counting it as 0. A labelled metric has no
series until its first observation, so a text-only run exports none of these.
"""

from typing import Any, Callable, Optional, Union

from prometheus_client import Counter, Histogram  # noqa: TID251 (declares each spec's metric_type)

from inference_perf.apis.base import RequestLifecycleMetric
from inference_perf.observability.metrics.registry import MetricSpec
from inference_perf.payloads import Audios, Images, Videos

from .core import stage_label

MediaByModality = dict[str, Optional[Union[Images, Videos, Audios]]]
Side = Callable[[RequestLifecycleMetric], MediaByModality]

BYTES_BUCKETS = (1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9)
# 256^2, 512^2, 1024^2, 1080p, 2048^2, 4K UHD, 4096^2.
PIXELS_BUCKETS = (65536.0, 262144.0, 1048576.0, 2073600.0, 4194304.0, 8294400.0, 16777216.0)
SECONDS_BUCKETS = (1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
FRAMES_BUCKETS = (1.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0)


def request_media(metric: RequestLifecycleMetric) -> MediaByModality:
    request = metric.info.request_metrics
    return {"image": request.image, "video": request.video, "audio": request.audio}


def response_media(metric: RequestLifecycleMetric) -> MediaByModality:
    response = metric.info.response_metrics
    if response is None:
        return {}
    return {"image": response.image, "video": response.video, "audio": response.audio}


def _count_instances(side: Side) -> Callable[[Counter, RequestLifecycleMetric], None]:
    def observe(counter: Counter, metric: RequestLifecycleMetric) -> None:
        if metric.error is not None:
            return
        for modality, payload in side(metric).items():
            if payload is not None and payload.count:
                counter.labels(stage_label(metric), modality).inc(payload.count)

    return observe


def _observe_field(side: Side, field: str, modalities: tuple[str, ...]) -> Callable[[Histogram, RequestLifecycleMetric], None]:
    def observe(histogram: Histogram, metric: RequestLifecycleMetric) -> None:
        if metric.error is not None:
            return
        for modality, payload in side(metric).items():
            if payload is None or modality not in modalities:
                continue
            for instance in payload.instances:
                value = getattr(instance, field)
                if value is not None:
                    histogram.labels(stage_label(metric), modality).observe(value)

    return observe


def _observe_frames(side: Side) -> Callable[[Histogram, RequestLifecycleMetric], None]:
    def observe(histogram: Histogram, metric: RequestLifecycleMetric) -> None:
        videos = side(metric).get("video")
        if metric.error is not None or not isinstance(videos, Videos):
            return
        for instance in videos.instances:
            if instance.frames is not None:
                histogram.labels(stage_label(metric)).observe(instance.frames)

    return observe


def _media_specs(direction: str, verb: str, side: Side) -> tuple[MetricSpec[Any], ...]:
    """The five media metrics for one direction: input (sent) or output (received)."""
    return (
        MetricSpec(
            name=f"inference_perf_{direction}_media",
            documentation=f"Media instances {verb} on successful requests, by stage and modality (image, video, audio).",
            metric_type=Counter,
            labelnames=("stage", "modality"),
            on_request=_count_instances(side),
        ),
        MetricSpec(
            name=f"inference_perf_{direction}_media_bytes",
            documentation=(
                f"Encoded size of each media instance {verb} on a successful request, by stage and modality. "
                "An instance sent or returned as a URL has no size and is not observed."
            ),
            metric_type=Histogram,
            labelnames=("stage", "modality"),
            buckets=BYTES_BUCKETS,
            on_request=_observe_field(side, "bytes", ("image", "video", "audio")),
        ),
        MetricSpec(
            name=f"inference_perf_{direction}_media_pixels",
            documentation=(
                f"Pixels of each image, or of one frame of each video, {verb} on a successful request, by stage and modality."
            ),
            metric_type=Histogram,
            labelnames=("stage", "modality"),
            buckets=PIXELS_BUCKETS,
            on_request=_observe_field(side, "pixels", ("image", "video")),
        ),
        MetricSpec(
            name=f"inference_perf_{direction}_media_seconds",
            documentation=f"Duration of each audio or video instance {verb} on a successful request, by stage and modality.",
            metric_type=Histogram,
            labelnames=("stage", "modality"),
            buckets=SECONDS_BUCKETS,
            on_request=_observe_field(side, "seconds", ("audio", "video")),
        ),
        MetricSpec(
            name=f"inference_perf_{direction}_video_frames",
            documentation=f"Frames in each video {verb} on a successful request, by stage.",
            metric_type=Histogram,
            labelnames=("stage",),
            buckets=FRAMES_BUCKETS,
            on_request=_observe_frames(side),
        ),
    )


MEDIA_SPECS: tuple[MetricSpec[Any], ...] = (
    *_media_specs("input", "sent", request_media),
    *_media_specs("output", "received", response_media),
)
