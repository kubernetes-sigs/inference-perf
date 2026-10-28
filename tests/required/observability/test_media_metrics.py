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
"""The media metric set: per-modality counts and sizes, read off each request's record."""

from typing import Optional

from inference_perf.apis import ErrorResponseInfo, InferenceInfo, RequestLifecycleMetric, UnaryResponseMetrics
from inference_perf.config import Config
from inference_perf.observability.metrics import MetricsHub, build_metrics
from inference_perf.observability.metrics.sets import MEDIA_SPECS
from inference_perf.payloads import Audio, Audios, Image, Images, RequestMetrics, Text, Video, Videos

ONE_MEGAPIXEL = 1024 * 1024
FULL_HD = 1920 * 1080


# Builds one stage-0 request with the given request-side audio and response media.
# error=True marks it failed, which the media metrics skip on both sides.
def _request(
    request_audio: Optional[Audios] = None,
    response: Optional[UnaryResponseMetrics] = None,
    error: bool = False,
) -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=0.0,
        start_time=0.0,
        end_time=1.0,
        request_data="{}",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=4), audio=request_audio),
            response_metrics=response or UnaryResponseMetrics(output_tokens=3),
        ),
        error=ErrorResponseInfo(error_type="TimeoutError", error_msg="timed out") if error else None,
    )


# Reads one series out of the hub's registry by sample name and labels. Returns None
# when the series does not exist, which is how absence gets asserted.
def _sample(hub: MetricsHub, name: str, **labels: str) -> Optional[float]:
    return hub.registry.get_sample_value(name, labels)


# Every sample the media specs have exported so far, as (sample name, labels).
# Empty when no media series exists yet.
def _media_samples(hub: MetricsHub) -> list[tuple[str, dict[str, str]]]:
    names = {spec.name for spec in MEDIA_SPECS}
    return [
        (sample.name, sample.labels)
        for family in hub.registry.collect()
        if family.name in names
        for sample in family.samples
        if not sample.name.endswith("_created")
    ]


# One response returns two images (one 1 MP and 2000 bytes, one by URL with nothing
# measured) and a 2s, 48-frame 1080p video. Expects 2 images and 1 video counted,
# one bytes observation for the images, 1080p pixels observed once for the video
# (per frame), 2s of video and 48 frames, and nothing on the input side.
def test_response_media_counted_and_sized_by_modality() -> None:
    hub = build_metrics(Config())
    response = UnaryResponseMetrics(
        image=Images(count=2, instances=[Image(pixels=ONE_MEGAPIXEL, bytes=2000, aspect_ratio=1.0), Image()]),
        video=Videos(count=1, instances=[Video(pixels=FULL_HD, bytes=5_000_000, aspect_ratio=16 / 9, frames=48, seconds=2.0)]),
    )

    hub.observe_request(_request(response=response))

    assert _sample(hub, "inference_perf_output_media_total", stage="0", modality="image") == 2.0
    assert _sample(hub, "inference_perf_output_media_total", stage="0", modality="video") == 1.0
    assert _sample(hub, "inference_perf_output_media_total", stage="0", modality="audio") is None
    assert _sample(hub, "inference_perf_output_media_bytes_count", stage="0", modality="image") == 1.0
    assert _sample(hub, "inference_perf_output_media_bytes_sum", stage="0", modality="image") == 2000.0
    assert _sample(hub, "inference_perf_output_media_pixels_sum", stage="0", modality="video") == FULL_HD
    assert _sample(hub, "inference_perf_output_media_seconds_sum", stage="0", modality="video") == 2.0
    assert _sample(hub, "inference_perf_output_video_frames_sum", stage="0") == 48.0
    assert _sample(hub, "inference_perf_input_media_total", stage="0", modality="image") is None


# One request sends a 3s, 100-byte clip and gets text back. Expects the clip on the
# input metrics (1 audio, 100 bytes, 3s) and no output media series at all.
def test_request_media_read_from_the_request_side() -> None:
    hub = build_metrics(Config())

    hub.observe_request(_request(request_audio=Audios(count=1, instances=[Audio(bytes=100, seconds=3.0)])))

    assert _sample(hub, "inference_perf_input_media_total", stage="0", modality="audio") == 1.0
    assert _sample(hub, "inference_perf_input_media_bytes_sum", stage="0", modality="audio") == 100.0
    assert _sample(hub, "inference_perf_input_media_seconds_sum", stage="0", modality="audio") == 3.0
    samples = _media_samples(hub)
    assert ("inference_perf_input_media_total", {"stage": "0", "modality": "audio"}) in samples
    assert not [name for name, _ in samples if name.startswith("inference_perf_output_")]


# A text request, a request whose response carries an empty image record, then a
# failed request that sent a clip and got an image and a 4-frame video back. Expects
# no media series at all: nothing was counted, and failures are skipped on both sides
# like the token counters skip them.
def test_text_and_failed_requests_export_no_media_series() -> None:
    hub = build_metrics(Config())
    media = UnaryResponseMetrics(
        image=Images(count=1, instances=[Image(pixels=1, bytes=1, aspect_ratio=1.0)]),
        video=Videos(count=1, instances=[Video(pixels=1, bytes=1, aspect_ratio=1.0, frames=4, seconds=1.0)]),
    )

    hub.observe_request(_request())
    hub.observe_request(_request(response=UnaryResponseMetrics(image=Images())))
    hub.observe_request(
        _request(request_audio=Audios(count=1, instances=[Audio(bytes=1, seconds=1.0)]), response=media, error=True)
    )

    assert _media_samples(hub) == []
