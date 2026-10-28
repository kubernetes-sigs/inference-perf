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
"""Media a response returns, and media whose size the client never saw, in the summary report."""

from typing import Optional

from inference_perf.apis import ErrorResponseInfo, InferenceInfo, RequestLifecycleMetric, UnaryResponseMetrics
from inference_perf.payloads import Image, Images, RequestMetrics, Text, Video, Videos
from inference_perf.reportgen.base import summarize_requests

ONE_MEGAPIXEL = 1024 * 1024


# Builds one request from start to end seconds with the given request and response
# media. error=True marks it failed, so the report counts it under failures only.
def _request(
    start: float,
    end: float,
    request_image: Optional[Images] = None,
    response: Optional[UnaryResponseMetrics] = None,
    error: bool = False,
) -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=start,
        start_time=start,
        end_time=end,
        request_data="{}",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=4), image=request_image),
            response_metrics=response or UnaryResponseMetrics(output_tokens=3),
        ),
        error=ErrorResponseInfo(error_type="TimeoutError", error_msg="timed out") if error else None,
    )


# Two successful responses over a 4s window each return one image: one measured
# (1 MP, 2000 bytes), one returned as a URL (nothing measured). A failed request also
# carries an image. Expects count mean 1, pixels and bytes over the one measured image
# only, and output_images_per_sec = 2 / 4 = 0.5 with the failure left out.
def test_response_images_get_an_output_block_and_rate() -> None:
    measured = UnaryResponseMetrics(
        image=Images(count=1, instances=[Image(pixels=ONE_MEGAPIXEL, bytes=2000, aspect_ratio=1.0)])
    )
    by_url = UnaryResponseMetrics(image=Images(count=1, instances=[Image()]))
    metrics = [
        _request(0.0, 4.0, response=measured),
        _request(1.0, 3.0, response=by_url),
        _request(1.0, 2.0, response=measured, error=True),
    ]

    successes = summarize_requests(metrics, percentiles=[50]).successes

    image = successes["output_image"]
    assert image["count"]["mean"] == 1.0
    assert image["pixels"]["mean"] == image["pixels"]["max"] == ONE_MEGAPIXEL
    assert image["bytes"]["mean"] == 2000.0
    assert image["aspect_ratio"]["mean"] == 1.0
    assert successes["throughput"]["output_images_per_sec"] == 0.5
    assert "output_video" not in successes and "output_audio" not in successes
    assert "output_videos_per_sec" not in successes["throughput"]


# A text-only run: one request, 3 output tokens, no media either way. Expects no
# output_* block or rate anywhere, and the input image block still present, so a
# text run's report keeps the shape it had before response media existed.
def test_text_run_report_has_no_output_media() -> None:
    successes = summarize_requests([_request(0.0, 1.0)], percentiles=[50]).successes

    assert not [key for key in successes if key.startswith("output_") and key not in ("output_len", "output_tokens")]
    assert not [key for key in successes["throughput"] if key.startswith("output_") and key != "output_tokens_per_sec"]
    assert successes["image"]["count"]["mean"] == 0.0


# One request sends two images: one measured at 100 pixels and 10 bytes, one sent by
# URL with nothing measured. Expects count mean 2 but pixels mean 100 and bytes mean
# 10, where averaging the URL image in as 0 would give 50 and 5.
def test_unmeasured_request_image_stays_out_of_size_distributions() -> None:
    images = Images(count=2, instances=[Image(pixels=100, bytes=10, aspect_ratio=1.0), Image()])

    successes = summarize_requests([_request(0.0, 1.0, request_image=images)], percentiles=[50]).successes

    assert successes["image"]["count"]["mean"] == 2.0
    assert successes["image"]["pixels"]["mean"] == 100.0
    assert successes["image"]["bytes"]["mean"] == 10.0


# One response returns a 2s, 48-frame 1080p video. Expects output_video to report
# seconds 2 and frames 48, and the input video block to carry a seconds key that is
# None because no request sent a video.
def test_video_seconds_reported_on_both_sides() -> None:
    video = Video(pixels=1920 * 1080, bytes=5_000_000, aspect_ratio=16 / 9, frames=48, seconds=2.0)
    response = UnaryResponseMetrics(video=Videos(count=1, instances=[video]))

    successes = summarize_requests([_request(0.0, 10.0, response=response)], percentiles=[50]).successes

    assert successes["output_video"]["seconds"]["mean"] == 2.0
    assert successes["output_video"]["frames"]["mean"] == 48.0
    assert successes["throughput"]["output_videos_per_sec"] == 0.1
    assert "seconds" in successes["video"] and successes["video"]["seconds"] is None
