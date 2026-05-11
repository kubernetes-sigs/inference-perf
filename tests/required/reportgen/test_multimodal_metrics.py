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
import pytest
import typing
from unittest.mock import Mock
from inference_perf.apis import InferenceInfo, StreamedResponseMetrics
from inference_perf.payloads import (
    RequestMetrics,
    Text,
    Images,
    Videos,
    Image,
    Video,
)
from inference_perf.reportgen.base import summarize_requests


def test_summarize_requests_multimodal_metrics() -> None:
    # Create mock metrics
    metric1 = Mock()
    metric1.start_time = 0.0
    metric1.scheduled_time = 0.0
    metric1.end_time = 2.0
    metric1.error = None
    metric1.ttft_slo_sec = None
    metric1.tpot_slo_sec = None
    metric1.request_data = "short request"
    metric1.info = Mock(spec=InferenceInfo)
    metric1.info.request_metrics = RequestMetrics(
        text=Text(input_tokens=10),
        image=Images(
            count=2,
            instances=[
                Image(pixels=100, bytes=1000, aspect_ratio=1.0),
                Image(pixels=200, bytes=2000, aspect_ratio=2.0),
            ],
        ),
    )
    metric1.info.response_metrics = StreamedResponseMetrics(
        response_chunks=[],
        chunk_times=[0.5, 1.5],
        output_tokens=20,
        output_token_times=[0.5, 1.5],
    )
    metric1.info.extra_info = {}

    metric2 = Mock()
    metric2.start_time = 1.0
    metric2.scheduled_time = 1.0
    metric2.end_time = 3.0
    metric2.error = None
    metric2.ttft_slo_sec = None
    metric2.tpot_slo_sec = None
    metric2.request_data = "a much longer request with more data"
    metric2.info = Mock(spec=InferenceInfo)
    metric2.info.request_metrics = RequestMetrics(
        text=Text(input_tokens=15),
        video=Videos(
            count=1,
            instances=[Video(pixels=640 * 480, bytes=500000, aspect_ratio=4 / 3, frames=10)],
        ),
        image=Images(count=1, instances=[Image(pixels=300, bytes=3000, aspect_ratio=0.5)]),
    )
    metric2.info.response_metrics = StreamedResponseMetrics(
        response_chunks=[],
        chunk_times=[1.5, 2.5],
        output_tokens=25,
        output_token_times=[1.5, 2.5],
    )
    metric2.info.extra_info = {}

    metrics = [metric1, metric2]

    # total_time = max(end) - min(start) = 3.0 - 0.0 = 3.0

    summary = summarize_requests(typing.cast(typing.Any, metrics), percentiles=[50, 90])

    assert summary.benchmark_time_seconds == 3.0
    successes = summary.successes

    # Throughput checks
    throughput = successes["throughput"]
    assert throughput["requests_per_sec"] == 2 / 3.0

    # Images: 2 from metric1, 1 from metric2. Total = 3. Rate = 3 / 3.0
    assert pytest.approx(throughput["images_per_sec"]) == 3 / 3.0

    # Videos: 0 from metric1, 1 from metric2. Total = 1. Rate = 1 / 3.0
    assert pytest.approx(throughput["videos_per_sec"]) == 1 / 3.0

    # No audio on either request
    assert throughput["audios_per_sec"] == 0.0

    # Image count per request: metric1 has 2, metric2 has 1. Mean = 1.5
    img = successes["image"]
    assert img["count"]["mean"] == 1.5
    assert img["count"]["min"] == 1.0
    assert img["count"]["max"] == 2.0

    # Individual image pixels: 100, 200, 300. Mean = 200.0
    assert img["pixels"]["mean"] == 200.0
    # Individual image bytes: 1000, 2000, 3000. Mean = 2000.0
    assert img["bytes"]["mean"] == 2000.0
    # Individual image aspect ratio: 1.0, 2.0, 0.5
    assert pytest.approx(img["aspect_ratio"]["mean"]) == (1.0 + 2.0 + 0.5) / 3.0

    # Video: 1 instance (metric2 only)
    video = successes["video"]
    assert video["count"]["mean"] == 0.5  # metric1 has 0, metric2 has 1
    assert video["frames"]["mean"] == 10.0
    assert video["pixels"]["mean"] == 640 * 480
    assert video["bytes"]["mean"] == 500000
    assert pytest.approx(video["aspect_ratio"]["mean"]) == 4 / 3

    # Request size checks
    size_summary = successes["request_size_bytes"]
    assert size_summary is not None

    size1 = len("short request".encode("utf-8"))
    size2 = len("a much longer request with more data".encode("utf-8"))

    assert size_summary["mean"] == (size1 + size2) / 2.0
    assert size_summary["min"] == min(size1, size2)
    assert size_summary["max"] == max(size1, size2)
