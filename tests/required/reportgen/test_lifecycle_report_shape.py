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
"""Asserts the full lifecycle-metrics report shape, exercising every field
that should be populated when a mix of multimodal requests is observed."""

import typing
from unittest.mock import Mock

from inference_perf.apis import InferenceInfo, StreamedResponseMetrics
from inference_perf.payloads import (
    RequestMetrics,
    Text,
    Images,
    Videos,
    Audios,
    Image,
    Video,
    Audio,
)
from inference_perf.reportgen.base import summarize_requests


def _mock_metric(
    *,
    start_time: float,
    end_time: float,
    scheduled_time: float,
    input_tokens: int,
    output_tokens: int,
    request_data: str,
    images: typing.List[Image],
    videos: typing.List[Video],
    audios: typing.List[Audio],
    output_token_times: typing.List[float],
) -> Mock:
    m = Mock()
    m.start_time = start_time
    m.end_time = end_time
    m.scheduled_time = scheduled_time
    m.error = None
    m.ttft_slo_sec = None
    m.tpot_slo_sec = None
    m.request_data = request_data
    m.info = Mock(spec=InferenceInfo)
    m.info.request_metrics = RequestMetrics(
        text=Text(input_tokens=input_tokens),
        image=Images(count=len(images), instances=images) if images else None,
        video=Videos(count=len(videos), instances=videos) if videos else None,
        audio=Audios(count=len(audios), instances=audios) if audios else None,
    )
    m.info.response_metrics = StreamedResponseMetrics(
        response_chunks=[],
        chunk_times=output_token_times,
        output_tokens=output_tokens,
        output_token_times=output_token_times,
    )
    m.info.extra_info = {}
    return m


def _assert_summary(d: typing.Any) -> None:
    """Every summary dict should carry mean/min/median/max."""
    assert isinstance(d, dict), d
    for k in ("mean", "min", "median", "max"):
        assert k in d, f"missing {k} in {d}"
        assert d[k] is not None


def test_lifecycle_report_shape_populated() -> None:
    # Three successful requests covering all modalities.
    metrics = [
        _mock_metric(
            start_time=0.0,
            end_time=0.5,
            scheduled_time=-0.001,
            input_tokens=200,
            output_tokens=60,
            request_data="a" * 50000,
            images=[
                Image(pixels=1920 * 1080, bytes=50000, aspect_ratio=16 / 9),
                Image(pixels=1280 * 720, bytes=30000, aspect_ratio=16 / 9),
            ],
            videos=[Video(pixels=1920 * 1080, bytes=1_300_000, aspect_ratio=16 / 9, frames=32)],
            audios=[Audio(bytes=60000, seconds=15)],
            output_token_times=[0.03, 0.2, 0.4, 0.5],
        ),
        _mock_metric(
            start_time=1.0,
            end_time=1.49,
            scheduled_time=0.999,
            input_tokens=205,
            output_tokens=64,
            request_data="b" * 180000,
            images=[Image(pixels=1024 * 1024, bytes=40000, aspect_ratio=1.0)],
            videos=[],
            audios=[Audio(bytes=90000, seconds=25)],
            output_token_times=[1.03, 1.2, 1.4, 1.49],
        ),
        _mock_metric(
            start_time=2.0,
            end_time=2.48,
            scheduled_time=2.001,
            input_tokens=196,
            output_tokens=36,
            request_data="c" * 1_800_000,
            images=[
                Image(pixels=3840 * 2160, bytes=110000, aspect_ratio=16 / 9),
                Image(pixels=1920 * 1080, bytes=60000, aspect_ratio=16 / 9),
                Image(pixels=800 * 600, bytes=22000, aspect_ratio=4 / 3),
            ],
            videos=[
                Video(pixels=3840 * 2160, bytes=3_000_000, aspect_ratio=16 / 9, frames=64),
                Video(pixels=960 * 720, bytes=500_000, aspect_ratio=4 / 3, frames=16),
            ],
            audios=[],
            output_token_times=[2.04, 2.2, 2.4, 2.48],
        ),
    ]

    summary = summarize_requests(
        typing.cast(typing.Any, metrics),
        percentiles=[50],
        stage_rate=1.0,
    )
    report = summary.model_dump()

    # Top-level
    assert set(report.keys()) >= {"benchmark_time_seconds", "load_summary", "successes", "failures"}
    assert report["benchmark_time_seconds"] > 0

    # load_summary
    load = report["load_summary"]
    assert load["count"] == 3
    _assert_summary(load["schedule_delay"])
    assert load["send_duration"] > 0
    assert load["requested_rate"] == 1.0
    assert load["achieved_rate"] > 0

    # successes
    s = report["successes"]
    assert s["count"] == 3

    # latency
    _assert_summary(s["latency"]["request_latency"])
    _assert_summary(s["latency"]["time_to_first_token"])

    # throughput
    t = s["throughput"]
    for k in (
        "requests_per_sec",
        "input_tokens_per_sec",
        "output_tokens_per_sec",
        "total_tokens_per_sec",
        "images_per_sec",
        "videos_per_sec",
        "audios_per_sec",
    ):
        assert k in t, f"missing throughput key {k}"
        assert t[k] > 0, f"expected {k} > 0"

    # request_size_bytes, prompt_len, output_len
    _assert_summary(s["request_size_bytes"])
    _assert_summary(s["prompt_len"])
    _assert_summary(s["output_len"])

    # image nested
    img = s["image"]
    for k in ("count", "pixels", "bytes", "aspect_ratio"):
        _assert_summary(img[k])

    # video nested
    vid = s["video"]
    for k in ("count", "frames", "pixels", "bytes", "aspect_ratio"):
        _assert_summary(vid[k])

    # audio nested
    aud = s["audio"]
    for k in ("count", "seconds", "bytes"):
        _assert_summary(aud[k])

    # failures
    f = report["failures"]
    assert f["count"] == 0
    assert f["request_latency"] is None
    assert f["prompt_len"] is None


def test_lifecycle_report_shape_with_failures() -> None:
    success = _mock_metric(
        start_time=0.0,
        end_time=0.5,
        scheduled_time=0.0,
        input_tokens=100,
        output_tokens=50,
        request_data="ok",
        images=[Image(pixels=100, bytes=1000, aspect_ratio=1.0)],
        videos=[Video(pixels=100, bytes=1000, aspect_ratio=1.0, frames=8)],
        audios=[Audio(bytes=500, seconds=3)],
        output_token_times=[0.1, 0.3, 0.5],
    )

    failure = Mock()
    failure.start_time = 1.0
    failure.end_time = 1.2
    failure.scheduled_time = 1.0
    failure.error = Mock()
    failure.ttft_slo_sec = None
    failure.tpot_slo_sec = None
    failure.request_data = "bad"
    failure.info = Mock(spec=InferenceInfo)
    failure.info.request_metrics = RequestMetrics(text=Text(input_tokens=80))
    failure.info.response_metrics = None
    failure.info.extra_info = {}

    summary = summarize_requests(typing.cast(typing.Any, [success, failure]), percentiles=[50])
    report = summary.model_dump()

    assert report["successes"]["count"] == 1
    assert report["failures"]["count"] == 1
    _assert_summary(report["failures"]["request_latency"])
    _assert_summary(report["failures"]["prompt_len"])
