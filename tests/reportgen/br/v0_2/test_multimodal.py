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
from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from inference_perf.apis import ErrorResponseInfo, InferenceInfo, RequestLifecycleMetric
from inference_perf.apis import UnaryResponseMetrics
from inference_perf.payloads import Audio, Audios, Image, Images, RequestMetrics, Text
from inference_perf.payloads import Video, Videos
from inference_perf.reportgen.base import summarize_requests
from inference_perf.reportgen.br.v0_2 import build_partial_report
from inference_perf.reportgen.br.v0_2.schema import BenchmarkReportV021


@pytest.fixture
def mixed_requests() -> list[RequestLifecycleMetric]:
    media = RequestMetrics(
        text=Text(input_tokens=4),
        image=Images(
            count=2,
            instances=[
                Image(pixels=100, bytes=10, aspect_ratio=1),
                Image(pixels=300, bytes=30, aspect_ratio=2),
            ],
        ),
        video=Videos(
            count=2,
            instances=[
                Video(pixels=200, bytes=20, aspect_ratio=1, frames=2),
                Video(pixels=600, bytes=60, aspect_ratio=2, frames=6),
            ],
        ),
        audio=Audios(
            count=2,
            instances=[
                Audio(bytes=100, seconds=1),
                Audio(bytes=300, seconds=3),
            ],
        ),
    )
    first = RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=10,
        start_time=10,
        end_time=12,
        request_data='{"prompt":"é"}',
        response_data="ok",
        info=InferenceInfo(
            request_metrics=media,
            response_metrics=UnaryResponseMetrics(output_tokens=2),
        ),
        error=None,
    )
    text = first.model_copy(deep=True)
    text.start_time = 11
    text.end_time = 14
    text.request_data = "{}"
    text.info.request_metrics = RequestMetrics(text=Text(input_tokens=2))
    failed = first.model_copy(deep=True)
    failed.end_time = 40
    failed.request_data = "excluded" * 100
    failed.error = ErrorResponseInfo(error_type="timeout", error_msg="timed out")
    return [first, text, failed]


def _aggregate(metrics: list[RequestLifecycleMetric]) -> dict[str, Any]:
    report = build_partial_report(metrics, None, run_uid="test")
    # Validate the serialized form consumed by downstream report composers.
    decoded = yaml.safe_load(yaml.safe_dump(report))
    BenchmarkReportV021.model_validate(decoded)
    return cast(dict[str, Any], decoded["results"]["request_performance"]["aggregate"])


def test_request_size_uses_successful_utf8_payloads(
    mixed_requests: list[RequestLifecycleMetric],
) -> None:
    sizes = [len(m.request_data.encode("utf-8")) for m in mixed_requests[:2]]
    stats = _aggregate(mixed_requests)["requests"]["request_size"]
    assert stats["units"] == "bytes"
    assert stats["mean"] == sum(sizes) / 2
    assert stats["min"] == min(sizes)
    assert stats["max"] == max(sizes)
    assert stats["p50"] == sum(sizes) / 2


@pytest.mark.parametrize(
    "modality,field,native_field,unit,expected",
    [
        ("image", "count", "count", "count", 1),
        ("image", "filesize", "bytes", "bytes", 20),
        ("image", "pixels", "pixels", "pixels", 200),
        ("image", "aspect_ratio", "aspect_ratio", "ratio", 1.5),
        ("video", "count", "count", "count", 1),
        ("video", "filesize", "bytes", "bytes", 40),
        ("video", "pixels", "pixels", "pixels", 400),
        ("video", "aspect_ratio", "aspect_ratio", "ratio", 1.5),
        ("video", "frames", "frames", "count", 4),
        ("audio", "count", "count", "count", 1),
        ("audio", "filesize", "bytes", "bytes", 200),
        ("audio", "duration", "seconds", "s", 2),
    ],
)
def test_media_statistics_match_native_report(
    mixed_requests: list[RequestLifecycleMetric],
    modality: str,
    field: str,
    native_field: str,
    unit: str,
    expected: float,
) -> None:
    stats = _aggregate(mixed_requests)["requests"]["multimodal"][modality][field]
    native = summarize_requests(mixed_requests, [50]).successes
    assert stats["units"] == unit
    assert stats["mean"] == expected
    assert stats["mean"] == native[modality][native_field]["mean"]


@pytest.mark.parametrize(
    "modality,unit",
    [
        ("image", "images/s"),
        ("video", "videos/s"),
        ("audio", "audios/s"),
    ],
)
def test_media_rates_exclude_failures_but_use_full_window(
    mixed_requests: list[RequestLifecycleMetric],
    modality: str,
    unit: str,
) -> None:
    stats = _aggregate(mixed_requests)["throughput"][modality + "_rate"]
    assert stats == {"units": unit, "mean": pytest.approx(2 / 30)}
    native = summarize_requests(mixed_requests, [50]).successes
    assert stats["mean"] == native["throughput"][modality + "s_per_sec"]


def test_text_only_omits_media_fields(mixed_requests: list[RequestLifecycleMetric]) -> None:
    aggregate = _aggregate([mixed_requests[1]])
    assert "multimodal" not in aggregate["requests"]
    assert "request_size" in aggregate["requests"]
    for modality in ("image", "video", "audio"):
        assert modality + "_rate" not in aggregate["throughput"]


def test_all_failed_omits_new_measurements(mixed_requests: list[RequestLifecycleMetric]) -> None:
    aggregate = _aggregate([mixed_requests[2]])
    assert "request_size" not in aggregate["requests"]
    assert "multimodal" not in aggregate["requests"]
    assert "throughput" not in aggregate


def test_zero_window_omits_rates(mixed_requests: list[RequestLifecycleMetric]) -> None:
    metric = mixed_requests[0]
    metric.end_time = metric.start_time
    aggregate = _aggregate([metric])
    assert "multimodal" in aggregate["requests"]
    assert "throughput" not in aggregate


def test_recorded_zero_count_is_preserved(mixed_requests: list[RequestLifecycleMetric]) -> None:
    metric = mixed_requests[1]
    metric.info.request_metrics.image = Images()
    aggregate = _aggregate([metric])
    image = aggregate["requests"]["multimodal"]["image"]
    assert image["count"]["mean"] == 0
    assert "filesize" not in image
    assert "video" not in aggregate["requests"]["multimodal"]
    assert aggregate["throughput"]["image_rate"]["mean"] == 0


def test_instance_statistics_are_not_averages_of_requests(
    mixed_requests: list[RequestLifecycleMetric],
) -> None:
    second = mixed_requests[1]
    second.info.request_metrics.image = Images(
        count=1,
        instances=[Image(pixels=800, bytes=80, aspect_ratio=3)],
    )
    image = _aggregate(mixed_requests)["requests"]["multimodal"]["image"]
    assert image["count"]["mean"] == 1.5
    assert image["pixels"]["mean"] == 400
    assert image["filesize"]["mean"] == 40
    assert image["filesize"]["p50"] == 30
    assert image["aspect_ratio"]["mean"] == 2
