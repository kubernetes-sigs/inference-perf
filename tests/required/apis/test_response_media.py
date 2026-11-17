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
"""Response media on ResponseMetrics, as it lands in per_request_lifecycle_metrics.json."""

from inference_perf.apis import InferenceInfo, RequestLifecycleMetric, StreamedResponseMetrics, UnaryResponseMetrics
from inference_perf.payloads import Audio, Audios, Image, Images, RequestMetrics, Text


# A unary text response with 3 output tokens, and a streamed one with no chunks.
# Expects neither dump to carry image, video or audio keys, so a text run's
# per-request JSON is unchanged.
def test_text_response_serializes_without_media_keys() -> None:
    assert UnaryResponseMetrics(output_tokens=3).model_dump() == {"output_tokens": 3, "server_usage": None}
    streamed = StreamedResponseMetrics(output_tokens=3).model_dump()
    assert not {"image", "video", "audio"} & streamed.keys()


# A response returning one 1 MP image and one 2s clip, inside a full lifecycle
# record. Expects both to survive a JSON round trip unchanged and video, which the
# response did not return, to stay out of the dump.
def test_response_media_round_trips_through_the_lifecycle_record() -> None:
    response = UnaryResponseMetrics(
        output_tokens=0,
        image=Images(count=1, instances=[Image(pixels=1024 * 1024, bytes=2000, aspect_ratio=1.0)]),
        audio=Audios(count=1, instances=[Audio(bytes=64_000, seconds=2.0)]),
    )
    record = RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=0.0,
        start_time=0.0,
        end_time=1.0,
        request_data="{}",
        info=InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=4)), response_metrics=response),
        error=None,
    )

    dumped = record.model_dump(mode="json")
    restored = RequestLifecycleMetric.model_validate_json(record.model_dump_json())

    assert "video" not in dumped["info"]["response_metrics"]
    assert restored.info.response_metrics is not None
    assert restored.info.response_metrics.image == response.image
    assert restored.info.response_metrics.audio == response.audio
