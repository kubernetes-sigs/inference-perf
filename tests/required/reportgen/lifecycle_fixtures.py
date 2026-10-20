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
"""Request lifecycle metric builders shared by the reportgen tests.

Each builder returns a `Mock` shaped like the per-request record that
`summarize_requests` consumes, so tests in this directory describe a request the
same way instead of keeping private copies that drift apart.
"""

import typing
from unittest.mock import Mock

from inference_perf.apis import ErrorResponseInfo, InferenceInfo, StreamedResponseMetrics
from inference_perf.payloads import Audio, Audios, Image, Images, RequestMetrics, Text, Video, Videos


# Builds one successful request: the given timestamps, token counts and media, with
# no error, no SLOs and zero retries. Media lists that are empty become None.
def mock_metric(
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
    # Real int/bool, not auto-specced Mocks: the retry rollup does arithmetic on these.
    m.info.retries_attempted = 0
    m.info.retries_recovered = False
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


# Builds one request that failed with HTTP 500 "Internal Server Error": the given
# timestamps and prompt size, no response metrics, zero retries.
def failed_metric(
    *,
    start_time: float,
    end_time: float,
    scheduled_time: float,
    input_tokens: int,
    request_data: str = "bad",
) -> Mock:
    m = Mock()
    m.start_time = start_time
    m.end_time = end_time
    m.scheduled_time = scheduled_time
    m.error = ErrorResponseInfo(error_type="HTTP Error 500", error_msg="Internal Server Error")
    m.session_id = None
    m.ttft_slo_sec = None
    m.tpot_slo_sec = None
    m.request_data = request_data
    m.info = Mock(spec=InferenceInfo)
    m.info.retries_attempted = 0
    m.info.retries_recovered = False
    m.info.request_metrics = RequestMetrics(text=Text(input_tokens=input_tokens))
    m.info.response_metrics = None
    m.info.extra_info = {}
    return m


# Builds one successful 1s request (10 tokens in, 5 out) carrying the given retry
# counters, e.g. retry_metric(2, True, 4.0) = retried twice, recovered, 4.0s wasted.
def retry_metric(
    retries_attempted: int,
    retries_recovered: bool,
    retry_wasted_sec: typing.Optional[float] = None,
) -> Mock:
    """A minimal successful request metric carrying retry counters."""
    m = mock_metric(
        start_time=0.0,
        end_time=1.0,
        scheduled_time=0.0,
        input_tokens=10,
        output_tokens=5,
        request_data="req",
        images=[],
        videos=[],
        audios=[],
        output_token_times=[0.5, 1.0],
    )
    m.info.retries_attempted = retries_attempted
    m.info.retries_recovered = retries_recovered
    # Real float or None, never an auto-specced Mock: the retry rollup feeds this
    # straight into summarize().
    m.info.retry_wasted_sec = retry_wasted_sec
    return m
