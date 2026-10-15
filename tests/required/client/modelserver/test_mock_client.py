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
"""The mock client records what the real client records, including cancellations."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from inference_perf.apis.base import STAGE_TEARDOWN_CANCELLED_ERROR_TYPE
from inference_perf.client.modelserver.mock_client import MockModelServerClient
from inference_perf.config import APIConfig, APIType


# A mock request with 60s latency, cancelled 50ms in. Expects one metric recorded
# as a StageTeardownCancelled failure for stage 2, and the cancellation re-raised.
@pytest.mark.asyncio
async def test_cancelled_in_flight_request_is_recorded_then_reraised() -> None:
    collector = MagicMock()
    client = MockModelServerClient(collector, APIConfig(type=APIType.Completion), mock_latency=60)
    data = MagicMock()
    data.to_request_body = AsyncMock(return_value={"prompt": "x"})

    task = asyncio.create_task(client.process_request(data, stage_id=2, scheduled_time=0.0))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    collector.record_metric.assert_called_once()
    metric = collector.record_metric.call_args[0][0]
    assert metric.stage_id == 2
    assert metric.error.error_type == STAGE_TEARDOWN_CANCELLED_ERROR_TYPE
