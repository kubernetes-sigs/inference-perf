import pytest

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from inference_perf.distributed.worker import DistributedWorker
from inference_perf.distributed.redis_client import RedisClient
from inference_perf.client.modelserver import ModelServerClient
from inference_perf.apis import RequestLifecycleMetric, InferenceInfo


@pytest.fixture
def mock_redis_client() -> MagicMock:
    client = MagicMock(spec=RedisClient)
    client.redis = MagicMock()
    client.redis.get = AsyncMock()
    client.redis.hget = AsyncMock()
    client.redis.incr = AsyncMock()
    client.connect = AsyncMock()
    client.create_consumer_group = AsyncMock()
    client.read_tasks = AsyncMock()
    client.ack_task = AsyncMock()
    client.add_result = AsyncMock()
    client.publish_telemetry = AsyncMock()
    return client


@pytest.fixture
def mock_model_server_client() -> MagicMock:
    client = MagicMock(spec=ModelServerClient)
    client.process_request = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_worker_run(mock_redis_client: MagicMock, mock_model_server_client: MagicMock) -> None:
    worker = DistributedWorker(
        worker_id="w1",
        redis_client=mock_redis_client,
        client=mock_model_server_client,
        stream_name="s1",
        group_name="g1",
        results_stream="r1",
        telemetry_channel="t1",
    )

    mock_redis_client.redis.get.return_value = "1000.0"  # global_start_time

    worker.process_task = AsyncMock()  # type: ignore[method-assign]

    async def mock_read_tasks(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        if mock_read_tasks.calls == 0:
            mock_read_tasks.calls += 1
            return [{"_id": "123-0", "prompt_field": "p1", "scheduled_offset": 0.1}]
        else:
            await asyncio.sleep(0.1)  # Let the created task run
            raise asyncio.CancelledError()

    mock_read_tasks.calls = 0
    mock_redis_client.read_tasks.side_effect = mock_read_tasks

    try:
        await worker.run()
    except asyncio.CancelledError:
        pass

    worker.process_task.assert_called_once()


@pytest.mark.asyncio
async def test_process_task(mock_redis_client: MagicMock, mock_model_server_client: MagicMock) -> None:
    worker = DistributedWorker(
        worker_id="w1",
        redis_client=mock_redis_client,
        client=mock_model_server_client,
        stream_name="s1",
        group_name="g1",
        results_stream="r1",
        telemetry_channel="t1",
    )

    task = {"_id": "123-0", "prompt_field": "p1", "scheduled_offset": 0.1}

    mock_redis_client.redis.get.return_value = None  # No cancellation
    mock_redis_client.redis.hget.return_value = "Prompt text"

    # Mock collector to return a metric
    mock_metric = MagicMock(spec=RequestLifecycleMetric)
    mock_metric.model_dump.return_value = {"info": {}}
    mock_metric.start_time = 1000.0
    mock_metric.info = InferenceInfo(output_token_times=[1000.1, 1000.2])
    worker.collector.get_metrics = MagicMock(return_value=[mock_metric])  # type: ignore[method-assign]

    with patch("asyncio.sleep", AsyncMock()):
        await worker.process_task(task, global_start_time=1000.0)

    mock_model_server_client.process_request.assert_called_once()
    mock_redis_client.add_result.assert_called_once()
    mock_redis_client.redis.incr.assert_any_call("completed_requests")
    mock_redis_client.ack_task.assert_called_once_with("s1", "g1", "123-0")
