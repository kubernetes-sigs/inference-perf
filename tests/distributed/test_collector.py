import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from inference_perf.distributed.collector import RedisRequestDataCollector
from inference_perf.distributed.redis_client import RedisClient
from inference_perf.apis import RequestLifecycleMetric


@pytest.fixture
def mock_redis_client() -> MagicMock:
    client = MagicMock(spec=RedisClient)
    client.redis = MagicMock()
    client.redis.xread = AsyncMock()
    client.connect = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_reload_metrics(mock_redis_client: MagicMock) -> None:
    collector = RedisRequestDataCollector(mock_redis_client, "results_stream")

    # Mock data to be returned by xread
    metric_data = {
        "info": {"input_tokens": 10, "output_tokens": 20},
        "scheduled_time": 100.0,
        "start_time": 101.0,
        "end_time": 102.0,
        "request_data": "req",
        "task_id": "t1",
        "worker_id": "w1",
        "status": "success",
    }

    # xread returns [ (stream_name, [ (msg_id, {b"data": json_str}) ]) ]
    mock_redis_client.redis.xread.side_effect = [
        [("results_stream", [("1-0", {b"data": json.dumps(metric_data)})])],
        [],  # Second call returns empty to break the loop
    ]

    await collector.reload_metrics()

    assert len(collector.get_metrics()) == 1
    metric = collector.get_metrics()[0]
    assert isinstance(metric, RequestLifecycleMetric)
    assert metric.info.input_tokens == 10
    assert metric.scheduled_time == 100.0

    mock_redis_client.connect.assert_not_called()


@pytest.mark.asyncio
async def test_reload_metrics_not_connected(mock_redis_client: MagicMock) -> None:
    mock_redis_client.redis = None  # Simulate not connected

    async def mock_connect() -> None:
        mock_redis_client.redis = MagicMock()
        mock_redis_client.redis.xread = AsyncMock(return_value=[])

    mock_redis_client.connect.side_effect = mock_connect

    collector = RedisRequestDataCollector(mock_redis_client, "results_stream")
    await collector.reload_metrics()

    mock_redis_client.connect.assert_called_once()


@pytest.mark.asyncio
async def test_reload_metrics_exception_handling(mock_redis_client: MagicMock) -> None:
    collector = RedisRequestDataCollector(mock_redis_client, "results_stream")

    # Return invalid JSON
    mock_redis_client.redis.xread.side_effect = [[("results_stream", [("1-0", {b"data": "invalid json"})])], []]

    await collector.reload_metrics()

    # Should not raise exception, but metrics should be empty
    assert len(collector.get_metrics()) == 0


@pytest.mark.asyncio
async def test_start(mock_redis_client: MagicMock) -> None:
    collector = RedisRequestDataCollector(mock_redis_client, "results_stream")
    async with collector.start():
        pass  # Just verify it works as context manager
