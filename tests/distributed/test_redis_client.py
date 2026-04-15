import pytest
import json
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch
from inference_perf.distributed.redis_client import RedisClient
from redis.exceptions import ResponseError


@pytest.fixture
def mock_redis() -> Generator[MagicMock, None, None]:
    with patch("redis.asyncio.Redis") as mock:
        instance = mock.return_value
        instance.xadd = AsyncMock()
        instance.xgroup_create = AsyncMock()
        instance.xreadgroup = AsyncMock()
        instance.xack = AsyncMock()
        instance.publish = AsyncMock()
        instance.close = AsyncMock()
        instance.pubsub = MagicMock()
        yield instance


@pytest.mark.asyncio
async def test_connect() -> None:
    with patch("redis.asyncio.Redis") as mock_redis_class:
        client = RedisClient()
        await client.connect()
        assert client.redis is not None
        mock_redis_class.assert_called_once_with(host="localhost", port=6379, db=0, password=None, decode_responses=True)


@pytest.mark.asyncio
async def test_close(mock_redis: MagicMock) -> None:
    client = RedisClient()
    await client.connect()
    await client.close()
    mock_redis.close.assert_called_once()


@pytest.mark.asyncio
async def test_push_task(mock_redis: MagicMock) -> None:
    client = RedisClient()
    await client.connect()
    mock_redis.xadd.return_value = "123-0"

    task_id = await client.push_task("mystream", {"key": "value"})

    assert task_id == "123-0"
    mock_redis.xadd.assert_called_once_with("mystream", {"data": json.dumps({"key": "value"})})


@pytest.mark.asyncio
async def test_create_consumer_group_success(mock_redis: MagicMock) -> None:
    client = RedisClient()
    await client.connect()

    await client.create_consumer_group("mystream", "mygroup")

    mock_redis.xgroup_create.assert_called_once_with("mystream", "mygroup", id="0", mkstream=True)


@pytest.mark.asyncio
async def test_create_consumer_group_busygroup(mock_redis: MagicMock) -> None:
    client = RedisClient()
    await client.connect()
    mock_redis.xgroup_create.side_effect = ResponseError("BUSYGROUP Consumer Group name already exists")

    # Should not raise error
    await client.create_consumer_group("mystream", "mygroup")


@pytest.mark.asyncio
async def test_create_consumer_group_other_error(mock_redis: MagicMock) -> None:
    client = RedisClient()
    await client.connect()
    mock_redis.xgroup_create.side_effect = ResponseError("Some other error")

    with pytest.raises(ResponseError):
        await client.create_consumer_group("mystream", "mygroup")


@pytest.mark.asyncio
async def test_read_tasks(mock_redis: MagicMock) -> None:
    client = RedisClient()
    await client.connect()
    mock_redis.xreadgroup.return_value = [("mystream", [("123-0", {"data": json.dumps({"key": "value"})})])]

    tasks = await client.read_tasks("mystream", "mygroup", "myconsumer")

    assert len(tasks) == 1
    assert tasks[0]["key"] == "value"
    assert tasks[0]["_id"] == "123-0"
    mock_redis.xreadgroup.assert_called_once_with(
        groupname="mygroup", consumername="myconsumer", streams={"mystream": ">"}, count=1
    )


@pytest.mark.asyncio
async def test_ack_task(mock_redis: MagicMock) -> None:
    client = RedisClient()
    await client.connect()

    await client.ack_task("mystream", "mygroup", "123-0")

    mock_redis.xack.assert_called_once_with("mystream", "mygroup", "123-0")


@pytest.mark.asyncio
async def test_publish_telemetry(mock_redis: MagicMock) -> None:
    client = RedisClient()
    await client.connect()

    await client.publish_telemetry("mychannel", {"key": "value"})

    mock_redis.publish.assert_called_once_with("mychannel", json.dumps({"key": "value"}))


@pytest.mark.asyncio
async def test_add_result(mock_redis: MagicMock) -> None:
    client = RedisClient()
    await client.connect()

    await client.add_result("mystream", {"key": "value"})

    mock_redis.xadd.assert_called_once_with("mystream", {"data": json.dumps({"key": "value"})})


@pytest.mark.asyncio
async def test_get_subscriber(mock_redis: MagicMock) -> None:
    client = RedisClient()
    await client.connect()
    mock_redis.pubsub.return_value = MagicMock()

    pubsub = await client.get_subscriber()

    assert pubsub is not None
    mock_redis.pubsub.assert_called_once()


@pytest.mark.asyncio
async def test_not_connected() -> None:
    client = RedisClient()
    with pytest.raises(RuntimeError, match="Redis client not connected"):
        await client.push_task("s", {})
