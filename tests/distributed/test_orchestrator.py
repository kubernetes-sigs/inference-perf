import pytest
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, ANY
from inference_perf.distributed.orchestrator import Orchestrator
from inference_perf.distributed.redis_client import RedisClient
from inference_perf.config import LoadConfig, DataConfig, APIConfig, StandardLoadStage


@pytest.fixture
def mock_redis_client() -> MagicMock:
    client = MagicMock(spec=RedisClient)
    client.redis = MagicMock()
    client.redis.set = AsyncMock()
    client.redis.delete = AsyncMock()
    client.redis.hset = AsyncMock()
    client.redis.get = AsyncMock()
    client.redis.xread = AsyncMock()
    client.redis.publish = AsyncMock()
    client.push_task = AsyncMock()
    client.connect = AsyncMock()
    return client


@pytest.fixture
def mock_configs() -> tuple[LoadConfig, DataConfig, APIConfig]:
    load_config = LoadConfig(stages=[StandardLoadStage(rate=10, duration=2)])
    data_config = DataConfig(type="mock")
    api_config = APIConfig(type="completion")
    return load_config, data_config, api_config


@pytest.mark.asyncio
@patch("inference_perf.distributed.orchestrator.CustomTokenizer")
@patch("inference_perf.distributed.orchestrator.SharedPrefixDataGenerator")
async def test_run_job_with_configs(
    mock_datagen_class: MagicMock,
    mock_tokenizer_class: MagicMock,
    mock_redis_client: MagicMock,
    mock_configs: tuple[LoadConfig, DataConfig, APIConfig],
) -> None:
    load_config, data_config, api_config = mock_configs

    mock_datagen = mock_datagen_class.return_value
    mock_datagen.prompts = ["prompt1", "prompt2"]
    mock_datagen.flat_output_lens = [10, 20]

    orchestrator = Orchestrator(mock_redis_client)

    # Mock wait_for_completion to avoid infinite loop
    orchestrator.wait_for_completion = AsyncMock()  # type: ignore[method-assign]

    await orchestrator.run_job_with_configs(load_config, data_config, api_config, tokenizer_config=None)

    # Verify Redis interactions
    mock_redis_client.redis.set.assert_any_call("global_start_time", ANY)
    mock_redis_client.redis.delete.assert_any_call("results_stream")

    # Verify tasks were dispatched
    assert mock_redis_client.push_task.call_count == 20  # 10 rate * 2 duration

    # Verify prime_redis called hset
    assert mock_redis_client.redis.hset.call_count == 2


@pytest.mark.asyncio
async def test_wait_for_completion(mock_redis_client: MagicMock) -> None:
    orchestrator = Orchestrator(mock_redis_client)

    # Mock redis.get to simulate completion
    # First call for cancel, second for completed, third for cancel, fourth for completed
    mock_redis_client.redis.get.side_effect = [None, "5", None, "20"]

    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        await orchestrator.wait_for_completion(total_requests=20)

        assert mock_sleep.call_count == 2


@pytest.mark.asyncio
async def test_wait_for_completion_cancellation(mock_redis_client: MagicMock) -> None:
    orchestrator = Orchestrator(mock_redis_client)

    # Mock redis.get to simulate cancellation
    mock_redis_client.redis.get.side_effect = ["1", "0"]  # cancel_job is "1"

    with patch("asyncio.sleep", AsyncMock()):
        await orchestrator.wait_for_completion(total_requests=20)

        # Should break loop early
        mock_redis_client.redis.get.assert_called_with("cancel_job")


@pytest.mark.asyncio
@patch("inference_perf.distributed.orchestrator.Config")
async def test_start_daemon(mock_config_class: MagicMock, mock_redis_client: MagicMock) -> None:
    orchestrator = Orchestrator(mock_redis_client)

    # Mock xread to return a job message, then raise CancelledError to stop daemon
    mock_redis_client.redis.xread.return_value = [
        ("job_stream", [("msg-1", {"config": json.dumps({"load": {}, "data": {}, "api": {}})})])
    ]

    mock_cfg = MagicMock()
    mock_cfg.load = MagicMock()
    mock_cfg.data = MagicMock()
    mock_cfg.api = MagicMock()
    mock_cfg.tokenizer = MagicMock()
    mock_config_class.model_validate_json.return_value = mock_cfg

    orchestrator.run_job_with_configs = AsyncMock()  # type: ignore[method-assign]

    # We need to make it stop after one iteration
    # Let's make xread raise CancelledError on second call
    mock_redis_client.redis.xread.side_effect = [
        [("job_stream", [("msg-1", {"config": json.dumps({"load": {}, "data": {}, "api": {}})})])],
        asyncio.CancelledError(),
    ]

    try:
        await orchestrator.start_daemon()
    except asyncio.CancelledError:
        pass

    orchestrator.run_job_with_configs.assert_called_once()
