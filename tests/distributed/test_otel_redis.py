import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from inference_perf.datagen.otel_trace_replay_datagen import RedisEventOutputRegistry, RedisWorkerSessionTracker
from inference_perf.distributed.redis_client import RedisClient

@pytest.mark.asyncio
async def test_redis_event_output_registry_record():
    mock_redis_client = MagicMock(spec=RedisClient)
    mock_redis_client.record_event_output = AsyncMock()
    mock_redis_client.signal_event_completion = AsyncMock()
    
    registry = RedisEventOutputRegistry(mock_redis_client, job_id="test_job")
    
    registry.record("event_1", "output_text", [{"role": "user", "content": "hi"}])
    
    # Wait for background task to complete
    await asyncio.sleep(0.1)
    
    mock_redis_client.record_event_output.assert_called_once_with("test_job", "event_1", "output_text", [{"role": "user", "content": "hi"}])
    mock_redis_client.signal_event_completion.assert_called_once_with("test_job", "event_1")
    
    # Verify it is also recorded locally
    assert registry.get_output_by_event_id("event_1") == "output_text"

@pytest.mark.asyncio
async def test_redis_event_output_registry_require_async_hit():
    mock_redis_client = MagicMock(spec=RedisClient)
    mock_redis_client.get_event_output = AsyncMock(return_value="cached_output")
    mock_redis_client.get_event_messages = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
    
    registry = RedisEventOutputRegistry(mock_redis_client, job_id="test_job")
    
    output = await registry.require_async("event_1")
    
    assert output == "cached_output"
    mock_redis_client.get_event_output.assert_called_once_with("test_job", "event_1")
    mock_redis_client.get_event_messages.assert_called_once_with("test_job", "event_1")
    
    # Verify it is recorded locally
    assert registry.get_output_by_event_id("event_1") == "cached_output"

@pytest.mark.asyncio
async def test_redis_worker_session_tracker():
    mock_redis_client = MagicMock(spec=RedisClient)
    mock_redis = AsyncMock()
    mock_redis_client.redis = mock_redis
    
    tracker = RedisWorkerSessionTracker(mock_redis_client, job_id="test_job", total_events=2)
    
    # Test is_session_failed
    mock_redis.sismember.return_value = True
    assert await tracker.is_session_failed("session_1") is True
    mock_redis.sismember.assert_called_once_with("otel:test_job:failed_sessions", "session_1")
    
    # Test record_event_completed (not full)
    mock_redis.hlen.return_value = 1
    tracker.record_event_completed("session_1", "event_1", 123.45)
    
    await asyncio.sleep(0.1) # Wait for background task
    
    mock_redis.hset.assert_called_once_with("otel:test_job:session:session_1:completions", "event_1", "123.45")
    mock_redis.publish.assert_not_called()
    
    # Test record_event_completed (full)
    mock_redis.hlen.return_value = 2
    mock_redis.sismember.return_value = False # Not failed
    
    tracker.record_event_completed("session_1", "event_2", 123.46)
    
    await asyncio.sleep(0.1)
    
    mock_redis.publish.assert_called_once_with("otel:test_job:session_completed:session_1", "completed")

    # Test mark_session_failed
    await tracker.mark_session_failed("session_1")
    mock_redis.sadd.assert_called_once_with("otel:test_job:failed_sessions", "session_1")
