import json
import logging
from typing import Any, Dict, List, Optional
import redis.asyncio as redis
from redis.exceptions import ResponseError

logger = logging.getLogger(__name__)


class RedisClient:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, password: Optional[str] = None):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.redis: Optional[redis.Redis] = None

    async def connect(self):
        self.redis = redis.Redis(host=self.host, port=self.port, db=self.db, password=self.password, decode_responses=True)
        return self

    async def close(self):
        if self.redis:
            await self.redis.close()

    # Task Queue Operations
    async def push_task(self, stream_name: str, task_data: Dict[str, Any]) -> str:
        """Push a task to a Redis Stream."""
        if not self.redis:
            raise RuntimeError("Redis client not connected")
        return await self.redis.xadd(stream_name, {"data": json.dumps(task_data)})

    async def create_consumer_group(self, stream_name: str, group_name: str):
        """Create a consumer group, ignoring error if it already exists."""
        if not self.redis:
            raise RuntimeError("Redis client not connected")
        try:
            await self.redis.xgroup_create(stream_name, group_name, id="0", mkstream=True)
        except ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    async def read_tasks(self, stream_name: str, group_name: str, consumer_name: str, count: int = 1) -> List[Dict[str, Any]]:
        """Read tasks from a Stream using a consumer group."""
        if not self.redis:
            raise RuntimeError("Redis client not connected")
        entries = await self.redis.xreadgroup(
            groupname=group_name, consumername=consumer_name, streams={stream_name: ">"}, count=count
        )
        tasks = []
        for _stream, messages in entries:
            for msg_id, data in messages:
                task = json.loads(data["data"])
                task["_id"] = msg_id
                tasks.append(task)
        return tasks

    async def ack_task(self, stream_name: str, group_name: str, task_id: str):
        """Acknowledge a task."""
        if not self.redis:
            raise RuntimeError("Redis client not connected")
        await self.redis.xack(stream_name, group_name, task_id)

    # Telemetry Operations
    async def publish_telemetry(self, channel: str, data: Dict[str, Any]):
        """Publish real-time telemetry."""
        if not self.redis:
            raise RuntimeError("Redis client not connected")
        await self.redis.publish(channel, json.dumps(data))

    async def add_result(self, stream_name: str, result_data: Dict[str, Any]):
        """Add rich result to results stream."""
        if not self.redis:
            raise RuntimeError("Redis client not connected")
        await self.redis.xadd(stream_name, {"data": json.dumps(result_data)})

    async def get_subscriber(self):
        """Get a pubsub object for subscribing to channels."""
        if not self.redis:
            raise RuntimeError("Redis client not connected")
        return self.redis.pubsub()
