import json
import logging
from typing import List, AsyncIterator
from contextlib import asynccontextmanager
from inference_perf.apis import RequestLifecycleMetric, InferenceInfo, ErrorResponseInfo
from inference_perf.client.requestdatacollector.base import RequestDataCollector
from inference_perf.distributed.redis_client import RedisClient

logger = logging.getLogger(__name__)


class RedisRequestDataCollector(RequestDataCollector):
    def __init__(self, redis_client: RedisClient, stream_name: str):
        self.redis = redis_client
        self.stream_name = stream_name
        self.metrics: List[RequestLifecycleMetric] = []

    def record_metric(self, metric: RequestLifecycleMetric) -> None:
        # Not used by orchestrator to record, as workers push to Redis directly.
        pass

    def get_metrics(self) -> List[RequestLifecycleMetric]:
        return self.metrics

    async def reload_metrics(self) -> None:
        """Fetch all metrics from Redis results stream with pagination and optimization."""
        if not self.redis.redis:
            await self.redis.connect()
        assert self.redis.redis is not None

        logger.info(f"Reloading metrics from Redis stream {self.stream_name}...")

        self.metrics = []
        last_id = "0"
        count = 5000

        while True:
            # Read messages from stream in chunks
            entries = await self.redis.redis.xread({self.stream_name: last_id}, count=count)
            if not entries:
                break

            messages_processed = 0
            for _stream, messages in entries:
                for msg_id, data in messages:
                    messages_processed += 1
                    last_id = msg_id  # Update last_id to continue from here

                    try:
                        # Redis stores data as bytes, and xread returns dict with field:value
                        json_str = data.get(b"data") or data.get("data")
                        if not json_str:
                            logger.warning(f"No data field in message {msg_id}")
                            continue

                        result_data = json.loads(json_str)

                        # Pop custom fields not in RequestLifecycleMetric schema before validation
                        result_data.pop("task_id", None)
                        result_data.pop("worker_id", None)
                        result_data.pop("status", None)

                        # Reconstruct RequestLifecycleMetric with model_construct for speed
                        info_data = result_data.pop("info", {})
                        info = InferenceInfo.model_construct(**info_data)

                        error_data = result_data.pop("error", None)
                        error = ErrorResponseInfo.model_construct(**error_data) if error_data else None

                        metric = RequestLifecycleMetric.model_construct(info=info, error=error, **result_data)
                        self.metrics.append(metric)

                    except Exception as e:
                        logger.error(f"Failed to parse metric from message {msg_id}: {e}")

            if messages_processed == 0:
                break

        logger.info(f"Successfully reloaded {len(self.metrics)} metrics from Redis.")

    @asynccontextmanager
    async def start(self) -> AsyncIterator[None]:
        # Nothing to start/stop for this collector as it polls on demand
        yield
