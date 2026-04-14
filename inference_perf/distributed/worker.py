import asyncio
from typing import Any
import logging
import time
from inference_perf.client.modelserver import ModelServerClient
from inference_perf.distributed.redis_client import RedisClient
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.client.requestdatacollector import LocalRequestDataCollector

logger = logging.getLogger(__name__)


class DistributedWorker:
    def __init__(
        self,
        worker_id: str,
        redis_client: RedisClient,
        client: ModelServerClient,
        stream_name: str,
        group_name: str,
        results_stream: str,
        telemetry_channel: str,
        max_concurrency: int = 10,
        publish_interval: int = 1,  # Publish telemetry every N requests
    ):
        self.worker_id = worker_id
        self.redis = redis_client
        self.client = client
        self.stream_name = stream_name
        self.group_name = group_name
        self.results_stream = results_stream
        self.telemetry_channel = telemetry_channel
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.stop_event = asyncio.Event()
        self.publish_interval = publish_interval
        self.request_counter = 0
        self.collector = LocalRequestDataCollector()
        self.client.metrics_collector = self.collector

    async def run(self) -> None:
        await self.redis.connect()
        assert self.redis.redis is not None
        await self.redis.create_consumer_group(self.stream_name, self.group_name)

        # Fetch global start time from Redis
        global_start_time_str = await self.redis.redis.get("global_start_time")
        if not global_start_time_str:
            logger.error("Global start time not found in Redis! Waiting...")
            while not global_start_time_str and not self.stop_event.is_set():
                await asyncio.sleep(1)
                global_start_time_str = await self.redis.redis.get("global_start_time")

            if self.stop_event.is_set():
                return

        global_start_time = float(global_start_time_str)
        logger.info(f"Worker {self.worker_id} started. Global start time: {global_start_time}")

        while not self.stop_event.is_set():
            try:
                # Read tasks
                tasks = await self.redis.read_tasks(self.stream_name, self.group_name, self.worker_id, count=1)
                if not tasks:
                    await asyncio.sleep(0.1)
                    continue

                for task in tasks:
                    # Claiming is handled by XREADGROUP (it reads pending if crashed, or new)
                    # We might need XAUTOCLAIM for claiming tasks from OTHER dead workers,
                    # but let's stick to XREADGROUP for now as requested.
                    asyncio.create_task(self.process_task(task, global_start_time))

            except Exception as e:
                if "NOGROUP" in str(e):
                    logger.info("Consumer group not found, trying to create it...")
                    await self.redis.create_consumer_group(self.stream_name, self.group_name)
                else:
                    logger.error(f"Error in worker loop: {e}")
                await asyncio.sleep(1)

    async def process_task(self, task: dict[str, Any], global_start_time: float) -> None:
        assert self.redis.redis is not None
        await self.semaphore.acquire()
        task_id = task["_id"]
        stage_id = task.get("stage_id", 0)
        lora_adapter = task.get("lora_adapter")

        try:
            scheduled_offset = task["scheduled_offset"]
            # Use global_start_time from task if available, fallback to cached one
            task_global_start_time = task.get("global_start_time", global_start_time)
            target_time = task_global_start_time + scheduled_offset

            # Wait until scheduled time
            now = time.time()
            sleep_time = target_time - now
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            # Check for cancellation
            cancel_requested = await self.redis.redis.get("cancel_job")
            if cancel_requested == b"1" or cancel_requested == "1":
                logger.info(f"Task {task_id} canceled because job was canceled.")
                await self.redis.ack_task(self.stream_name, self.group_name, task_id)
                return

            # Check task type
            if task.get("type") == "otel_trace_replay":
                # OTel replay mode
                event_id = task["event_id"]
                messages = task["messages"]
                predecessor_event_ids = task["predecessor_event_ids"]
                wait_ms = task["wait_ms"]
                input_segments = task["input_segments"]
                expected_output_tokens = task.get("expected_output_tokens", 50)
                total_events = task.get("total_events_in_session", 1)

                # Initialize Redis registries
                job_id = await self.redis.redis.get("current_job_id")
                if job_id:
                    job_id = job_id.decode("utf-8") if isinstance(job_id, bytes) else job_id
                else:
                    job_id = "default_job"
                
                from inference_perf.datagen.otel_trace_replay_datagen import RedisEventOutputRegistry, RedisWorkerSessionTracker, OTelChatCompletionAPIData
                from inference_perf.datagen.otel_trace_to_replay_graph import InputSegment
                from inference_perf.apis.chat import ChatMessage
                
                registry = RedisEventOutputRegistry(self.redis, job_id)
                tracker = RedisWorkerSessionTracker(self.redis, job_id, total_events)

                # Reconstruct InputSegment objects
                segments = []
                for seg in input_segments:
                    segments.append(InputSegment(
                        type=seg["type"],
                        message_count=seg["message_count"],
                        token_count=seg["token_count"],
                        source_event_id=seg.get("source_event_id")
                    ))

                chat_messages = [ChatMessage(role=m["role"], content=m["content"]) for m in messages]

                api_data = OTelChatCompletionAPIData(
                    messages=chat_messages,
                    max_tokens=expected_output_tokens,
                    event_id=event_id,
                    registry=registry,
                    worker_tracker=tracker,
                    completion_queue=None,
                    total_events_in_session=total_events,
                    input_segments=segments,
                )

                # Wait for predecessors and substitute!
                await api_data.wait_for_predecessors_and_substitute()
                
                if api_data.skip_request:
                    logger.info(f"Task {task_id} skipped due to predecessor failure or session skip.")
                    await self.redis.ack_task(self.stream_name, self.group_name, task_id)
                    return
            else:
                # Legacy mode
                # Fetch prompt from Redis Hash
                prompt_field = task["prompt_field"]
                prompt_text = await self.redis.redis.hget("test_prompts", prompt_field)  # type: ignore[misc]

                if not prompt_text:
                    logger.error(f"Prompt not found for field {prompt_field}")
                    await self.redis.ack_task(self.stream_name, self.group_name, task_id)
                    return

                # Construct API Data
                api_data = CompletionAPIData(prompt=prompt_text, max_tokens=task.get("output_len", 50))

            # Execute request
            start_perf = time.perf_counter()
            start_wall = time.time()

            # Note: process_request logs metrics to its own collector if configured.
            # Here we want to capture results for Redis.
            # We might need to wrap it or modify it to return metrics.
            # Let's assume process_request handles it and we can access results,
            # or we implement the call here directly if process_request is too tight.
            # Looking at load_generator.py, it calls client.process_request directly.

            try:
                # We need to pass a valid InferenceAPIData and scheduled_time
                await self.client.process_request(api_data, stage_id, target_time, lora_adapter)
                status = "success"
            except Exception as e:
                logger.error(f"Request failed: {e}")
                status = "failed"

            end_perf = time.perf_counter()
            latency = end_perf - start_perf
            end_wall = start_wall + latency

            # Get metric from collector
            metrics = self.collector.get_metrics()
            metric_data = {}
            metric = None
            if metrics:
                metric = metrics.pop()  # Get the latest recorded metric
                metric_data = metric.model_dump()
                metric_data["task_id"] = task_id
                metric_data["worker_id"] = self.worker_id
                metric_data["status"] = status
            else:
                # Fallback if no metric was recorded
                metric_data = {
                    "task_id": task_id,
                    "node_id": task.get("node_id"),
                    "status": status,
                    "latency": latency,
                    "start_time": start_wall,
                    "end_time": end_wall,
                    "worker_id": self.worker_id,
                }

            await self.redis.add_result(self.results_stream, metric_data)
            await self.redis.redis.incr("completed_requests")
            await self.redis.redis.incr(f"completed_requests_stage_{stage_id}")

            # Report live telemetry to Pub/Sub
            self.request_counter += 1
            if self.request_counter % self.publish_interval == 0:
                ttft = None
                itl = None
                if metric:
                    if metric.info.output_token_times and len(metric.info.output_token_times) > 0:
                        ttft = metric.info.output_token_times[0] - metric.start_time
                        if len(metric.info.output_token_times) > 1:
                            dur = metric.info.output_token_times[-1] - metric.info.output_token_times[0]
                            itl = dur / (len(metric.info.output_token_times) - 1)

                telemetry_data = {"worker_id": self.worker_id, "status": status, "latency": latency, "ttft": ttft, "itl": itl}
                await self.redis.publish_telemetry(self.telemetry_channel, telemetry_data)

            # Acknowledge task
            await self.redis.ack_task(self.stream_name, self.group_name, task_id)

        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
        finally:
            self.semaphore.release()

    def stop(self) -> None:
        self.stop_event.set()
