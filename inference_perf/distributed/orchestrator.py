import asyncio
import logging
import time
import json
from typing import Any, Dict, List
from inference_perf.config import Config, LoadConfig, DataConfig, APIConfig
from inference_perf.datagen.shared_prefix_datagen import SharedPrefixDataGenerator
from inference_perf.distributed.redis_client import RedisClient
from inference_perf.utils.custom_tokenizer import CustomTokenizer

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(
        self,
        redis_client: RedisClient,
        load_config: LoadConfig = None,
        data_config: DataConfig = None,
        api_config: APIConfig = None,
        tokenizer_config: Any = None
    ):
        self.redis = redis_client
        self.load_config = load_config
        self.data_config = data_config
        self.api_config = api_config
        self.tokenizer_config = tokenizer_config
        self.stream_name = "task_stream"
        self.results_stream = "results_stream"
        self.telemetry_channel = "telemetry_channel"
        self.group_name = "worker_group"

    async def run(self):
        """Original run method, kept for backward compatibility if needed."""
        await self.redis.connect()
        await self.run_job_with_configs(self.load_config, self.data_config, self.api_config, self.tokenizer_config)

    async def run_job_with_configs(self, load_config, data_config, api_config, tokenizer_config, job_id=None):
        # Initialize tokenizer
        tokenizer = CustomTokenizer(tokenizer_config)
        
        # Initialize data generator
        datagen = SharedPrefixDataGenerator(api_config, data_config, tokenizer)
        
        # Prime Redis with prompts
        await self.prime_redis(datagen)
        
        # Set global start time
        global_start_time = time.time() + 5  # Give workers time to start
        await self.redis.redis.set("global_start_time", str(global_start_time))
        await self.redis.redis.set("completed_requests", "0")
        await self.redis.redis.delete(self.results_stream)  # Clear results from previous runs
        await self.redis.redis.delete("cancel_job")  # Clear cancellation flag
        
        # Clear per-stage counters
        for i in range(len(load_config.stages)):
            await self.redis.redis.delete(f"completed_requests_stage_{i}")
            
        # Generate and push tasks
        total_requests = await self.dispatch_tasks(datagen, load_config, global_start_time)
        
        # Wait for completion
        await self.wait_for_completion(total_requests)
        
        # Post-processing
        await self.post_process()
        
        # Cleanup
        await self.cleanup()
        
        if job_id:
            logger.info(f"Publishing completion for job {job_id}")
            await self.redis.redis.publish("job_status", json.dumps({"job_id": job_id.decode('utf-8') if isinstance(job_id, bytes) else job_id, "status": "completed"}))

    async def start_daemon(self):
        await self.redis.connect()
        logger.info("Orchestrator daemon started, waiting for jobs...")
        
        # Start reading from the end of the stream
        last_id = "$"
        
        while True:
            try:
                # Read from job_stream
                streams = await self.redis.redis.xread({"job_stream": last_id}, block=0, count=1)
                
                if streams:
                    for stream_name, messages in streams:
                        for message_id, message_data in messages:
                            if b"config" in message_data:
                                config_json = message_data[b"config"].decode("utf-8")
                            elif "config" in message_data:
                                config_json = message_data["config"]
                                if isinstance(config_json, bytes):
                                    config_json = config_json.decode("utf-8")
                            else:
                                logger.error(f"Message data missing config key: {message_data}")
                                continue
                            logger.info(f"Received job with message_id: {message_id}")
                            
                            # Parse config
                            cfg = Config.model_validate_json(config_json)
                            
                            # Run the job
                            await self.run_job_with_configs(cfg.load, cfg.data, cfg.api, cfg.tokenizer, message_id)
                            
                            # Update last_id to continue from this message
                            last_id = message_id
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in orchestrator daemon: {e}")
                await asyncio.sleep(1)

    async def prime_redis(self, datagen: SharedPrefixDataGenerator):
        logger.info("Priming Redis with prompts...")
        # Store prompts in hash
        prompts = datagen.prompts
        for i, prompt in enumerate(prompts):
            await self.redis.redis.hset("test_prompts", f"prompt_{i}", prompt)
        logger.info(f"Stored {len(prompts)} prompts in Redis.")

    async def dispatch_tasks(self, datagen: SharedPrefixDataGenerator, load_config: LoadConfig, global_start_time: float) -> int:
        logger.info("Dispatching tasks...")
        
        total_requests = 0
        current_offset = 0.0
        
        for stage_id, stage in enumerate(load_config.stages):
            rate = stage.rate
            duration = stage.duration
            num_requests = int(rate * duration)
            
            logger.info(f"Staging stage {stage_id}: rate={rate}, duration={duration}, requests={num_requests}")
            
            for i in range(num_requests):
                data_index = (total_requests + i) % len(datagen.prompts)
                task = {
                    "prompt_field": f"prompt_{data_index}",
                    "output_len": datagen.flat_output_lens[data_index],
                    "scheduled_offset": current_offset + (i / rate),
                    "stage_id": stage_id,
                    "node_id": f"stage_{stage_id}_req_{i}",
                    "global_start_time": global_start_time
                }
                await self.redis.push_task(self.stream_name, task)
                
            current_offset += duration
            total_requests += num_requests
            
        logger.info(f"Dispatched {total_requests} tasks in total.")
        return total_requests

    async def wait_for_completion(self, total_requests: int):
        logger.info("Waiting for completion...")
        completed = 0
        while completed < total_requests:
            await asyncio.sleep(1)
            
            # Check for cancellation
            cancel_requested = await self.redis.redis.get("cancel_job")
            if cancel_requested == b"1" or cancel_requested == "1":
                logger.info("Job cancellation requested!")
                break
                
            completed_str = await self.redis.redis.get("completed_requests")
            completed = int(completed_str) if completed_str else 0
            logger.info(f"Completed {completed}/{total_requests} requests")
            
        logger.info("All tasks completed.")

    async def post_process(self):
        logger.info("Running post-processing...")
        # TODO: Implement summary report generation by draining the results stream.
        # This will involve aggregating metrics like TTFT, ITL, and latency.

    async def cleanup(self):
        logger.info("Cleaning up...")
        await self.redis.redis.delete(self.stream_name)
        await self.redis.redis.delete("test_prompts")
        await self.redis.redis.delete("global_start_time")
        await self.redis.redis.delete("completed_requests")
        # Do not close connection here if we want to continue running as daemon!
        # await self.redis.close()
