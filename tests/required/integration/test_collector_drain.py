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
"""A slow collector must still receive every metric the workers recorded (#913).

A worker's ``record_metric`` returns once the metric is in its ``mp.Queue`` feeder buffer, not once it is in the pipe.
When the main process collects slower than the workers produce, the buffers still hold metrics when the run ends, and the collector's ``None`` sentinel can reach the pipe ahead of them.
The collector stops at the sentinel, so those requests leave the report without an error.
"""

import asyncio
import time
from multiprocessing.queues import JoinableQueue
from typing import List, Optional, Union

from fake_openai_server import FakeOpenAIServer, StreamEvent
from test_worker_matrix import FakeServerClient, make_datagen

from inference_perf.apis import RequestLifecycleMetric
from inference_perf.config import APIConfig, APIType, LoadConfig, LoadType, StandardLoadStage
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.metrics.request_collector import MultiprocessRequestMetricCollector
from inference_perf.utils.mp_context import MP_CONTEXT

RATE = 40
DURATION = 1
NUM_WORKERS = 2

# Each response carries about 128 KB of text, so one metric fills a Linux pipe (64 KB) and a backlog stays in the workers' feeder buffers instead of in the pipe.
RESPONSE_SCRIPT = [StreamEvent("content", f" w{i}" + "x" * 32_000, 0.0) for i in range(4)]

# Each get on the main-process side waits this long, which is a collector that drains about 10 metrics a second against 40 produced.
SLOW_GET_SEC = 0.1


CollectedItem = Optional[Union[RequestLifecycleMetric, List[RequestLifecycleMetric]]]


# The collector's queue with every get in the main process delayed by SLOW_GET_SEC; workers put into it unchanged.
# Queue.get_nowait calls get(False), so overriding get slows both read paths once.
class SlowGetQueue(JoinableQueue[CollectedItem]):
    def __init__(self) -> None:
        super().__init__(ctx=MP_CONTEXT)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> CollectedItem:
        time.sleep(SLOW_GET_SEC)
        return super().get(block, timeout)


# 40 requests on 2 workers, each answered with about 128 KB, collected at about 10 metrics a second: the report must still hold all 40.
async def test_slow_collector_receives_every_recorded_metric() -> None:
    api_config = APIConfig(type=APIType.Chat, streaming=True)
    collector = MultiprocessRequestMetricCollector()
    collector.queue = SlowGetQueue()
    load_config = LoadConfig(
        type=LoadType.CONSTANT,
        num_workers=NUM_WORKERS,
        worker_max_concurrency=64,
        interval=0,
        stages=[StandardLoadStage(rate=RATE, duration=DURATION)],
        base_seed=42,
    )
    async with FakeOpenAIServer(RESPONSE_SCRIPT) as server:
        client = FakeServerClient(collector, api_config, f"http://127.0.0.1:{server.port}")
        load_gen = LoadGenerator(make_datagen(api_config), load_config)
        try:
            async with collector.start():
                await asyncio.wait_for(load_gen.run(client), timeout=120)
        finally:
            await load_gen.stop()
            await client.close()
        served = len(server.served)

    assert served == RATE * DURATION
    assert len(collector.get_metrics()) == served
