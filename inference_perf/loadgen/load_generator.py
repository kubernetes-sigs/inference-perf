# Copyright 2025 The Kubernetes Authors.
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
from pydantic import BaseModel
from .load_timer import LoadTimer, ConstantLoadTimer, PoissonLoadTimer
from inference_perf.datagen import DataGenerator
from inference_perf.apis import InferenceAPIData
from inference_perf.client.modelserver import ModelServerClient
from inference_perf.config import LoadType, LoadConfig
from asyncio import Semaphore, TaskGroup, create_task, gather, run, sleep, set_event_loop_policy, get_event_loop
from typing import List, Tuple, TypeAlias, Optional
import time
import multiprocessing as mp
from multiprocessing.synchronize import Event as SyncEvent
from multiprocessing.sharedctypes import Synchronized
from concurrent.futures import TimeoutError
from functools import partial
import logging
import uvloop

logger = logging.getLogger(__name__)

RequestQueueData: TypeAlias = Tuple[int, InferenceAPIData | int, float]


class Worker(mp.Process):
    def __init__(
        self,
        id: int,
        client: ModelServerClient,
        request_queue: mp.Queue,  # type: ignore[type-arg]
        datagen: DataGenerator,
        max_concurrency: int,
        stop_signal: SyncEvent,
        request_phase: SyncEvent,
        finished_requests_counter: "Synchronized[int]",
    ):
        super().__init__()
        self.id = id
        self.client = client
        self.request_queue = request_queue
        self.max_concurrency = max_concurrency
        self.datagen = datagen
        self.stop_signal = stop_signal
        self.request_phase = request_phase
        self.finished_requests_counter = finished_requests_counter

    async def loop(self) -> None:
        semaphore = Semaphore(self.max_concurrency)
        tasks = []
        event_loop = get_event_loop()
        item = None
        timeout = 0.5

        while not self.stop_signal.is_set():
            while self.request_phase.is_set():
                await semaphore.acquire()
                try:
                    # Use partial to pass named arg
                    get = partial(self.request_queue.get, timeout=timeout)
                    item = await event_loop.run_in_executor(None, get)
                    if item is None:
                        semaphore.release()
                        continue
                except TimeoutError:
                    semaphore.release()
                    continue
                except mp.queues.Empty:
                    semaphore.release()
                    continue
                except Exception as e:
                    logger.info(f"[Worker {self.id}] hit exception {e}")
                    semaphore.release()
                    continue

                async def schedule_client(
                    queue: mp.Queue,  # type: ignore[type-arg]
                    request_data: InferenceAPIData,
                    request_time: float,
                    stage_id: int,
                ) -> None:
                    current_time = time.perf_counter()
                    sleep_time = request_time - current_time
                    if sleep_time > 0:
                        await sleep(sleep_time)
                    else:
                        logger.debug(f"Worker {self.id} missed scheduled request time by {-1.0 * sleep_time:0.2f}")
                    await self.client.process_request(request_data, stage_id, request_time)
                    with self.finished_requests_counter.get_lock():
                        self.finished_requests_counter.value += 1
                    queue.task_done()
                    semaphore.release()

                stage_id, request, request_time = item
                request_data = self.datagen.get_request(request) if isinstance(request, int) else request
                task = create_task(schedule_client(self.request_queue, request_data, request_time, stage_id))
                tasks.append(task)
                await sleep(0)

            await gather(*tasks)
            tasks = []
            self.request_phase.wait()

        logger.debug(f"[Worker {self.id}] stopped")

    def run(self) -> None:
        set_event_loop_policy(uvloop.EventLoopPolicy())
        run(self.loop())


class StageRuntimeInfo(BaseModel):
    stage_id: int
    rate: float
    end_time: float
    start_time: float


class LoadGenerator:
    def __init__(self, datagen: DataGenerator, load_config: LoadConfig) -> None:
        self.datagen = datagen
        self.stageInterval = load_config.interval
        self.load_type = load_config.type
        self.stages = load_config.stages
        self.stage_runtime_info = dict[int, StageRuntimeInfo]()
        self.num_workers = load_config.num_workers
        self.workers: List[Worker] = []
        self.worker_max_concurrency = load_config.worker_max_concurrency

    def get_timer(self, rate: float, duration: float) -> LoadTimer:
        if self.load_type == LoadType.POISSON:
            return PoissonLoadTimer(rate=rate, duration=duration)
        return ConstantLoadTimer(rate=rate, duration=duration)

    async def run_stage(
        self,
        stage_id: int,
        rate: float,
        duration: int,
        request_queue: mp.Queue, # type: ignore[type-arg]
        finished_requests_counter: "Synchronized[int]",
        request_phase: SyncEvent,
        timeout: Optional[float] = None,
    ) -> None:
        logger.info("Stage %d - run started", stage_id)
        request_phase.set()
        with finished_requests_counter.get_lock():
            finished_requests_counter.value = 0
        timer = self.get_timer(rate, duration)

        # Allow generation a second to begin populating the queue so the workers
        # don't miss the initial scheuled request times
        start_time_epoch = time.time()
        start_time = time.perf_counter() + 1
        num_requests = int(rate * duration)

        time_generator = timer.start_timer(start_time)
        if hasattr(self.datagen, "get_request"):
            # Datagen supports deferring to workers, enqueue request number
            for request_number in range(num_requests):
                request_time = next(time_generator)
                request_queue.put((stage_id, request_number, request_time))
        else:
            # Datagen requires queueing request_data
            data_generator = self.datagen.get_data()
            for _ in range(num_requests):
                request_queue.put((stage_id, next(data_generator), next(time_generator)))

        logger.debug("Loadgen sleeping until end of stage")
        await sleep(start_time + duration - time.perf_counter())

        # Wait until all requests are finished processing
        while finished_requests_counter.value < num_requests:
            logger.debug(f"Loadgen waiting for all requests to finish: {finished_requests_counter.value}/{num_requests}")
            await sleep(1)

        # Clear the request_phase event to force worker gather
        request_phase.clear()
        request_queue.join()

        self.stage_runtime_info[stage_id] = StageRuntimeInfo(
            stage_id=stage_id, rate=rate, start_time=start_time_epoch, end_time=time.time()
        )
        logger.info("Stage %d - run completed", stage_id)

    async def mp_run(self, client: ModelServerClient) -> None:
        request_queue: mp.Queue[RequestQueueData] = mp.JoinableQueue()
        finished_requests_counter: "Synchronized[int]" = mp.Value("i", 0)
        request_phase: SyncEvent = mp.Event()
        stop_signal: SyncEvent = mp.Event()
        # start workers in the request phase
        request_phase.set()

        for id in range(self.num_workers):
            self.workers.append(
                Worker(
                    id,
                    client,
                    request_queue,
                    self.datagen,
                    self.worker_max_concurrency,
                    stop_signal,
                    request_phase,
                    finished_requests_counter,
                )
            )
            self.workers[-1].start()

        for stage_id, stage in enumerate(self.stages):
            await self.run_stage(stage_id, stage.rate, stage.duration, request_queue, finished_requests_counter, request_phase)
            if self.stageInterval:
                await sleep(self.stageInterval)

        # Reset the request phase to get workers out of their final wait()
        request_phase.set()
        stop_signal.set()

    async def run(self, client: ModelServerClient) -> None:
        if self.num_workers > 0:
            return await self.mp_run(client)

        for stage_id, stage in enumerate(self.stages):
            timer = self.get_timer(stage.rate, stage.duration)
            start_time_epoch = time.time()
            start_time = time.perf_counter()
            end_time = start_time + stage.duration
            logger.info("Stage %d - run started", stage_id)
            async with TaskGroup() as tg:
                time_generator = timer.start_timer(start_time)
                for _, (data, time_index) in enumerate(zip(self.datagen.get_data(), time_generator, strict=True)):
                    now = time.perf_counter()
                    if time_index < end_time and now < end_time:
                        if time_index > now:
                            await sleep(time_index - time.perf_counter())
                        tg.create_task(client.process_request(data, stage_id, time_index))
                        continue
                    else:
                        break
            self.stage_runtime_info[stage_id] = StageRuntimeInfo(
                stage_id=stage_id, rate=stage.rate, start_time=start_time_epoch, end_time=time.time()
            )
            logger.info("Stage %d - run completed", stage_id)
            if self.stageInterval and stage_id < len(self.stages) - 1:
                await sleep(self.stageInterval)

    async def stop(self) -> None:
        for worker in self.workers:
            worker.join(timeout=1.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=0.0)
