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
from inference_perf.config import LoadConfig, LoadStage, LoadType, StageGenType
from asyncio import (
    CancelledError,
    Semaphore,
    TaskGroup,
    create_task,
    gather,
    run,
    sleep,
    set_event_loop_policy,
    get_event_loop,
)
from typing import List, Tuple, TypeAlias, Optional
import time
import multiprocessing as mp
from multiprocessing.synchronize import Event as SyncEvent
from multiprocessing.sharedctypes import Synchronized
from concurrent.futures import TimeoutError
from functools import partial
import logging
import uvloop
import numpy as np
from tqdm import tqdm

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
        cancel_signal: SyncEvent,
        request_phase: SyncEvent,
        finished_requests_counter: "Synchronized[int]",
        active_requests_counter: "Synchronized[int]",
    ):
        super().__init__()
        self.id = id
        self.client = client
        self.request_queue = request_queue
        self.max_concurrency = max_concurrency
        self.datagen = datagen
        self.stop_signal = stop_signal
        self.cancel_signal = cancel_signal
        self.request_phase = request_phase
        self.finished_requests_counter = finished_requests_counter
        self.active_requests_counter = active_requests_counter

    async def loop(self) -> None:
        semaphore = Semaphore(self.max_concurrency)
        tasks = []
        event_loop = get_event_loop()
        item = None
        timeout = 0.5

        while not self.stop_signal.is_set():
            while self.request_phase.is_set() and not self.cancel_signal.is_set():
                await semaphore.acquire()
                try:
                    # Use partial to pass named arg
                    get = partial(self.request_queue.get, timeout=timeout)
                    item = await event_loop.run_in_executor(None, get)
                    if item is None:
                        semaphore.release()
                        continue
                except TimeoutError:
                    logger.debug(f"[Worker {self.id}] timed out getting request from queue")
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
                    inflight = False
                    try:
                        current_time = time.perf_counter()
                        sleep_time = request_time - current_time
                        if sleep_time > 0:
                            await sleep(sleep_time)
                        with self.active_requests_counter.get_lock():
                            self.active_requests_counter.value += 1
                            inflight = True
                        await self.client.process_request(request_data, stage_id, request_time)
                    except CancelledError:
                        pass
                    finally:
                        with self.active_requests_counter.get_lock():
                            if inflight:
                                self.active_requests_counter.value -= 1
                        with self.finished_requests_counter.get_lock():
                            self.finished_requests_counter.value += 1
                        queue.task_done()
                        semaphore.release()

                stage_id, request, request_time = item
                request_data = self.datagen.get_request(request) if isinstance(request, int) else request
                task = create_task(schedule_client(self.request_queue, request_data, request_time, stage_id))
                tasks.append(task)
                await sleep(0)

            if self.cancel_signal.is_set():
                logger.debug(f"[Worker {self.id}] cancelling tasks with {self.active_requests_counter.value} active requests")
                for task in tasks:
                    task.cancel()
                while self.request_phase.is_set():
                    await sleep(0)
                logger.debug(f"[Worker {self.id}] done cancelling")
            if not self.request_phase.is_set():
                await gather(*tasks)
                tasks = []
                logger.debug(f"[Worker {self.id}] waiting for next phase")
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
        self.auto_stage_config = load_config.auto_stage

    def get_timer(self, rate: float, duration: float) -> LoadTimer:
        if self.load_type == LoadType.POISSON:
            return PoissonLoadTimer(rate=rate, duration=duration)
        return ConstantLoadTimer(rate=rate, duration=duration)

    async def drain(self, queue: mp.Queue) -> None:  # type: ignore[type-arg]
        while True:
            try:
                _ = queue.get_nowait()
                queue.task_done()
            except mp.queues.Empty:
                if queue.qsize() == 0:
                    logger.debug("Drain finished")
                    return

    async def run_stage(
        self,
        stage_id: int,
        rate: float,
        duration: int,
        request_queue: mp.Queue,  # type: ignore[type-arg]
        active_requests_counter: "Synchronized[int]",
        finished_requests_counter: "Synchronized[int]",
        request_phase: SyncEvent,
        cancel_signal: Optional[SyncEvent] = None,
        timeout: Optional[float] = None,
    ) -> None:
        logger.info("Stage %d - run started", stage_id)

        if timeout is not None and cancel_signal is None:
            raise Exception("run_stage timeout requires cancel_signal to be not None!")

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

        # Wait until all requests are finished processing
        with tqdm(total=1.0, desc=f"Stage {stage_id} progress") as pbar:
            timed_out = False
            timeout_progress = (time.perf_counter() - start_time) / timeout if timeout else 0
            prog, last = min(1.0, max(timeout_progress, finished_requests_counter.value / num_requests)), 0.0
            pbar.update(prog)
            last = prog
            while finished_requests_counter.value < num_requests:
                if timeout and start_time + timeout < time.perf_counter():
                    logger.info(f"Loadgen timed out after {timeout:0.2f}s")
                    timed_out = True
                    break
                await sleep(1)
                timeout_progress = (time.perf_counter() - start_time) / timeout if timeout else 0
                prog = min(1.0, max(timeout_progress, finished_requests_counter.value / num_requests))
                pbar.update(prog - last)
                last = prog

        if timed_out and cancel_signal:
            # Cancel signal must be set before request_phase
            # Allow time for workers to process the signal
            cancel_signal.set()
            await sleep(1)
            while active_requests_counter.value > 0:
                await sleep(1)
            await self.drain(request_queue)
            cancel_signal.clear()
        # Clear the request_phase event to force worker gather
        request_phase.clear()
        request_queue.join()

        self.stage_runtime_info[stage_id] = StageRuntimeInfo(
            stage_id=stage_id, rate=rate, start_time=start_time_epoch, end_time=time.time()
        )
        logger.info("Stage %d - run completed", stage_id)

    async def preprocess(
        self,
        client: ModelServerClient,
        request_queue: mp.Queue,  # type: ignore[type-arg]
        active_requests_counter: "Synchronized[int]",
        finished_requests_counter: "Synchronized[int]",
        request_phase: SyncEvent,
        cancel_signal: SyncEvent,
    ) -> None:
        """
        Runs a preliminary load test to automatically determine the server's saturation point
        and generate a suitable series of load stages for the main benchmark.

        An aggregator task samples the active requests and then the burn down rate is
        calculated from the samples. Saturation is derived from a percentile of the
        sampled burn down rates.
        """
        logger.info("Running preprocessing stage")
        results: List[Tuple[float, int]] = []

        if self.auto_stage_config is None:
            raise Exception("auto_stage_config cannot be none")

        # Aggregator collects timestamped value of active_requests throughout the preprocessing
        async def aggregator() -> None:
            while True:
                results.append((time.perf_counter(), active_requests_counter.value))
                await sleep(0.5)

        aggregator_task = create_task(aggregator())

        stage_id = -1
        duration = 5
        rate = self.auto_stage_config.num_requests / duration
        timeout = self.auto_stage_config.timeout
        start_time = time.perf_counter()
        await self.run_stage(
            stage_id,
            rate,
            duration,
            request_queue,
            active_requests_counter,
            finished_requests_counter,
            request_phase,
            timeout=timeout,
            cancel_signal=cancel_signal,
        )

        aggregator_task.cancel()
        try:
            await aggregator_task
        except CancelledError:
            pass

        # Ensure that we don't calculate saturation based on the post-timeout drain
        results = [(timestamp, requests) for timestamp, requests in results if timestamp < start_time + timeout]
        # Calculate the sampled QPS by interval between the samples
        rates = [
            abs((current_requests - previous_requests) / (current_timestamp - previous_timestamp))
            for (current_timestamp, current_requests), (previous_timestamp, previous_requests) in zip(
                results[1:], results[:-1], strict=True
            )
            if current_requests - previous_requests < 0
        ]

        if len(rates) <= 1:
            raise Exception(
                "Loadgen preprocessing failed to gather enough samples to determine saturation, try increasing the num_requests or timeout"
            )

        # Generate new stages
        saturation_point = float(np.percentile(rates, self.auto_stage_config.saturation_percentile))
        logger.info(f"Saturation point estimated at {saturation_point:0.2f} concurrent requests.")

        def generateRates(target_request_rate: float, size: int, gen_type: StageGenType) -> List[float]:
            if gen_type == StageGenType.GEOM:
                return [float(round(1 + target_request_rate - rr, 2)) for rr in np.geomspace(target_request_rate, 1, num=size)]
            elif gen_type == StageGenType.LINEAR:
                return [float(round(r)) for r in np.linspace(1, target_request_rate, size)]

        rates = generateRates(saturation_point, self.auto_stage_config.num_stages, self.auto_stage_config.gen_type)
        self.stages = [LoadStage(rate=r, duration=self.auto_stage_config.stage_duration) for r in rates]
        logger.info(f"Generated load stages: {[s.rate for s in self.stages]}")

    async def mp_run(self, client: ModelServerClient) -> None:
        request_queue: mp.Queue[RequestQueueData] = mp.JoinableQueue()
        finished_requests_counter: "Synchronized[int]" = mp.Value("i", 0)
        active_requests_counter: "Synchronized[int]" = mp.Value("i", 0)
        request_phase: SyncEvent = mp.Event()
        stop_signal: SyncEvent = mp.Event()
        cancel_signal: SyncEvent = mp.Event()
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
                    cancel_signal,
                    request_phase,
                    finished_requests_counter,
                    active_requests_counter,
                )
            )
            self.workers[-1].start()

        if self.auto_stage_config:
            await self.preprocess(
                client, request_queue, active_requests_counter, finished_requests_counter, request_phase, cancel_signal
            )

        for stage_id, stage in enumerate(self.stages):
            await self.run_stage(
                stage_id,
                stage.rate,
                stage.duration,
                request_queue,
                active_requests_counter,
                finished_requests_counter,
                request_phase,
            )
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
