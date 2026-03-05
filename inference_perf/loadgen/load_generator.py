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
from pathlib import Path
from inference_perf.client.metricsclient.base import StageRuntimeInfo, StageStatus
from inference_perf.utils.trace_reader import AzurePublicDatasetReader
from inference_perf.utils.request_queue import RequestQueue
from .load_timer import LoadTimer, ConstantLoadTimer, PoissonLoadTimer, TraceReplayLoadTimer
from inference_perf.datagen import DataGenerator, LazyLoadDataMixin
from inference_perf.apis import InferenceAPIData
from inference_perf.client.modelserver import ModelServerClient
from inference_perf.circuit_breaker import get_circuit_breaker
from inference_perf.config import (
    LoadConfig,
    LoadType,
    StageGenType,
    TraceFormat,
    ConcurrentLoadStage,
    StandardLoadStage,
)
from asyncio import (
    CancelledError,
    Semaphore,
    create_task,
    gather,
    run,
    sleep,
    set_event_loop_policy,
    get_event_loop,
)
import sys

if sys.version_info >= (3, 11):
    from asyncio import TaskGroup
else:
    # Python 3.9 compatibility: TaskGroup was added in 3.11
    # This is a dummy for import-time compatibility.
    # Runtime usage will still require Python 3.11+.
    TaskGroup = object

from typing import List, Tuple, Optional, Union

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    # Python 3.9 compatibility: TypeAlias was added in 3.10
    from typing import Any

    TypeAlias = Any
from types import FrameType
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
import signal

logger = logging.getLogger(__name__)

RequestQueueData: TypeAlias = Tuple[int, Union[InferenceAPIData, int], float, Optional[str]]


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
        shared_max_concurrency: Optional["Synchronized[int]"],
        base_seed: int,
    ):
        super().__init__(daemon=True)  # kill worker process if main process exit unexpected
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
        self.shared_max_concurrency = shared_max_concurrency
        self.skip = False
        self.base_seed = base_seed

    async def loop(self) -> None:
        # The self.shared_max_concurrency is initialized to self.max_concurrency
        semaphore = Semaphore(self.max_concurrency)
        current_concurrency = self.max_concurrency
        tasks = []
        event_loop = get_event_loop()
        item = None
        timeout = 0.5

        while not self.stop_signal.is_set():
            # Check if max_concurrency has been updated and recreate semaphore if needed (concurrent load type)
            if self.shared_max_concurrency and not self.skip:
                with self.shared_max_concurrency.get_lock():
                    new_concurrency = self.shared_max_concurrency.value
                if new_concurrency == 0:
                    self.skip = True
                elif new_concurrency != current_concurrency:
                    logger.debug(f"[Worker {self.id}] updating semaphore from {current_concurrency} to {new_concurrency}")
                    # Wait for all current semaphore permits to be released
                    for _ in range(current_concurrency):
                        await semaphore.acquire()
                    # Create new semaphore with updated limit
                    semaphore = Semaphore(new_concurrency)
                    current_concurrency = new_concurrency

            if not self.skip:
                logger.debug(f"Worker {self.id} is currently working")
            else:
                await sleep(0)

            # Process requests in loop
            while self.request_phase.is_set() and not self.cancel_signal.is_set() and not self.skip:
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
                    semaphore: Semaphore,
                    lora_adapter: Optional[str],
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
                        await self.client.process_request(request_data, stage_id, request_time, lora_adapter)
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

                stage_id, request, request_time, lora_adapter = item
                request_data = LazyLoadDataMixin.get_request(self.datagen, request)
                task = create_task(
                    schedule_client(self.request_queue, request_data, request_time, stage_id, semaphore, lora_adapter)
                )
                tasks.append(task)
                await sleep(0)

            # Reset skip
            self.skip = False

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
        # Seed with current time + worker id to ensure unique random sequences per worker
        seed = (self.base_seed + self.id) % 2**32
        np.random.seed(seed)
        logger.debug(f"[Worker {self.id}] seeded numpy with {seed} and base seed {self.base_seed}")

        # Ignore SIGINT in workers to prevent multiple calls to SIGINT handler
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        set_event_loop_policy(uvloop.EventLoopPolicy())
        run(self.loop())


class LoadGenerator:
    def __init__(self, datagen: DataGenerator, load_config: LoadConfig) -> None:
        self.datagen = datagen
        self.stageInterval = load_config.interval
        self.load_type = load_config.type
        self.stages = load_config.stages
        self.stage_runtime_info = dict[int, StageRuntimeInfo]()
        self.num_workers = load_config.num_workers
        self.worker_max_concurrency = load_config.worker_max_concurrency
        self.workers: List[Worker] = []
        self.circuit_breakers = [get_circuit_breaker(breaker_name) for breaker_name in load_config.circuit_breakers]
        self.sweep_config = load_config.sweep
        self.interrupt_sig = False
        signal.signal(signal.SIGINT, self._sigint_handler)
        if self.load_type == LoadType.TRACE_REPLAY:
            self.trace = load_config.trace

            if self.trace is None:
                raise ValueError("Trace file is required for trace replay load generator")

            if self.trace.format == TraceFormat.AZURE_PUBLIC_DATASET:
                self.trace_reader = AzurePublicDatasetReader()
            else:
                raise ValueError(f"Unsupported trace format: {self.trace.format}")
        self.lora_adapters: Optional[List[str]] = None
        self.lora_weights: Optional[List[float]] = None
        if load_config.lora_traffic_split is not None:
            self.lora_adapters = [config.name for config in load_config.lora_traffic_split]
            self.lora_weights = [config.split for config in load_config.lora_traffic_split]
        self.base_seed: int = load_config.base_seed

    def _sigint_handler(self, _signum: int, _frame: Optional[FrameType]) -> None:
        """SIGINT handler that sets interrup_sig flag to True"""
        self.interrupt_sig = True

    def _set_worker_concurrency(self, concurrency_level: int) -> None:
        """Determines the per worker concurrency, handling cases where concurrency_level % num_workers != 0."""
        # Calculate new concurrency for worker (concurrency_level will always be > 0)
        new_concurrency = concurrency_level // self.num_workers
        # Calculate index cutoff for workers with +1 concurrency
        remainder = concurrency_level % self.num_workers
        for worker in self.workers:
            worker_concurrency = new_concurrency + 1 if worker.id < remainder else new_concurrency
            # Update the shared concurrency value to signal the worker to update its semaphore (needs to be synchronized with main process)
            if worker.shared_max_concurrency:
                with worker.shared_max_concurrency.get_lock():
                    worker.shared_max_concurrency.value = worker_concurrency

    def _get_lora_adapter(self) -> Optional[str]:
        """Returns a randomly selected LoRA adapter based on configured probability weights, or None if not configured."""
        if self.lora_adapters is not None and self.lora_weights is not None:
            return str(np.random.choice(self.lora_adapters, p=self.lora_weights))
        return None

    def get_timer(self, rate: float, duration: float) -> LoadTimer:
        if self.load_type == LoadType.POISSON:
            return PoissonLoadTimer(rate=rate, duration=duration)
        elif self.load_type == LoadType.TRACE_REPLAY:
            if self.trace is None:
                raise ValueError("Trace configuration is required for trace replay load generator")
            return TraceReplayLoadTimer(trace_reader=self.trace_reader, trace_file=Path(self.trace.file))
        # For concurrent and constant load types (rate is adjusted in main.py for concurrent load type)
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
        request_queue: RequestQueue[RequestQueueData],
        active_requests_counter: "Synchronized[int]",
        finished_requests_counter: "Synchronized[int]",
        request_phase: SyncEvent,
        cancel_signal: Optional[SyncEvent] = None,
        timeout: Optional[float] = None,
        concurrency_level: Optional[int] = None,
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

        if self.datagen.trace is not None:
            num_requests = self.datagen.get_request_count()
        else:
            num_requests = int(rate * duration)

        stage_status = StageStatus.RUNNING

        time_generator = timer.start_timer(start_time)
        data_generator = self.datagen.get_data()
        active_workers = self.num_workers
        if concurrency_level:
            # If concurrency_level is set, some worker may get 0 concurrency, then we should re-evaluate workers we can assign reqeusts to.
            active_workers = min(self.num_workers, concurrency_level)

        for _ in range(num_requests):
            request_data = next(data_generator)
            lora_adapter = self._get_lora_adapter()
            worker_id = request_data.prefered_worker_id
            if worker_id >= 0:
                worker_id = worker_id % active_workers
            request_queue.put((stage_id, request_data, next(time_generator), lora_adapter), worker_id)

        # Wait until all requests are finished processing
        with tqdm(total=1.0, desc=f"Stage {stage_id} progress") as pbar:
            timed_out = False
            timeout_progress = (time.perf_counter() - start_time) / timeout if timeout else 0
            prog, last = min(1.0, max(timeout_progress, finished_requests_counter.value / num_requests)), 0.0
            pbar.update(prog)
            last = prog
            while finished_requests_counter.value < num_requests:
                if timeout and start_time + timeout < time.perf_counter():
                    pbar.close()
                    logger.info(f"Loadgen timed out after {timeout:0.2f}s")
                    timed_out = True
                    break
                if self.interrupt_sig:
                    pbar.close()
                    logger.info("Loadgen encountered SIGINT")
                    break
                if cb := next((cb for cb in self.circuit_breakers if cb.is_open()), None):
                    logger.warning(f'Loadgen detects circuit breakers "{cb.name}" open, exit the stage.')
                    timed_out = True
                    break
                await sleep(1)
                timeout_progress = (time.perf_counter() - start_time) / timeout if timeout else 0
                prog = min(1.0, max(timeout_progress, finished_requests_counter.value / num_requests))
                pbar.update(prog - last)
                last = prog

        # Trigger cleanup if timed out or received SIGINT
        if (timed_out or self.interrupt_sig) and cancel_signal:
            # Cancel signal must be set before request_phase
            # Allow time for workers to process the signal
            cancel_signal.set()
            await sleep(1)
            while active_requests_counter.value > 0:
                await sleep(1)
            request_queue.drain()
            cancel_signal.clear()
            stage_status = StageStatus.FAILED
        else:
            stage_status = StageStatus.COMPLETED
        # Clear the request_phase event to force worker gather
        request_phase.clear()
        request_queue.join()

        self.stage_runtime_info[stage_id] = StageRuntimeInfo(
            stage_id=stage_id,
            rate=rate,
            start_time=start_time_epoch,
            end_time=time.time(),
            status=stage_status,
            concurrency_level=concurrency_level,
        )
        logger.info("Stage %d - run completed" if stage_status == StageStatus.COMPLETED else "Stage %d - run failed", stage_id)

    async def preprocess(
        self,
        client: ModelServerClient,
        request_queue: RequestQueue[RequestQueueData],
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

        if self.sweep_config is None:
            raise Exception("sweep_config cannot be none")

        # Aggregator collects timestamped value of active_requests throughout the preprocessing
        async def aggregator() -> None:
            while True:
                results.append((time.perf_counter(), active_requests_counter.value))
                await sleep(0.5)

        aggregator_task = create_task(aggregator())

        stage_id = -1
        duration = 5
        rate = self.sweep_config.num_requests / duration
        timeout = self.sweep_config.timeout
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
        logger.debug(f"Determining saturation from rates: {[f'{rate:0.2f}' for rate in sorted(rates)]}")
        saturation_point = float(np.percentile(rates, self.sweep_config.saturation_percentile))
        logger.info(f"Saturation point estimated at {saturation_point:0.2f} concurrent requests.")

        def generateRates(target_request_rate: float, size: int, gen_type: StageGenType) -> List[float]:
            if gen_type == StageGenType.GEOM:
                return [float(round(1 + target_request_rate - rr, 2)) for rr in np.geomspace(target_request_rate, 1, num=size)]
            elif gen_type == StageGenType.LINEAR:
                return [float(round(r, 2)) for r in np.linspace(1, target_request_rate, size)]

        rates = generateRates(saturation_point, self.sweep_config.num_stages, self.sweep_config.type)
        self.stages = [StandardLoadStage(rate=r, duration=self.sweep_config.stage_duration) for r in rates]
        logger.info(f"Generated load stages: {[s.rate for s in self.stages]}")

    async def mp_run(self, client: ModelServerClient) -> None:
        request_queue: RequestQueue[RequestQueueData] = RequestQueue(
            self.num_workers if self.datagen.is_prefered_worker_requested() else 1
        )
        finished_requests_counter: "Synchronized[int]" = mp.Value("i", 0)
        active_requests_counter: "Synchronized[int]" = mp.Value("i", 0)
        request_phase: SyncEvent = mp.Event()
        stop_signal: SyncEvent = mp.Event()
        cancel_signal: SyncEvent = mp.Event()
        # start workers in the request phase
        request_phase.set()

        # Create list of workers to process requests
        for id in range(self.num_workers):
            # Create shared value for each worker's max concurrency if concurrent load type
            if self.load_type == LoadType.CONCURRENT:
                shared_max_concurrency = mp.Value("i", self.worker_max_concurrency)
            else:
                shared_max_concurrency = None

            self.workers.append(
                Worker(
                    id,
                    client,
                    request_queue.get_channel(id),
                    self.datagen,
                    self.worker_max_concurrency,
                    stop_signal,
                    cancel_signal,
                    request_phase,
                    finished_requests_counter,
                    active_requests_counter,
                    shared_max_concurrency,
                    self.base_seed,
                )
            )
            self.workers[-1].start()

        if self.sweep_config:
            try:
                await self.preprocess(
                    client, request_queue, active_requests_counter, finished_requests_counter, request_phase, cancel_signal
                )
            except Exception as e:
                logger.error(f"Preprocessing exception: {e}")
                stop_signal.set()
                return

        for stage_id, stage in enumerate(self.stages):
            # Update worker concurrency for concurrent load type
            if self.load_type == LoadType.CONCURRENT and isinstance(stage, ConcurrentLoadStage):
                logger.debug(f"Setting worker concurrency to {stage.concurrency_level} for stage {stage_id}")
                self._set_worker_concurrency(stage.concurrency_level)

                # Use the dynamically set rate/duration from main.py
                rate = getattr(stage, "rate", stage.num_requests)
                duration = getattr(stage, "duration", 1)
                concurrency_level = stage.concurrency_level
            elif self.load_type != LoadType.CONCURRENT and isinstance(stage, StandardLoadStage):
                rate = stage.rate
                duration = stage.duration
                concurrency_level = None
            else:
                raise Exception(f"Stage {stage_id} has the wrong load type")

            await self.run_stage(
                stage_id,
                rate,
                duration,
                request_queue,
                active_requests_counter,
                finished_requests_counter,
                request_phase,
                cancel_signal,
                concurrency_level=concurrency_level,
            )
            # If we encountered a SIGINT, we can break out of run stages loop
            if self.interrupt_sig:
                break
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
            stage_status = StageStatus.RUNNING
            logger.info("Stage %d - run started", stage_id)
            async with TaskGroup() as tg:
                time_generator = timer.start_timer(start_time)
                for _, (data, time_index) in enumerate(zip(self.datagen.get_data(), time_generator, strict=True)):
                    request_data = LazyLoadDataMixin.get_request(self.datagen, data)
                    lora_adapter = self._get_lora_adapter()
                    if cb := next((cb for cb in self.circuit_breakers if cb.is_open()), None):
                        logger.warning(f'Loadgen detects circuit breakers "{cb.name}" open, clean up stage and exit early.')
                        stage_status = StageStatus.FAILED
                        break
                    now = time.perf_counter()
                    if time_index < end_time and now < end_time:
                        if cb := next((cb for cb in self.circuit_breakers if cb.is_open()), None):
                            logger.warning(
                                f'Loadgen detects circuit breakers "{cb.name}" open, clean up stage and exit early.'
                            )
                            stage_status = StageStatus.FAILED
                            break
                        if time_index > now:
                            await sleep(time_index - time.perf_counter())
                        tg.create_task(client.process_request(request_data, stage_id, time_index, lora_adapter))
                        continue
                    else:
                        break
            if stage_status == StageStatus.RUNNING:
                stage_status = StageStatus.COMPLETED
                logger.info("Stage %d - run completed", stage_id)
            else:
                logger.info("Stage %d - run failed", stage_id)
            self.stage_runtime_info[stage_id] = StageRuntimeInfo(
                stage_id=stage_id,
                rate=stage.rate,
                start_time=start_time_epoch,
                end_time=time.time(),
                status=stage_status,
                concurrency_level=None,
            )
            if self.stageInterval and stage_id < len(self.stages) - 1:
                await sleep(self.stageInterval)

    async def stop(self) -> None:
        for worker in self.workers:
            worker.join(timeout=1.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=0.0)
