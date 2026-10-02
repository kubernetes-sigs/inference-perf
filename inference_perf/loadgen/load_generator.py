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
from pathlib import Path
from inference_perf.client.server_metrics.base import StageRuntimeInfo, StageStatus
from inference_perf.datagen.base import BaseGenerator
from inference_perf.utils.numeric.request_schedule import RequestSchedule
from inference_perf.utils.numeric.rate_schedule import RateSchedule
from inference_perf.utils.trace_reader import AzurePublicDatasetReader
from inference_perf.utils.request_queue import RequestQueue
from .load_timer import LoadTimer, ConstantLoadTimer, RequestIntervalLoadTimer, PoissonLoadTimer, TraceReplayLoadTimer
from inference_perf.datagen import DataGenerator, SessionGenerator, LazyLoadDataMixin
from inference_perf.apis import InferenceAPIData, RequestLifecycleMetric
from inference_perf.apis.user_session import LocalUserSession
from inference_perf.client.modelserver import ModelServerClient
from inference_perf.client.modelserver.otel_instrumentation import get_otel_instrumentation
from inference_perf.circuit_breaker import get_circuit_breaker
from inference_perf.loadgen.stop_condition import StopConditionMonitor
from inference_perf.metrics import SessionMetricsCollector
from inference_perf.config import (
    LoadConfig,
    LoadType,
    StageGenType,
    TraceFormat,
    ConcurrentLoadStage,
    StandardLoadStage,
    TraceSessionReplayLoadStage,
)
from asyncio import (
    CancelledError,
    Event,
    Semaphore,
    Task,
    TimeoutError as AsyncioTimeoutError,
    create_task,
    gather,
    run,
    sleep,
    set_event_loop_policy,
    get_event_loop,
    wait_for,
)
import sys

if sys.version_info >= (3, 11):
    from asyncio import TaskGroup
else:
    # Python 3.9 compatibility: TaskGroup was added in 3.11
    # This is a dummy for import-time compatibility.
    # Runtime usage will still require Python 3.11+.
    TaskGroup = object

from typing import Awaitable, List, Tuple, Optional, NamedTuple, Protocol, Union, Set, Dict, cast
from types import FrameType
import ctypes
import time
import traceback
import multiprocessing as mp
from queue import Empty
from multiprocessing.synchronize import Event as SyncEvent
from multiprocessing.sharedctypes import Synchronized
from concurrent.futures import TimeoutError
from functools import partial
import logging
import uvloop
import numpy as np
from prometheus_client import CollectorRegistry
import signal

from inference_perf.observability.context import StageContext
from inference_perf.observability.logging import get_console
from inference_perf.observability.progress import (
    OVERALL_BAR,
    STAGE_ELAPSED_BAR,
    STAGE_REQUESTS_BAR,
    STAGE_SESSIONS_BAR,
    BarHandle,
    ProgressBars,
)

logger = logging.getLogger(__name__)

# Stage-teardown tuning. The teardown protocol is bounded end to end: workers
# give in-flight requests `stage_teardown_grace_seconds` (LoadConfig) to finish
# naturally, then cancel and reap the remainder within _WIND_DOWN_REAP_SECONDS.
# The main process allows the worker grace plus _TEARDOWN_MARGIN_SECONDS before
# broadcasting the force-stop signal, then _FORCE_REAP_SECONDS more before
# terminating (and respawning) workers that still have not reached the stage
# rendezvous. Report generation therefore always runs.
_WIND_DOWN_REAP_SECONDS = 10.0
_TEARDOWN_MARGIN_SECONDS = 15.0
_FORCE_REAP_SECONDS = 20.0

# ConstantLoadTimer scales its gaps to sum to the stage duration, so its last request is scheduled at the stage end
# up to float rounding. A request scheduled within this many seconds past the end still belongs to the stage.
_STAGE_END_TOLERANCE = 1e-6

# Log-line verb for each terminal StageStatus; anything not listed reads as "failed".
_STAGE_STATUS_LOG_VERB: dict[StageStatus, str] = {
    StageStatus.COMPLETED: "completed",
    StageStatus.TIMED_OUT: "timed out",
    StageStatus.INTERRUPTED: "interrupted",
}


def _counter_value_nolock(counter: "Synchronized[int]") -> int:
    """Read a shared counter without acquiring its lock.

    `Synchronized.value` acquires the mutex, so a worker terminated while
    holding it would strand the lock and hang any later locked access. The
    main process must only read worker-owned counters through this helper so
    that teardown always returns.
    """
    return int(cast("ctypes.c_int", counter.get_obj()).value)


class TeardownResult(NamedTuple):
    clean: bool
    dropped_requests: int
    dropped_requests_exact: bool


class StageLifecycleObserver(Protocol):
    """Receives stage transitions from the load generator.

    Satisfied structurally by ``inference_perf.observability.metrics.MetricsHub``;
    kept as a Protocol so the load generator does not import the metrics package.
    """

    def on_stage_start(self, context: StageContext) -> None: ...

    def on_stage_end(self, context: StageContext) -> None: ...


class RequestQueueData(NamedTuple):
    stage_id: int
    request_data: InferenceAPIData
    request_time: float
    lora_adapter: Optional[str]


class WorkerCrash(NamedTuple):
    """What a worker managed to record about its own unhandled exception before dying."""

    worker_id: int
    stage_id: Optional[int]
    exc_type: str
    message: str
    traceback_text: str
    in_flight: int


class WorkerFailure(NamedTuple):
    """A worker that is no longer alive, with whatever the parent can say about why."""

    worker_id: int
    exitcode: Optional[int]
    crash: Optional[WorkerCrash]

    def describe(self) -> str:
        if self.exitcode is not None and self.exitcode < 0:
            try:
                signame = signal.Signals(-self.exitcode).name
            except ValueError:
                signame = f"signal {-self.exitcode}"
            cause = f"was killed by {signame}"
        elif self.exitcode == 0:
            cause = "exited cleanly before the stage finished"
        else:
            cause = f"exited with code {self.exitcode}"
        if self.crash is not None:
            # stage_id is None when the worker died before it pulled its first request,
            # which is the usual shape for a failure during startup.
            where = "before serving any stage" if self.crash.stage_id is None else f"during stage {self.crash.stage_id}"
            # in_flight is read from a counter shared by every worker, so it is the
            # run-wide figure at the moment of the crash, not this worker's own share.
            return (
                f"Worker {self.worker_id} {cause} after an unhandled "
                f"{self.crash.exc_type} {where} "
                f"with {self.crash.in_flight} request(s) in flight run-wide: {self.crash.message}\n"
                f"{self.crash.traceback_text.rstrip()}"
            )
        return (
            f"Worker {self.worker_id} {cause} without reporting a Python traceback (segfault, OOM kill, or forced termination)"
        )

    def cause_label(self) -> str:
        """Why this worker died, as a bounded string fit for a metric label.

        Prefers the exception class the worker reported; falls back to the
        signal name, and then to the exit code, which are the only things a
        worker killed without warning leaves behind.
        """
        if self.crash is not None:
            return self.crash.exc_type
        if self.exitcode is not None and self.exitcode < 0:
            try:
                return signal.Signals(-self.exitcode).name
            except ValueError:
                return f"signal_{-self.exitcode}"
        if self.exitcode == 0:
            return "clean_exit"
        return f"exit_{self.exitcode}"


def collect_worker_failures(
    workers: List["Worker"],
    crash_queue: Optional["mp.SimpleQueue[WorkerCrash]"],
) -> List[WorkerFailure]:
    """Pair every dead worker with its crash record, if the worker lived long enough to send one.

    A worker killed by a signal never reports, so exitcode is the only evidence
    available for that case and the record is None.
    """
    crashes: Dict[int, WorkerCrash] = {}
    if crash_queue is not None:
        while True:
            try:
                if crash_queue.empty():
                    break
                record = crash_queue.get()
            except (OSError, EOFError, ValueError):
                break
            crashes[record.worker_id] = record
    failures = []
    for worker in workers:
        if worker.is_alive() or worker.exitcode is None:
            continue
        failures.append(WorkerFailure(worker_id=worker.id, exitcode=worker.exitcode, crash=crashes.get(worker.id)))
    return failures


class Worker(mp.Process):
    def __init__(
        self,
        id: int,
        client: ModelServerClient,
        request_queue: "mp.Queue[Optional[RequestQueueData]]",
        datagen: BaseGenerator,
        max_concurrency: int,
        stop_signal: SyncEvent,
        cancel_signal: SyncEvent,
        request_phase: SyncEvent,
        finished_requests_counter: "Synchronized[int]",
        active_requests_counter: "Synchronized[int]",
        skipped_requests_counter: "Synchronized[int]",
        shared_max_concurrency: Optional["Synchronized[int]"],
        base_seed: int,
        force_stop_signal: Optional[SyncEvent] = None,
        stage_done_counter: Optional["Synchronized[int]"] = None,
        stage_boundary_seq: Optional["Synchronized[int]"] = None,
        teardown_grace_seconds: float = 120.0,
        crash_queue: Optional["mp.SimpleQueue[WorkerCrash]"] = None,
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
        self.skipped_requests_counter = skipped_requests_counter
        self.shared_max_concurrency = shared_max_concurrency
        self.skip = False
        self.base_seed = base_seed
        self.force_stop_signal = force_stop_signal
        self.stage_done_counter = stage_done_counter
        self.stage_boundary_seq = stage_boundary_seq
        self.teardown_grace_seconds = teardown_grace_seconds
        self.crash_queue = crash_queue
        # Stage this worker is currently serving, reported alongside a crash so the
        # failure can be attributed to a stage rather than to the run as a whole.
        self._current_stage_id: Optional[int] = None
        # Snapshot the parent's effective root log level so the worker
        # interpreter (which under forkserver/spawn does not inherit the
        # parent's basicConfig) can configure its own handler to surface
        # logs at the same level the user asked for.
        self._log_level = logging.getLogger().getEffectiveLevel()

    async def loop(self) -> None:
        # The self.shared_max_concurrency is initialized to self.max_concurrency
        semaphore = Semaphore(self.max_concurrency)
        draining = Event()
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
            while (
                self.request_phase.is_set()
                and not self.cancel_signal.is_set()
                and not self.skip
                and not self.stop_signal.is_set()
            ):
                permit_preacquired = False
                if not isinstance(self.datagen, SessionGenerator):
                    # Preserve bounded queue admission for ordinary request
                    # loads. The timeout lets a worker saturated with hung
                    # requests re-check the loop condition, reach the stage
                    # boundary, and cancel them after the teardown grace.
                    # Session replay acquires only after dependencies.
                    try:
                        await wait_for(semaphore.acquire(), timeout=timeout)
                        permit_preacquired = True
                    except (AsyncioTimeoutError, TimeoutError):
                        continue
                try:
                    # Non-blocking get: a blocking get(timeout=...) holds the
                    # queue's shared reader lock across the poll, so a worker
                    # killed abruptly (OOM, terminate) mid-poll would strand
                    # the lock and starve every consumer of this channel,
                    # including its own respawned replacement. get_nowait
                    # holds the lock only while actually transferring an item.
                    item = await event_loop.run_in_executor(None, self.request_queue.get_nowait)
                    if item is not None:
                        self._current_stage_id = item.stage_id
                    if item is None:
                        if permit_preacquired:
                            semaphore.release()
                        continue
                except TimeoutError:
                    logger.debug(f"[Worker {self.id}] timed out getting request from queue")
                    if permit_preacquired:
                        semaphore.release()
                    continue
                except Empty:
                    if permit_preacquired:
                        semaphore.release()
                    await sleep(0.02)
                    continue
                except Exception as e:
                    logger.info(f"[Worker {self.id}] hit exception {e}")
                    if permit_preacquired:
                        semaphore.release()
                    continue

                async def schedule_client(
                    request_data: InferenceAPIData,
                    request_time: float,
                    stage_id: int,
                    semaphore: Semaphore,
                    lora_adapter: Optional[str],
                    permit_acquired: bool,
                ) -> None:
                    inflight = False
                    try:
                        current_time = time.perf_counter()
                        sleep_time = request_time - current_time
                        if sleep_time > 0:
                            try:
                                await wait_for(draining.wait(), timeout=sleep_time)
                            except (AsyncioTimeoutError, TimeoutError):
                                pass
                            else:
                                logger.debug(f"[Worker {self.id}] stage tearing down, dropping scheduled-ahead request")
                                return

                        # Wait for dependencies before dispatching (OTel trace replay)
                        if hasattr(request_data, "wait_for_predecessors_and_substitute"):
                            await request_data.wait_for_predecessors_and_substitute()

                        # Check if request should be skipped (e.g., session failed in OTel replay)
                        if hasattr(request_data, "skip_request") and request_data.skip_request:
                            logger.debug(
                                f"Skipping request - session failure detected: {getattr(request_data, 'event_id', 'unknown')}"
                            )
                            return  # Exit this task, finally block will clean up

                        # Stage is winding down: in-flight requests may finish,
                        # but nothing new is sent to the model server. This also
                        # unwinds session-replay dependency chains quickly - a
                        # successor woken by its predecessor exits here instead
                        # of dispatching.
                        if draining.is_set():
                            logger.debug(f"[Worker {self.id}] stage tearing down, not dispatching new request")
                            return

                        # Dependency and schedule waits do not consume request
                        # concurrency. Poll so stage teardown can stop tasks that
                        # are waiting for a permit without dispatching them.
                        while (
                            not permit_acquired
                            and not draining.is_set()
                            and self.request_phase.is_set()
                            and not self.cancel_signal.is_set()
                        ):
                            try:
                                await wait_for(semaphore.acquire(), timeout=timeout)
                                permit_acquired = True
                            except (AsyncioTimeoutError, TimeoutError):
                                continue
                        # The stage signals may change while acquire() is
                        # pending, before the worker loop observes teardown.
                        if (
                            not permit_acquired
                            or draining.is_set()
                            or not self.request_phase.is_set()
                            or self.cancel_signal.is_set()
                        ):
                            return

                        # A parallel branch may have failed the session while
                        # this request was waiting for request capacity.
                        if hasattr(request_data, "skip_if_session_failed") and request_data.skip_if_session_failed(
                            "post-permit check"
                        ):
                            return

                        with self.active_requests_counter.get_lock():
                            self.active_requests_counter.value += 1
                            inflight = True

                        await self.client.process_request(request_data, stage_id, request_time, lora_adapter)
                    except CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"[DEBUG] Exception in task: {type(e).__name__}: {e}", exc_info=True)
                        raise
                    finally:
                        with self.active_requests_counter.get_lock():
                            if inflight:
                                self.active_requests_counter.value -= 1
                        if not inflight:
                            # Retired without being sent: the session had already
                            # failed, the stage was winding down, or the task was
                            # cancelled while waiting for its scheduled time or its
                            # predecessors. No lifecycle metric will exist for it, so
                            # it is counted here, on every such path, so the finished
                            # total the progress bar reads still reconciles against
                            # the outcome counters the report is built from.
                            with self.skipped_requests_counter.get_lock():
                                self.skipped_requests_counter.value += 1
                        with self.finished_requests_counter.get_lock():
                            self.finished_requests_counter.value += 1
                        if permit_acquired:
                            semaphore.release()

                try:
                    stage_id, request, request_time, lora_adapter = item
                    request_data = LazyLoadDataMixin.get_request(self.datagen, request)
                except Exception as e:
                    logger.error(f"[Worker {self.id}] Failed to get request: {e}", exc_info=True)
                    with self.skipped_requests_counter.get_lock():
                        self.skipped_requests_counter.value += 1
                    with self.finished_requests_counter.get_lock():
                        self.finished_requests_counter.value += 1
                    if permit_preacquired:
                        semaphore.release()
                    continue

                task = create_task(
                    schedule_client(
                        request_data,
                        request_time,
                        stage_id,
                        semaphore,
                        lora_adapter,
                        permit_preacquired,
                    )
                )
                logging.debug(
                    f"creating inference task with request data {request_data}", extra={"request_data": request_data}
                )
                tasks.append(task)
                await sleep(0)

            # Reset skip
            self.skip = False

            if self.cancel_signal.is_set() or not self.request_phase.is_set():
                await self._wind_down_stage(tasks, draining)
                tasks = []
                LocalUserSession.clear_instances()
                if self.stage_done_counter is not None and self.stage_boundary_seq is not None:
                    # Assign the main-published boundary sequence rather than
                    # incrementing: a worker that enters this block without
                    # having served the stage (e.g. respawned between stages)
                    # then converges on the current boundary instead of
                    # running permanently ahead of the rendezvous.
                    with self.stage_done_counter.get_lock():
                        self.stage_done_counter.value = _counter_value_nolock(self.stage_boundary_seq)
                    logger.debug(f"[Worker {self.id}] waiting for next phase")
                    # Hold until the next stage is running, or until a newer
                    # boundary is published. The latter covers a stage that
                    # started and finished entirely between two samples of
                    # these events (e.g. zero requests, or a circuit breaker
                    # opening immediately): the worker then re-enters the
                    # boundary block and acknowledges the new sequence instead
                    # of waiting out a signal edge it never observed.
                    while not self.stop_signal.is_set():
                        if self.request_phase.is_set() and not self.cancel_signal.is_set():
                            break
                        if _counter_value_nolock(self.stage_boundary_seq) > _counter_value_nolock(self.stage_done_counter):
                            break
                        self.request_phase.wait(timeout=0.05)
                else:
                    while self.cancel_signal.is_set() and not self.stop_signal.is_set():
                        await sleep(0.05)
                    logger.debug(f"[Worker {self.id}] waiting for next phase")
                    while not self.request_phase.is_set() and not self.stop_signal.is_set():
                        self.request_phase.wait(timeout=0.5)

        logger.debug(f"[Worker {self.id}] stopped")

    async def _wind_down_stage(self, tasks: List["Task[None]"], draining: Event) -> None:
        """Bounded end-of-stage wind-down; always returns.

        In-flight requests get up to teardown_grace_seconds to finish naturally
        (recording their metrics). Tasks still parked on session predecessors
        unwind quickly because new dispatches are gated off by the draining event.
        Whatever remains after the grace is cancelled and reaped within a fixed
        bound, so the worker always reaches the stage rendezvous.
        """
        draining.set()
        try:
            pending = [t for t in tasks if not t.done()]
            if pending:
                logger.debug(
                    f"[Worker {self.id}] stage teardown: waiting up to {self.teardown_grace_seconds:.0f}s "
                    f"for {len(pending)} in-flight tasks"
                )
                deadline = time.perf_counter() + self.teardown_grace_seconds
                while pending and time.perf_counter() < deadline:
                    if self.force_stop_signal is not None and self.force_stop_signal.is_set():
                        break
                    await sleep(0.25)
                    pending = [t for t in pending if not t.done()]
            if pending:
                logger.warning(f"[Worker {self.id}] cancelling {len(pending)} tasks still running at stage teardown")
                for task in pending:
                    task.cancel()
                try:
                    await wait_for(gather(*pending, return_exceptions=True), timeout=_WIND_DOWN_REAP_SECONDS)
                except (AsyncioTimeoutError, TimeoutError):
                    stuck = sum(1 for t in pending if not t.done())
                    logger.error(f"[Worker {self.id}] abandoning {stuck} tasks that did not respond to cancellation")
            # Surface real task failures. Before this rework they were re-raised
            # by an unguarded gather at teardown, killing the worker process and
            # stranding the main process at the stage rendezvous.
            for task in tasks:
                if task.done() and not task.cancelled():
                    exc = task.exception()
                    if exc is not None:
                        logger.error(f"[Worker {self.id}] task failed during stage: {type(exc).__name__}: {exc}")
        finally:
            draining.clear()
            # Sessions still open at this point end with the stage and never reach a
            # terminal state, so they never report themselves to the main process. This
            # worker's tracker holds the only record of how far each of them got; hand it
            # over so they can be reported as truncated with real numbers instead of zeros.
            # In the finally so it runs on every wind-down path, including cancellation -
            # which is the path that leaves sessions unfinished in the first place.
            flush = getattr(self.datagen, "flush_incomplete_sessions", None)
            if flush is not None:
                try:
                    flush()
                except Exception as e:
                    logger.error(f"[Worker {self.id}] failed to flush partial session progress: {e}")

    def run(self) -> None:
        """Entry point in the worker process. Nothing below may escape unexplained."""
        try:
            self._run()
        except BaseException as exc:
            # Without this the process dies with the traceback on its own stderr,
            # which races the parent's terminate() and is attributed to no worker.
            # Record it first, then re-raise so the exit code still reflects the failure.
            self._report_crash(exc)
            raise

    def _run(self) -> None:
        # forkserver/spawn workers start from a fresh interpreter without the
        # parent's logging.basicConfig, so logger.info() calls are silently
        # dropped by the lastResort handler (WARNING+). Install a minimal
        # handler at the parent-configured level so worker-side logs surface.
        logging.basicConfig(
            level=self._log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Seed with current time + worker id to ensure unique random sequences per worker
        seed = (self.base_seed + self.id) % 2**32
        np.random.seed(seed)
        # Also reseed any np.random.Generator on the datagen (e.g. RandomDataGenerator.rng),
        # since np.random.seed() only affects the legacy module, not Generator instances.
        if hasattr(self.datagen, "rng") and isinstance(self.datagen.rng, np.random.Generator):
            self.datagen.rng = np.random.default_rng(seed)
        logger.debug(f"[Worker {self.id}] seeded numpy with {seed} and base seed {self.base_seed}")

        # Ignore SIGINT in workers to prevent multiple calls to SIGINT handler
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # Forked with the load generator's SIGTERM handler; terminate() needs the default action.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        set_event_loop_policy(uvloop.EventLoopPolicy())
        run(self.loop())

    def _report_crash(self, exc: BaseException) -> None:
        """Log the failure locally and hand a structured copy to the parent before exiting."""
        try:
            in_flight = int(self.active_requests_counter.value)
        except (AttributeError, OSError, ValueError):
            in_flight = -1
        record = WorkerCrash(
            worker_id=self.id,
            stage_id=self._current_stage_id,
            exc_type=type(exc).__name__,
            message=str(exc),
            traceback_text="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            in_flight=in_flight,
        )
        logger.error(
            "[Worker %d] unhandled %s %s with %d request(s) in flight run-wide: %s\n%s",
            record.worker_id,
            record.exc_type,
            "before serving any stage" if record.stage_id is None else f"during stage {record.stage_id}",
            record.in_flight,
            record.message,
            record.traceback_text.rstrip(),
        )
        if self.crash_queue is None:
            return
        try:
            # SimpleQueue writes straight to the pipe, so the record is not lost to an
            # unflushed feeder thread when the process exits immediately after this.
            self.crash_queue.put(record)
        except Exception:
            logger.error("[Worker %d] could not report its crash to the parent process", self.id)


class LoadGenerator:
    def __init__(
        self,
        datagen: BaseGenerator,
        load_config: LoadConfig,
        session_metrics_collector: Optional[SessionMetricsCollector] = None,
        stage_observer: Optional[StageLifecycleObserver] = None,
        metrics_registry: Optional[CollectorRegistry] = None,
        use_server_output_tokens: bool = False,
    ) -> None:
        self.datagen = datagen
        # The report's output-token rule, so a stop condition's tpot and ntpot are the report's.
        self.use_server_output_tokens = use_server_output_tokens
        self.stage_observer = stage_observer
        # The registry the progress bars read. Without one the run is silent
        # rather than showing bars built from some second set of counters.
        self.metrics_registry = metrics_registry
        # Requests sent and not yet finished. Multiprocess runs read the shared
        # counter the workers maintain; the in-process run() path counts locally.
        self._active_requests_counter: Optional["Synchronized[int]"] = None
        self._local_in_flight = 0
        # The in-process run() path has no worker counters to sample, so it
        # keeps its own. Reset per stage, like the shared ones.
        self._local_finished = 0
        self.stageInterval = load_config.interval
        self.load_type = load_config.type
        self.stages = load_config.stages
        self.stage_runtime_info = dict[int, StageRuntimeInfo]()
        self.num_workers = load_config.num_workers
        self.worker_max_concurrency = load_config.worker_max_concurrency
        self.workers: List[Worker] = []
        self.circuit_breakers = [get_circuit_breaker(breaker_name) for breaker_name in load_config.circuit_breakers]
        # The running stage's measured stop condition, fed by observe_result;
        # None while no stage with one is running.
        self._stop_monitor: Optional[StopConditionMonitor] = None
        self.sweep_config = load_config.sweep
        self.interrupt_sig = False
        self.session_metrics_collector = session_metrics_collector
        self.teardown_grace_seconds = load_config.stage_teardown_grace_seconds
        # Set by mp_run; broadcast to workers when the teardown grace expires
        # so they cancel whatever is still in flight.
        self._force_stop_signal: Optional[SyncEvent] = None
        # Number of stage teardowns initiated so far; each worker's
        # stage_done_counter must reach this value at the stage rendezvous.
        # Published to workers through _stage_boundary_seq so they assign
        # (not increment) their counter at the boundary.
        self._expected_stage_done = 0
        self._stage_boundary_seq: Optional["Synchronized[int]"] = None
        self.worker_crash_queue: Optional["mp.SimpleQueue[WorkerCrash]"] = None
        self.worker_failures: List[WorkerFailure] = []
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, self._interrupt_handler)

        # Validate that datagen type matches load_type
        if self.load_type == LoadType.TRACE_SESSION_REPLAY:
            if not isinstance(datagen, SessionGenerator):
                raise TypeError(
                    f"LoadType.TRACE_SESSION_REPLAY requires SessionGenerator, "
                    f"but got {type(datagen).__name__}. "
                    f"Please use a SessionGenerator-based data generator (e.g., OTelTraceReplayDataGenerator)."
                )
        else:  # CONSTANT, POISSON, CONCURRENT, TRACE_REPLAY
            if not isinstance(datagen, DataGenerator):
                raise TypeError(
                    f"LoadType {self.load_type.value} requires DataGenerator, "
                    f"but got {type(datagen).__name__}. "
                    f"Use LoadType.TRACE_SESSION_REPLAY for SessionGenerator-based data generators."
                )
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
        self._session_cursor: int = 0
        # Read by _run_reported_stage to close out a stage its runner left open.
        self._open_stage: Optional[StageContext] = None
        self._stages_started = 0

    def in_flight_requests(self) -> int:
        """Requests sent to the server and not yet finished, sampled live."""
        if self._active_requests_counter is not None:
            return int(self._active_requests_counter.value)
        return self._local_in_flight

    def stage_count(self) -> int:
        """Stages this run executes, sampled live: a sweep replaces them after preprocess."""
        return len(self.stages)

    async def _track_in_flight(self, request: Awaitable[None]) -> None:
        self._local_in_flight += 1
        try:
            await request
        finally:
            self._local_in_flight -= 1
            self._local_finished += 1

    def _stage_started(self, context: StageContext) -> None:
        self._open_stage = context
        self._stages_started += 1
        if self.stage_observer is not None:
            self.stage_observer.on_stage_start(context)

    def _stage_ended(self, context: StageContext) -> None:
        self._open_stage = None
        if self.stage_observer is not None:
            self.stage_observer.on_stage_end(context)

    def observe_result(self, metric: RequestLifecycleMetric) -> None:
        """Feed one observed result to the running stage's stop condition.

        Registered as a request metric collector observer, next to the circuit
        breakers, so it runs in this process as results are drained.
        """
        if self._stop_monitor is not None:
            self._stop_monitor.observe(metric, time.perf_counter(), self.in_flight_requests())

    def _stop_condition_held(self, stage_id: int, stop_monitor: Optional[StopConditionMonitor]) -> bool:
        """True once the stage's measured stop condition has held; logs when and with what values.

        Called from the stage loop, so it also gives the condition its once-a-second
        evaluation while no result arrives.
        """
        if stop_monitor is None:
            return False
        stop_monitor.tick(time.perf_counter(), self.in_flight_requests())
        if stop_monitor.met_at is None:
            return False
        logger.info(
            "Stage %d: stop condition %r held at t=%.2fs with %s; ending the stage",
            stage_id,
            stop_monitor.predicate.raw,
            stop_monitor.met_at,
            stop_monitor.values_when_met,
        )
        return True

    def _stop_monitor_for(self, stage_id: int, stage: StandardLoadStage) -> Optional[StopConditionMonitor]:
        """A monitor for a stage whose stop condition reads measured terms; None when it ends on its deadline alone."""
        predicate = stage.predicate
        if not predicate.measured_symbols:
            return None
        return StopConditionMonitor(stage_id, predicate, stage.stop_settings.hold, self.use_server_output_tokens)

    async def _run_reported_stage(self, stage_id: int, runner: Awaitable[None]) -> None:
        """Run one stage so its observer sees exactly one start and one end, however the runner exits."""
        # The overall bar reads stages ended against the stage count, so a runner
        # that returns before reporting its stage, or raises after starting it,
        # would leave the bar short for the rest of the run.
        started_before = self._stages_started
        try:
            await runner
        finally:
            if self._stages_started == started_before:
                self._stage_started(StageContext(stage_id=stage_id))
            if self._open_stage is not None:
                self._stage_ended(self._open_stage)

    def _interrupt_handler(self, _signum: int, _frame: Optional[FrameType]) -> None:
        """Ends the run after the current stage's teardown, so its reports are still written."""
        self.interrupt_sig = True

    def _active_worker_count(self, concurrency_level: Optional[int]) -> int:
        """Number of workers that can receive requests in a stage.

        Routing (worker_id % active) and the concurrency split both use this
        number, so a worker that gets no requests also gets no share of
        concurrency_level. Two things shrink it below num_workers: a
        concurrency_level smaller than num_workers, and a data generator
        whose preferred_worker_id values reach fewer workers than exist.
        """
        active = self.num_workers
        if concurrency_level:
            active = min(active, concurrency_level)
        if self.datagen.is_preferred_worker_requested():
            pinned = self.datagen.preferred_worker_count()
            if pinned:
                active = min(active, pinned)
        return max(1, active)

    def _set_worker_concurrency(self, concurrency_level: int) -> None:
        """Determines the per worker concurrency, handling cases where concurrency_level % num_workers != 0."""
        active_workers = self._active_worker_count(concurrency_level)
        # Calculate new concurrency for worker (concurrency_level will always be > 0)
        new_concurrency = concurrency_level // active_workers
        # Calculate index cutoff for workers with +1 concurrency
        remainder = concurrency_level % active_workers
        for worker in self.workers:
            if worker.id >= active_workers:
                # Never routed to, so it must not hold any of the budget.
                worker_concurrency = 0
            else:
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

    def get_timer(self, rate: Union[float, RateSchedule, RequestSchedule], duration: float) -> LoadTimer:
        # A request_interval stage carries its own arrival process; load.type is
        # not set alongside it.
        if isinstance(rate, RequestSchedule):
            return RequestIntervalLoadTimer(rate)
        if self.load_type == LoadType.POISSON:
            return PoissonLoadTimer(rate=rate, duration=duration)
        elif self.load_type == LoadType.TRACE_REPLAY:
            if self.trace is None:
                raise ValueError("Trace configuration is required for trace replay load generator")
            return TraceReplayLoadTimer(trace_reader=self.trace_reader, trace_file=Path(self.trace.file))
        # For concurrent and constant load types (rate is adjusted in main.py for concurrent load type)
        return ConstantLoadTimer(rate=rate, duration=duration)

    async def run_session_stage(
        self,
        stage_id: int,
        stage: TraceSessionReplayLoadStage,
        request_queue: RequestQueue[RequestQueueData],
        active_requests_counter: "Synchronized[int]",
        finished_requests_counter: "Synchronized[int]",
        request_phase: SyncEvent,
        cancel_signal: Optional[SyncEvent] = None,
        progress_ctx: Optional[ProgressBars] = None,
    ) -> None:
        """Run a session-based trace replay stage.

        LoadGen manages a session pool with max size = stage.concurrent_sessions.
        Sessions are dispatched with optional rate limiting (stage.session_rate).
        When a session completes, the next pending session is started to fill the pool.

        ``worker_max_concurrency`` limits requests executing against the model
        server in each worker. Events waiting for their scheduled time or for
        predecessors do not consume those permits.
        """
        logger.info("Stage %d - session-based run started", stage_id)

        request_phase.set()
        with finished_requests_counter.get_lock():
            finished_requests_counter.value = 0

        start_time_epoch = time.time()
        start_time = time.perf_counter()

        # Get total number of sessions

        if not isinstance(self.datagen, SessionGenerator):
            raise ValueError("Session-based replay requires a SessionGenerator")

        total_sessions = self.datagen.get_session_count()
        stage_status = StageStatus.RUNNING
        active_workers = self.num_workers

        # Session pool management
        concurrent_sessions = stage.concurrent_sessions
        session_rate = stage.session_rate

        max_stage_duration = (
            stage.max_stage_duration if stage.max_stage_duration else stage.timeout
        )  # timeout is deprecated, kept for legacy
        duration = stage.duration

        # A duration-bounded stage draws sessions without limit: indices past the corpus
        # resolve to further plays, so it fills its window instead of ending with the corpus.
        can_cycle = self.datagen.supports_corpus_cycling()
        cycle_corpus = duration is not None and can_cycle

        available_sessions = total_sessions - self._session_cursor
        if available_sessions <= 0 and not cycle_corpus:
            # Same symptom, two causes needing different fixes: a corpus that is genuinely
            # used up, or a replayable one this count-bounded stage simply won't replay.
            if can_cycle:
                logger.warning(
                    f"Stage {stage_id}: corpus used up at index {self._session_cursor}, skipping - "
                    f"this generator can replay its corpus, but only a duration-bounded stage does. "
                    f"Give this stage a duration instead of num_sessions to replay, or run it before "
                    f"the stage that used the corpus up."
                )
            else:
                logger.warning(f"Stage {stage_id}: no sessions remaining in trace files, skipping")
            # Reported as a stage with nothing planned so dashboards can tell it
            # was skipped rather than never reached.
            skipped_stage = StageContext(stage_id=stage_id, planned_sessions=0)
            self._stage_started(skipped_stage)
            self._stage_ended(skipped_stage)
            return
        effective_num_sessions = (
            0
            if cycle_corpus
            else (min(stage.num_sessions, available_sessions) if stage.num_sessions is not None else available_sessions)
        )

        stage_start_cursor = self._session_cursor
        next_session_index = stage_start_cursor
        # Where the supply runs out, or None while cycling - which is what leaves the
        # deadline as the only thing that can end the stage.
        session_supply_end: Optional[int] = None if cycle_corpus else stage_start_cursor + effective_num_sessions

        def sessions_left_to_start() -> Optional[int]:
            """Sessions not yet started, or None when the supply is unbounded."""
            return None if session_supply_end is None else session_supply_end - next_session_index

        logger.info(
            f"Session pool: concurrent_sessions={concurrent_sessions}, "
            f"session_rate={session_rate}, max_stage_duration={max_stage_duration}, "
            f"duration={duration}, "
            f"num_sessions={'unbounded (corpus cycling)' if cycle_corpus else effective_num_sessions} "
            f"(corpus offset {stage_start_cursor}), "
            f"total_sessions={total_sessions}"
        )

        # Track active sessions
        active_session_indices: Set[int] = set()  # Session indices currently active
        completed_session_ids: Set[str] = set()  # Session IDs that have completed
        skipped_session_count = 0  # Sessions counted complete that dispatched nothing
        session_dispatch_times: Dict[str, float] = {}  # session_id → wall-clock dispatch time
        session_dispatch_perf_counters: Dict[str, float] = {}  # session_id → perf_counter dispatch time

        # Reported from here, not from the driver loop, because the planned
        # count the progress metrics need is only known once it is computed.
        # worker_failures accumulates across the whole run, so remember where this
        # stage starts and report only what it loses.
        failures_before_stage = len(self.worker_failures)
        stage_context = StageContext(
            stage_id=stage_id,
            planned_sessions=effective_num_sessions,
            planned_duration_seconds=duration,
            # Capped at the window: dispatching stops at the deadline, and the bar
            # should read full there rather than run on through teardown.
            elapsed_seconds=lambda: min(time.perf_counter() - start_time, duration or 0.0),
            sessions_finished=lambda: len(completed_session_ids),
            sessions_skipped=lambda: skipped_session_count,
            workers_lost=lambda: tuple(f.cause_label() for f in self.worker_failures[failures_before_stage:]),
        )
        self._stage_started(stage_context)

        # Cache OTEL instrumentation to avoid redundant calls
        otel_instr = get_otel_instrumentation()
        session_spans: Dict[str, object] = {}  # session_id → OTEL span object
        stage_span = None  # Stage-level span (if trace_per_stage is enabled)
        stage_context_dict = None  # Stage-level context for propagation

        # Start stage-level span if trace_per_stage is enabled
        if otel_instr.trace_per_stage:
            stage_info: Dict[str, Union[int, float]] = {
                "concurrent_sessions": concurrent_sessions,
            }
            # Omitted while cycling: the stage has no session count to report.
            if session_supply_end is not None:
                stage_info["num_sessions"] = effective_num_sessions
            if session_rate is not None:
                stage_info["session_rate"] = session_rate
            if max_stage_duration is not None:
                stage_info["max_stage_duration"] = max_stage_duration
            if duration is not None:
                stage_info["duration"] = duration

            stage_span, stage_context_dict = otel_instr.start_stage_span(stage_id, stage_info)
            logger.info(f"Started stage-level OTEL span for stage {stage_id}")

        # Track dispatch timing
        sessions_dispatched = 0
        last_dispatch_time = start_time
        next_dispatch_time = start_time
        # Skips in a row, and the point at which they mean the corpus itself is the problem.
        # A skipped session never enters the pool, so it never counts against
        # concurrent_sessions and never slows admission down. With a fixed slice that is
        # harmless - the stage skips its share and ends. While cycling there is always
        # another index, so a corpus with nothing buildable would be skipped over as fast as
        # the process can loop, for the whole window, admitting nothing and growing the
        # generator's index tables the entire time. One full pass with no session admitted
        # is proof enough; anything buildable resets the count well before this.
        consecutive_skipped_sessions = 0
        unbuildable_corpus_limit = max(total_sessions, 1)

        def should_start_next_session() -> bool:
            """Check if we should start the next session."""
            # Past the planned stop: admit nothing new. Checked here as well as at the
            # loop exit because dispatch runs earlier in the same iteration, so without
            # this a session could be started microseconds after the deadline only to be
            # cut short immediately.
            if duration is not None and time.perf_counter() - start_time >= duration:
                return False

            # Check concurrency limit (0 = unlimited)
            if concurrent_sessions > 0 and len(active_session_indices) >= concurrent_sessions:
                return False

            # Check if the supply is exhausted (never, while cycling)
            if sessions_left_to_start() == 0:
                return False

            # Check rate limit
            if session_rate is not None:
                now = time.perf_counter()
                if now < next_dispatch_time:
                    return False

            return True

        def dispatch_session(session_idx: int) -> int:
            """Dispatch all events for a session. Returns number of events dispatched."""
            nonlocal sessions_dispatched, last_dispatch_time, next_dispatch_time
            nonlocal consecutive_skipped_sessions

            if not isinstance(self.datagen, SessionGenerator):
                raise TypeError("Expected SessionGenerator for session-based operations")

            # Skip sessions whose graph couldn't be built (malformed spans, or all calls
            # errored with include_errors=False). This matches the eager path, which dropped
            # such sessions at load time. NOTE: skipped sessions are currently UNREPORTED —
            # they are counted as complete (so the pool drains and the stage finishes) but do
            # NOT appear in the session lifecycle metrics. To surface them later, record a
            # skipped/failed session metric here instead of only marking them complete.
            if hasattr(self.datagen, "is_session_buildable") and not self.datagen.is_session_buildable(session_idx):
                nonlocal skipped_session_count
                skipped_session_count += 1
                completed_session_ids.add(self.datagen._session_ids[session_idx])  # type: ignore[attr-defined]
                consecutive_skipped_sessions += 1
                return 0

            consecutive_skipped_sessions = 0

            # Get session info
            session_info = self.datagen.get_session_info(session_idx)
            session_id = session_info["session_id"]

            logger.debug(
                f"Starting session {session_idx}: {session_id} "
                f"({len(active_session_indices)} active, {sessions_left_to_start()} pending)"
            )

            # Start OTEL session span (as child of stage span if trace_per_stage is enabled)
            parent_ctx = stage_context_dict if otel_instr.trace_per_stage else None
            span, context_dict = otel_instr.start_session_span(session_id, session_info, parent_ctx)
            if span is not None:
                session_spans[session_id] = span

            # Activate session in DataGen (marks root events as ready)
            self.datagen.activate_session(session_id)

            # Record dispatch time for session duration tracking
            session_dispatch_times[session_id] = time.time()
            session_dispatch_perf_counters[session_id] = time.perf_counter()

            # Get all events for this session
            events = self.datagen.get_session_events(session_idx)

            # Dispatch all events
            dispatched_count = 0
            for lazy_data in events:
                lora_adapter = self._get_lora_adapter()
                worker_id = lazy_data.preferred_worker_id
                if worker_id >= 0:
                    worker_id = worker_id % active_workers

                # Stamp session_id, OTEL context, and stage_id so workers/generators can use them
                lazy_data.session_id = session_id
                lazy_data.otel_context = context_dict  # Embed OTEL context in data
                lazy_data.stage_id = stage_id

                event_time = time.perf_counter()
                queue_data = RequestQueueData(stage_id, lazy_data, event_time, lora_adapter)
                request_queue.put(queue_data, worker_id)
                dispatched_count += 1

            # Update session pool
            active_session_indices.add(session_idx)
            sessions_dispatched += 1

            # Update timing for rate limiting
            now = time.perf_counter()
            last_dispatch_time = now
            if session_rate is not None:
                session_interval = 1.0 / session_rate
                next_dispatch_time = max(next_dispatch_time + session_interval, now)

            logger.debug(f"Dispatched {dispatched_count} events for session {session_idx}")
            return dispatched_count

        # Main dispatch and wait loop
        stage_task: Optional[BarHandle] = None
        if progress_ctx:
            # A duration-bounded stage plans no session count, so it shows its window.
            bar = STAGE_ELAPSED_BAR if duration is not None else STAGE_SESSIONS_BAR
            stage_task = progress_ctx.open(bar, stage=str(stage_id))

        while True:
            # Check for interrupts
            if self.interrupt_sig:
                if stage_task:
                    stage_task.close()
                    stage_task = None
                logger.info("Loadgen interrupted by a signal")
                stage_status = StageStatus.INTERRUPTED
                # Clean up any active session spans (using cached otel_instr)
                for sid in list(session_spans.keys()):
                    otel_instr.end_session_span(session_spans[sid], "Session interrupted by a signal")
                    del session_spans[sid]
                break

            if cb := next((cb for cb in self.circuit_breakers if cb.is_open()), None):
                if stage_task:
                    stage_task.close()
                    stage_task = None
                logger.warning(f'Loadgen detects circuit breakers "{cb.name}" open, exit the stage.')
                stage_status = StageStatus.FAILED
                # Clean up any active session spans (using cached otel_instr)
                for sid in list(session_spans.keys()):
                    otel_instr.end_session_span(session_spans[sid], f"Session failed due to circuit breaker: {cb.name}")
                    del session_spans[sid]
                break

            if max_stage_duration is not None and time.perf_counter() - start_time >= max_stage_duration:
                if stage_task:
                    stage_task.close()
                    stage_task = None
                logger.warning(f"Stage {stage_id}: max_stage_duration ({max_stage_duration:.1f}s) exceeded")
                stage_status = StageStatus.TIMED_OUT
                # Clean up any active session spans (using cached otel_instr)
                for sid in list(session_spans.keys()):
                    otel_instr.end_session_span(session_spans[sid], "Session cancelled: max_stage_duration exceeded")
                    del session_spans[sid]
                break

            # Fail fast on worker death instead of waiting out the stage
            # timeout: sessions assigned to a dead worker can never complete.
            if self.workers and any(not w.is_alive() for w in self.workers):
                if stage_task:
                    stage_task.close()
                    stage_task = None
                failures = collect_worker_failures(self.workers, self.worker_crash_queue)
                for failure in failures:
                    logger.error("%s", failure.describe())
                self.worker_failures.extend(failures)
                if not failures:
                    logger.error(f"Stage {stage_id}: a worker process died unexpectedly and left no exit status behind")
                logger.error(f"Stage {stage_id}: failing stage after worker death")
                stage_status = StageStatus.FAILED
                for sid in list(session_spans.keys()):
                    otel_instr.end_session_span(session_spans[sid], "Worker process died")
                    del session_spans[sid]
                break

            # Check for completed sessions
            newly_completed = []
            for session_idx in list(active_session_indices):
                session_info = self.datagen.get_session_info(session_idx)
                session_id = session_info["session_id"]

                # Check if this session completed
                if self.datagen.check_session_completed(session_id):
                    if session_id not in completed_session_ids:
                        completed_session_ids.add(session_id)
                        newly_completed.append(session_idx)

                        # End OTEL session span (using cached otel_instr)
                        if session_id in session_spans:
                            # Check if session failed from SessionGenerator state
                            session_state = self.datagen.get_session_state(session_id)
                            session_failed = session_state.failed if session_state else False
                            error_msg = "Session failed" if session_failed else None
                            otel_instr.end_session_span(session_spans[session_id], error_msg)
                            del session_spans[session_id]

                        logger.debug(
                            f"Session {session_idx} ({session_id}) completed "
                            f"({len(completed_session_ids)}/{effective_num_sessions or 'unbounded'} total)"
                        )

            # Remove completed sessions from active pool and clean up memory
            for session_idx in newly_completed:
                active_session_indices.discard(session_idx)

                # Build and record session-level metric before cleanup
                session_info = self.datagen.get_session_info(session_idx)
                session_id = session_info["session_id"]
                dispatch_time = session_dispatch_times.get(session_id)
                if dispatch_time is None:
                    logger.warning(
                        f"Session {session_id} has no recorded dispatch time; "
                        f"falling back to stage start for session start_time."
                    )
                session_metric = self.datagen.build_session_metric(
                    session_id=session_id,
                    stage_id=stage_id,
                    start_time=dispatch_time if dispatch_time is not None else start_time_epoch,
                    end_time=time.time(),
                )
                session_metric.dispatch_perf_counter = session_dispatch_perf_counters.get(session_id)

                # Record in collector instead of datagen
                if self.session_metrics_collector:
                    self.session_metrics_collector.record_metric(session_metric)

                # Clean up completed session data to prevent memory leaks
                self.datagen.cleanup_session(session_id)

            # Try to start new sessions to fill the pool
            while should_start_next_session():
                session_idx = next_session_index
                next_session_index += 1
                dispatch_session(session_idx)
                if cycle_corpus and consecutive_skipped_sessions >= unbuildable_corpus_limit:
                    break

            if cycle_corpus and consecutive_skipped_sessions >= unbuildable_corpus_limit:
                logger.error(
                    f"Stage {stage_id}: no session in the corpus could be built - "
                    f"{consecutive_skipped_sessions} in a row were skipped, which is the whole corpus "
                    f"of {total_sessions}. Replaying it would skip every session again, so the stage is "
                    "stopped instead of running to its duration. Check that the trace files are the "
                    "format this generator expects, and if include_errors is false, that not every "
                    "recorded call errored."
                )
                stage_status = StageStatus.FAILED
                break

            # Only meaningful for a fixed slice: a cycling stage has no count to finish.
            if session_supply_end is not None and len(completed_session_ids) >= effective_num_sessions:
                logger.info(f"All {effective_num_sessions} sessions completed")
                skipped = getattr(self.datagen, "_skipped_session_count", 0)
                if skipped:
                    logger.warning(f"{skipped} session(s) were skipped (failed to build or had no schedulable events)")
                break

            # Planned stop. stage_status is deliberately left RUNNING so the fall-through
            # below marks it COMPLETED: this is a stop we asked for, not a failure. Any
            # session still active is recorded as truncated after teardown rather than
            # here, so that a session whose last request lands while the stage winds down
            # is recorded as complete instead of truncated. Checked before the
            # pending/active exit so a corpus that runs dry exactly at the deadline
            # reports the planned stop rather than the short-corpus warning.
            if duration is not None and time.perf_counter() - start_time >= duration:
                logger.info(
                    f"Stage {stage_id}: duration {duration:.1f}s reached; "
                    f"{len(completed_session_ids)} session(s) completed, "
                    f"{len(active_session_indices)} still running"
                )
                break

            # Nothing left to start or wait for. While cycling this is None, never 0, so
            # the stage runs to its deadline instead.
            if sessions_left_to_start() == 0 and not active_session_indices:
                logger.info("No more sessions to dispatch or wait for")
                break

            # Sleep and update progress
            await sleep(0)

            # Refresh is throttled internally, so driving it from this hot loop
            # costs a clock read per iteration rather than a registry collect.
            if progress_ctx:
                progress_ctx.refresh()

        # Clean up progress task
        if stage_task:
            stage_task.close()
            stage_task = None

        # A duration-bounded stage that exited before its deadline ran out of corpus, so
        # it measured a shorter window than was asked for. Only a generator that cannot
        # replay its corpus reaches here: a cycling one draws further plays instead of
        # running out. Checked here rather than at each break so every early-exit path is
        # covered, and only while the status is still RUNNING — on a failure the shortfall
        # is not the story.
        if duration is not None and stage_status == StageStatus.RUNNING:
            elapsed = time.perf_counter() - start_time
            if elapsed < duration:
                # Kept generic: weka_trace_replay grows its corpus with
                # duplicate_sessions_target, and another non-cycling generator may size
                # its corpus by some other setting.
                logger.warning(
                    f"Stage {stage_id}: corpus exhausted after {elapsed:.1f}s but duration="
                    f"{duration:.1f}s was requested - the stage ran {duration - elapsed:.1f}s short "
                    f"of the requested window. Grow the session corpus in the data config until "
                    f"it covers the window."
                )

        # Mark stage as completed if we finished normally
        if stage_status == StageStatus.RUNNING:
            stage_status = StageStatus.COMPLETED

        # Sessions this stage never got to dispatch, dropped by an early exit
        # (max_stage_duration exceeded, SIGINT, an open circuit breaker, or a dead worker).
        # Taken from the cursor rather than a materialized pending list, because a cycling
        # duration-bounded stage has an unbounded supply: no fixed set was promised, so
        # nothing is left outstanding and the count is 0.
        sessions_left = sessions_left_to_start()
        sessions_not_completed_pending = max(0, sessions_left) if sessions_left is not None else 0
        if sessions_not_completed_pending:
            logger.warning(f"Stage {stage_id}: {sessions_not_completed_pending} session(s) never started when the stage ended")
        # The stranded-active count is settled after teardown, by the truncation sweep below:
        # a session whose last request lands inside the grace really does finish, so counting
        # the pool here would contradict the lifecycle rows the report is built from.

        # The metrics window ends here: the teardown tail carries no offered
        # load, so including it would stretch every server-side rate average.
        end_time_epoch = time.time()

        # Bounded teardown: stop dispatching, give in-flight requests the
        # configured grace to complete (their metrics are kept), then force
        # the stage boundary. Guaranteed to return, so reports always generate.
        teardown_start = time.perf_counter()
        teardown = await self._teardown_stage(stage_id, request_queue, request_phase, cancel_signal)
        teardown_duration = time.perf_counter() - teardown_start
        if not teardown.clean and stage_status == StageStatus.COMPLETED:
            stage_status = StageStatus.FAILED

        # Sessions still active when the loop exited never reached the completion path
        # above, so they produced no lifecycle metric and their OTEL span was never
        # ended. Both are handled here — deliberately after teardown, because a session
        # whose last request lands while in-flight work is winding down completes, and
        # recording earlier would label it truncated when it is not. check_session_
        # completed() drains the worker completion queue, so those late completions are
        # picked up.
        #
        # This runs on every exit path, not just the duration one: a timeout, an
        # interrupt or an open circuit breaker drops sessions the same way, and marking
        # them truncated keeps them out of the success/failure counts either way. The
        # requests they did complete are already in the request-level metrics.
        sessions_truncated = 0
        for session_idx in list(active_session_indices):
            session_id = self.datagen.get_session_info(session_idx)["session_id"]
            if session_id in completed_session_ids:
                continue
            session_completed = self.datagen.check_session_completed(session_id)
            # Never the current time: teardown waits on every in-flight request, so reading
            # the clock here would charge this session for time spent draining others.
            # A session cut short ends at the boundary captured above. One that finished
            # in the grace, successfully or not, ends when its own last activity did, so
            # its duration is what it really took rather than capped at the boundary.
            session_start = session_dispatch_times.get(session_id, start_time_epoch)
            dispatch_perf = session_dispatch_perf_counters.get(session_id)
            session_end = end_time_epoch
            if session_completed and dispatch_perf is not None:
                end_perf = self.datagen.session_end_perf_counter(session_id)
                if end_perf is not None:
                    session_end = session_start + (end_perf - dispatch_perf)
            session_metric = self.datagen.build_session_metric(
                session_id=session_id,
                stage_id=stage_id,
                start_time=session_start,
                end_time=session_end,
            )
            session_metric.dispatch_perf_counter = dispatch_perf
            if not session_completed:
                session_metric.truncated = True
                sessions_truncated += 1
            if self.session_metrics_collector:
                self.session_metrics_collector.record_metric(session_metric)
            if session_id in session_spans:
                if not session_completed:
                    span_error: Optional[str] = "Session cut short at stage boundary"
                else:
                    # Same check as the in-loop completion path: a session that failed
                    # during the grace is terminal, but must not be traced as OK.
                    session_state = self.datagen.get_session_state(session_id)
                    span_error = "Session failed" if session_state is not None and session_state.failed else None
                otel_instr.end_session_span(session_spans[session_id], span_error)
                del session_spans[session_id]
            self.datagen.cleanup_session(session_id)

        # One source of truth for stranded actives: every session still unfinished after the
        # grace got a truncated row just above, and every row counted here has one.
        sessions_not_completed_active = sessions_truncated
        if sessions_truncated:
            logger.warning(
                f"Stage {stage_id}: {sessions_truncated} session(s) cut short at the stage boundary; "
                f"recorded as truncated and excluded from the session success/failure counts and "
                f"duration percentiles"
            )

        # The next index this stage would have handed out, so a stage that stopped early does
        # not burn sessions it never started, and slots whose graph failed to build still
        # count as consumed rather than being retried by the next stage.
        self._session_cursor = next_session_index

        # End stage-level span if trace_per_stage is enabled
        if stage_span is not None:
            error_msg = (
                None
                if stage_status == StageStatus.COMPLETED
                else f"Stage {_STAGE_STATUS_LOG_VERB.get(stage_status, 'failed')}"
            )
            otel_instr.end_stage_span(stage_span, error_msg)
            logger.info(f"Ended stage-level OTEL span for stage {stage_id}")

        self.stage_runtime_info[stage_id] = StageRuntimeInfo(
            stage_id=stage_id,
            rate=session_rate if session_rate else 0.0,
            start_time=start_time_epoch,
            end_time=end_time_epoch,
            status=stage_status,
            concurrency_level=concurrent_sessions,
            max_stage_duration=max_stage_duration,
            duration=duration,
            teardown_duration=teardown_duration,
            dropped_requests=teardown.dropped_requests,
            dropped_requests_exact=teardown.dropped_requests_exact,
            sessions_not_completed_active=sessions_not_completed_active,
            sessions_not_completed_pending=sessions_not_completed_pending,
        )
        logger.info("Stage %d - session-based run %s", stage_id, _STAGE_STATUS_LOG_VERB.get(stage_status, "failed"))
        self._stage_ended(stage_context)

    async def _teardown_stage(
        self,
        stage_id: int,
        request_queue: RequestQueue[RequestQueueData],
        request_phase: SyncEvent,
        cancel_signal: Optional[SyncEvent],
    ) -> TeardownResult:
        """Bounded stage teardown; always returns, so reports always generate.

        Stops dispatch immediately, gives in-flight requests the configured
        grace to finish (their metrics are kept), then forces the boundary:
        the force-stop signal makes workers cancel whatever is left, and any
        worker that still fails to reach the stage rendezvous (died, or its
        event loop is wedged) is terminated and respawned so subsequent stages
        run at full capacity. Returns clean=True when every worker wound down
        cleanly, clean=False when the stage had to be forced, plus the number
        of never-dispatched requests dropped from the queue.
        """
        event_loop = get_event_loop()
        forced: Set[int] = set()

        clean = True
        if self.workers:
            # Publish the boundary sequence before signalling so any worker
            # observing the boundary reads the current value.
            self._expected_stage_done += 1
            expected = self._expected_stage_done
            if self._stage_boundary_seq is not None:
                self._stage_boundary_seq.value = expected
        if cancel_signal is not None:
            cancel_signal.set()
        request_phase.clear()

        if self.workers:
            grace_deadline = time.perf_counter() + self.teardown_grace_seconds + _TEARDOWN_MARGIN_SECONDS
            force_deadline: Optional[float] = None
            while True:
                pending = [
                    w
                    for w in self.workers
                    if w.is_alive()
                    and (w.stage_done_counter is None or _counter_value_nolock(w.stage_done_counter) < expected)
                ]
                if not pending:
                    break
                now = time.perf_counter()
                if force_deadline is None and now >= grace_deadline:
                    logger.warning(
                        "Stage %d: teardown grace expired; forcing cancellation of in-flight work on %d worker(s)",
                        stage_id,
                        len(pending),
                    )
                    if self._force_stop_signal is not None:
                        self._force_stop_signal.set()
                    force_deadline = now + _FORCE_REAP_SECONDS
                elif force_deadline is not None and now >= force_deadline:
                    for worker in pending:
                        logger.error("Stage %d: terminating worker %d stuck in teardown", stage_id, worker.id)
                        worker.terminate()
                        forced.add(worker.id)
                    clean = False
                    break
                await sleep(0.25)

        # Nobody is pulling anymore: remove undispatched items left in the queue.
        dropped = request_queue.drain()
        if dropped:
            logger.warning(
                "Stage %d: dropped %s%d request(s) that were never dispatched to the model server",
                stage_id,
                "" if dropped.exact else "at least ",
                dropped.removed,
            )
        if self._force_stop_signal is not None:
            self._force_stop_signal.clear()
        if cancel_signal is not None:
            cancel_signal.clear()

        # Replace workers that did not survive the stage (terminated above, or
        # died earlier e.g. from an OOM kill) so later stages keep full capacity.
        # A terminated worker may have died holding its counter's lock, so its
        # counter is never re-read here: forced membership already implies the
        # worker missed the rendezvous.
        for idx, worker in enumerate(self.workers):
            if not worker.is_alive() and worker.exitcode is None:
                continue  # never started; leave to caller
            if (
                worker.id in forced
                or not worker.is_alive()
                or (
                    worker.stage_done_counter is not None
                    and _counter_value_nolock(worker.stage_done_counter) < self._expected_stage_done
                )
            ):
                clean = False
                await event_loop.run_in_executor(None, partial(worker.join, 5.0))
                if worker.is_alive():
                    worker.kill()
                    await event_loop.run_in_executor(None, partial(worker.join, 5.0))
                logger.warning("Stage %d: worker %d did not survive the stage; respawning", stage_id, worker.id)
                # A per-worker channel has no other consumers, so swap it out
                # in case the dead worker stranded the old queue's locks. A
                # shared channel is kept: surviving workers still hold it.
                fresh_channel = request_queue.replace_channel(worker.id) if request_queue.num_channels > 1 else None
                self.workers[idx] = self._respawn_worker(worker, request_channel=fresh_channel)
                self.workers[idx].start()
        return TeardownResult(clean=clean, dropped_requests=dropped.removed, dropped_requests_exact=dropped.exact)

    def _respawn_worker(
        self, dead: Worker, request_channel: Optional["mp.Queue[Optional[RequestQueueData]]"] = None
    ) -> Worker:
        """Build a replacement for a dead worker, sharing the same IPC objects.

        The replacement forks from the main process' current state, and its
        stage-done counter starts at the current expected value so it is
        considered up to date at the next rendezvous. request_channel, when
        given, replaces the dead worker's queue channel (see _teardown_stage).
        """
        stage_done_counter: "Synchronized[int]" = mp.Value("i", self._expected_stage_done)
        return Worker(
            dead.id,
            dead.client,
            request_channel if request_channel is not None else dead.request_queue,
            dead.datagen,
            dead.max_concurrency,
            dead.stop_signal,
            dead.cancel_signal,
            dead.request_phase,
            dead.finished_requests_counter,
            dead.active_requests_counter,
            dead.skipped_requests_counter,
            dead.shared_max_concurrency,
            dead.base_seed,
            force_stop_signal=dead.force_stop_signal,
            stage_done_counter=stage_done_counter,
            stage_boundary_seq=dead.stage_boundary_seq,
            teardown_grace_seconds=dead.teardown_grace_seconds,
            crash_queue=dead.crash_queue,
        )

    async def run_stage(
        self,
        stage_id: int,
        rate: Union[float, RateSchedule, RequestSchedule],
        duration: float,
        request_queue: RequestQueue[RequestQueueData],
        active_requests_counter: "Synchronized[int]",
        finished_requests_counter: "Synchronized[int]",
        skipped_requests_counter: "Synchronized[int]",
        request_phase: SyncEvent,
        cancel_signal: Optional[SyncEvent] = None,
        timeout: Optional[float] = None,
        concurrency_level: Optional[int] = None,
        progress_ctx: Optional[ProgressBars] = None,
        report_stage: bool = True,
        stop_monitor: Optional[StopConditionMonitor] = None,
    ) -> None:
        logger.info("Stage %d - run started", stage_id)

        if timeout is not None and cancel_signal is None:
            raise Exception("run_stage timeout requires cancel_signal to be not None!")

        request_phase.set()
        with finished_requests_counter.get_lock():
            finished_requests_counter.value = 0
        with skipped_requests_counter.get_lock():
            skipped_requests_counter.value = 0
        timer = self.get_timer(rate, duration)

        # Allow generation a second to begin populating the queue so the workers
        # don't miss the initial scheuled request times
        start_time_epoch = time.time()
        start_time = time.perf_counter() + 1
        if stop_monitor is not None:
            stop_monitor.start(start_time)
            self._stop_monitor = stop_monitor

        if isinstance(self.datagen, DataGenerator) and self.datagen.trace is not None:
            num_requests = self.datagen.get_request_count()
        else:
            num_requests = rate.expected_requests if not isinstance(rate, (float, int)) else int(rate * duration)
        # Reports carry one rate per stage; a time-varying rate reports its mean.
        report_rate = rate.mean_rate if not isinstance(rate, (float, int)) else rate

        stage_status = StageStatus.RUNNING
        # Reported from here, not from the driver loop, because the planned
        # count the progress metrics need is only known once it is computed.
        # worker_failures accumulates across the whole run, so remember where this
        # stage starts and report only what it loses.
        failures_before_stage = len(self.worker_failures)
        stage_context = StageContext(
            stage_id=stage_id,
            planned_requests=num_requests,
            requests_finished=lambda: int(finished_requests_counter.value),
            requests_skipped=lambda: int(skipped_requests_counter.value),
            workers_lost=lambda: tuple(f.cause_label() for f in self.worker_failures[failures_before_stage:]),
        )
        if report_stage:
            self._stage_started(stage_context)

        time_generator = timer.start_timer(start_time)
        if isinstance(self.datagen, DataGenerator):
            data_generator = self.datagen.get_data()
        else:
            raise TypeError("run_stage requires DataGenerator, use run_session_stage for SessionGenerator")
        # Same count _set_worker_concurrency split the level over, so pinned
        # requests only ever land on workers that hold a share of it.
        active_workers = self._active_worker_count(concurrency_level)

        for _ in range(num_requests):
            request_data = next(data_generator)
            request_data.stage_id = stage_id
            lora_adapter = self._get_lora_adapter()
            worker_id = request_data.preferred_worker_id
            if worker_id >= 0:
                worker_id = worker_id % active_workers
            scheduled_time = next(time_generator)
            if stop_monitor is not None:
                stop_monitor.dispatched(scheduled_time)
            request_queue.put(
                RequestQueueData(stage_id, request_data, scheduled_time, lora_adapter),
                worker_id,
            )

        # Wait until all requests are finished processing
        stage_task = None
        if progress_ctx:
            stage_task = progress_ctx.open(STAGE_REQUESTS_BAR, stage=str(stage_id))

        stage_status = StageStatus.RUNNING
        while finished_requests_counter.value < num_requests:
            if self._stop_condition_held(stage_id, stop_monitor):
                break
            if timeout and start_time + timeout < time.perf_counter():
                logger.info(f"Loadgen timed out after {timeout:0.2f}s")
                stage_status = StageStatus.TIMED_OUT
                break
            if self.interrupt_sig:
                logger.info("Loadgen interrupted by a signal")
                stage_status = StageStatus.INTERRUPTED
                break
            if cb := next((cb for cb in self.circuit_breakers if cb.is_open()), None):
                logger.warning(f'Loadgen detects circuit breakers "{cb.name}" open, exit the stage.')
                stage_status = StageStatus.FAILED
                break
            if self.workers and len([w for w in self.workers if w.is_alive()]) < self.num_workers:
                failures = collect_worker_failures(self.workers, self.worker_crash_queue)
                for failure in failures:
                    logger.error("%s", failure.describe())
                self.worker_failures.extend(failures)
                if not failures:
                    logger.error("A worker process died unexpectedly and left no exit status behind!")
                stage_status = StageStatus.FAILED  # Trigger cleanup
                break
            await sleep(1)
            if progress_ctx:
                progress_ctx.refresh()

        if stage_task:
            stage_task.close()
        self._stop_monitor = None

        if stage_status == StageStatus.RUNNING:
            stage_status = StageStatus.COMPLETED

        # The metrics window ends here: the teardown tail carries no offered
        # load, so including it would stretch every server-side rate average.
        end_time_epoch = time.time()

        # Bounded teardown: stop dispatching, give in-flight requests the
        # configured grace to complete, then force the stage boundary.
        teardown_start = time.perf_counter()
        teardown = await self._teardown_stage(stage_id, request_queue, request_phase, cancel_signal)
        teardown_duration = time.perf_counter() - teardown_start
        if not teardown.clean and stage_status == StageStatus.COMPLETED:
            stage_status = StageStatus.FAILED

        self.stage_runtime_info[stage_id] = StageRuntimeInfo(
            stage_id=stage_id,
            rate=report_rate,
            start_time=start_time_epoch,
            end_time=end_time_epoch,
            status=stage_status,
            concurrency_level=concurrency_level,
            max_stage_duration=timeout,
            teardown_duration=teardown_duration,
            dropped_requests=teardown.dropped_requests,
            dropped_requests_exact=teardown.dropped_requests_exact,
            stop_condition_met_at=stop_monitor.met_at if stop_monitor is not None else None,
        )
        logger.info("Stage %d - run %s", stage_id, _STAGE_STATUS_LOG_VERB.get(stage_status, "failed"))
        if report_stage:
            self._stage_ended(stage_context)

    async def preprocess(
        self,
        client: ModelServerClient,
        request_queue: RequestQueue[RequestQueueData],
        active_requests_counter: "Synchronized[int]",
        finished_requests_counter: "Synchronized[int]",
        skipped_requests_counter: "Synchronized[int]",
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
            skipped_requests_counter,
            request_phase,
            timeout=timeout,
            cancel_signal=cancel_signal,
            report_stage=False,
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
            self.num_workers if self.datagen.is_preferred_worker_requested() else 1
        )
        finished_requests_counter: "Synchronized[int]" = mp.Value("i", 0)
        active_requests_counter: "Synchronized[int]" = mp.Value("i", 0)
        skipped_requests_counter: "Synchronized[int]" = mp.Value("i", 0)
        self._active_requests_counter = active_requests_counter
        request_phase: SyncEvent = mp.Event()
        stop_signal: SyncEvent = mp.Event()
        cancel_signal: SyncEvent = mp.Event()
        # Broadcast when the teardown grace expires so workers cancel whatever
        # is still in flight (see _teardown_stage).
        force_stop_signal: SyncEvent = mp.Event()
        self._force_stop_signal = force_stop_signal
        # Stage-boundary sequence number, written only by the main process.
        # Workers assign it to their stage_done_counter at each rendezvous
        # (see _teardown_stage).
        stage_boundary_seq: "Synchronized[int]" = mp.Value("i", 0)
        self._stage_boundary_seq = stage_boundary_seq
        # Channel a dying worker uses to hand its traceback to the parent.
        self.worker_crash_queue = mp.SimpleQueue()
        # start workers in the request phase
        request_phase.set()

        # Create list of workers to process requests
        for id in range(self.num_workers):
            # Create shared value for each worker's max concurrency if concurrent load type
            if self.load_type == LoadType.CONCURRENT:
                shared_max_concurrency = mp.Value("i", self.worker_max_concurrency)
            else:
                shared_max_concurrency = None

            # Per-worker stage-done counter for the stage rendezvous: the worker
            # sets it to the published boundary sequence after winding down each
            # stage (see _teardown_stage).
            stage_done_counter: "Synchronized[int]" = mp.Value("i", 0)

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
                    skipped_requests_counter,
                    shared_max_concurrency,
                    self.base_seed,
                    force_stop_signal=force_stop_signal,
                    stage_done_counter=stage_done_counter,
                    stage_boundary_seq=stage_boundary_seq,
                    teardown_grace_seconds=self.teardown_grace_seconds,
                    crash_queue=self.worker_crash_queue,
                )
            )
            self.workers[-1].start()

        if self.sweep_config:
            try:
                await self.preprocess(
                    client,
                    request_queue,
                    active_requests_counter,
                    finished_requests_counter,
                    skipped_requests_counter,
                    request_phase,
                    cancel_signal,
                )
            except Exception as e:
                logger.error(f"Preprocessing exception: {e}")
                stop_signal.set()
                return

        if self.load_type == LoadType.TRACE_SESSION_REPLAY:
            if isinstance(self.datagen, SessionGenerator):
                total_sessions = self.datagen.get_session_count()
                total_requested = sum(
                    s.num_sessions
                    for s in self.stages
                    if isinstance(s, TraceSessionReplayLoadStage) and s.num_sessions is not None
                )
                has_open_ended = any(
                    isinstance(s, TraceSessionReplayLoadStage) and s.num_sessions is None for s in self.stages
                )
                if not has_open_ended and total_requested > total_sessions:
                    raise ValueError(
                        f"Stages request {total_requested} sessions total but corpus only has "
                        f"{total_sessions}. Reduce num_sessions across stages, add more trace files, "
                        f"or set 'duplicate_sessions_target: {total_requested}' in data.otel_trace_replay config "
                        f"to duplicate sessions to meet the total required across all stages."
                    )

        # Create progress context for all stages
        with ProgressBars(self.metrics_registry, console=get_console()) as progress:
            progress.open(OVERALL_BAR)
            for stage_id, stage in enumerate(self.stages):
                # Handle session-based trace replay
                if self.load_type == LoadType.TRACE_SESSION_REPLAY and isinstance(stage, TraceSessionReplayLoadStage):
                    await self._run_reported_stage(
                        stage_id,
                        self.run_session_stage(
                            stage_id,
                            stage,
                            request_queue,
                            active_requests_counter,
                            finished_requests_counter,
                            request_phase,
                            cancel_signal,
                            progress_ctx=progress,
                        ),
                    )
                # Update worker concurrency for concurrent load type
                elif self.load_type == LoadType.CONCURRENT and isinstance(stage, ConcurrentLoadStage):
                    logger.debug(f"Setting worker concurrency to {stage.concurrency_level} for stage {stage_id}")
                    self._set_worker_concurrency(stage.concurrency_level)

                    # Use the dynamically set rate/duration from main.py
                    rate = getattr(stage, "rate", stage.num_requests)
                    duration = getattr(stage, "duration", 1)
                    concurrency_level = stage.concurrency_level
                    await self._run_reported_stage(
                        stage_id,
                        self.run_stage(
                            stage_id,
                            rate,
                            duration,
                            request_queue,
                            active_requests_counter,
                            finished_requests_counter,
                            skipped_requests_counter,
                            request_phase,
                            cancel_signal,
                            concurrency_level=concurrency_level,
                            progress_ctx=progress,
                        ),
                    )
                elif self.load_type != LoadType.CONCURRENT and isinstance(stage, StandardLoadStage):
                    duration = stage.effective_duration
                    concurrency_level = None
                    await self._run_reported_stage(
                        stage_id,
                        self.run_stage(
                            stage_id,
                            stage.schedule,
                            duration,
                            request_queue,
                            active_requests_counter,
                            finished_requests_counter,
                            skipped_requests_counter,
                            request_phase,
                            cancel_signal,
                            concurrency_level=concurrency_level,
                            progress_ctx=progress,
                            stop_monitor=self._stop_monitor_for(stage_id, stage),
                        ),
                    )
                else:
                    raise Exception(f"Stage {stage_id} has the wrong load type")

                # Stage rendezvous already happened inside _teardown_stage
                # (bounded, liveness-aware), so no barrier is needed here.

                # If we were interrupted by a signal, we can break out of run stages loop
                if self.interrupt_sig:
                    break
                progress.refresh(force=True)
                if self.stageInterval:
                    await sleep(self.stageInterval)

        # Reset the request phase to get workers out of their final wait()
        request_phase.set()
        stop_signal.set()

    async def run(self, client: ModelServerClient) -> None:
        if self.num_workers > 0:
            return await self.mp_run(client)

        # Create progress context
        with ProgressBars(self.metrics_registry, console=get_console()) as progress:
            progress.open(OVERALL_BAR)

            for stage_id, stage in enumerate(self.stages):
                if not isinstance(stage, StandardLoadStage):
                    raise TypeError(f"Non-multiprocessing run() only supports StandardLoadStage, got {type(stage)}")

                await self._run_reported_stage(stage_id, self._run_in_process_stage(client, stage_id, stage, progress))
                progress.refresh(force=True)
                LocalUserSession.clear_instances()
                if self.interrupt_sig:
                    break
                if self.stageInterval and stage_id < len(self.stages) - 1:
                    await sleep(self.stageInterval)

    async def _run_in_process_stage(
        self, client: ModelServerClient, stage_id: int, stage: StandardLoadStage, progress: ProgressBars
    ) -> None:
        """Send one stage's requests from this process and record its StageRuntimeInfo."""
        duration = stage.effective_duration
        timer = self.get_timer(stage.schedule, duration)
        start_time_epoch = time.time()
        start_time = time.perf_counter()
        end_time = start_time + duration
        stop_monitor = self._stop_monitor_for(stage_id, stage)
        if stop_monitor is not None:
            stop_monitor.start(start_time)
            self._stop_monitor = stop_monitor
        stage_status = StageStatus.RUNNING
        logger.info("Stage %d - run started", stage_id)

        # This path has no worker processes to sample, so it counts its
        # own completions; reset per stage like the shared counters are.
        self._local_finished = 0
        self._stage_started(
            StageContext(
                stage_id=stage_id,
                planned_requests=stage.expected_requests,
                requests_finished=lambda: self._local_finished,
            )
        )
        stage_task = progress.open(STAGE_REQUESTS_BAR, stage=str(stage_id))

        if not isinstance(self.datagen, DataGenerator):
            raise TypeError("Non-multiprocessing run() requires DataGenerator")

        async with TaskGroup() as tg:
            time_generator = timer.start_timer(start_time)
            for data, time_index in zip(self.datagen.get_data(), time_generator, strict=False):
                if self.interrupt_sig:
                    logger.info("Loadgen interrupted by a signal")
                    stage_status = StageStatus.INTERRUPTED
                    break
                progress.refresh()
                data.stage_id = stage_id
                request_data = LazyLoadDataMixin.get_request(self.datagen, data)
                lora_adapter = self._get_lora_adapter()
                if cb := next((cb for cb in self.circuit_breakers if cb.is_open()), None):
                    logger.warning(f'Loadgen detects circuit breakers "{cb.name}" open, clean up stage and exit early.')
                    stage_status = StageStatus.FAILED
                    break
                if self._stop_condition_held(stage_id, stop_monitor):
                    break
                now = time.perf_counter()
                if time_index < end_time + _STAGE_END_TOLERANCE:
                    if cb := next((cb for cb in self.circuit_breakers if cb.is_open()), None):
                        logger.warning(f'Loadgen detects circuit breakers "{cb.name}" open, clean up stage and exit early.')
                        stage_status = StageStatus.FAILED
                        break
                    if time_index > now:
                        await sleep(time_index - time.perf_counter())
                    if stop_monitor is not None:
                        stop_monitor.dispatched(time_index)
                    tg.create_task(
                        self._track_in_flight(client.process_request(request_data, stage_id, time_index, lora_adapter))
                    )
                    continue
                else:
                    break

        progress.refresh(force=True)
        stage_task.close()  # Clean up after completion
        self._stop_monitor = None

        if stage_status == StageStatus.RUNNING:
            stage_status = StageStatus.COMPLETED
        logger.info("Stage %d - run %s", stage_id, _STAGE_STATUS_LOG_VERB.get(stage_status, "failed"))
        self.stage_runtime_info[stage_id] = StageRuntimeInfo(
            stage_id=stage_id,
            rate=stage.mean_rate,
            start_time=start_time_epoch,
            end_time=time.time(),
            status=stage_status,
            concurrency_level=None,
            stop_condition_met_at=stop_monitor.met_at if stop_monitor is not None else None,
        )

    async def stop(self) -> None:
        for worker in self.workers:
            worker.join(timeout=1.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=0.0)
