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
"""End-to-end probe preprocessing tests with real worker processes.

The main test runs the closed-loop capacity probe against the mock client
(fixed 50ms latency, so closed-loop throughput is ideally X(N) = 20N and never
saturates) and then the generated stages, locking these invariants:

- Every probe rung's phase boundary pairs a main-side stage_barrier.wait()
  with one arrival per worker, so the run completes instead of deadlocking
  (asyncio.wait_for turns a pairing bug into a test failure).
- Rung measurements obey Little's law against the known mock latency.
- Probe traffic stays on negative stage ids, which the report generator
  drops, and the generated stages run to completion afterwards.
- A ladder stopped by its ceiling is reported as censored, and the run's
  measured window starts after the probe rather than with it.

The rung-outcome tests below pin the three ways a rung can turn bad: a backend
that saturates and queues (overloaded rungs must leave the fit), a backend that
fails every request of one rung (that rung ends the ladder instead of entering
the curve), and a backend that stops answering (the tail is cancelled and the
rung ends the ladder) rather than silently reading any of them as the server's
capacity. When no rung is measurable at all, the probe still fails loudly.
"""

import asyncio
import multiprocessing as mp
import sys
import time
import unittest
from typing import Any, Optional
from unittest.mock import patch

from inference_perf.client.modelserver.mock_client import MockModelServerClient
from inference_perf.config import (
    APIConfig,
    APIType,
    DataConfig,
    DataGenType,
    Distribution,
    LoadConfig,
    LoadType,
    ProbeConfig,
    StageGenType,
    StandardLoadStage,
    SweepConfig,
)
from inference_perf.datagen.synthetic.random_datagen import RandomDataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.loadgen.probe import ProbeStopReason
from inference_perf.metrics.request_collector import MultiprocessRequestMetricCollector
from inference_perf.utils.custom_tokenizer import CustomTokenizer

# Match the start method used in production (main.py) so the tokenizer and
# datagen objects are inherited rather than pickled into worker processes.
if sys.platform == "darwin":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

MOCK_LATENCY = 0.05


class _DummyHFTokenizer:
    """Minimal HuggingFace-shaped tokenizer backed by space-delimited integers."""

    vocab_size = 1000
    all_special_ids = [1, 2, 3]

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        return " ".join(str(t) for t in tokens)

    def encode(self, text: str) -> list[int]:
        try:
            return [int(t) for t in text.split()]
        except ValueError:
            return list(range(10, 10010))


class _DummyCustomTokenizer(CustomTokenizer):
    def __init__(self) -> None:
        pass

    def get_tokenizer(self) -> Any:
        return _DummyHFTokenizer()

    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        # Each space-separated integer is one token.
        return len(text.split()) if text else 0


class _CapacityLimitedMockClient(MockModelServerClient):
    """Mock server with a fixed number of service slots: past it, requests queue.

    Models a backend that saturates: throughput flattens at capacity/latency
    while the client keeps more requests in flight than the server is serving.
    The semaphore is created lazily so it belongs to the worker process's event
    loop instead of being pickled across the process boundary.
    """

    def __init__(self, *args: Any, capacity: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.capacity = capacity
        self._slots: Optional[asyncio.Semaphore] = None

    async def process_request(
        self, data: Any, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        if self._slots is None:
            self._slots = asyncio.Semaphore(self.capacity)
        async with self._slots:
            await super().process_request(data, stage_id, scheduled_time, lora_adapter)


class _HangingMockClient(MockModelServerClient):
    """Mock server that measures the first two rungs and then never answers."""

    async def process_request(
        self, data: Any, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        if stage_id <= -3:
            # Only a cancellation ends this, which is what the drain cancel does.
            await asyncio.sleep(3600.0)
            return
        await super().process_request(data, stage_id, scheduled_time, lora_adapter)


class _FailingRungMockClient(MockModelServerClient):
    """Mock server that times out every request from a given rung onward."""

    def __init__(self, *args: Any, fail_from_stage: int = -3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fail_from_stage = fail_from_stage

    async def process_request(
        self, data: Any, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        if stage_id <= self.fail_from_stage:
            # The mock records a failed lifecycle metric for this timeout, which
            # is what an all-error rung looks like from the probe's side.
            self.timeout = 1e-9
        await super().process_request(data, stage_id, scheduled_time, lora_adapter)


def make_probe_load_generator(
    probe_config: ProbeConfig, worker_max_concurrency: int = 16
) -> tuple[LoadGenerator, MultiprocessRequestMetricCollector]:
    """Build a one-worker load generator whose sweep is measured by the probe."""
    api_config = APIConfig(type=APIType.Completion, streaming=False)
    data_config = DataConfig(
        type=DataGenType.Random,
        input_distribution=Distribution(min=10, max=10, mean=10.0, std_dev=0.0, total_count=2000),
        output_distribution=Distribution(min=5, max=5, mean=5.0, std_dev=0.0, total_count=2000),
    )
    collector = MultiprocessRequestMetricCollector()
    load_config = LoadConfig(
        type=LoadType.CONSTANT,
        num_workers=1,
        worker_max_concurrency=worker_max_concurrency,
        interval=0,
        stages=[],
        sweep=SweepConfig(
            type=StageGenType.LINEAR,
            num_stages=2,
            stage_duration=1,
            probe=probe_config,
        ),
        # The probe drains its own tail before teardown, so the stage grace only
        # bounds genuinely stuck work; keep it short so these tests stay fast.
        stage_teardown_grace_seconds=1.0,
        base_seed=42,
    )
    datagen = RandomDataGenerator(api_config, data_config, _DummyCustomTokenizer())
    return LoadGenerator(datagen, load_config, request_metric_collector=collector), collector


class TestProbePreprocess(unittest.IsolatedAsyncioTestCase):
    async def test_probe_measures_generates_and_runs_stages(self) -> None:
        num_stages = 2
        api_config = APIConfig(type=APIType.Completion, streaming=False)
        data_config = DataConfig(
            type=DataGenType.Random,
            input_distribution=Distribution(min=10, max=10, mean=10.0, std_dev=0.0, total_count=2000),
            output_distribution=Distribution(min=5, max=5, mean=5.0, std_dev=0.0, total_count=2000),
        )
        datagen = RandomDataGenerator(api_config, data_config, _DummyCustomTokenizer())

        collector = MultiprocessRequestMetricCollector()
        client = MockModelServerClient(collector, api_config, mock_latency=MOCK_LATENCY)

        load_config = LoadConfig(
            type=LoadType.CONSTANT,
            num_workers=1,
            worker_max_concurrency=16,
            interval=0,
            stages=[],
            sweep=SweepConfig(
                type=StageGenType.LINEAR,
                num_stages=num_stages,
                stage_duration=1,
                probe=ProbeConfig(rung_duration=1.0, settle_duration=0.25, start_concurrency=1, max_concurrency=4),
            ),
            base_seed=42,
        )
        load_gen = LoadGenerator(datagen, load_config, request_metric_collector=collector)

        wall_before = time.time()
        async with collector.start():
            # A broken barrier pairing deadlocks instead of failing, so bound
            # the whole run.
            await asyncio.wait_for(load_gen.mp_run(client), timeout=120)
        await load_gen.stop()

        # The ladder never saturates against a fixed-latency backend, so it
        # climbs to the cap: rungs at N = 1, 2, 4.
        probe_result = load_gen.probe_result
        assert probe_result is not None
        self.assertEqual([r.concurrency for r in probe_result.rungs], [1, 2, 4])
        for rung in probe_result.rungs:
            ideal = rung.concurrency / MOCK_LATENCY
            self.assertGreater(rung.throughput, 0.5 * ideal)
            self.assertLessEqual(rung.throughput, 1.5 * ideal)
            self.assertLess(rung.littles_law_residual, 0.5)
        # A ladder stopped by its ceiling is censored: r_sat is a lower bound.
        self.assertIs(probe_result.stop_reason, ProbeStopReason.MAX_CONCURRENCY)
        self.assertTrue(probe_result.censored)
        self.assertTrue(probe_result.usable_concurrencies)
        self.assertTrue(set(probe_result.usable_concurrencies) <= {r.concurrency for r in probe_result.rungs})

        # The measured window starts when the stages do, not when calibration
        # did: the probe's own runtime must not count toward the run duration.
        assert load_gen.load_start_time is not None
        self.assertGreaterEqual(load_gen.load_start_time - wall_before, 3.0)

        r_sat = probe_result.constant("r_sat")
        self.assertGreaterEqual(r_sat.ci.low, probe_result.rungs[0].throughput)

        # Generated stages are spaced up to r_sat and actually ran.
        stage_rates = [s.rate for s in load_gen.stages if isinstance(s, StandardLoadStage)]
        self.assertEqual(len(stage_rates), num_stages)
        self.assertAlmostEqual(max(stage_rates), round(r_sat.value, 2), places=2)
        for stage_id in range(num_stages):
            self.assertEqual(load_gen.stage_runtime_info[stage_id].status.name, "COMPLETED")

        # Probe traffic is confined to negative stage ids (dropped from
        # reports) and no probe rung leaks into stage_runtime_info.
        metrics = collector.get_metrics()
        probe_metrics = [m for m in metrics if m.stage_id is not None and m.stage_id < 0]
        self.assertGreater(len(probe_metrics), 0)
        self.assertEqual({m.stage_id for m in probe_metrics}, {-1, -2, -3})
        self.assertTrue(all(stage_id >= 0 for stage_id in load_gen.stage_runtime_info))


class TestProbeRungOutcomes(unittest.IsolatedAsyncioTestCase):
    async def test_overloaded_rungs_are_dropped_from_the_fit(self) -> None:
        """A backend that saturates at `capacity` slots queues the excess, so
        rungs above it cannot hold their concurrency. They must be reported as
        unusable and kept out of r_sat instead of becoming clean points."""
        capacity = 4
        probe_config = ProbeConfig(rung_duration=1.0, settle_duration=0.25, start_concurrency=1, max_concurrency=16)
        load_gen, collector = make_probe_load_generator(probe_config, worker_max_concurrency=32)
        client = _CapacityLimitedMockClient(
            collector, APIConfig(type=APIType.Completion, streaming=False), mock_latency=MOCK_LATENCY, capacity=capacity
        )

        async with collector.start():
            await asyncio.wait_for(load_gen.mp_run(client), timeout=180)
        await load_gen.stop()

        probe_result = load_gen.probe_result
        self.assertIsNotNone(probe_result)
        assert probe_result is not None
        # Queueing is not the generator's fault, so the ladder must never end
        # with the generator blamed for the server's ceiling. Whether it stops
        # on a plateau or at the cap depends on the noise of these 1s rungs.
        self.assertIsNot(probe_result.stop_reason, ProbeStopReason.CLIENT_BOUND)
        over_capacity = [rung for rung in probe_result.rungs if rung.concurrency > capacity]
        self.assertTrue(over_capacity)
        for rung in over_capacity:
            self.assertGreater(rung.littles_law_residual, 0.2)
            self.assertNotIn(rung.concurrency, probe_result.usable_concurrencies)
        for rung in probe_result.rungs:
            if rung.concurrency <= capacity:
                self.assertIn(rung.concurrency, probe_result.usable_concurrencies)
        # The measured rungs sit at the server's ceiling, not below it.
        self.assertGreaterEqual(probe_result.constant("r_sat").value, 0.7 * capacity / MOCK_LATENCY)

    async def test_all_requests_failing_leaves_nothing_to_fit(self) -> None:
        """When every request of every rung times out there is no capacity to
        report: the probe must fail loudly instead of estimating from errors."""
        probe_config = ProbeConfig(rung_duration=1.0, settle_duration=0.25, start_concurrency=1, max_concurrency=4)
        load_gen, collector = make_probe_load_generator(probe_config)
        client = MockModelServerClient(collector, APIConfig(type=APIType.Completion, streaming=False), timeout=0.001)

        with self.assertLogs("inference_perf.loadgen.load_generator", level="ERROR") as logs:
            async with collector.start():
                await asyncio.wait_for(load_gen.mp_run(client), timeout=120)
        await load_gen.stop()

        self.assertIsNone(load_gen.probe_result)
        self.assertEqual(load_gen.stages, [])
        self.assertTrue(any("could not estimate capacity" in line for line in logs.output), logs.output)

    async def test_a_rung_that_measures_nothing_ends_the_ladder_not_the_run(self) -> None:
        """A rung where every request fails has no throughput to report, but the
        rungs below it did measure the server: the ladder stops there and r_sat
        comes from those, labelled as a lower bound and naming the bad rung."""
        probe_config = ProbeConfig(rung_duration=1.0, settle_duration=0.25, start_concurrency=1, max_concurrency=8)
        load_gen, collector = make_probe_load_generator(probe_config)
        client = _FailingRungMockClient(
            collector, APIConfig(type=APIType.Completion, streaming=False), mock_latency=MOCK_LATENCY
        )

        with self.assertLogs("inference_perf.loadgen.load_generator", level="WARNING") as logs:
            async with collector.start():
                await asyncio.wait_for(load_gen.mp_run(client), timeout=120)
        await load_gen.stop()

        probe_result = load_gen.probe_result
        self.assertIsNotNone(probe_result)
        assert probe_result is not None
        self.assertIs(probe_result.stop_reason, ProbeStopReason.MEASUREMENT_FAILED)
        self.assertTrue(probe_result.censored)
        self.assertEqual(probe_result.failed_concurrency, 4)
        # Only the rungs that were actually measured appear in the curve, and
        # they are still fitted rather than discarded with the bad rung.
        self.assertEqual([rung.concurrency for rung in probe_result.rungs], [1, 2])
        self.assertEqual(probe_result.usable_concurrencies, (1, 2))
        self.assertGreater(probe_result.constant("r_sat").value, 0.0)
        self.assertTrue(any("produced no usable measurement" in line for line in logs.output), logs.output)
        # The sweep is still configured from the surviving measurements.
        self.assertTrue(load_gen.stages)

    async def test_hung_server_drain_cancels_and_ends_the_ladder(self) -> None:
        """A rung whose tail never drains is bounded by cancelling it; that rung
        then has no completions to measure, so it ends the ladder instead of
        being reported as capacity."""
        probe_config = ProbeConfig(rung_duration=1.0, settle_duration=0.25, start_concurrency=1, max_concurrency=8)
        load_gen, collector = make_probe_load_generator(probe_config)
        client = _HangingMockClient(collector, APIConfig(type=APIType.Completion, streaming=False), mock_latency=MOCK_LATENCY)

        # The production drain floor is a minute; shorten it so the cancel path
        # is exercised in seconds rather than stalling the suite.
        with patch("inference_perf.loadgen.load_generator._PROBE_DRAIN_FLOOR_SEC", 1.0):
            with self.assertLogs("inference_perf.loadgen.load_generator", level="WARNING") as logs:
                async with collector.start():
                    await asyncio.wait_for(load_gen.mp_run(client), timeout=180)
        await load_gen.stop()

        self.assertTrue(any("cancelling" in line for line in logs.output), logs.output)
        probe_result = load_gen.probe_result
        self.assertIsNotNone(probe_result)
        assert probe_result is not None
        self.assertIs(probe_result.stop_reason, ProbeStopReason.MEASUREMENT_FAILED)
        self.assertTrue(probe_result.censored)
        self.assertEqual(probe_result.failed_concurrency, 4)
        self.assertEqual([rung.concurrency for rung in probe_result.rungs], [1, 2])


if __name__ == "__main__":
    unittest.main()
