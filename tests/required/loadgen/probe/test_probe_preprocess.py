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
"""End-to-end probe preprocessing test with real worker processes.

Runs the closed-loop capacity probe against the mock client (fixed 50ms
latency, so closed-loop throughput is ideally X(N) = 20N and never
saturates) and then the generated stages, locking three invariants:

- Every probe rung's phase boundary pairs a main-side stage_barrier.wait()
  with one arrival per worker, so the run completes instead of deadlocking
  (asyncio.wait_for turns a pairing bug into a test failure).
- Rung measurements obey Little's law against the known mock latency.
- Probe traffic stays on negative stage ids, which the report generator
  drops, and the generated stages run to completion afterwards.
"""

import asyncio
import multiprocessing as mp
import sys
import unittest
from typing import Any

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


if __name__ == "__main__":
    unittest.main()
