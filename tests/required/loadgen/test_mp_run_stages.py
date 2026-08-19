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
"""Multi-stage mp_run integration test with real worker processes.

Locks the stage-boundary protocol invariant: every stage end pairs one
main-side stage_barrier.wait() with one arrival from each worker, so a
multi-stage run completes without deadlock and every stage's requests are
accounted. Any stage-driving code path that runs a stage outside the
mp_run stage loop (as the sweep pre-pass does) breaks this pairing and
hangs the run — asyncio.wait_for turns that hang into a test failure.
"""

import asyncio
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
    StandardLoadStage,
)
from inference_perf.datagen.synthetic.random_datagen import RandomDataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.metrics.request_collector import MultiprocessRequestMetricCollector
from inference_perf.utils.custom_tokenizer import CustomTokenizer


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


class TestMpRunMultiStage(unittest.IsolatedAsyncioTestCase):
    async def test_two_stages_complete_with_paired_barriers(self) -> None:
        requests_per_stage = 3
        num_stages = 2

        api_config = APIConfig(type=APIType.Completion, streaming=False)
        data_config = DataConfig(
            type=DataGenType.Random,
            input_distribution=Distribution(min=10, max=10, mean=10.0, std_dev=0.0, total_count=requests_per_stage),
            output_distribution=Distribution(min=5, max=5, mean=5.0, std_dev=0.0, total_count=requests_per_stage),
        )
        datagen = RandomDataGenerator(api_config, data_config, _DummyCustomTokenizer())

        collector = MultiprocessRequestMetricCollector()
        client = MockModelServerClient(collector, api_config, mock_latency=0)

        load_config = LoadConfig(
            type=LoadType.CONSTANT,
            num_workers=1,
            worker_max_concurrency=10,
            interval=0,
            stages=[StandardLoadStage(rate=requests_per_stage, duration=1) for _ in range(num_stages)],
            base_seed=42,
        )
        load_gen = LoadGenerator(datagen, load_config)

        async with collector.start():
            # A broken barrier pairing deadlocks instead of failing, so bound
            # the whole run.
            await asyncio.wait_for(load_gen.mp_run(client), timeout=120)
        await load_gen.stop()

        metrics = collector.get_metrics()
        self.assertEqual(len(metrics), requests_per_stage * num_stages, "every stage's requests must be accounted")
        for stage_id in range(num_stages):
            stage_metrics = [m for m in metrics if m.stage_id == stage_id]
            self.assertEqual(len(stage_metrics), requests_per_stage, f"stage {stage_id} must process its own requests")
            self.assertEqual(load_gen.stage_runtime_info[stage_id].status.name, "COMPLETED")


if __name__ == "__main__":
    unittest.main()
