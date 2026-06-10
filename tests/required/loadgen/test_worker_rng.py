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
import ast
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
    StandardLoadStage,
)
from inference_perf.datagen.random_datagen import RandomDataGenerator
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


class _DummyHFTokenizer:
    """Minimal HuggingFace-shaped tokenizer backed by space-delimited integers."""

    vocab_size = 1000
    all_special_ids = [1, 2, 3]

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        return " ".join(str(t) for t in tokens)


class _DummyCustomTokenizer(CustomTokenizer):
    def __init__(self) -> None:
        pass

    def get_tokenizer(self) -> Any:
        return _DummyHFTokenizer()

    def count_tokens(self, text: str) -> int:
        # Each space-separated integer is one token.
        return len(text.split()) if text else 0


class TestWorkerRNGUniqueness(unittest.IsolatedAsyncioTestCase):
    """Integration test: multi-worker LoadGenerator must produce unique prompts per request.

    Regression test for the bug where every worker inherited the same
    np.random.Generator state from the parent process, causing bitwise-duplicate
    prompts whenever std_dev=0 (all input lengths identical).
    """

    async def test_multiworker_prompts_unique_with_zero_stdev(self) -> None:
        """All prompts must be distinct even when std_dev=0 forces identical input lengths."""
        num_requests = 6
        num_workers = 2

        api_config = APIConfig(type=APIType.Completion, streaming=False)
        data_config = DataConfig(
            type=DataGenType.Random,
            input_distribution=Distribution(min=10, max=10, mean=10.0, std_dev=0.0, total_count=num_requests),
            output_distribution=Distribution(min=5, max=5, mean=5.0, std_dev=0.0, total_count=num_requests),
        )
        datagen = RandomDataGenerator(api_config, data_config, _DummyCustomTokenizer())

        collector = MultiprocessRequestMetricCollector()
        client = MockModelServerClient(collector, api_config, mock_latency=0)

        load_config = LoadConfig(
            type=LoadType.CONSTANT,
            num_workers=num_workers,
            worker_max_concurrency=10,
            stages=[StandardLoadStage(rate=num_requests, duration=1)],
            base_seed=42,
        )
        load_gen = LoadGenerator(datagen, load_config)

        async with collector.start():
            await load_gen.mp_run(client)

        metrics = collector.get_metrics()
        self.assertEqual(len(metrics), num_requests, f"Expected {num_requests} completed requests")

        prompts = [ast.literal_eval(m.request_data)["prompt"] for m in metrics]
        self.assertEqual(
            len(set(prompts)),
            num_requests,
            f"Expected {num_requests} unique prompts across {num_workers} workers, got duplicates:\n"
            + "\n".join(f"  [{i}] {p!r}" for i, p in enumerate(prompts)),
        )


if __name__ == "__main__":
    unittest.main()
