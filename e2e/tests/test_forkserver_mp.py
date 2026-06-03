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
"""End-to-end guard for the fork -> forkserver multiprocessing switch.

The load generator starts workers with a ``forkserver`` context instead of the
(Python 3.14 deprecated) ``fork`` default. Unlike fork, forkserver pickles the
worker payload and runs it in a fresh interpreter, so the entire production
payload -- the model server client, the datagen, and its CustomTokenizer -- must
be picklable. The unit suite only exercises this boundary with picklable
doubles; these tests drive the real ``inference-perf`` CLI through the
multiprocessing path (num_workers > 0) with a real tokenizer-backed datagen and
the mock server (so no real model is required).
"""

from typing import Any, Dict

import pytest

from utils.benchmark import run_benchmark_minimal

# A real, tiny tokenizer so the datagen carries a genuine CustomTokenizer across
# the forkserver boundary (rather than a mock that only survives fork).
_TOKENIZER = "gpt2"

# rate * duration -> total requests issued across the stage.
_RATE = 2
_DURATION = 5
_EXPECTED_REQUESTS = _RATE * _DURATION


def _config(datagen_type: str) -> Dict[str, Any]:
    distribution = {"min": 10, "max": 50, "mean": 30, "std_dev": 10}
    return {
        "data": {
            "type": datagen_type,
            "input_distribution": dict(distribution),
            "output_distribution": dict(distribution),
        },
        "load": {
            "type": "constant",
            "stages": [{"rate": _RATE, "duration": _DURATION}],
            # > 0 forces the mp_run path, which starts forkserver workers and
            # pickles the datagen + tokenizer into them.
            "num_workers": 2,
        },
        "api": {"type": "completion"},
        "server": {"type": "mock", "base_url": "http://0.0.0.0:8000"},
        "tokenizer": {"pretrained_model_name_or_path": _TOKENIZER},
        "report": {"request_lifecycle": {"summary": True, "per_stage": True, "per_request": True}},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("datagen_type", ["synthetic", "random"])
async def test_real_datagen_pickles_into_forkserver_workers(datagen_type: str) -> None:
    result = await run_benchmark_minimal(_config(datagen_type), timeout_sec=180)

    assert result.success, (
        f"Benchmark failed for datagen={datagen_type} "
        f"(rc={result.return_code}, timed_out={result.timed_out}).\n"
        f"If this is a PicklingError or a 'SemLock created in a fork context' "
        f"error, the worker payload is not forkserver-safe.\n{result.stdout}"
    )
    assert result.reports, "No reports generated from benchmark"

    summary = result.reports["summary_lifecycle_metrics.json"]
    assert summary["successes"]["count"] == _EXPECTED_REQUESTS, (
        f"expected {_EXPECTED_REQUESTS} successful requests, got {summary['successes']['count']}"
    )

    # The deprecated fork() warning must not reappear: workers now start from the
    # single-threaded forkserver, not a fork of this multi-threaded process.
    assert "use of fork() may lead to deadlocks" not in result.stdout, (
        f"fork() deprecation warning present for datagen={datagen_type}; "
        f"workers are not using the forkserver context.\n{result.stdout}"
    )
