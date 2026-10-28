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
"""
The runtime metrics carry the report's per-request values (#943).

The sim's token count and the client tokenizer's disagree on nearly every
response. With use_server_output_tokens off, the report divides by the client's
count, and the runtime metrics must too.

The endpoint stops as soon as the run's last result is collected, so a scrape
never sees the final state of the last stage. Stage 1 exists only to keep the
endpoint up: by then stage 0's series are final, and its sums are compared
exactly with the report's per-request values.
"""

import asyncio
from typing import Any, Dict, List, Optional

import aiohttp
import pytest
from prometheus_client.parser import text_string_to_metric_families

from utils.benchmark import BenchmarkResult, run_benchmark_minimal
from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.net import get_free_port
from utils.testdata import extract_tarball

TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"
STAGE_0_REQUESTS = 60
STAGE_0 = frozenset({("stage", "0")})

Samples = Dict[tuple[str, frozenset[tuple[str, str]]], float]


# Polls http://127.0.0.1:<port>/metrics every 50ms until the run ends.
# Returns the body of the last 200, the latest state any scrape saw.
async def _last_scrape(run: "asyncio.Task[BenchmarkResult]", port: int) -> Optional[str]:
    last = None
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1)) as session:
        while not run.done():
            try:
                async with session.get(f"http://127.0.0.1:{port}/metrics") as resp:
                    if resp.status == 200:
                        last = await resp.text()
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(0.05)
    return last


# Sample values by (sample name, frozenset of labels), e.g. ("x_sum", {("stage", "0")}) -> 1.5.
def _samples(body: str) -> Samples:
    return {
        (sample.name, frozenset(sample.labels.items())): sample.value
        for family in text_string_to_metric_families(body)
        for sample in family.samples
    }


# A fixed-length distribution, e.g. 64 -> min = max = mean = 64 with no spread.
def _fixed(n: int) -> Dict[str, int]:
    return {"min": n, "max": n, "mean": n, "std_dev": 0, "total_count": STAGE_0_REQUESTS}


# A stage-0 histogram's (count, sum) next to the same over the report's values, skipping Nones.
# E.g. a histogram of 2 observations summing 0.3 against report values [0.1, None, 0.2] -> ((2, 0.3), (2, 0.3)).
def _compare(samples: Samples, metric: str, values: List[Optional[float]]) -> tuple[tuple[float, float], tuple[int, float]]:
    present = [v for v in values if v is not None]
    runtime = (samples.get((f"{metric}_count", STAGE_0), 0.0), samples.get((f"{metric}_sum", STAGE_0), 0.0))
    return runtime, (len(present), sum(present))


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
# 60 streamed requests of 64 sim tokens in stage 0, nearly all counted differently by the sim and the client.
# Stage 0's latency, TTFT, TPOT, NTPOT and schedule delay histograms and its output token counter
# equal the report's per-request values summed, and its last completion falls after its start.
async def test_runtime_metrics_match_the_report() -> None:
    model_path = extract_tarball(TEST_MODEL_TARBALL)
    port = get_free_port()

    async with LLMDInferenceSimRunner(
        TEST_MODEL_NAME, "--time-to-first-token", "20", "--inter-token-latency", "2", port=get_free_port()
    ) as sim:
        run = asyncio.create_task(
            run_benchmark_minimal(
                {
                    "data": {"type": "random", "input_distribution": _fixed(32), "output_distribution": _fixed(64)},
                    "load": {
                        "type": "constant",
                        "stages": [{"rate": 10, "duration": STAGE_0_REQUESTS // 10}, {"rate": 1, "duration": 2}],
                        "interval": 3,
                        "num_workers": 4,
                    },
                    "api": {"type": "completion", "streaming": True},
                    "server": {
                        "type": "vllm",
                        "model_name": TEST_MODEL_NAME,
                        "base_url": f"http://{sim.host}:{sim.port}",
                        "ignore_eos": True,
                    },
                    "tokenizer": {"pretrained_model_name_or_path": str(model_path)},
                    "observability": {"metrics": {"port": port}},
                    "report": {
                        "request_lifecycle": {
                            "summary": True,
                            "per_stage": True,
                            "per_request": True,
                            "per_request_fields": {"computed_metrics": True},
                        },
                    },
                }
            )
        )
        body = await _last_scrape(run, port)
        result = await run

    assert result.success, f"Benchmark failed: {result.stdout}"
    assert body is not None, "the endpoint never answered"
    records: List[Dict[str, Any]] = [r for r in result.reports["per_request_lifecycle_metrics.json"] if r["stage_id"] == 0]
    assert len(records) == STAGE_0_REQUESTS
    computed = [r["computed_metrics"] for r in records]

    # Guards the premise: with matching counts, TPOT and the output token counter would agree on main too.
    mismatched = [
        r
        for r in records
        if r["info"]["response_metrics"]["server_usage"]["completion_tokens"] != r["computed_metrics"]["output_tokens"]
    ]
    assert len(mismatched) >= STAGE_0_REQUESTS // 2, (
        f"only {len(mismatched)} of {STAGE_0_REQUESTS} requests had differing counts"
    )

    samples = _samples(body)
    for metric, key in [
        ("inference_perf_request_latency_seconds", "request_latency"),
        ("inference_perf_time_to_first_token_seconds", "time_to_first_token"),
        ("inference_perf_time_per_output_token_seconds", "time_per_output_token"),
        ("inference_perf_normalized_time_per_output_token_seconds", "normalized_time_per_output_token"),
    ]:
        runtime, report = _compare(samples, metric, [c[key] for c in computed])
        assert runtime == pytest.approx(report, rel=1e-9), metric

    runtime, report = _compare(
        samples, "inference_perf_schedule_delay_seconds", [r["start_time"] - r["scheduled_time"] for r in records]
    )
    assert runtime == pytest.approx(report, rel=1e-9), "inference_perf_schedule_delay_seconds"

    assert samples.get(("inference_perf_output_tokens_total", STAGE_0)) == sum(c["output_tokens"] for c in computed)

    started = samples[("inference_perf_stage_start_timestamp_seconds", STAGE_0)]
    finished = samples.get(("inference_perf_last_completion_timestamp_seconds", STAGE_0))
    assert finished is not None and finished > started
