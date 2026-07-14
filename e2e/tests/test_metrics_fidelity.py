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
End-to-end fidelity test for the reported request-lifecycle metrics.

llm-d-inference-sim is started with deterministic timing (a fixed
time-to-first-token and a fixed inter-token latency, std-dev 0) and streams
exactly one token per SSE chunk plus a trailing usage chunk. Combined with the
`random` datagen pinning max_tokens per request, every ground-truth quantity
(TTFT, per-token latency, tokens per response, request count) is set by this
test's own config, so the report can be checked against known values instead
of only asserting that a run succeeds.

Unlike the unit tests in tests/required/reportgen/, which hand-construct
StreamedResponseMetrics and verify the summarization math, this exercises the
full pipeline (HTTP client, SSE parsing, timestamp capture, summarization),
guarding the seam where bugs like #364 (tokens counted per chunk) and #564
(per-chunk BOS inflation) lived.

Tolerance philosophy: the sim sleeps at least the configured durations and
client/network overhead only adds on top, so measured latencies can only
exceed ground truth. Lower bounds are therefore strict, while upper bounds
leave generous slack for loaded CI runners executing e2e tests in parallel
under pytest-xdist.
"""

import pytest

from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.benchmark import run_benchmark_minimal
from utils.net import get_free_port
from utils.testdata import extract_tarball

TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"

# Upper-bound slack (see the module docstring; lower bounds are strict).
TTFT_SLACK_SEC = 0.25
PER_TOKEN_SLACK_SEC = 0.025
REQUEST_LATENCY_SLACK_SEC = 0.5
# Per-token metrics derive from timestamp deltas, where a late read of the
# first chunk can slightly shrink the measured span, so give the strict floor
# a small allowance instead.
PER_TOKEN_EPSILON_SEC = 0.005


def _exact_length_distribution(tokens: int) -> dict:
    """A degenerate length distribution: every request gets exactly `tokens`."""
    return {
        "min": tokens,
        "max": tokens,
        "mean": tokens,
        "std_dev": 0,
        # Sample pool size; only needs to exceed the number of requests issued.
        "total_count": 100,
    }


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
# Two deliberately contrasting timing profiles: a metric that is scaled or
# offset yet happens to pass one profile cannot also pass the other. Values
# must stay well above scheduling jitter and within the CI time budget.
@pytest.mark.parametrize(
    ("ttft_sec", "itl_sec", "output_tokens_per_request"),
    [
        pytest.param(0.5, 0.05, 25, id="ttft500ms_itl50ms_z25"),
        pytest.param(0.2, 0.1, 8, id="ttft200ms_itl100ms_z8"),
    ],
)
async def test_streaming_metrics_match_simulated_ground_truth(ttft_sec: float, itl_sec: float, output_tokens_per_request: int):
    model_path = extract_tarball(TEST_MODEL_TARBALL)

    # The sim waits TTFT before the first token, then ITL before each of the
    # remaining tokens, so each request's wall time has a hard server-side floor.
    request_latency_floor_sec = ttft_sec + (output_tokens_per_request - 1) * itl_sec

    load = {
        "type": "constant",
        "stages": [{"rate": 2, "duration": 5}],
        "num_workers": 2,
    }
    num_requests = sum(stage["rate"] * stage["duration"] for stage in load["stages"])

    async with LLMDInferenceSimRunner(
        TEST_MODEL_NAME,
        *("--time-to-first-token", str(int(ttft_sec * 1000))),
        *("--inter-token-latency", str(int(itl_sec * 1000))),
        # Deterministic sleeps: no jitter around the configured latencies.
        *("--time-to-first-token-std-dev", "0"),
        *("--inter-token-latency-std-dev", "0"),
        # Queueing would inflate TTFT beyond its slack; keep every in-flight
        # request scheduled immediately (default max-num-seqs is only 5).
        *("--max-num-seqs", "64"),
        *("--seed", "42"),
        port=get_free_port(),
    ) as sim:
        result = await run_benchmark_minimal(
            {
                "api": {"type": "completion", "streaming": True},
                # The degenerate output distribution pins the exact per-request
                # max_tokens in this config rather than relying on the client's
                # internal default.
                "data": {
                    "type": "random",
                    # Prompt length is arbitrary; prompt tokens are only asserted > 0.
                    "input_distribution": _exact_length_distribution(8),
                    "output_distribution": _exact_length_distribution(output_tokens_per_request),
                },
                "load": load,
                "server": {
                    "type": "vllm",
                    "model_name": TEST_MODEL_NAME,
                    "base_url": f"http://{sim.host}:{sim.port}",
                    "ignore_eos": True,
                },
                "tokenizer": {"pretrained_model_name_or_path": str(model_path)},
                "report": {
                    "request_lifecycle": {
                        "summary": True,
                        "per_stage": False,
                        "per_request": True,
                        # Normalize TPOT by the server's exact completion_tokens
                        # so its expected value is itl_sec independent of how the
                        # client tokenizer re-splits the sim's word chunks.
                        "use_server_output_tokens": True,
                    },
                },
            }
        )

    assert result.success, "Benchmark failed"
    assert result.reports, "No reports generated from benchmark"

    per_request = result.reports["per_request_lifecycle_metrics.json"]
    assert per_request and len(per_request) == num_requests, "Unexpected number of requests in report"

    summary = result.reports["summary_lifecycle_metrics.json"]
    successes = summary["successes"]
    assert summary["failures"]["count"] == 0
    assert successes["count"] == num_requests

    # Exact assertions: token counts are deterministic (ignore_eos plus a
    # pinned max_tokens makes the sim emit exactly Z tokens, and its usage
    # chunk reports them), so the whole per-request distribution collapses to Z.
    output_tokens = successes["output_tokens"]
    assert output_tokens["total"] == num_requests * output_tokens_per_request
    distribution_stats = {k: v for k, v in output_tokens.items() if k != "total"}
    assert set(distribution_stats.values()) == {float(output_tokens_per_request)}, distribution_stats
    # Input-side usage is plumbed through too. Its exact value is the sim's own
    # tokenization of the random prompts, which this test cannot predict.
    assert successes["prompt_tokens"]["total"] > 0

    latency = successes["latency"]

    # TTFT: first-chunk arrival relative to request start. The min is the
    # sharpest fidelity signal, as any mis-anchored timestamp (e.g. measured
    # from first-token parse instead of request start) breaks the floor.
    ttft = latency["time_to_first_token"]
    assert ttft["min"] >= ttft_sec, ttft
    assert ttft["mean"] <= ttft_sec + TTFT_SLACK_SEC, ttft

    # TPOT: (last - first token timestamp) / (server tokens - 1) == ITL.
    tpot = latency["time_per_output_token"]
    assert itl_sec - PER_TOKEN_EPSILON_SEC <= tpot["mean"] <= itl_sec + PER_TOKEN_SLACK_SEC, tpot

    # ITL: individual timestamp gaps. The mean is not asserted because chunks
    # the client tokenizer splits into several tokens pad zero-length gaps
    # into the series, dragging it below ground truth; the median stays on the
    # true cadence while most chunks re-tokenize to a single token (true for
    # the pinned gemma tokenizer over the sim's word bank).
    itl = latency["inter_token_latency"]
    assert itl_sec - PER_TOKEN_EPSILON_SEC <= itl["median"] <= itl_sec + PER_TOKEN_SLACK_SEC, itl

    # Request latency: hard floor of TTFT + (Z-1) * ITL enforced by the sim.
    request_latency = latency["request_latency"]
    assert request_latency["min"] >= request_latency_floor_sec, request_latency
    assert request_latency["mean"] <= request_latency_floor_sec + REQUEST_LATENCY_SLACK_SEC, request_latency

    # NTPOT: request latency normalized by the server token count.
    ntpot = latency["normalized_time_per_output_token"]
    assert ntpot["mean"] >= request_latency_floor_sec / output_tokens_per_request, ntpot
    assert ntpot["mean"] <= (request_latency_floor_sec + REQUEST_LATENCY_SLACK_SEC) / output_tokens_per_request, ntpot
