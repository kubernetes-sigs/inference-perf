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
"""Sim-backed worker-matrix smoke test (#632, #606 e2e tier).

One multi-stage run at a pinned worker count against llm-d-inference-sim, with
the token-accounting helpers from ``utils/accuracy.py`` attached.

Why it exists: the Integration matrix in
``tests/required/integration/test_worker_matrix.py`` drives the same worker
architecture against an in-process fake server. That is faster and more
controllable, but it is only worth trusting if a real server agrees. This test
is the cross-check on the two properties the fake is least able to vouch for,
because both are about the worker processes rather than about the HTTP
exchange:

- worker teardown: the CLI must exit 0 with every worker joined, rather than
  leaving a daemon child behind or hanging on the stage barrier at the end of a
  multi-stage run (#469)
- the liveness check: ``run_stage`` polls ``Worker.is_alive()`` and aborts the
  stage if a worker died. It must never fire on a healthy run, or the matrix
  is measuring an aborted stage and calling it a pass (#593)

Ground truth is the sim's own ``usage`` accounting: ``ignore_eos`` plus a
degenerate output distribution pins exactly ``OUTPUT_TOKENS`` generated tokens
per request, and the sim streams one token per chunk, so both are asserted
exactly. The condition (server timing, worker count, stage layout) is
configured; the oracle (the server's token counts) is not.

Client-side re-tokenization is deliberately NOT held to zero tolerance here.
The sim generates from a word bank and the pinned gemma tokenizer re-splits
some of those words, so the client count legitimately runs a token or two above
the server's (measured 0 to 2 over this fixture). Exact client/server agreement
needs a server whose text is built from known token ids, which is what #631's
golden sim and #627's real vLLM provide; asserting it here would be pinning the
tokenizer, not the worker matrix.

Retired once the Integration fake has proved out, per the #606 row.
"""

import pytest

from utils.accuracy import (
    assert_output_token_accounting,
    assert_streaming_bookkeeping,
    assert_successful_run,
    server_completion_tokens,
    ttft,
)
from utils.benchmark import run_benchmark_minimal
from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.net import get_free_port
from utils.testdata import extract_tarball

TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"

# Pinned, not inherited from cpu_count(). The default num_workers is
# max(1, cpu_count()), so an unpinned run tests whichever cell the runner
# happens to have; N > 1 is the cell that exercises work distribution across
# workers on top of the multiprocessing machinery itself.
PINNED_N = 4

RATE = 4
DURATION = 3
NUM_STAGES = 2
REQUESTS_PER_STAGE = RATE * DURATION
EXPECTED_REQUESTS = REQUESTS_PER_STAGE * NUM_STAGES

# Deterministic server timing, std-dev 0, so the sim sleeps at least these
# durations and TTFT has a hard floor.
TTFT_SEC = 0.2
ITL_SEC = 0.05
OUTPUT_TOKENS = 8

# Slack on the client's re-tokenization of the sim's word bank (see the module
# docstring). Loose on purpose: it is a sanity bound, not the token oracle.
CLIENT_RETOKENIZATION_SLACK = 3

# The liveness check in run_stage. If this appears, a worker died and the stage
# was cut short, so every count below would be measuring an aborted run.
WORKER_DEATH_LOG = "A worker process died unexpectedly!"


def _exact_length_distribution(tokens: int) -> dict:
    """A degenerate length distribution: every request gets exactly `tokens`."""
    return {"min": tokens, "max": tokens, "mean": tokens, "std_dev": 0, "total_count": 100}


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
async def test_multi_stage_run_at_pinned_worker_count():
    model_path = extract_tarball(TEST_MODEL_TARBALL)

    async with LLMDInferenceSimRunner(
        TEST_MODEL_NAME,
        *("--time-to-first-token", str(int(TTFT_SEC * 1000))),
        *("--inter-token-latency", str(int(ITL_SEC * 1000))),
        *("--time-to-first-token-std-dev", "0"),
        *("--inter-token-latency-std-dev", "0"),
        # PINNED_N workers each dispatch concurrently; the default
        # max-num-seqs of 5 would queue them and inflate TTFT past its floor.
        *("--max-num-seqs", "64"),
        *("--seed", "42"),
        port=get_free_port(),
    ) as sim:
        result = await run_benchmark_minimal(
            {
                "api": {"type": "completion", "streaming": True},
                "data": {
                    "type": "random",
                    "input_distribution": _exact_length_distribution(8),
                    "output_distribution": _exact_length_distribution(OUTPUT_TOKENS),
                },
                "load": {
                    "type": "constant",
                    "interval": 1,
                    "stages": [{"rate": RATE, "duration": DURATION} for _ in range(NUM_STAGES)],
                    "num_workers": PINNED_N,
                },
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
                        "per_stage": True,
                        "per_request": True,
                    },
                },
            },
            timeout_sec=240,
        )

    # Clean teardown: a run that leaves a worker wedged on the stage barrier
    # times out here rather than returning a non-zero code, and either way the
    # helper refuses the report.
    entries = assert_successful_run(result, EXPECTED_REQUESTS)
    assert WORKER_DEATH_LOG not in result.stdout, (
        f"the worker liveness check fired, so at least one stage was aborted:\n{result.stdout}"
    )

    # Both stages ran and both carried their own load. A stage-barrier
    # regression shows up here as a missing stage report or an empty one, well
    # before it shows up as a wrong aggregate.
    for stage_id in range(NUM_STAGES):
        stage_report_name = f"stage_{stage_id}_lifecycle_metrics.json"
        assert stage_report_name in result.reports, f"missing {stage_report_name} in {sorted(result.reports)}"
        stage_report = result.reports[stage_report_name]
        assert stage_report["successes"]["count"] == REQUESTS_PER_STAGE, (
            f"stage {stage_id} reported {stage_report['successes']['count']} successes, expected {REQUESTS_PER_STAGE}"
        )
        assert stage_report["failures"]["count"] == 0

    # Token accounting, per request, against the server's own usage counts.
    # PINNED_N workers each build their own tokenizer and their own HTTP
    # session, so a worker that came up misconfigured shows up as a
    # per-request break rather than as a shifted average.
    for entry in entries:
        # Exact: the server generated exactly OUTPUT_TOKENS, and streamed them
        # one per chunk, for every request from every worker.
        assert server_completion_tokens(entry) == OUTPUT_TOKENS
        assert_streaming_bookkeeping(entry, expected_chunks=OUTPUT_TOKENS)
        # Bounded: client re-tokenization drift, see the module docstring.
        assert_output_token_accounting(entry, expected=OUTPUT_TOKENS, tolerance=CLIENT_RETOKENIZATION_SLACK)
        # The streamed timestamps have to survive the trip back from the
        # worker: a dropped or defaulted first-chunk time reads as a TTFT
        # below the floor the sim enforces.
        assert ttft(entry) >= TTFT_SEC, f"TTFT {ttft(entry):.4f}s is below the sim's configured {TTFT_SEC}s"

    summary = result.reports["summary_lifecycle_metrics.json"]["successes"]
    assert summary["output_tokens"]["total"] == EXPECTED_REQUESTS * OUTPUT_TOKENS
