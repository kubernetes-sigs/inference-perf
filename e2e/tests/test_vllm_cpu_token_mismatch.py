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
"""Client vs server token accounting against a real vLLM in CPU mode (#580).

The #410 mismatch detector only logged, so #564 stayed silent for months;
this test turns disagreement with a real server into a CI failure. Two
independent server-side ledgers must both agree with the client:

- in-band: per-request ``usage`` from the response, and the summary's
  ``token_count_mismatches`` (the detector's own output) must be zero
- out-of-band: the server's cumulative ``/metrics`` counters
  (``vllm:prompt_tokens_total`` / ``vllm:generation_tokens_total``) must
  grow by exactly the totals the client reports for the run

The /metrics deltas assume no other traffic reaches the server during the
run: fine for a spawned server, and for an external (shared) server only
when tests run sequentially, as the CPU-vLLM CI job does.
"""

import pytest

from utils.accuracy import (
    SUMMARY_REPORT,
    assert_output_token_accounting,
    assert_successful_run,
    response_metrics,
    server_completion_tokens,
)
from utils.benchmark import run_benchmark_minimal
from utils.net import get_free_port
from utils.vllm_server import VLLMServerRunner

RATE = 2
DURATION = 5
EXPECTED_REQUESTS = RATE * DURATION


def _counter_total(metrics_text: str, base_name: str) -> float:
    """Sum of all samples of a counter family, accepting the OpenMetrics
    ``_total``-suffixed sample name as well as the bare name."""
    total = 0.0
    found = False
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        sample = line.split("{")[0].split(" ")[0]
        if sample in (base_name, f"{base_name}_total"):
            total += float(line.rsplit(" ", 1)[1])
            found = True
    assert found, f"counter {base_name} not found in /metrics exposition"
    return total


@pytest.mark.asyncio
@pytest.mark.skipif(not VLLMServerRunner.is_available(), reason="no vLLM server or executable available")
async def test_token_counts_agree_with_server():
    async with VLLMServerRunner(port=get_free_port()) as server:
        before = await server.fetch_metrics()

        result = await run_benchmark_minimal(
            {
                "data": {"type": "mock"},
                "load": {
                    "type": "constant",
                    "stages": [{"rate": RATE, "duration": DURATION}],
                    "num_workers": 2,
                },
                "api": {"type": "completion", "streaming": True},
                "server": {
                    "type": "vllm",
                    "model_name": server.model,
                    "base_url": server.base_url,
                    "ignore_eos": True,
                },
                "tokenizer": {"pretrained_model_name_or_path": server.model},
                "report": {
                    "request_lifecycle": {
                        "summary": True,
                        "per_stage": True,
                        "per_request": True,
                    },
                },
            },
            timeout_sec=300,
        )

        after = await server.fetch_metrics()

    entries = assert_successful_run(result, EXPECTED_REQUESTS)

    # In-band ledger: every request's client count must equal the server's
    # usage, and the report's own mismatch detector must agree there was
    # nothing to flag. A mismatch here fails CI instead of logging (#580).
    for entry in entries:
        assert_output_token_accounting(entry, tolerance=0)
    summary = result.reports[SUMMARY_REPORT]["successes"]
    assert summary["token_count_mismatches"] == 0, f"{summary['token_count_mismatches']} requests flagged token mismatches"

    # Out-of-band ledger: the server's cumulative counters. Exact equality
    # holds because ignore_eos pins generation and nothing else talked to
    # the server between the two scrapes.
    generated = sum(server_completion_tokens(e) or 0 for e in entries)
    prompted = sum(e["info"]["request_metrics"]["text"]["input_tokens"] for e in entries)
    gen_delta = _counter_total(after, "vllm:generation_tokens") - _counter_total(before, "vllm:generation_tokens")
    prompt_delta = _counter_total(after, "vllm:prompt_tokens") - _counter_total(before, "vllm:prompt_tokens")
    assert gen_delta == generated, f"server generated {gen_delta} tokens, client accounted {generated}"
    assert prompt_delta == prompted, f"server saw {prompt_delta} prompt tokens, client accounted {prompted}"

    # The client-side output totals must line up with the same ledger.
    client_generated = sum(response_metrics(e)["output_tokens"] for e in entries)
    assert client_generated == generated, f"client output_len total {client_generated} != server usage total {generated}"
