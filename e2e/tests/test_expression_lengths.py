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
"""Do expression-valued lengths reach the wire as the lengths they describe?

The unit tests sample an expression and check the numbers. Nothing there shows
that a config with ``input_distribution: "<expression>"`` loads through the
CLI, survives the hand-off to the worker processes and ends up as the prompt
and ``max_tokens`` of a real request.

The run here points the CLI at the absorber, which records every request body,
and reads the lengths back off those bodies. Prompt lengths are counted with a
raw transformers tokenizer, not inference_perf's own, so a counting bug in the
product cannot cancel itself out.
"""

from typing import Any, Dict, List

from transformers import AutoTokenizer

from utils.absorber import AbsorbedRequest, AbsorberServer
from utils.benchmark import run_benchmark_minimal
from utils.net import get_free_port
from utils.testdata import extract_tarball

MODEL_PATH = str(extract_tarball("e2e/testdata/models/google_gemma-3-270m.tar.gz"))

RATE = 20
DURATION = 2
EXPECTED_REQUESTS = RATE * DURATION

# A clamped normal for the prompt and a flat range for the output, so each has
# a hard floor and ceiling the wire values can be held to.
INPUT_EXPRESSION = "Min(Max(Normal(60, 15), 30), 90)"
INPUT_BOUNDS = (30, 90)
OUTPUT_EXPRESSION = "Uniform(8, 24)"
OUTPUT_BOUNDS = (8, 24)

# 40 draws that all came out equal, or nearly so, mean the expression was
# collapsed to one number somewhere on the way. 40 draws over 17 output values
# or 61 input values give fewer than 5 distinct ones with negligible probability.
MIN_DISTINCT = 5


# 40 synthetic completion requests at 20 per second for 2 seconds, one worker,
# both length fields set to expression strings. Only base_url is filled in,
# since the port belongs to the test.
def _config(base_url: str) -> Dict[str, Any]:
    return {
        "api": {"type": "completion"},
        "data": {
            "type": "synthetic",
            "input_distribution": INPUT_EXPRESSION,
            "output_distribution": OUTPUT_EXPRESSION,
        },
        "load": {"type": "constant", "stages": [{"rate": RATE, "duration": DURATION}], "num_workers": 1},
        "server": {"type": "vllm", "model_name": MODEL_PATH, "base_url": base_url},
        "tokenizer": {"pretrained_model_name_or_path": MODEL_PATH},
        "report": {"request_lifecycle": {"summary": True, "per_stage": False, "per_request": False}},
    }


# Token count of each recorded prompt, e.g. three requests with 41, 77 and 58
# token prompts give [41, 77, 58]. Counted the way a server counts a prompt,
# with the tokenizer's leading special token, which is the length the config sets.
def _prompt_token_counts(requests: List[AbsorbedRequest]) -> List[int]:
    oracle = AutoTokenizer.from_pretrained(MODEL_PATH)
    counts = []
    for request in requests:
        text = request.prompt_text
        assert text is not None, f"request carried no text prompt: {sorted(request.body)}"
        counts.append(len(oracle.encode(text, add_special_tokens=True)))
    return counts


# Runs the config above against the absorber. All 40 requests arrive, every
# max_tokens is a whole number in [8, 24], every prompt is 30 to 90 tokens
# long, and both vary across requests (at least 5 distinct values each).
async def test_expression_lengths_reach_the_wire() -> None:
    async with AbsorberServer(port=get_free_port(), model=MODEL_PATH) as absorber:
        result = await run_benchmark_minimal(_config(absorber.base_url), timeout_sec=None)

    assert result.success, f"run failed (rc={result.return_code}):\n{result.stdout[-4000:]}"
    requests = absorber.requests
    assert len(requests) == EXPECTED_REQUESTS

    outputs = [request.max_tokens for request in requests]
    assert all(isinstance(value, int) for value in outputs), f"max_tokens missing or not a whole number: {outputs}"
    low, high = OUTPUT_BOUNDS
    assert all(low <= value <= high for value in outputs if value is not None), (
        f"max_tokens outside [{low}, {high}]: {outputs}"
    )
    assert len(set(outputs)) >= MIN_DISTINCT, f"max_tokens barely varies: {sorted(set(outputs))}"

    inputs = _prompt_token_counts(requests)
    low, high = INPUT_BOUNDS
    assert all(low <= value <= high for value in inputs), f"prompt length outside [{low}, {high}] tokens: {inputs}"
    assert len(set(inputs)) >= MIN_DISTINCT, f"prompt length barely varies: {sorted(set(inputs))}"
