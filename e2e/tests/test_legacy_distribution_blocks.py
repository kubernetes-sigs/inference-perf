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
"""Does a legacy distribution block still run, and does the run say what to write instead?

A ``{type, min, max, mean, std_dev}`` block is compiled to an expression
string when the config loads. The unit tests check the compiled string. The
run here checks the two things a user with an old config sees: the run still
sends requests with lengths inside the block's ``[min, max]``, and the log
names the expression that replaces each block.

The CLI is pointed at the absorber, which records every request body. Prompt
lengths are counted with a raw transformers tokenizer, not inference_perf's
own, so a counting bug in the product cannot cancel itself out.
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

INPUT_BLOCK = {"type": "normal", "min": 30, "max": 90, "mean": 60, "std_dev": 15}
INPUT_REPLACEMENT = 'Replace it with `input_distribution: "floor(Min(Max(Normal(60, 15), 30), 90) + 1/2)"`'
OUTPUT_BLOCK = {"type": "uniform", "min": 8, "max": 24}
OUTPUT_REPLACEMENT = 'Replace it with `output_distribution: "floor(Uniform(8, 24) + 1/2)"`'

# 40 draws that all came out equal, or nearly so, mean the block was collapsed
# to one number on the way. 40 draws over 17 output values or 61 input values
# give fewer than 5 distinct ones with negligible probability.
MIN_DISTINCT = 5


# 40 synthetic completion requests at 20 per second for 2 seconds, one worker,
# both length fields written as legacy blocks. Only base_url is filled in,
# since the port belongs to the test.
def _config(base_url: str) -> Dict[str, Any]:
    return {
        "api": {"type": "completion"},
        "data": {"type": "synthetic", "input_distribution": INPUT_BLOCK, "output_distribution": OUTPUT_BLOCK},
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


# Runs the config above against the absorber. The log names the replacement
# expression for each block, all 40 requests arrive, every max_tokens is in
# [8, 24], every prompt is 30 to 90 tokens long, and both vary across requests.
async def test_a_legacy_block_runs_and_names_its_replacement() -> None:
    async with AbsorberServer(port=get_free_port(), model=MODEL_PATH) as absorber:
        result = await run_benchmark_minimal(_config(absorber.base_url), timeout_sec=180)

    assert result.success, f"run failed (rc={result.return_code}):\n{result.stdout[-4000:]}"
    assert INPUT_REPLACEMENT in result.stdout, "the log does not name the replacement for input_distribution"
    assert OUTPUT_REPLACEMENT in result.stdout, "the log does not name the replacement for output_distribution"

    requests = absorber.requests
    assert len(requests) == EXPECTED_REQUESTS

    outputs = [request.max_tokens for request in requests]
    low, high = OUTPUT_BLOCK["min"], OUTPUT_BLOCK["max"]
    assert all(value is not None and low <= value <= high for value in outputs), (
        f"max_tokens outside [{low}, {high}]: {outputs}"
    )
    assert len(set(outputs)) >= MIN_DISTINCT, f"max_tokens barely varies: {sorted(set(outputs), key=str)}"

    inputs = _prompt_token_counts(requests)
    low, high = INPUT_BLOCK["min"], INPUT_BLOCK["max"]
    assert all(low <= value <= high for value in inputs), f"prompt length outside [{low}, {high}] tokens: {inputs}"
    assert len(set(inputs)) >= MIN_DISTINCT, f"prompt length barely varies: {sorted(set(inputs))}"
