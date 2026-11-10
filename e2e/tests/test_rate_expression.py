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
"""Does a rate expression shape the traffic a real run sends?

The unit tests check the schedule a rate expression produces. Nothing there
shows that a config with ``rate: "<expression>"`` loads through the CLI, that
the stage dispatches the count the expression integrates to, or that the
requests arrive when the expression says they should.

The run here points the CLI at the absorber, which records when every request
arrived, and reads the count and the shape back off that recording.
"""

from typing import Any, Dict, List

from utils.absorber import AbsorberServer
from utils.benchmark import run_benchmark_minimal
from utils.net import get_free_port
from utils.testdata import extract_tarball

MODEL_PATH = str(extract_tarball("e2e/testdata/models/google_gemma-3-270m.tar.gz"))

# 5 requests per second for the first 4 seconds, then 30 per second for the
# last 4: 20 slow requests and 120 fast ones.
RATE_EXPRESSION = "Piecewise((5, t < 4), (30, True))"
DURATION = 8
EXPECTED_REQUESTS = 5 * 4 + 30 * 4

# The stage start is not visible from the server, so the shape is read off the
# arrivals alone: the pace of the first 20 against the pace of the last 100.
# The step makes the late pace 6 times the early one and a flat rate makes
# them equal, so 2.5 separates the two with room for the timer's random spacing.
EARLY_REQUESTS = 20
LATE_REQUESTS = 100
MIN_PACE_RATIO = 2.5


# One constant-type stage whose rate is the step above, for 8 seconds, on one
# worker with mock prompts. Only base_url is filled in, since the port belongs
# to the test.
def _config(base_url: str) -> Dict[str, Any]:
    return {
        "api": {"type": "completion"},
        "data": {"type": "mock"},
        "load": {"type": "constant", "stages": [{"rate": RATE_EXPRESSION, "duration": DURATION}], "num_workers": 1},
        "server": {"type": "vllm", "model_name": MODEL_PATH, "base_url": base_url},
        "tokenizer": {"pretrained_model_name_or_path": MODEL_PATH},
        "report": {"request_lifecycle": {"summary": True, "per_stage": False, "per_request": False}},
    }


# Requests per second across a run of arrivals, e.g. arrivals at 1.0, 1.5 and
# 2.0 are 2 gaps in 1 second, so 2.0.
def _pace(arrivals: List[float]) -> float:
    return (len(arrivals) - 1) / (arrivals[-1] - arrivals[0])


# Runs the config above against the absorber. Exactly 140 requests arrive, and
# the last 100 arrive at more than 2.5 times the pace of the first 20.
async def test_rate_expression_sets_the_count_and_the_shape() -> None:
    async with AbsorberServer(port=get_free_port(), model=MODEL_PATH) as absorber:
        result = await run_benchmark_minimal(_config(absorber.base_url), timeout_sec=180)

    assert result.success, f"run failed (rc={result.return_code}):\n{result.stdout[-4000:]}"
    arrivals = sorted(request.arrival_s for request in absorber.requests)
    assert len(arrivals) == EXPECTED_REQUESTS

    early = _pace(arrivals[:EARLY_REQUESTS])
    late = _pace(arrivals[-LATE_REQUESTS:])
    assert late >= MIN_PACE_RATIO * early, f"late pace {late:.1f}/s is not well above early pace {early:.1f}/s"
