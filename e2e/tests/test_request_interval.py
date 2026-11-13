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
"""Does request_interval space the requests a real run sends?

The unit tests check the schedule a request_interval produces. Nothing there
shows that a config with ``request_interval`` and no ``load.type`` loads
through the CLI and puts requests on the wire with the gaps it describes.

Each run here points the CLI at the absorber, which records when every request
arrived, and reads the count and the gaps back off that recording. Wall-clock
gaps over a real socket are never exact, so the gap checks are on the middle
gap and on how much the gaps vary, not on each gap.
"""

import statistics
from typing import Any, Dict, List

from utils.absorber import AbsorberServer
from utils.benchmark import run_benchmark_minimal
from utils.net import get_free_port
from utils.testdata import extract_tarball

MODEL_PATH = str(extract_tarball("e2e/testdata/models/google_gemma-3-270m.tar.gz"))

# Spread of the gaps over their mean. Evenly spaced requests give about 0 and
# a Poisson process gives about 1, so 0.3 and 0.6 leave each side clear of the other.
MAX_EVEN_SPREAD = 0.3
MIN_RANDOM_SPREAD = 0.6


# One stage with the given request_interval and duration, mock prompts, one
# worker, and no load.type, which request_interval replaces. Only base_url is
# filled in, since the port belongs to the test.
def _config(base_url: str, request_interval: Any, duration: int) -> Dict[str, Any]:
    return {
        "api": {"type": "completion"},
        "data": {"type": "mock"},
        "load": {"stages": [{"request_interval": request_interval, "duration": duration}], "num_workers": 1},
        "server": {"type": "vllm", "model_name": MODEL_PATH, "base_url": base_url},
        "tokenizer": {"pretrained_model_name_or_path": MODEL_PATH},
        "report": {"request_lifecycle": {"summary": True, "per_stage": False, "per_request": False}},
    }


# Runs one stage against the absorber and returns the seconds between
# consecutive arrivals, e.g. arrivals at 0.0, 0.1 and 0.3 give [0.1, 0.2].
async def _arrival_gaps(request_interval: Any, duration: int) -> List[float]:
    async with AbsorberServer(port=get_free_port(), model=MODEL_PATH) as absorber:
        result = await run_benchmark_minimal(_config(absorber.base_url, request_interval, duration), timeout_sec=180)

    assert result.success, f"run failed (rc={result.return_code}):\n{result.stdout[-4000:]}"
    arrivals = sorted(request.arrival_s for request in absorber.requests)
    return [later - earlier for earlier, later in zip(arrivals, arrivals[1:], strict=False)]


# Spread of the gaps over their mean: gaps of 0.1, 0.1 and 0.1 give 0, and
# gaps drawn from an exponential give about 1.
def _relative_spread(gaps: List[float]) -> float:
    return statistics.pstdev(gaps) / statistics.fmean(gaps)


# request_interval: 0.1 for 3 seconds. Exactly 30 requests arrive (29 gaps),
# the middle gap is within 20ms of 0.1s, and the gaps barely vary.
async def test_a_numeric_request_interval_sends_evenly_spaced_requests() -> None:
    gaps = await _arrival_gaps(0.1, 3)

    assert len(gaps) + 1 == 30
    assert abs(statistics.median(gaps) - 0.1) <= 0.02, f"middle gap {statistics.median(gaps):.3f}s is not near 0.1s"
    assert _relative_spread(gaps) < MAX_EVEN_SPREAD, f"gaps vary too much for even spacing: {sorted(gaps)}"


# request_interval: "Exponential(10)" for 6 seconds, a Poisson process at 10
# per second. About 60 requests arrive (30 to 90 allowed), the mean gap is
# near 0.1s, and the gaps vary the way random arrivals do.
async def test_an_exponential_request_interval_sends_randomly_spaced_requests() -> None:
    gaps = await _arrival_gaps("Exponential(10)", 6)

    assert 30 <= len(gaps) + 1 <= 90, f"{len(gaps) + 1} requests arrived, expected about 60"
    assert 0.06 <= statistics.fmean(gaps) <= 0.16, f"mean gap {statistics.fmean(gaps):.3f}s is not near 0.1s"
    assert _relative_spread(gaps) > MIN_RANDOM_SPREAD, f"gaps are too even for random arrivals: {sorted(gaps)}"
