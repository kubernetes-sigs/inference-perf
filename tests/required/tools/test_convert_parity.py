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
"""The parity case pairs are the converter's acceptance oracle (#755).

Converting each case's checked-in peer args must reproduce the checked-in
``inference-perf.yaml``, modulo the fields the harness owns. Conversion is a
pure function, so this runs in the unit tier even though the fixtures live
under ``e2e/``; until the parity harness lands on main the discovery is
empty and the tests skip.

Harness-owned fields (fixture values the peer args cannot produce):

- ``load.num_workers``: the config default is the CPU count, machine
  dependent, so the fixtures pin it for the live run.
- ``load.base_seed``: defaults to the current time; the converter emits the
  peer seed while a fixture without one gets a fresh default at every load.
- ``report``: report shape is harness policy, not a workload property.
- ``data.*.total_count``: the runtime raises it to the request count when
  unset, so the converter leaves it unset.
- ``load.type`` between ``constant`` and ``poisson`` when ``expected.yaml``
  declares per-tool ``arrival`` spacing: vllm bench cannot produce even
  spacing at any finite rate, so the fixture pair intentionally differs in
  arrival while the offered average rate matches. The converter emits the
  arrival the args actually mean; the load-shape comparison in the harness
  is tolerance-based per declared arrival.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import yaml

from inference_perf.config import read_config
from inference_perf.tools.convert import convert_vllm_bench
from inference_perf.tools.convert.versions import VLLM_BENCH_PINNED

REPO_ROOT = Path(__file__).resolve().parents[3]
PARITY_DIR = REPO_ROOT / "e2e" / "tests" / "parity"
CASES_DIR = PARITY_DIR / "cases"

BASE_URL = "http://127.0.0.1:8000"
MODEL = "google/gemma-3-270m"
RESULT = "result.json"

# The tail e2e/utils/vllm_bench.py appends to every case's args.
HARNESS_TAIL = [
    "--base-url",
    BASE_URL,
    "--model",
    MODEL,
    "--tokenizer",
    MODEL,
    "--save-result",
    "--result-filename",
    RESULT,
]

HARNESS_OWNED = (
    ("load", "num_workers"),
    ("load", "base_seed"),
    ("report",),
    ("data", "input_distribution", "total_count"),
    ("data", "output_distribution", "total_count"),
)


def _discover_cases() -> List[Path]:
    if not CASES_DIR.is_dir():
        return []
    return sorted(path.parent for path in CASES_DIR.glob("*/vllm-bench.args"))


def _read_args(path: Path) -> List[str]:
    args: List[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        args.append(stripped.split("#", 1)[0].strip())
    return args


def _prune(tree: Dict[str, Any], path: Tuple[str, ...]) -> None:
    node: Any = tree
    for key in path[:-1]:
        if not isinstance(node, dict) or key not in node:
            return
        node = node[key]
    if isinstance(node, dict):
        node.pop(path[-1], None)


def _declared_arrivals(case_dir: Path) -> Optional[Dict[str, str]]:
    expected_path = case_dir / "expected.yaml"
    if not expected_path.is_file():
        return None
    expected = yaml.safe_load(expected_path.read_text()) or {}
    tools = expected.get("tools") or {}
    arrivals: Dict[str, str] = {}
    for name, spec in tools.items():
        if isinstance(spec, dict) and isinstance(spec.get("arrival"), str):
            arrivals[str(name)] = str(spec["arrival"])
    return arrivals or None


@pytest.mark.parametrize("case_dir", _discover_cases(), ids=lambda path: path.name if isinstance(path, Path) else str(path))
def test_convert_reproduces_fixture(case_dir: Path) -> None:
    argv = _read_args(case_dir / "vllm-bench.args") + HARNESS_TAIL
    conversion = convert_vllm_bench(argv, VLLM_BENCH_PINNED)
    assert conversion.refusals == [], conversion.refusals
    assert conversion.config is not None

    want_config = read_config(str(case_dir / "inference-perf.yaml"))
    got = conversion.config.model_dump(mode="json")
    want = want_config.model_dump(mode="json")

    for path in HARNESS_OWNED:
        _prune(got, path)
        _prune(want, path)

    arrivals = _declared_arrivals(case_dir)
    if arrivals and {got["load"]["type"], want["load"]["type"]} <= set(arrivals.values()) | {"constant", "poisson"}:
        if got["load"]["type"] != want["load"]["type"]:
            assert got["load"]["type"] in ("constant", "poisson")
            assert want["load"]["type"] in ("constant", "poisson")
            got["load"]["type"] = want["load"]["type"]

    assert got == want


def test_no_cases_yet_is_visible() -> None:
    # Until the parity harness lands, the discovery above is empty; this
    # placeholder documents that the oracle is pending rather than passing.
    if _discover_cases():
        pytest.skip("parity cases present; the parametrized oracle runs")


def test_pin_matches_parity_harness() -> None:
    harness = REPO_ROOT / "e2e" / "utils" / "vllm_bench.py"
    if not harness.is_file():
        pytest.skip("parity harness not in tree yet")
    for line in harness.read_text().splitlines():
        if line.startswith("VLLM_PINNED_REF"):
            pinned = line.split("=", 1)[1].strip().strip("\"'")
            assert pinned == VLLM_BENCH_PINNED, (
                "the converter's verified vllm version and the parity harness pin have drifted apart:"
                f" converter {VLLM_BENCH_PINNED}, harness {pinned}. Re-verify the flag tables against the"
                " new pin before bumping either."
            )
            return
    pytest.fail("VLLM_PINNED_REF not found in e2e/utils/vllm_bench.py")
