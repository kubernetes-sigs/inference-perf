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
"""Offered-load parity between inference-perf and `vllm bench serve`.

Every subdirectory of ``cases/`` describes one workload twice (an
inference-perf config and a vllm bench arg file) plus the invariants both
must satisfy (``expected.yaml``); see the README next to this file for the
drop-in contract. Each tool is run against the absorber
(``e2e/utils/absorber.py``), and the absorber's record of what actually
arrived is the oracle: request counts, per-request prompt token lengths and
``max_tokens``, sampling flags, realized arrival rate, and peak concurrency.

Two tests per case: each tool against ``expected.yaml`` (the vllm side skips
when no vllm is available, see ``e2e/utils/vllm_bench.py``), and the two
tools against each other. Tool runs are cached per case, so a case costs one
run per tool regardless of how many tests read it.
"""

import statistics
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest
import yaml

from utils.absorber import AbsorbedRequest, AbsorberServer
from utils.benchmark import run_benchmark_minimal
from utils.net import get_free_port
from utils.testdata import extract_tarball
from utils.vllm_bench import VllmUnavailable, ensure_vllm_bench_bin, run_vllm_bench, warn_if_pin_stale

MODEL_NAME = "google/gemma-3-270m"
TOKENIZER_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"

PARITY_DIR = Path(__file__).resolve().parent
CASE_DIRS = sorted(p for p in (PARITY_DIR / "cases").iterdir() if p.is_dir())
TOOLS = ("inference-perf", "vllm-bench")

# One xdist worker for the whole module: profiles are cached in-process, and
# absorber timing assertions should not compete with sibling tests for CPU.
pytestmark = pytest.mark.xdist_group(name="tool-parity")


def test_cases_are_committed() -> None:
    # An empty cases/ directory would silently deselect every test below, and
    # a gate that silently skips gates nothing.
    assert CASE_DIRS, f"no parity cases committed under {PARITY_DIR / 'cases'}"


@dataclass(frozen=True)
class OfferedLoad:
    """What one tool actually offered the absorber, after trimming declared warmup."""

    n: int
    prompt_token_lens: List[int]
    max_tokens: List[Optional[int]]
    stream_flags: Set[bool]
    ignore_eos_flags: Set[bool]
    max_in_flight: int
    duration_s: float

    @property
    def realized_rate(self) -> float:
        if self.n < 2 or self.duration_s <= 0:
            return 0.0
        return (self.n - 1) / self.duration_s


@lru_cache(maxsize=1)
def _tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(extract_tarball(TOKENIZER_TARBALL)))


def _prompt_tokens(request: AbsorbedRequest) -> int:
    ids = request.prompt_token_ids
    if ids is not None:
        return len(ids)
    text = request.prompt_text
    assert text is not None, f"cannot interpret prompt as text or token ids: {request.body.get('prompt')!r:.100}"
    return len(_tokenizer().encode(text, add_special_tokens=False))


def _expected(case_dir: Path) -> Dict[str, Any]:
    loaded = yaml.safe_load((case_dir / "expected.yaml").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _profile(requests: List[AbsorbedRequest], trim: int) -> OfferedLoad:
    assert len(requests) > trim, f"tool sent {len(requests)} requests, all trimmed as leading extras ({trim})"
    kept = sorted(requests, key=lambda r: r.arrival_s)[trim:]
    return OfferedLoad(
        n=len(kept),
        prompt_token_lens=[_prompt_tokens(r) for r in kept],
        max_tokens=[r.max_tokens for r in kept],
        stream_flags={r.stream for r in kept},
        ignore_eos_flags={r.ignore_eos for r in kept},
        max_in_flight=max(r.in_flight_at_arrival for r in kept),
        duration_s=kept[-1].arrival_s - kept[0].arrival_s,
    )


async def _drive_inference_perf(case_dir: Path, base_url: str) -> None:
    config = yaml.safe_load((case_dir / "inference-perf.yaml").read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    # The harness owns endpoint wiring (see README); everything else is verbatim.
    config["server"]["base_url"] = base_url
    config["tokenizer"] = {"pretrained_model_name_or_path": str(extract_tarball(TOKENIZER_TARBALL))}
    result = await run_benchmark_minimal(config)
    assert result.success, f"inference-perf run failed (rc={result.return_code}):\n{result.stdout[-4000:]}"


async def _drive_vllm_bench(case_dir: Path, base_url: str, vllm_bin: str) -> None:
    lines = (case_dir / "vllm-bench.args").read_text(encoding="utf-8").splitlines()
    args = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    args += ["--base-url", base_url, "--model", MODEL_NAME, "--tokenizer", str(extract_tarball(TOKENIZER_TARBALL))]
    args += ["--save-result", "--result-filename", "vllm_bench_result.json"]
    warn_if_pin_stale()
    result = await run_vllm_bench(args, vllm_bin=vllm_bin)
    assert result.success, f"vllm bench run failed (rc={result.return_code}):\n{result.stdout[-4000:]}"


_PROFILES: Dict[Tuple[str, str], OfferedLoad] = {}


async def _offered_load(case_dir: Path, tool: str) -> OfferedLoad:
    key = (case_dir.name, tool)
    if key in _PROFILES:
        return _PROFILES[key]

    vllm_bin: Optional[str] = None
    if tool == "vllm-bench":
        try:
            vllm_bin = ensure_vllm_bench_bin()
        except VllmUnavailable as e:
            pytest.skip(str(e))

    expected = _expected(case_dir)
    pacing = expected.get("absorber", {})
    absorber = AbsorberServer(
        port=get_free_port(),
        model=MODEL_NAME,
        ttft_s=pacing.get("ttft_ms", 40) / 1000.0,
        itl_s=pacing.get("itl_ms", 5) / 1000.0,
    )
    async with absorber:
        if tool == "inference-perf":
            await _drive_inference_perf(case_dir, absorber.base_url)
        else:
            assert vllm_bin is not None
            await _drive_vllm_bench(case_dir, absorber.base_url, vllm_bin)

    trim = int(expected.get("tools", {}).get(tool, {}).get("leading_extra_requests", 0))
    _PROFILES[key] = _profile(absorber.requests, trim)
    return _PROFILES[key]


def _assert_prompt_lens(lens: List[int], target: int, rel_tol: float, who: str) -> None:
    bound = max(1, int(target * rel_tol))
    off = [n for n in lens if abs(n - target) > bound]
    assert not off, (
        f"{who}: {len(off)}/{len(lens)} prompts outside {target}±{bound} tokens "
        f"(min={min(lens)}, mean={statistics.mean(lens):.1f}, max={max(lens)}); "
        f"check the length knobs (input_distribution vs --random-input-len/--random-range-ratio)"
    )


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("case_dir", CASE_DIRS, ids=lambda p: p.name)
async def test_offered_load_matches_expected(case_dir: Path, tool: str) -> None:
    expected = _expected(case_dir)
    offered = await _offered_load(case_dir, tool)
    workload, load = expected["workload"], expected["load"]

    assert offered.n == workload["num_requests"], (
        f"{tool} offered {offered.n} requests, expected {workload['num_requests']} "
        f"(warmup traffic is declared via tools.{tool}.leading_extra_requests)"
    )
    assert set(offered.max_tokens) == {workload["max_tokens"]}, (
        f"{tool} max_tokens values {sorted(set(offered.max_tokens), key=str)} != {workload['max_tokens']} "
        f"(check the output-length knobs, and --random-range-ratio semantics for vllm)"
    )
    assert offered.stream_flags == {workload["stream"]}, f"{tool} stream flags {offered.stream_flags}"
    assert offered.ignore_eos_flags == {workload["ignore_eos"]}, f"{tool} ignore_eos flags {offered.ignore_eos_flags}"
    _assert_prompt_lens(offered.prompt_token_lens, workload["prompt_tokens"], workload["prompt_tokens_rel_tol"], tool)

    if load["mode"] == "rate":
        rate, tol = float(load["rate"]), float(load["rate_rel_tol"])
        assert offered.realized_rate == pytest.approx(rate, rel=tol), (
            f"{tool} realized arrival rate {offered.realized_rate:.2f}/s vs configured {rate}/s "
            f"(±{tol:.0%}); check rate semantics and worker caps"
        )
    elif load["mode"] == "concurrency":
        level = int(load["concurrency"])
        assert offered.max_in_flight == level, (
            f"{tool} peak in-flight {offered.max_in_flight} != configured concurrency {level}; "
            f"check closed-loop semantics (concurrency_level vs --max-concurrency)"
        )
    else:
        pytest.fail(f"unknown load.mode {load['mode']!r} in {case_dir / 'expected.yaml'}")


@pytest.mark.parametrize("case_dir", CASE_DIRS, ids=lambda p: p.name)
async def test_tools_offer_the_same_workload(case_dir: Path) -> None:
    expected = _expected(case_dir)
    ip = await _offered_load(case_dir, "inference-perf")
    vb = await _offered_load(case_dir, "vllm-bench")
    workload, load = expected["workload"], expected["load"]

    assert ip.n == vb.n, f"request counts differ: inference-perf {ip.n} vs vllm-bench {vb.n}"
    assert sorted(ip.max_tokens, key=str) == sorted(vb.max_tokens, key=str), (
        f"max_tokens multisets differ: inference-perf {sorted(set(ip.max_tokens), key=str)} "
        f"vs vllm-bench {sorted(set(vb.max_tokens), key=str)}"
    )
    assert ip.stream_flags == vb.stream_flags and ip.ignore_eos_flags == vb.ignore_eos_flags, (
        f"sampling flags differ: stream {ip.stream_flags} vs {vb.stream_flags}, "
        f"ignore_eos {ip.ignore_eos_flags} vs {vb.ignore_eos_flags}"
    )

    ip_mean, vb_mean = statistics.mean(ip.prompt_token_lens), statistics.mean(vb.prompt_token_lens)
    rel_tol = float(workload["prompt_tokens_rel_tol"])
    assert ip_mean == pytest.approx(vb_mean, rel=rel_tol), (
        f"mean prompt tokens differ: inference-perf {ip_mean:.1f} vs vllm-bench {vb_mean:.1f} "
        f"(the two configs describe different workloads; check the length knobs)"
    )

    if load["mode"] == "rate":
        tol = float(load["rate_rel_tol"])
        assert ip.realized_rate == pytest.approx(vb.realized_rate, rel=tol), (
            f"realized arrival rates differ: inference-perf {ip.realized_rate:.2f}/s "
            f"vs vllm-bench {vb.realized_rate:.2f}/s (mean rate should agree even though "
            f"arrival shape legitimately differs: constant vs Poisson)"
        )
    elif load["mode"] == "concurrency":
        assert ip.max_in_flight == vb.max_in_flight, (
            f"peak in-flight differs: inference-perf {ip.max_in_flight} vs vllm-bench {vb.max_in_flight}"
        )
