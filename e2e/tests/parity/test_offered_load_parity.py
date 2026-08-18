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
"""Do inference-perf and `vllm bench serve` send the same traffic when configured for the same workload?

Each subdirectory of ``cases/`` holds one workload written down twice (an
inference-perf config and a vllm bench args file) plus ``expected.yaml``, the
numbers both must hit. See the README next to this file.

Each tool is pointed at the absorber (``e2e/utils/absorber.py``), a fake server
that records every request it receives. That recording is what gets checked:
how many requests arrived, how many prompt tokens and ``max_tokens`` each had,
the ``stream`` / ``ignore_eos`` flags, how fast they arrived, and how many were
in flight at once. The numbers the tools themselves print are never read here.

Rate and concurrency are worked out from the absorber's arrival/finish times by
the same helpers the load-shape accuracy test uses (``e2e/utils/load_shape.py``,
#633): the same sweep line for "how many in flight", and the same
``rate_tolerance(n, arrival)`` for "how far off the configured rate is still
fine". So there is one definition of each in the e2e tier, and the tolerance
comes from the request count and the tool's spacing pattern, not from a number
in a case file.

Two kinds of test per case: each tool against ``expected.yaml`` (the vllm side
skips when no vllm is installed, see ``e2e/utils/vllm_bench.py``), and the two
tools directly against each other. Each tool is run once per case and the
result is cached, so the number of tests reading it does not matter.
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
from utils.load_shape import (
    Segment,
    assert_delivered_concurrency,
    inflight_segments,
    max_inflight,
    mean_inflight,
    observed_send_rate,
    plateau_window,
    rate_tolerance,
)
from utils.net import get_free_port
from utils.testdata import extract_tarball
from utils.vllm_bench import VllmUnavailable, ensure_vllm_bench_bin, run_vllm_bench, warn_if_pin_stale

MODEL_NAME = "google/gemma-3-270m"
TOKENIZER_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"

PARITY_DIR = Path(__file__).resolve().parent
CASE_DIRS = sorted(p for p in (PARITY_DIR / "cases").iterdir() if p.is_dir())
TOOLS = ("inference-perf", "vllm-bench")

# Run this whole file on one pytest-xdist worker. The per-tool results are cached
# in this process, and the timing checks below should not share a CPU with
# unrelated tests running at the same time.
pytestmark = pytest.mark.xdist_group(name="tool-parity")


# Guard: at least one case directory exists. If cases/ were empty, every
# parametrized test below would silently produce zero tests, and this file
# would pass while checking nothing.
def test_cases_are_committed() -> None:
    assert CASE_DIRS, f"no parity cases committed under {PARITY_DIR / 'cases'}"


# Everything the checks below need to know about one tool's run, computed from
# the absorber's recording after dropping the tool's declared warmup requests.
# `lifecycle` is the list of {start_time, end_time} pairs (arrival at the
# absorber, reply finished) that e2e/utils/load_shape.py reads; the rate and
# in-flight numbers below are its functions applied to that list, nothing else.
@dataclass(frozen=True)
class OfferedLoad:
    """What one tool actually sent, after dropping declared warmup requests."""

    n: int
    prompt_token_lens: List[int]
    max_tokens: List[Optional[int]]
    stream_flags: Set[bool]
    ignore_eos_flags: Set[bool]
    lifecycle: List[Dict[str, float]]

    # Average arrival rate: request count over the time from first to last
    # arrival. 600 requests whose first and last arrivals are 15s apart = 40/s.
    @property
    def realized_rate(self) -> float:
        return observed_send_rate(self.lifecycle)[1]

    # In-flight count over time as a step function (load_shape's sweep line).
    @property
    def segments(self) -> List[Segment]:
        return inflight_segments(self.lifecycle)

    # Highest number of requests in flight at any moment.
    @property
    def peak_in_flight(self) -> int:
        return max_inflight(self.segments)

    # Time-weighted average in flight over the steady-state window for a
    # configured concurrency of `level` (from the level-th arrival to the last
    # arrival, so ramp-up and drain are excluded). At level=8 with the cap
    # honoured, this sits just under 8.0.
    def plateau_mean_in_flight(self, level: int) -> float:
        return mean_inflight(self.segments, plateau_window(self.lifecycle, level))


# Loads the bundled gemma-3-270m tokenizer once. Used to count prompt tokens
# the same way for both tools.
@lru_cache(maxsize=1)
def _tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(extract_tarball(TOKENIZER_TARBALL)))


# Prompt length of one recorded request, in tokens. If the tool sent token ids
# directly, that is just the list length; if it sent text, re-tokenize it
# (without adding BOS/EOS). Fails if the prompt is neither.
def _prompt_tokens(request: AbsorbedRequest) -> int:
    ids = request.prompt_token_ids
    if ids is not None:
        return len(ids)
    text = request.prompt_text
    assert text is not None, f"cannot interpret prompt as text or token ids: {request.body.get('prompt')!r:.100}"
    return len(_tokenizer().encode(text, add_special_tokens=False))


# Reads cases/<name>/expected.yaml into a dict.
def _expected(case_dir: Path) -> Dict[str, Any]:
    loaded = yaml.safe_load((case_dir / "expected.yaml").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


# Turns the absorber's raw recording into an OfferedLoad. Sorts by arrival time,
# drops the first `trim` requests (declared warmup traffic), then summarizes the
# rest. 601 recorded requests with trim=1 gives n=600, with the rate measured
# from the 2nd arrival to the 601st.
def _profile(requests: List[AbsorbedRequest], trim: int) -> OfferedLoad:
    assert len(requests) > trim, f"tool sent {len(requests)} requests, all trimmed as leading extras ({trim})"
    kept = sorted(requests, key=lambda r: r.arrival_s)[trim:]
    return OfferedLoad(
        n=len(kept),
        prompt_token_lens=[_prompt_tokens(r) for r in kept],
        max_tokens=[r.max_tokens for r in kept],
        stream_flags={r.stream for r in kept},
        ignore_eos_flags={r.ignore_eos for r in kept},
        lifecycle=[r.lifecycle_entry() for r in kept],
    )


# The per-tool block of expected.yaml (tools.<tool>), or {} if there is none.
def _tool_settings(expected: Dict[str, Any], tool: str) -> Dict[str, Any]:
    settings = expected.get("tools", {}).get(tool, {})
    assert isinstance(settings, dict)
    return settings


# How far a tool's measured rate may sit from the configured one, from
# load_shape.rate_tolerance(n, arrival). `arrival` is the tool's spacing pattern
# declared in expected.yaml under tools.<tool>.arrival ("constant" for evenly
# spaced, "poisson" for random). At n=600: 5% for constant, about 16% for
# poisson. Fails with a pointer to the case file if the pattern is missing.
def _rate_tol(expected: Dict[str, Any], tool: str, n: int, case_dir: Path) -> float:
    arrival = _tool_settings(expected, tool).get("arrival")
    assert isinstance(arrival, str), (
        f"{case_dir / 'expected.yaml'}: load.mode is rate, so tools.{tool}.arrival must say how {tool} "
        f"spaces its requests (constant or poisson); the tolerance is derived from it"
    )
    return rate_tolerance(n, arrival)


# Runs inference-perf with the case's inference-perf.yaml, pointed at the
# absorber. Only server.base_url and the tokenizer path are replaced (see
# README); every other setting is used as written. Fails if the run itself
# fails.
async def _drive_inference_perf(case_dir: Path, base_url: str) -> None:
    config = yaml.safe_load((case_dir / "inference-perf.yaml").read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    config["server"]["base_url"] = base_url
    config["tokenizer"] = {"pretrained_model_name_or_path": str(extract_tarball(TOKENIZER_TARBALL))}
    result = await run_benchmark_minimal(config)
    assert result.success, f"inference-perf run failed (rc={result.return_code}):\n{result.stdout[-4000:]}"


# Runs `vllm bench serve` with the case's vllm-bench.args, pointed at the
# absorber. Blank lines and # comments in the args file are skipped; the
# harness adds --base-url/--model/--tokenizer/--save-result/--result-filename
# at the end. Fails if the run itself fails.
async def _drive_vllm_bench(case_dir: Path, base_url: str, vllm_bin: str) -> None:
    lines = (case_dir / "vllm-bench.args").read_text(encoding="utf-8").splitlines()
    args = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    args += ["--base-url", base_url, "--model", MODEL_NAME, "--tokenizer", str(extract_tarball(TOKENIZER_TARBALL))]
    args += ["--save-result", "--result-filename", "vllm_bench_result.json"]
    warn_if_pin_stale()
    result = await run_vllm_bench(args, vllm_bin=vllm_bin)
    assert result.success, f"vllm bench run failed (rc={result.return_code}):\n{result.stdout[-4000:]}"


# Cache of (case name, tool) -> OfferedLoad, so each tool runs once per case.
_PROFILES: Dict[Tuple[str, str], OfferedLoad] = {}


# The one place a tool actually gets run. Starts a fresh absorber on a free port
# with the pacing from expected.yaml (default 40ms first chunk, 5ms between
# chunks), runs the tool against it, then summarizes the recording with the
# tool's leading_extra_requests dropped. Skips (not fails) the vllm side when
# no vllm is available. Cached, so a second call for the same case+tool is free.
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

    trim = int(_tool_settings(expected, tool).get("leading_extra_requests", 0))
    _PROFILES[key] = _profile(absorber.requests, trim)
    return _PROFILES[key]


# Every prompt length must be within target*rel_tol of target (at least +-1).
# target=128, rel_tol=0.15 accepts 109..147 tokens. The message lists how many
# were outside and the min/mean/max, and points at the length settings.
def _assert_prompt_lens(lens: List[int], target: int, rel_tol: float, who: str) -> None:
    bound = max(1, int(target * rel_tol))
    off = [n for n in lens if abs(n - target) > bound]
    assert not off, (
        f"{who}: {len(off)}/{len(lens)} prompts outside {target}±{bound} tokens "
        f"(min={min(lens)}, mean={statistics.mean(lens):.1f}, max={max(lens)}); "
        f"check the prompt-length settings (input_distribution vs --random-input-len/--random-range-ratio)"
    )


# One tool against expected.yaml. For case a_fixed_rate that means: exactly 600
# requests (after dropping warmup), every max_tokens == 64, every request has
# stream=true and ignore_eos=true, every prompt is 128 tokens +-15%, and the
# average arrival rate is 40/s within rate_tolerance(600, arrival): +-5% for
# inference-perf (even spacing), +-16% for vllm bench (random spacing). For a
# concurrency case, load_shape.assert_delivered_concurrency applies: never more
# than the configured number in flight, and on average within half a slot of it
# over the steady-state window. Runs once per (case, tool); the vllm side skips
# when no vllm is installed.
@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("case_dir", CASE_DIRS, ids=lambda p: p.name)
async def test_offered_load_matches_expected(case_dir: Path, tool: str) -> None:
    expected = _expected(case_dir)
    offered = await _offered_load(case_dir, tool)
    workload, load = expected["workload"], expected["load"]

    assert offered.n == workload["num_requests"], (
        f"{tool} offered {offered.n} requests, expected {workload['num_requests']} "
        f"(if the extra ones are warmup traffic, declare them in tools.{tool}.leading_extra_requests)"
    )
    assert set(offered.max_tokens) == {workload["max_tokens"]}, (
        f"{tool} max_tokens values {sorted(set(offered.max_tokens), key=str)} != {workload['max_tokens']} "
        f"(check the output-length settings; for vllm, check what --random-range-ratio means at this pin)"
    )
    assert offered.stream_flags == {workload["stream"]}, f"{tool} stream flags {offered.stream_flags}"
    assert offered.ignore_eos_flags == {workload["ignore_eos"]}, f"{tool} ignore_eos flags {offered.ignore_eos_flags}"
    _assert_prompt_lens(offered.prompt_token_lens, workload["prompt_tokens"], workload["prompt_tokens_rel_tol"], tool)

    if load["mode"] == "rate":
        rate, tol = float(load["rate"]), _rate_tol(expected, tool, offered.n, case_dir)
        assert offered.realized_rate == pytest.approx(rate, rel=tol), (
            f"{tool} realized arrival rate {offered.realized_rate:.2f}/s vs configured {rate}/s "
            f"(±{tol:.0%} for {offered.n} requests); check what the rate setting means for this tool, "
            f"and whether a worker cap is throttling it"
        )
    elif load["mode"] == "concurrency":
        level = int(load["concurrency"])
        try:
            assert_delivered_concurrency(offered.lifecycle, level)
        except AssertionError as e:
            pytest.fail(f"{tool}: {e}; check the concurrency settings (concurrency_level vs --max-concurrency)")
    else:
        pytest.fail(f"unknown load.mode {load['mode']!r} in {case_dir / 'expected.yaml'}")


# The two tools directly against each other, same quantities as above but with
# no reference numbers involved: same request count, same multiset of
# max_tokens, same stream/ignore_eos flags, mean prompt length within
# prompt_tokens_rel_tol, and by load mode either average rates within the
# looser of the two tools' rate tolerances (the evenly spaced tool adds almost
# no wobble of its own, so the randomly spaced tool's tolerance covers the
# difference), or the same peak in flight and steady-state averages within half
# a slot of each other. Skips when the vllm side is unavailable.
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
        f"(the two configs describe different workloads; check the prompt-length settings)"
    )

    if load["mode"] == "rate":
        tol = max(_rate_tol(expected, "inference-perf", ip.n, case_dir), _rate_tol(expected, "vllm-bench", vb.n, case_dir))
        assert ip.realized_rate == pytest.approx(vb.realized_rate, rel=tol), (
            f"realized arrival rates differ: inference-perf {ip.realized_rate:.2f}/s "
            f"vs vllm-bench {vb.realized_rate:.2f}/s (±{tol:.0%}; the average should agree even though "
            f"the spacing between requests differs on purpose: even vs random)"
        )
    elif load["mode"] == "concurrency":
        level = int(load["concurrency"])
        assert ip.peak_in_flight == vb.peak_in_flight, (
            f"peak in-flight differs: inference-perf {ip.peak_in_flight} vs vllm-bench {vb.peak_in_flight}"
        )
        ip_mean, vb_mean = ip.plateau_mean_in_flight(level), vb.plateau_mean_in_flight(level)
        assert abs(ip_mean - vb_mean) <= 0.5, (
            f"steady-state in-flight differs: inference-perf {ip_mean:.3f} vs vllm-bench {vb_mean:.3f} "
            f"(more than half a slot apart at configured concurrency {level})"
        )
