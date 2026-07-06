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
Tool-parity end-to-end test: inference-perf vs `vllm bench serve`.

WHY THIS EXISTS
We observed inference-perf and vllm's serving benchmark report materially
different numbers when pointed at the *same* server. This test pins down that
discrepancy by running both tools against a single configurable mock
(llm-d-inference-sim) whose TTFT / ITL we set explicitly, across a table of
scenarios that each isolate one axis (load mode, sequence lengths, tokenizer).

The PRIMARY assertion is strict tool-vs-tool agreement within reasonable
bounds. As a SECONDARY sanity check, where it is meaningful, each tool's TPOT
is compared against the configured per-token latency (ground truth). TPOT is
used for ground truth rather than TTFT because TTFT absorbs queue wait under
load, whereas per-token spacing stays ~constant.

This doubles as the regression fixture for the perf-diff investigation: on
failure, the logged comparison table shows which metric diverged and by how
much.

REQUIREMENTS (cases auto-skip if unmet, like the other e2e tests):
  * llm-d-inference-sim on PATH (nix develop provides it).
  * a runnable vllm: either $VLLM_BENCH_BIN, or $VLLM_BENCH_PROVISION=1 (see
    utils/vllm_bench.py).
  * for non-bundled tokenizers: network to fetch them from HF (the case skips
    if offline). Bundle a tokenizer-only tarball under e2e/testdata/models and
    add it to BUNDLED_TOKENIZERS to make a tokenizer hermetic.
"""

import logging
import os
import socket
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import pytest

from utils.benchmark import run_benchmark_minimal
from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.scenario import (
    Scenario,
    compare,
    format_comparison_table,
    ground_truth,
    normalize_inference_perf,
    normalize_vllm_bench,
)
from utils.testdata import extract_tarball
from utils.vllm_bench import (
    VllmUnavailable,
    ensure_vllm_bench_bin,
    run_vllm_bench,
    warn_if_pin_stale,
)

logger = logging.getLogger(__name__)

# Server model is held constant (bundled gemma tokenizer) so the sim is always
# offline-reliable; only the *client* tokenizer varies across cases.
SERVER_MODEL = "google/gemma-3-270m"

# Tokenizers resolvable without network (tokenizer-only tarballs in testdata).
# Add entries here (and drop the tarball in) to make a tokenizer hermetic.
BUNDLED_TOKENIZERS: Dict[str, str] = {
    "google/gemma-3-270m": "e2e/testdata/models/google_gemma-3-270m.tar.gz",
}

# Default tool-vs-tool tolerances (relative). Strict, but mindful that neither
# tool recovers a configured latency exactly and ITL is the noisiest metric
# (per-chunk effects, cf. inference-perf #566). Tune as real deltas come in.
DEFAULT_TOL = {
    "median_ttft_ms": 0.20,
    "median_tpot_ms": 0.20,
    "median_itl_ms": 0.30,
    "output_throughput_tok_s": 0.20,
}
GROUND_TRUTH_TPOT_TOL = 0.40


@dataclass
class ParityCase:
    """A scenario plus how strictly to judge it."""

    scenario: Scenario
    # TPOT-vs-configured is only meaningful when the client tokenizer matches
    # what the server generated; with a mismatched tokenizer the client
    # re-tokenizes to a different count and TPOT drifts (which is exactly the
    # tokenizer-driven divergence we want tool-vs-tool to still agree on).
    check_tpot_ground_truth: bool = True
    tol: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TOL))


def _base(**overrides: object) -> Scenario:
    """Shared baseline; override per case."""
    defaults: Dict[str, object] = dict(
        model_name=SERVER_MODEL,
        num_prompts=40,
        input_len=128,
        output_len=64,
        load_mode="rate",
        request_rate=8.0,
        num_workers=32,
        ttft_ms=200.0,
        itl_ms=10.0,
    )
    defaults.update(overrides)
    return Scenario(**defaults)  # type: ignore[arg-type]


# The table. Each case isolates one axis; runtimes are kept bounded by trimming
# num_prompts / itl_ms for the long-output cases.
CASES = [
    # (a) fixed request rate (open loop)
    pytest.param(ParityCase(_base(load_mode="rate", request_rate=8.0)), id="a_fixed_rate"),
    # (b) fixed concurrency (closed loop). TTFT absorbs queue wait, so judge on
    # tool-vs-tool + TPOT ground truth only.
    pytest.param(
        ParityCase(_base(load_mode="concurrency", concurrency=16, num_prompts=48)),
        id="b_fixed_concurrency",
    ),
    # (c) 1k in / 1k out
    pytest.param(
        ParityCase(_base(input_len=1024, output_len=1024, itl_ms=2.0, request_rate=4.0, num_prompts=20)),
        id="c_1k_in_1k_out",
    ),
    # (d) 1k in / 4k out
    pytest.param(
        ParityCase(_base(input_len=1024, output_len=4096, itl_ms=1.0, request_rate=2.0, num_prompts=12)),
        id="d_1k_in_4k_out",
    ),
    # (e) multiple tokenizers. Gemma matches the server (ground truth holds);
    # gpt2 is a mismatched tokenizer (no BOS injection) so only tool-vs-tool is
    # asserted -- this is the case most likely to expose tokenizer-driven drift.
    pytest.param(ParityCase(_base(tokenizer="google/gemma-3-270m")), id="e_tokenizer_gemma"),
    pytest.param(
        ParityCase(_base(tokenizer="gpt2"), check_tpot_ground_truth=False),
        id="e_tokenizer_gpt2",
    ),
]


def _free_port() -> int:
    """Pick an ephemeral free port so parametrized cases never collide."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _resolve_tokenizer(name: str) -> str:
    """Return a local path (bundled) or HF id for the client tokenizer.

    Skips the case if a non-bundled tokenizer cannot be loaded (e.g. offline),
    so the suite degrades gracefully rather than failing on a network hiccup.
    """
    if name in BUNDLED_TOKENIZERS:
        return str(extract_tarball(BUNDLED_TOKENIZERS[name]))
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(name)
    except Exception as e:  # noqa: BLE001 - offline / gated / missing
        pytest.skip(f"tokenizer {name!r} unavailable (offline?): {e}")
    return name


@pytest.fixture(scope="module")
def vllm_bin() -> str:
    """Provision a runnable vllm or skip the whole module."""
    warn_if_pin_stale(check_upstream=False)
    try:
        return ensure_vllm_bench_bin()
    except VllmUnavailable as e:
        pytest.skip(f"vllm bench unavailable: {e}")


@pytest.mark.asyncio
@pytest.mark.skipif(
    not LLMDInferenceSimRunner.is_available(),
    reason="local environment missing llm-d-inference-sim",
)
@pytest.mark.parametrize("case", CASES)
async def test_inference_perf_vs_vllm_bench_parity(case: ParityCase, vllm_bin: str) -> None:
    scenario = case.scenario
    tokenizer = _resolve_tokenizer(scenario.tokenizer or scenario.model_name)
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"

    async with LLMDInferenceSimRunner(scenario.model_name, *scenario.sim_args(), port=port):
        # --- inference-perf ---
        ip_result = await run_benchmark_minimal(
            scenario.inference_perf_config(base_url=base_url, tokenizer_path=tokenizer),
        )
        assert ip_result.success, f"inference-perf run failed:\n{ip_result.stdout}"
        assert ip_result.reports, "inference-perf produced no reports"

        # --- vllm bench (same server, same workload, same tokenizer) ---
        with tempfile.TemporaryDirectory(prefix="vllm-bench-parity-") as td:
            result_file = str(Path(td) / "vllm_bench_result.json")
            vb_result = await run_vllm_bench(
                scenario.vllm_bench_args(base_url=base_url, result_path=result_file, tokenizer=tokenizer),
                vllm_bin=vllm_bin,
                work_dir=Path(td),
                result_filename=result_file,
            )
            assert vb_result.success, f"vllm bench run failed:\n{vb_result.stdout}"
            assert vb_result.result_json, "vllm bench produced no result JSON"
            vb = normalize_vllm_bench(vb_result.result_json)

    ip = normalize_inference_perf(ip_result.reports["summary_lifecycle_metrics.json"])
    gt = ground_truth(scenario)

    # --- Always emit the comparison artifact (the regression record). ---
    tables = {
        "inference-perf vs vllm-bench": compare(ip, vb),
        "ground-truth vs inference-perf": compare(gt, ip),
        "ground-truth vs vllm-bench": compare(gt, vb),
    }
    rendered = format_comparison_table(tables)
    logger.info("tool parity comparison:%s", rendered)
    artifact = Path(os.environ.get("PARITY_ARTIFACT_DIR", tempfile.gettempdir())) / "tool_parity.txt"
    with artifact.open("a") as f:
        f.write(rendered + "\n")

    # --- Primary gate: the two tools must agree within bounds. ---
    tool_table = tables["inference-perf vs vllm-bench"]
    failures = []
    for metric, tol in case.tol.items():
        rel = tool_table[metric]["rel_diff"]
        if rel is None:
            failures.append(f"{metric}: missing from one tool ({tool_table[metric]})")
        elif rel > tol:
            failures.append(f"{metric}: rel_diff {rel:.1%} > tol {tol:.0%} ({tool_table[metric]})")
    assert not failures, "inference-perf and vllm-bench disagree:\n" + "\n".join(failures) + rendered

    # --- Secondary: TPOT should track the configured per-token latency. ---
    # TPOT is load-invariant (unlike TTFT), so this holds even under
    # concurrency; it is skipped only when the client tokenizer is mismatched.
    if case.check_tpot_ground_truth:
        for name, tool in (("inference-perf", ip), ("vllm-bench", vb)):
            vals = compare(gt, tool, fields=("median_tpot_ms",))["median_tpot_ms"]
            rel = vals["rel_diff"]
            assert rel is not None and rel <= GROUND_TRUTH_TPOT_TOL, (
                f"{name} median_tpot_ms off from configured {scenario.itl_ms}ms by "
                f"{rel:.1%} (>{GROUND_TRUTH_TPOT_TOL:.0%}): {vals}{rendered}"
            )
