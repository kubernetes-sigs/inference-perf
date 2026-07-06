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
Single source of truth for a tool-parity benchmark scenario.

The whole point of the parity test is to drive *both* inference-perf and
`vllm bench serve` with an identical workload against the same configurable
mock server, then compare what each tool reports. The discrepancies we have
seen historically almost certainly live in how each tool *interprets* a
workload (request-rate vs concurrency semantics, warmup, tokenization) and how
each *computes* TTFT / ITL / TPOT from a stream -- not in the runners
themselves.

So the mapping from one `Scenario` to each tool's inputs is deliberately
centralized here, in plain sight, because that mapping is the thing under test.
If the two tools disagree, this file is the first place to look for an
asymmetry.

Units note (load-bearing):
  * inference-perf reports latency in SECONDS.
  * `vllm bench serve` reports latency in MILLISECONDS (`*_ms`).
  * `llm-d-inference-sim` is configured in MILLISECONDS.
The normalizers below convert everything to milliseconds so comparisons and
ground-truth assertions are apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Scenario:
    """A single workload, plus the latency the mock is configured to emulate.

    Because the mock is configured from the same object, `ttft_ms` / `itl_ms`
    are *ground truth*: the value each tool should recover. That is strictly
    stronger than only comparing the two tools to each other, since it tells us
    which tool drifts (and by how much), not merely that they disagree.
    """

    model_name: str  # model sent in requests; the sim is launched with this
    # --- Workload ---
    num_prompts: int  # total requests across the run
    input_len: int  # prompt tokens per request
    output_len: int  # generated tokens per request (with ignore_eos)

    # Load mode -- the two axes we most want to compare:
    #   "rate"        -> open-loop: arrivals at `request_rate` QPS (Poisson for
    #                    vllm, constant for inference-perf). num_workers must be
    #                    high enough not to bottleneck the offered rate.
    #   "concurrency" -> closed-loop: exactly `concurrency` requests in flight,
    #                    a new one launched as each completes.
    load_mode: str = "rate"  # "rate" | "concurrency"
    request_rate: float = 4.0  # used when load_mode == "rate"
    concurrency: int = 8  # used when load_mode == "concurrency"
    num_workers: int = 16  # inference-perf worker pool (cap on in-flight for rate mode)

    streaming: bool = True
    api: str = "completion"  # "completion" | "chat"

    # Client-side tokenizer: a bundled-tarball key or an HF id. Kept separate
    # from model_name so we can hold the server constant (the bundled gemma
    # tokenizer) and vary ONLY the tokenizer -- isolating tokenizer-driven
    # divergence (e.g. BOS/special-token handling, cf. #566) from everything
    # else. The test resolves this to a path/id; defaults to model_name.
    tokenizer: Optional[str] = None

    # vllm bench's --random-range-ratio controls how much input/output lengths
    # vary around the requested length. THIS IS THE KNOWN ROOT CAUSE of the
    # inference-perf-vs-vllm discrepancy we are chasing: a wrong value makes
    # vllm sample lengths from a wide range while inference-perf uses fixed
    # lengths, so the two tools benchmark different workloads.
    #
    # SEMANTICS ARE VERSION-DEPENDENT (reconcile when bumping the vllm pin):
    #   * old benchmarks/benchmark_serving.py: range is [len*ratio, len];
    #       ratio=1.0 -> FIXED, ratio=0.0 -> Uniform[0, len].
    #   * new `vllm bench serve`: range is [len*(1-ratio), len*(1+ratio)];
    #       ratio=0.0 -> FIXED, ratio=1.0 -> Uniform[0, 2*len].
    # Default below targets FIXED lengths for the pinned (new-CLI) ref so the
    # workload matches inference-perf. If this test ever passes only by
    # accident, check this value first.
    random_range_ratio: float = 0.0

    # --- Mock ground-truth latencies (milliseconds) ---
    ttft_ms: float = 0.0  # time-to-first-token
    itl_ms: float = 0.0  # inter-token latency == per-output-token time for the sim
    # Keep noise off by default so per-request timing is deterministic and the
    # ground-truth assertions are not fighting the mock's own variance.
    ttft_std_dev_ms: float = 0.0
    itl_std_dev_ms: float = 0.0

    @property
    def duration_sec(self) -> float:
        """For rate mode, inference-perf drives load by (rate, duration); derive
        the duration so it issues the same number of requests as vllm."""
        return self.num_prompts / self.request_rate

    def _inference_perf_load(self) -> Dict[str, Any]:
        """Load section for inference-perf, branched on load mode."""
        if self.load_mode == "concurrency":
            return {
                "type": "concurrent",
                "stages": [{"num_requests": self.num_prompts, "concurrency_level": self.concurrency}],
                "num_workers": max(self.num_workers, self.concurrency),
            }
        if self.load_mode == "rate":
            return {
                "type": "constant",
                "stages": [{"rate": self.request_rate, "duration": self.duration_sec}],
                "num_workers": self.num_workers,
            }
        raise ValueError(f"unknown load_mode: {self.load_mode!r}")

    # ------------------------------------------------------------------ mocks

    def sim_args(self) -> List[str]:
        """CLI flags for llm-d-inference-sim to emulate this scenario.

        Flag names verified against llm-d-inference-sim v0.6.1 (the version
        pinned in flake.nix). See its README "latency simulation" section.
        """
        # The sim's context window defaults to 1024 tokens and it 400s any
        # request where prompt + completion exceeds it, so size it to the
        # scenario. Doubled for headroom: the sim counts prompt tokens with its
        # own tokenizer, which can exceed the client's intended input_len when
        # the client tokenizer differs (and chat adds template tokens). The sim
        # is a mock, so an oversized window costs nothing.
        max_model_len = max(1024, 2 * (self.input_len + self.output_len))
        args = [
            "--max-model-len",
            str(max_model_len),
            "--time-to-first-token",
            str(int(self.ttft_ms)),
            "--inter-token-latency",
            str(int(self.itl_ms)),
        ]
        if self.ttft_std_dev_ms:
            args += ["--time-to-first-token-std-dev", str(int(self.ttft_std_dev_ms))]
        if self.itl_std_dev_ms:
            args += ["--inter-token-latency-std-dev", str(int(self.itl_std_dev_ms))]
        return args

    # ----------------------------------------------------------- inference-perf

    def inference_perf_config(self, *, base_url: str, tokenizer_path: str) -> Dict[str, Any]:
        """Render this scenario as an inference-perf config dict.

        Uses synthetic data with fixed input/output lengths so the workload is
        identical to vllm bench's `--dataset-name random`.
        """
        return {
            "data": {
                "type": "synthetic",
                # Fixed distributions => exactly input_len/output_len tokens,
                # matching vllm bench's fixed --random-{input,output}-len so the
                # two tools issue byte-for-byte comparable requests.
                "input_distribution": {
                    "type": "fixed",
                    "min": self.input_len,
                    "max": self.input_len,
                    "mean": self.input_len,
                },
                "output_distribution": {
                    "type": "fixed",
                    "min": self.output_len,
                    "max": self.output_len,
                    "mean": self.output_len,
                },
            },
            "load": self._inference_perf_load(),
            "api": {"type": self.api, "streaming": self.streaming},
            "server": {
                "type": "vllm",
                "model_name": self.model_name,
                "base_url": base_url,
                "ignore_eos": True,  # force exactly output_len tokens, like vllm bench
            },
            "tokenizer": {"pretrained_model_name_or_path": tokenizer_path},
            "report": {
                "request_lifecycle": {"summary": True, "per_stage": True, "per_request": True},
            },
        }

    # -------------------------------------------------------------- vllm bench

    def vllm_bench_args(self, *, base_url: str, result_path: str, tokenizer: str) -> List[str]:
        """Render this scenario as `vllm bench serve` CLI args.

        NOTE: flag names track the pinned vllm ref (see vllm_bench.py). If you
        bump the pin and the CLI changed, this is the place to reconcile.
        """
        endpoint = "/v1/chat/completions" if self.api == "chat" else "/v1/completions"
        backend = "openai-chat" if self.api == "chat" else "openai"
        args = [
            "--backend",
            backend,
            "--model",
            self.model_name,
            "--tokenizer",
            tokenizer,
            "--base-url",
            base_url,
            "--endpoint",
            endpoint,
            "--dataset-name",
            "random",
            "--random-input-len",
            str(self.input_len),
            "--random-output-len",
            str(self.output_len),
            # See Scenario.random_range_ratio: pin this explicitly, it is the
            # known root cause of workload mismatch between the two tools.
            "--random-range-ratio",
            str(self.random_range_ratio),
            "--num-prompts",
            str(self.num_prompts),
            "--ignore-eos",
            "--percentile-metrics",
            "ttft,tpot,itl",
            "--save-result",
            "--result-filename",
            result_path,
        ]
        # Load mode: rate (open-loop arrivals) vs concurrency (closed-loop).
        if self.load_mode == "concurrency":
            # Unbounded arrivals, capped to `concurrency` in flight == closed loop.
            args += ["--request-rate", "inf", "--max-concurrency", str(self.concurrency)]
        elif self.load_mode == "rate":
            args += ["--request-rate", str(self.request_rate)]
        else:
            raise ValueError(f"unknown load_mode: {self.load_mode!r}")

        if not self.streaming:
            # The pinned vllm bench cannot disable response streaming: its
            # request funcs hardcode "stream": True, and its only stream
            # flag (--no-stream) controls HF dataset loading, not responses.
            raise ValueError("vllm bench serve (pinned ref) only supports streaming scenarios")
        return args


@dataclass
class NormalizedMetrics:
    """Comparable metrics extracted from either tool. All latencies in ms."""

    source: str  # "inference-perf" | "vllm-bench" | "ground-truth"
    request_count: int
    mean_ttft_ms: Optional[float] = None
    median_ttft_ms: Optional[float] = None
    mean_itl_ms: Optional[float] = None
    median_itl_ms: Optional[float] = None
    mean_tpot_ms: Optional[float] = None
    median_tpot_ms: Optional[float] = None
    output_throughput_tok_s: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def _sec_to_ms(v: Optional[float]) -> Optional[float]:
    return None if v is None else v * 1000.0


def normalize_inference_perf(summary_report: Dict[str, Any]) -> NormalizedMetrics:
    """Pull comparable metrics out of inference-perf's summary_lifecycle_metrics.json.

    Shape (see inference_perf/reportgen/base.py):
      successes.latency.{time_to_first_token,inter_token_latency,
                         time_per_output_token}.{mean,median,...}   # SECONDS
      successes.throughput.output_tokens_per_sec
    """
    s = summary_report["successes"]
    lat = s["latency"]
    return NormalizedMetrics(
        source="inference-perf",
        request_count=s["count"],
        mean_ttft_ms=_sec_to_ms(lat["time_to_first_token"].get("mean")),
        median_ttft_ms=_sec_to_ms(lat["time_to_first_token"].get("median")),
        mean_itl_ms=_sec_to_ms(lat["inter_token_latency"].get("mean")),
        median_itl_ms=_sec_to_ms(lat["inter_token_latency"].get("median")),
        mean_tpot_ms=_sec_to_ms(lat["time_per_output_token"].get("mean")),
        median_tpot_ms=_sec_to_ms(lat["time_per_output_token"].get("median")),
        output_throughput_tok_s=s.get("throughput", {}).get("output_tokens_per_sec"),
        raw=summary_report,
    )


def normalize_vllm_bench(result: Dict[str, Any]) -> NormalizedMetrics:
    """Pull comparable metrics out of `vllm bench serve --save-result` JSON.

    vllm already reports milliseconds (`mean_ttft_ms`, etc.). Key names are
    stable across recent vllm releases but should be re-checked when the pin
    moves; unknown keys simply come back as None rather than KeyError.
    """
    return NormalizedMetrics(
        source="vllm-bench",
        request_count=result.get("completed", result.get("num_prompts", 0)),
        mean_ttft_ms=result.get("mean_ttft_ms"),
        median_ttft_ms=result.get("median_ttft_ms"),
        mean_itl_ms=result.get("mean_itl_ms"),
        median_itl_ms=result.get("median_itl_ms"),
        mean_tpot_ms=result.get("mean_tpot_ms"),
        median_tpot_ms=result.get("median_tpot_ms"),
        output_throughput_tok_s=result.get("output_throughput"),
        raw=result,
    )


def ground_truth(scenario: Scenario) -> NormalizedMetrics:
    """The latency the mock was configured to emulate -- the value both tools
    should recover. ITL and TPOT collapse to the same per-token figure for the
    sim, which generates one token per inter-token interval."""
    return NormalizedMetrics(
        source="ground-truth",
        request_count=scenario.num_prompts,
        mean_ttft_ms=scenario.ttft_ms,
        median_ttft_ms=scenario.ttft_ms,
        mean_itl_ms=scenario.itl_ms,
        median_itl_ms=scenario.itl_ms,
        mean_tpot_ms=scenario.itl_ms,
        median_tpot_ms=scenario.itl_ms,
    )


# Metrics compared by the parity test, with their per-metric tolerance. Strict
# tool-vs-tool parity, but with reasonable bounds: neither tool will recover a
# configured TTFT/TPOT perfectly, and ITL is the noisiest (per-chunk effects,
# cf. inference-perf #566). Tune these as real deltas come in.
PARITY_FIELDS = ("median_ttft_ms", "median_tpot_ms", "median_itl_ms", "output_throughput_tok_s")


def compare(
    a: NormalizedMetrics,
    b: NormalizedMetrics,
    *,
    fields: tuple[str, ...] = PARITY_FIELDS,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Return a per-field {a, b, abs_diff, rel_diff} table for reporting and
    assertions. rel_diff is relative to max(|a|,|b|) so it is symmetric and
    well-defined when one value is ~0."""
    table: Dict[str, Dict[str, Optional[float]]] = {}
    for f in fields:
        av = getattr(a, f)
        bv = getattr(b, f)
        if av is None or bv is None:
            table[f] = {"a": av, "b": bv, "abs_diff": None, "rel_diff": None}
            continue
        abs_diff = abs(av - bv)
        denom = max(abs(av), abs(bv))
        rel_diff = abs_diff / denom if denom > 0 else 0.0
        table[f] = {"a": av, "b": bv, "abs_diff": abs_diff, "rel_diff": rel_diff}
    return table


def format_comparison_table(rows: Dict[str, Dict[str, Dict[str, Optional[float]]]]) -> str:
    """Render named comparison tables (e.g. {"inference-perf vs vllm-bench": ...})
    as a human-readable artifact for the test log / regression record."""
    out: List[str] = []
    for title, table in rows.items():
        out.append(f"\n=== {title} ===")
        out.append(f"{'metric':<28}{'a':>14}{'b':>14}{'abs_diff':>14}{'rel_diff':>12}")
        for metric, vals in table.items():

            def fmt(x: Optional[float], pct: bool = False) -> str:
                if x is None:
                    return "n/a"
                return f"{x * 100:.1f}%" if pct else f"{x:.3f}"

            out.append(
                f"{metric:<28}{fmt(vals['a']):>14}{fmt(vals['b']):>14}"
                f"{fmt(vals['abs_diff']):>14}{fmt(vals['rel_diff'], pct=True):>12}"
            )
    return "\n".join(out)
