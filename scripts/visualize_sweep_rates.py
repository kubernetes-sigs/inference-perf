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
"""Visualize and sanity-check sweep stage placement.

Compares the current utilization-based sweep layout against the former
symmetric +/-1.5x rate window, for a given mu_max. Prints an ASCII number line
of where stages land relative to saturation and, with ``--plot``, writes a PNG
overlaying both layouts on the queueing latency curve (W ~ 1/(1-rho)).

This is a developer aid for reasoning about the methodology; it imports the same
``_generate_sweep_rates`` the load generator uses, so what you see is what runs.

Examples:
    python scripts/visualize_sweep_rates.py --mu-max 100 --num-stages 10
    python scripts/visualize_sweep_rates.py --mu-max 250 --timeout 30 --plot out.png
"""

from __future__ import annotations

import argparse
from typing import List

import numpy as np

from inference_perf.config import StageGenType
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.loadgen.saturation import ProbeResult, SaturationResult

# The factor the old layout used; kept here only for the side-by-side comparison.
SWEEP_RANGE_FACTOR_LEGACY = 1.5


def _legacy_rates(mu_max: float, size: int, gen_type: StageGenType) -> List[float]:
    """The former symmetric window: [mu_max / f, mu_max * f], geometric/linear."""
    bottom, top = mu_max / SWEEP_RANGE_FACTOR_LEGACY, mu_max * SWEEP_RANGE_FACTOR_LEGACY
    space = np.geomspace if gen_type == StageGenType.GEOM else np.linspace
    return [float(round(r, 3)) for r in space(bottom, top, num=size)]


def _synthetic_probes(mu_max: float, n_half: float = 10.0) -> List[ProbeResult]:
    """Probes from X(N) = mu_max * N / (N + n_half) on a doubling ladder."""
    probes = []
    n = 1
    while n <= 1024:
        x = mu_max * n / (n + n_half)
        probes.append(ProbeResult(concurrency=n, throughput=x, ci_half_width=0.0, n_batches=20, converged=True))
        n *= 2
    return probes


def _number_line(rates: List[float], mu_max: float, hi: float, width: int = 64) -> str:
    """ASCII number line; '|' marks a stage, 'S' marks mu_max (saturation)."""
    cells = [" "] * (width + 1)
    sat_col = min(width, round(width * mu_max / hi))
    cells[sat_col] = "S"
    for r in rates:
        col = min(width, round(width * r / hi))
        cells[col] = "+" if cells[col] in (" ", "S") else "*"
    return "".join(cells)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mu-max", type=float, default=100.0, help="Saturation throughput (req/s)")
    parser.add_argument("--num-stages", type=int, default=10)
    parser.add_argument("--stage-duration", type=float, default=60.0)
    parser.add_argument("--timeout", type=float, default=None, help="Per-request timeout (s); caps the upper bound")
    parser.add_argument("--linear", action="store_true", help="Use linear spacing instead of geometric")
    parser.add_argument("--plot", type=str, default=None, metavar="PNG", help="Write a comparison plot to this path")
    args = parser.parse_args()

    gen_type = StageGenType.LINEAR if args.linear else StageGenType.GEOM
    sat = SaturationResult(
        max_throughput=args.mu_max,
        knee_concurrency=0,
        harness_limited=False,
        confident=True,
        probes=_synthetic_probes(args.mu_max),
    )
    new = LoadGenerator._generate_sweep_rates(sat, args.num_stages, gen_type, args.stage_duration, args.timeout)
    old = _legacy_rates(args.mu_max, args.num_stages, gen_type)

    mu = args.mu_max
    hi = max(max(new), max(old)) * 1.02

    def summarize(name: str, rates: List[float]) -> None:
        below = [r for r in rates if r < mu]
        near = [r for r in rates if 0.9 * mu <= r <= mu]
        over = [r for r in rates if r > mu]
        print(f"\n{name}")
        print(f"  rates: {[round(r, 2) for r in rates]}")
        print(f"  {len(below)} below mu_max, {len(near)} in the [0.9, 1.0]*mu_max knee band, {len(over)} above")
        print(f"  {_number_line(rates, mu, hi)}")

    print(f"mu_max = {mu} req/s, num_stages = {args.num_stages}, spacing = {gen_type.value}, timeout = {args.timeout}")
    print(f"('S' = saturation; scale 0 .. {hi:.1f} req/s)")
    summarize("OLD  (symmetric +/-1.5x rate window)", old)
    summarize("NEW  (utilization, geometric-in-gap + 1 overload)", new)

    if args.plot:
        _plot(old, new, mu, hi, gen_type, args.plot)


def _plot(old: List[float], new: List[float], mu: float, hi: float, gen_type: StageGenType, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rho = np.linspace(0.01, 0.99, 400)
    w = 1.0 / (1.0 - rho)  # normalized queueing latency proxy

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rho * mu, w, color="0.6", label="latency proxy 1/(1-rho)")
    for r in old:
        ax.axvline(r / mu * mu, color="tab:red", alpha=0.35, linewidth=1)
    for r in new:
        ax.axvline(r, color="tab:blue", alpha=0.6, linewidth=1.2)
    ax.axvline(mu, color="black", linestyle="--", linewidth=1, label="mu_max")
    ax.plot([], [], color="tab:red", label=f"old window ({len(old)} stages)")
    ax.plot([], [], color="tab:blue", label=f"new layout ({len(new)} stages)")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, 1.0 / (1.0 - 0.97))
    ax.set_xlabel("offered rate (req/s)")
    ax.set_ylabel("relative latency")
    ax.set_title(f"Sweep stage placement ({gen_type.value}), mu_max={mu:g}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
