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
"""Orchestration tests for the closed-loop sweep ``preprocess``.

These drive the real ``LoadGenerator.preprocess`` but mock the single live
multiprocessing call (``_measure_concurrency``) with a synthetic server, so the
ladder -> mu_max -> stage-generation logic is exercised without a cluster. The
estimator math itself is covered in ``test_saturation.py``.
"""

from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inference_perf.config import LoadConfig, LoadType, StageGenType, StandardLoadStage, SweepConfig
from inference_perf.datagen import RandomDataGenerator
from inference_perf.loadgen.load_generator import (
    SWEEP_RHO_CEILING,
    LoadGenerator,
)
from inference_perf.loadgen.saturation import ProbeResult, SaturationResult


def _stage_rates(generator: LoadGenerator) -> List[float]:
    """Generated stage rates (all generated stages are StandardLoadStage)."""
    return [s.rate for s in generator.stages if isinstance(s, StandardLoadStage)]


def _make_generator(sweep: SweepConfig, num_workers: int = 4, worker_max_concurrency: int = 100) -> LoadGenerator:
    datagen = MagicMock(spec=RandomDataGenerator)
    datagen.trace = None
    load_config = LoadConfig(
        type=LoadType.CONSTANT,
        sweep=sweep,
        num_workers=num_workers,
        worker_max_concurrency=worker_max_concurrency,
        stages=[],
    )
    with patch("inference_perf.loadgen.load_generator.get_circuit_breaker"):
        return LoadGenerator(datagen, load_config)


def _saturating_measure(mu: float, n_half: float) -> AsyncMock:
    """Mock _measure_concurrency: throughput follows X(N) = mu*N/(N+n_half)."""

    async def measure(n: int, *args: Any, **kwargs: Any) -> ProbeResult:
        x = mu * n / (n + n_half)
        return ProbeResult(concurrency=n, throughput=x, ci_half_width=0.0, n_batches=20, converged=True)

    return AsyncMock(side_effect=measure)


async def _run_preprocess(generator: LoadGenerator) -> None:
    request_phase = MagicMock()
    stage_barrier = MagicMock()
    await generator.preprocess(
        MagicMock(),  # client
        MagicMock(),  # request_queue
        MagicMock(),  # active_requests_counter
        MagicMock(),  # finished_requests_counter
        request_phase,
        MagicMock(),  # cancel_signal
        stage_barrier,
    )


def _sat(mu_max: float, *, harness_limited: bool = False, probes: List[ProbeResult] | None = None) -> SaturationResult:
    """A SaturationResult for exercising the pure rate-layout logic."""
    return SaturationResult(
        max_throughput=mu_max,
        knee_concurrency=0,
        harness_limited=harness_limited,
        confident=True,
        probes=probes or [],
    )


@pytest.mark.asyncio
async def test_preprocess_concentrates_stages_at_knee() -> None:
    # Stages are placed by utilization with geometric spacing in the saturation
    # gap (1 - rho), so resolution clusters near saturation: consecutive rates
    # get closer together as they approach mu_max. Exactly one stage sits above
    # saturation to expose the latency cliff.
    sweep = SweepConfig(type=StageGenType.GEOM, num_stages=10, stage_duration=30)
    generator = _make_generator(sweep)
    generator._measure_concurrency = _saturating_measure(mu=100.0, n_half=10.0)  # type: ignore[method-assign]

    await _run_preprocess(generator)
    rates = sorted(_stage_rates(generator))
    assert len(rates) == 10
    assert all(isinstance(s, StandardLoadStage) and s.duration == 30 for s in generator.stages)

    stable, overload = rates[:-1], rates[-1]
    assert overload > stable[-1]  # one deliberate above-saturation stage
    # Spacing shrinks monotonically toward saturation (resolution at the knee).
    diffs = [b - a for a, b in zip(stable[:-1], stable[1:], strict=True)]
    assert all(later <= earlier for earlier, later in zip(diffs[:-1], diffs[1:], strict=True))
    assert diffs[0] > diffs[-1]


def test_generate_sweep_rates_clusters_below_saturation_and_caps_overload() -> None:
    rates = LoadGenerator._generate_sweep_rates(_sat(100.0), 10, StageGenType.GEOM, stage_duration=60, timeout=None)
    assert rates == sorted(rates)
    assert len(rates) == 10
    stable, overload = rates[:-1], rates[-1]
    # Top stable stage sits at the utilization ceiling; none sample the asymptote.
    assert max(stable) == pytest.approx(100.0 * SWEEP_RHO_CEILING, rel=1e-3)
    assert all(r <= 100.0 * SWEEP_RHO_CEILING + 1e-6 for r in stable)
    # No timeout -> overload backlog adds m_min/duration requests above mu_max.
    assert overload == pytest.approx(100.0 + 200.0 / 60.0, rel=1e-3)


def test_generate_sweep_rates_scale_invariant_in_utilization() -> None:
    # Holding mu_max * stage_duration constant, the normalized layout (rate/mu_max)
    # is identical across wildly different absolute rates: invariant to scale.
    a = LoadGenerator._generate_sweep_rates(_sat(10.0), 10, StageGenType.GEOM, stage_duration=600, timeout=None)
    b = LoadGenerator._generate_sweep_rates(_sat(5000.0), 10, StageGenType.GEOM, stage_duration=1.2, timeout=None)
    norm_a = [r / 10.0 for r in a]
    norm_b = [r / 5000.0 for r in b]
    # Identical pre-rounding; 3-decimal rate rounding leaves only ~1e-3 drift.
    assert norm_a == pytest.approx(norm_b, rel=3e-3)


def test_generate_sweep_rates_timeout_caps_upper_bound() -> None:
    # Probe latencies (R = N/X) above 50% of the timeout are excluded from the
    # stable band, pulling rho_max below the ceiling. With X(N)=100N/(N+10),
    # R=(N+10)/100; budget=0.5s admits N<=40. The cap snaps to the highest
    # *probed* point under budget (N=32 -> 76.19 req/s), since N=40 isn't on the
    # doubling ladder; the discrete, conservative cap is intended.
    probes = [
        ProbeResult(concurrency=n, throughput=100.0 * n / (n + 10), ci_half_width=0.0, n_batches=20, converged=True)
        for n in (1, 2, 4, 8, 16, 32, 64, 128)
    ]
    rates = LoadGenerator._generate_sweep_rates(
        _sat(100.0, probes=probes), 10, StageGenType.GEOM, stage_duration=60, timeout=1.0
    )
    stable = rates[:-1]
    assert max(stable) == pytest.approx(100.0 * 32 / 42, rel=1e-2)  # N=32 probe throughput


def test_generate_sweep_rates_harness_limited_ramps_to_lower_bound() -> None:
    # No plateau found: ramp up to the lower-bound mu_max with no overload stage.
    rates = LoadGenerator._generate_sweep_rates(
        _sat(100.0, harness_limited=True), 8, StageGenType.GEOM, stage_duration=60, timeout=None
    )
    assert rates == sorted(rates)
    assert len(rates) == 8
    assert max(rates) == pytest.approx(100.0, rel=1e-3)
    assert all(r <= 100.0 + 1e-6 for r in rates)  # no above-saturation stage


def test_generate_sweep_rates_linear_uses_same_data_derived_bounds() -> None:
    # LINEAR keeps the utilization bounds but spaces rho evenly between them.
    rates = LoadGenerator._generate_sweep_rates(_sat(100.0), 6, StageGenType.LINEAR, stage_duration=60, timeout=None)
    assert rates == sorted(rates)
    stable = rates[:-1]
    diffs = [b - a for a, b in zip(stable[:-1], stable[1:], strict=True)]
    assert max(diffs) - min(diffs) == pytest.approx(0.0, abs=1e-2)  # evenly spaced
    assert max(stable) == pytest.approx(100.0 * SWEEP_RHO_CEILING, rel=1e-3)


@pytest.mark.asyncio
async def test_preprocess_raises_when_no_throughput() -> None:
    sweep = SweepConfig(type=StageGenType.GEOM, num_stages=5)
    generator = _make_generator(sweep)

    async def zero(n: int, *args: Any, **kwargs: Any) -> ProbeResult:
        return ProbeResult(concurrency=n, throughput=0.0, ci_half_width=float("inf"), n_batches=0, converged=False)

    generator._measure_concurrency = AsyncMock(side_effect=zero)  # type: ignore[method-assign]

    with pytest.raises(Exception, match="failed to determine a valid saturation point"):
        await _run_preprocess(generator)


@pytest.mark.asyncio
async def test_preprocess_concurrency_ceiling_from_harness() -> None:
    # num_workers=2 * worker_max_concurrency=8 => ceiling 16; ladder tops out there.
    sweep = SweepConfig(type=StageGenType.LINEAR, num_stages=3)
    generator = _make_generator(sweep, num_workers=2, worker_max_concurrency=8)
    measure = _saturating_measure(mu=100.0, n_half=10.0)
    generator._measure_concurrency = measure  # type: ignore[method-assign]

    await _run_preprocess(generator)
    probed = [call.args[0] for call in measure.await_args_list]
    assert max(probed) <= 16  # never probes beyond the harness ceiling
