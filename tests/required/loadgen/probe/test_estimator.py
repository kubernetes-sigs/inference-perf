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
import unittest

import numpy as np

from inference_perf.loadgen.probe import (
    RungResult,
    SaturationSignal,
    batch_means_throughput,
    estimate_constants,
    fit_saturating_curve,
    isotonic_regression,
    make_rung,
    select_usable_rungs,
)


def saturating_throughput(concurrency: float, mu: float = 20.0, knee_constant: float = 8.0) -> float:
    """Closed-loop throughput of an idealized backend: X(N) = mu * N / (K + N)."""
    return mu * concurrency / (knee_constant + concurrency)


def ideal_rungs(concurrencies: list[int], mu: float = 20.0, knee_constant: float = 8.0) -> list[RungResult]:
    return [
        RungResult(concurrency=n, throughput=saturating_throughput(n, mu, knee_constant), throughput_se=0.0)
        for n in concurrencies
    ]


class TestBatchMeansThroughput(unittest.TestCase):
    def test_uniform_completions(self) -> None:
        times = np.arange(0.0, 10.0, 0.1)
        throughput, standard_error, batch_rates = batch_means_throughput(times, 0.0, 10.0, num_batches=8)
        self.assertAlmostEqual(throughput, 10.0)
        self.assertLess(standard_error, 0.5)
        self.assertEqual(batch_rates.size, 8)

    def test_window_filtering(self) -> None:
        times = np.arange(0.0, 20.0, 0.1)
        throughput, _, _ = batch_means_throughput(times, 5.0, 15.0)
        self.assertAlmostEqual(throughput, 10.0)

    def test_empty_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            batch_means_throughput([100.0], 0.0, 10.0)

    def test_invalid_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            batch_means_throughput([1.0], 10.0, 10.0)

    def test_too_few_batches_raises(self) -> None:
        with self.assertRaises(ValueError):
            batch_means_throughput([1.0], 0.0, 10.0, num_batches=1)


class TestMakeRung(unittest.TestCase):
    def test_littles_law_consistency(self) -> None:
        # Closed loop at N=5 with R=0.5s implies X=10/s; the residual must vanish.
        times = np.arange(0.0, 10.0, 0.1)
        latencies = np.full_like(times, 0.5)
        rung = make_rung(5, times, latencies, 0.0, 10.0)
        self.assertAlmostEqual(rung.throughput, 10.0)
        self.assertAlmostEqual(rung.latency, 0.5)
        self.assertAlmostEqual(rung.littles_law_residual, 0.0)
        self.assertTrue(rung.stationary)
        self.assertIs(rung.signal, SaturationSignal.NONE)

    def test_inconsistent_window_has_large_residual(self) -> None:
        # Claiming N=50 while X*R=5 leaves a residual of 0.9.
        times = np.arange(0.0, 10.0, 0.1)
        latencies = np.full_like(times, 0.5)
        rung = make_rung(50, times, latencies, 0.0, 10.0)
        self.assertGreater(rung.littles_law_residual, 0.85)

    def test_misaligned_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            make_rung(1, [1.0, 2.0], [0.5], 0.0, 10.0)


class TestIsotonicRegression(unittest.TestCase):
    def test_pools_violators(self) -> None:
        fitted = isotonic_regression([1.0, 3.0, 2.0])
        self.assertTrue(np.allclose(fitted, [1.0, 2.5, 2.5]))

    def test_decreasing_pools_to_mean(self) -> None:
        fitted = isotonic_regression([3.0, 2.0, 1.0])
        self.assertTrue(np.allclose(fitted, [2.0, 2.0, 2.0]))

    def test_monotone_input_unchanged(self) -> None:
        values = [1.0, 2.0, 4.0, 8.0]
        self.assertTrue(np.allclose(isotonic_regression(values), values))

    def test_weights_shift_pooled_mean(self) -> None:
        fitted = isotonic_regression([3.0, 1.0], weights=[3.0, 1.0])
        self.assertTrue(np.allclose(fitted, [2.5, 2.5]))

    def test_invalid_weights_raise(self) -> None:
        with self.assertRaises(ValueError):
            isotonic_regression([1.0, 2.0], weights=[1.0, 0.0])


class TestFitSaturatingCurve(unittest.TestCase):
    def test_recovers_noiseless_parameters(self) -> None:
        concurrencies = [1, 2, 4, 8, 16, 32, 64]
        throughputs = [saturating_throughput(n) for n in concurrencies]
        fit = fit_saturating_curve(concurrencies, throughputs)
        self.assertIsNotNone(fit)
        assert fit is not None
        mu, knee_constant = fit
        self.assertAlmostEqual(mu, 20.0, places=6)
        self.assertAlmostEqual(knee_constant, 8.0, places=6)

    def test_superlinear_data_rejected(self) -> None:
        concurrencies = [1.0, 2.0, 4.0, 8.0]
        throughputs = [n * n for n in concurrencies]
        self.assertIsNone(fit_saturating_curve(concurrencies, throughputs))

    def test_too_few_points_rejected(self) -> None:
        self.assertIsNone(fit_saturating_curve([1.0, 2.0], [1.0, 2.0]))

    def test_non_positive_raises(self) -> None:
        with self.assertRaises(ValueError):
            fit_saturating_curve([1.0, 2.0, 3.0], [1.0, -2.0, 3.0])


class TestEstimateConstants(unittest.TestCase):
    def test_noiseless_ladder_recovers_asymptote(self) -> None:
        rungs = ideal_rungs([1, 2, 4, 8, 16, 32, 64])
        constants = estimate_constants(rungs, rng=np.random.default_rng(0), num_bootstrap=50)
        self.assertAlmostEqual(constants["r_sat"].value, 20.0, places=5)
        self.assertAlmostEqual(constants["r_sat"].ci.width, 0.0, places=5)

    def test_knee_fraction_controls_knee(self) -> None:
        # X(32) = 16 = 0.8 * mu exactly, so the 0.8-knee sits at N=32.
        rungs = ideal_rungs([1, 2, 4, 8, 16, 32, 64])
        constants = estimate_constants(rungs, rng=np.random.default_rng(0), num_bootstrap=50, knee_fraction=0.8)
        self.assertAlmostEqual(constants["n_knee"].value, 32.0, places=5)

    def test_censored_knee_falls_back_to_largest_rung(self) -> None:
        # No measured rung reaches 0.9 * mu, so n_knee is censored at N=64.
        rungs = ideal_rungs([1, 2, 4, 8, 16, 32, 64])
        constants = estimate_constants(rungs, rng=np.random.default_rng(0), num_bootstrap=50, knee_fraction=0.9)
        self.assertAlmostEqual(constants["n_knee"].value, 64.0, places=5)

    def test_noisy_ladder_recovers_asymptote_within_ci(self) -> None:
        rng = np.random.default_rng(7)
        concurrencies = [1, 2, 4, 8, 16, 32, 64]
        rungs = [
            RungResult(
                concurrency=n,
                throughput=float(saturating_throughput(n) + rng.normal(0.0, 0.2)),
                throughput_se=0.2,
            )
            for n in concurrencies
        ]
        constants = estimate_constants(rungs, rng=np.random.default_rng(0))
        r_sat = constants["r_sat"]
        self.assertAlmostEqual(r_sat.value, 20.0, delta=2.0)
        self.assertLessEqual(r_sat.ci.low, 20.0)
        self.assertGreaterEqual(r_sat.ci.high, 19.0)

    def test_unsaturated_ladder_falls_back_to_plateau(self) -> None:
        # Linear X(N) = 2N carries no evidence of an asymptote; report the
        # measured plateau instead of a wild extrapolation.
        rungs = [RungResult(concurrency=n, throughput=2.0 * n, throughput_se=0.0) for n in [1, 2, 4, 8, 16, 32]]
        constants = estimate_constants(rungs, rng=np.random.default_rng(0), num_bootstrap=50)
        self.assertAlmostEqual(constants["r_sat"].value, 64.0, places=5)

    def test_unusable_rungs_excluded(self) -> None:
        rungs = ideal_rungs([1, 2, 4, 8, 16, 32, 64])
        rungs.append(RungResult(concurrency=128, throughput=100.0, throughput_se=0.0, stationary=False))
        rungs.append(RungResult(concurrency=256, throughput=1.0, throughput_se=0.0, signal=SaturationSignal.CLIENT_BOUND))
        constants = estimate_constants(rungs, rng=np.random.default_rng(0), num_bootstrap=50)
        self.assertAlmostEqual(constants["r_sat"].value, 20.0, places=5)

    def test_retried_rung_uses_latest_measurement(self) -> None:
        rungs = ideal_rungs([1, 2, 4, 8, 16, 32])
        stale = RungResult(concurrency=64, throughput=5.0, throughput_se=0.0)
        fresh = RungResult(concurrency=64, throughput=saturating_throughput(64), throughput_se=0.0)
        constants = estimate_constants([stale, *rungs, fresh], rng=np.random.default_rng(0), num_bootstrap=50)
        self.assertAlmostEqual(constants["r_sat"].value, 20.0, places=5)

    def test_too_few_rungs_raises(self) -> None:
        with self.assertRaises(ValueError):
            estimate_constants(ideal_rungs([1]), rng=np.random.default_rng(0))


class TestSelectUsableRungs(unittest.TestCase):
    # A rung whose window did not actually hold its concurrency (residual above
    # the gate) describes a broken measurement, so it must not reach the fit.
    def test_residual_above_gate_excluded(self) -> None:
        rungs = ideal_rungs([1, 2]) + [RungResult(concurrency=4, throughput=1000.0, littles_law_residual=0.5)]
        self.assertEqual([rung.concurrency for rung in select_usable_rungs(rungs)], [1, 2])

    # A mostly-failing rung measures the failure, not the server, so its
    # throughput must not become a point on the capacity curve.
    def test_error_rate_above_gate_excluded(self) -> None:
        rungs = ideal_rungs([1, 2]) + [RungResult(concurrency=4, throughput=1000.0, error_rate=0.5)]
        self.assertEqual([rung.concurrency for rung in select_usable_rungs(rungs)], [1, 2])

    # Exactly at the gate is still usable: the thresholds are inclusive.
    def test_rung_at_gate_stays_usable(self) -> None:
        rungs = [
            RungResult(concurrency=1, throughput=5.0, littles_law_residual=0.2, error_rate=0.1),
            RungResult(concurrency=2, throughput=9.0),
        ]
        self.assertEqual([rung.concurrency for rung in select_usable_rungs(rungs)], [1, 2])

    # Client-bound and non-stationary rungs are rejected for the same reason as
    # before the residual gate existed, and the gate must not have replaced them.
    def test_client_bound_and_non_stationary_excluded(self) -> None:
        rungs = ideal_rungs([1, 2]) + [
            RungResult(concurrency=4, throughput=1000.0, signal=SaturationSignal.CLIENT_BOUND),
            RungResult(concurrency=8, throughput=1000.0, stationary=False),
        ]
        self.assertEqual([rung.concurrency for rung in select_usable_rungs(rungs)], [1, 2])

    # A retried concurrency contributes only its latest measurement, in ladder
    # order, so a stale sample cannot drag the fit down.
    def test_latest_measurement_per_concurrency(self) -> None:
        rungs = [
            RungResult(concurrency=1, throughput=5.0),
            RungResult(concurrency=2, throughput=1.0),
            RungResult(concurrency=2, throughput=9.0),
        ]
        self.assertEqual([rung.throughput for rung in select_usable_rungs(rungs)], [5.0, 9.0])

    # Both thresholds are fractions; a passed-in 1.5 or -1 is a config bug.
    def test_out_of_range_thresholds_raise(self) -> None:
        with self.assertRaises(ValueError):
            select_usable_rungs([], max_littles_law_residual=1.5)
        with self.assertRaises(ValueError):
            select_usable_rungs([], max_error_rate=-0.1)

    # The two new gates are part of estimate_constants itself, not only of the
    # helper the caller happens to use: a broken or failing rung with a huge
    # throughput cannot pull r_sat up.
    def test_estimate_constants_gates_residual_and_error_rate(self) -> None:
        rungs = ideal_rungs([1, 2, 4, 8, 16, 32, 64])
        rungs.append(RungResult(concurrency=128, throughput=1000.0, littles_law_residual=0.9))
        rungs.append(RungResult(concurrency=256, throughput=1000.0, error_rate=0.9))
        constants = estimate_constants(rungs, rng=np.random.default_rng(0), num_bootstrap=50)
        self.assertAlmostEqual(constants["r_sat"].value, 20.0, places=5)

    # A config that rejects every rung must fail with a message that names the
    # gates, so the operator knows which knob to loosen.
    def test_all_rungs_rejected_message_names_the_gates(self) -> None:
        rungs = [
            RungResult(concurrency=1, throughput=5.0, error_rate=0.9),
            RungResult(concurrency=2, throughput=9.0, littles_law_residual=0.9),
        ]
        with self.assertRaises(ValueError) as raised:
            estimate_constants(rungs, rng=np.random.default_rng(0))
        message = str(raised.exception)
        self.assertIn("Little's law residual", message)
        self.assertIn("error rate", message)


if __name__ == "__main__":
    unittest.main()
