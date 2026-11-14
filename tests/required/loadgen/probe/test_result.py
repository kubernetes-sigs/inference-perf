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

from inference_perf.loadgen.probe import (
    RESERVED_SYMBOLS,
    BoundConstant,
    ConfidenceInterval,
    ProbeResult,
    ProbeStopReason,
    RungResult,
    SaturationSignal,
    classify_saturation,
)


class TestConfidenceInterval(unittest.TestCase):
    def test_valid_interval(self) -> None:
        ci = ConfidenceInterval(low=1.0, high=3.0)
        self.assertEqual(ci.width, 2.0)

    def test_inverted_interval_raises(self) -> None:
        with self.assertRaises(ValueError):
            ConfidenceInterval(low=3.0, high=1.0)


class TestBoundConstant(unittest.TestCase):
    def test_reserved_names_accepted(self) -> None:
        for name in RESERVED_SYMBOLS:
            constant = BoundConstant(name=name, value=1.0, ci=ConfidenceInterval(0.5, 1.5))
            self.assertEqual(constant.name, name)

    def test_unreserved_name_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BoundConstant(name="rate", value=1.0, ci=ConfidenceInterval(0.5, 1.5))


class TestRungResult(unittest.TestCase):
    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            RungResult(concurrency=0, throughput=1.0)
        with self.assertRaises(ValueError):
            RungResult(concurrency=1, throughput=0.0)
        with self.assertRaises(ValueError):
            RungResult(concurrency=1, throughput=1.0, throughput_se=-0.1)

    # error_rate is a fraction, so 1.5 or -0.1 must be rejected outright rather
    # than skewing the estimator's error gate.
    def test_error_rate_outside_unit_interval_raises(self) -> None:
        with self.assertRaises(ValueError):
            RungResult(concurrency=1, throughput=1.0, error_rate=1.5)
        with self.assertRaises(ValueError):
            RungResult(concurrency=1, throughput=1.0, error_rate=-0.1)

    # Loop lag is a duration: a negative sample is a clock bug, not lag.
    def test_negative_client_lag_raises(self) -> None:
        with self.assertRaises(ValueError):
            RungResult(concurrency=1, throughput=1.0, client_lag=-0.001)


class TestProbeResult(unittest.TestCase):
    def test_constant_lookup(self) -> None:
        r_sat = BoundConstant(name="r_sat", value=20.0, ci=ConfidenceInterval(19.0, 21.0))
        result = ProbeResult(rungs=(RungResult(concurrency=1, throughput=5.0),), constants={"r_sat": r_sat})
        self.assertIs(result.constant("r_sat"), r_sat)
        with self.assertRaises(KeyError):
            result.constant("n_knee")

    # Only a measured plateau is a capacity estimate; a ladder that stopped on
    # client saturation or the concurrency cap yields a lower bound.
    def test_censored_follows_stop_reason(self) -> None:
        rungs = (RungResult(concurrency=1, throughput=5.0),)
        plateau = ProbeResult(rungs=rungs, constants={}, stop_reason=ProbeStopReason.PLATEAU)
        client_bound = ProbeResult(rungs=rungs, constants={}, stop_reason=ProbeStopReason.CLIENT_BOUND)
        capped = ProbeResult(rungs=rungs, constants={}, stop_reason=ProbeStopReason.MAX_CONCURRENCY)
        unmeasurable = ProbeResult(rungs=rungs, constants={}, stop_reason=ProbeStopReason.MEASUREMENT_FAILED)
        self.assertFalse(plateau.censored)
        self.assertTrue(client_bound.censored)
        self.assertTrue(capped.censored)
        self.assertTrue(unmeasurable.censored)

    # The reported schema has to carry the evidence behind r_sat, including
    # which rungs the estimator actually fit, or a censored lower bound reads
    # exactly like a measured plateau.
    def test_to_dict_serializes_stop_reason_and_rungs(self) -> None:
        r_sat = BoundConstant(name="r_sat", value=20.0, ci=ConfidenceInterval(19.0, 21.0))
        n_knee = BoundConstant(name="n_knee", value=8.0, ci=ConfidenceInterval(7.0, 9.0))
        fitted = RungResult(concurrency=4, throughput=18.0, throughput_se=0.5, latency=0.2, error_rate=0.05, client_lag=0.002)
        rejected = RungResult(concurrency=8, throughput=19.0, littles_law_residual=0.6, error_rate=0.4)
        result = ProbeResult(
            rungs=(fitted, rejected),
            constants={"r_sat": r_sat, "n_knee": n_knee},
            stop_reason=ProbeStopReason.MAX_CONCURRENCY,
            usable_concurrencies=(4,),
        )

        payload = result.to_dict()

        self.assertEqual(payload["stop_reason"], "max_concurrency")
        self.assertTrue(payload["censored"])
        self.assertIsNone(payload["failed_concurrency"])
        self.assertEqual(
            payload["constants"],
            {
                "n_knee": {"value": 8.0, "ci": {"low": 7.0, "high": 9.0}},
                "r_sat": {"value": 20.0, "ci": {"low": 19.0, "high": 21.0}},
            },
        )
        self.assertEqual([rung["concurrency"] for rung in payload["rungs"]], [4, 8])
        self.assertTrue(payload["rungs"][0]["usable"])
        self.assertFalse(payload["rungs"][1]["usable"])
        self.assertEqual(payload["rungs"][0]["error_rate"], 0.05)
        self.assertEqual(payload["rungs"][0]["client_lag"], 0.002)
        self.assertEqual(payload["rungs"][1]["littles_law_residual"], 0.6)

    # A stop caused by an unmeasurable rung has to say which rung it was, or
    # the report reader cannot tell where the curve broke.
    def test_to_dict_reports_the_failed_concurrency(self) -> None:
        result = ProbeResult(
            rungs=(),
            constants={},
            stop_reason=ProbeStopReason.MEASUREMENT_FAILED,
            failed_concurrency=64,
        )
        payload = result.to_dict()
        self.assertEqual(payload["stop_reason"], "measurement_failed")
        self.assertEqual(payload["failed_concurrency"], 64)


class TestClassifySaturation(unittest.TestCase):
    def test_client_bound_wins(self) -> None:
        signal = classify_saturation(ttft_inflation=5.0, itl_inflation=5.0, client_lag_inflation=3.0)
        self.assertIs(signal, SaturationSignal.CLIENT_BOUND)

    def test_prefill_bound(self) -> None:
        signal = classify_saturation(ttft_inflation=4.0, itl_inflation=1.1)
        self.assertIs(signal, SaturationSignal.PREFILL_BOUND)

    def test_decode_bound(self) -> None:
        signal = classify_saturation(ttft_inflation=1.1, itl_inflation=4.0)
        self.assertIs(signal, SaturationSignal.DECODE_BOUND)

    def test_prefill_wins_ties(self) -> None:
        signal = classify_saturation(ttft_inflation=4.0, itl_inflation=4.0)
        self.assertIs(signal, SaturationSignal.PREFILL_BOUND)

    def test_unloaded_is_none(self) -> None:
        signal = classify_saturation(ttft_inflation=1.0, itl_inflation=1.0)
        self.assertIs(signal, SaturationSignal.NONE)

    def test_non_positive_ratio_raises(self) -> None:
        with self.assertRaises(ValueError):
            classify_saturation(ttft_inflation=0.0, itl_inflation=1.0)


if __name__ == "__main__":
    unittest.main()
