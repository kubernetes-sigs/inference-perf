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


class TestProbeResult(unittest.TestCase):
    def test_constant_lookup(self) -> None:
        r_sat = BoundConstant(name="r_sat", value=20.0, ci=ConfidenceInterval(19.0, 21.0))
        result = ProbeResult(rungs=(RungResult(concurrency=1, throughput=5.0),), constants={"r_sat": r_sat})
        self.assertIs(result.constant("r_sat"), r_sat)
        with self.assertRaises(KeyError):
            result.constant("n_knee")


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
