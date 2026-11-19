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

from inference_perf.loadgen.probe import cusum_drift_index, is_stationary


class TestCusumDriftIndex(unittest.TestCase):
    def test_constant_series_is_stationary(self) -> None:
        self.assertIsNone(cusum_drift_index([5.0] * 16))
        self.assertTrue(is_stationary([5.0] * 16))

    def test_stationary_noise_is_stationary(self) -> None:
        rng = np.random.default_rng(42)
        series = rng.normal(10.0, 1.0, 64)
        self.assertTrue(is_stationary(series))

    def test_warmup_transient_detected_early(self) -> None:
        series = [0.0] * 5 + [10.0] * 27
        index = cusum_drift_index(series)
        self.assertIsNotNone(index)
        assert index is not None
        self.assertLess(index, 8)
        self.assertFalse(is_stationary(series))

    def test_ramp_detected(self) -> None:
        series = np.linspace(0.0, 20.0, 64)
        self.assertIsNotNone(cusum_drift_index(series))
        self.assertFalse(is_stationary(series))

    def test_short_series_raises_and_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            cusum_drift_index([1.0, 2.0, 3.0])
        self.assertFalse(is_stationary([1.0, 2.0, 3.0]))

    def test_non_finite_raises(self) -> None:
        with self.assertRaises(ValueError):
            cusum_drift_index([1.0, float("nan"), 1.0, 1.0])

    def test_two_dimensional_raises(self) -> None:
        with self.assertRaises(ValueError):
            cusum_drift_index(np.ones((4, 4)))
        self.assertFalse(is_stationary(np.ones((4, 4))))


if __name__ == "__main__":
    unittest.main()
