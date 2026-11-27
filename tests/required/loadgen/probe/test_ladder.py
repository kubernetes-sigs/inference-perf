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

from inference_perf.loadgen.probe import ConcurrencyLadder, LadderConfig, RungResult, SaturationSignal


def saturating_rung(
    concurrency: int,
    mu: float = 20.0,
    knee_constant: float = 2.0,
    stationary: bool = True,
    signal: SaturationSignal = SaturationSignal.NONE,
) -> RungResult:
    throughput = mu * concurrency / (knee_constant + concurrency)
    return RungResult(concurrency=concurrency, throughput=throughput, throughput_se=0.0, stationary=stationary, signal=signal)


class TestLadderConfig(unittest.TestCase):
    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            LadderConfig(start_concurrency=0)
        with self.assertRaises(ValueError):
            LadderConfig(growth_factor=1.0)
        with self.assertRaises(ValueError):
            LadderConfig(max_concurrency=2, start_concurrency=4)
        with self.assertRaises(ValueError):
            LadderConfig(gain_threshold=0.0)


class TestConcurrencyLadder(unittest.TestCase):
    def test_starts_at_configured_concurrency(self) -> None:
        ladder = ConcurrencyLadder(config=LadderConfig(start_concurrency=2))
        self.assertEqual(ladder.next_concurrency([]), 2)

    def test_grows_geometrically_until_cap(self) -> None:
        # Linear throughput never plateaus, so the ladder doubles to the cap.
        ladder = ConcurrencyLadder(config=LadderConfig(max_concurrency=16, refine=False))
        history: list[RungResult] = []
        visited: list[int] = []
        concurrency = ladder.next_concurrency(history)
        while concurrency is not None:
            visited.append(concurrency)
            history.append(RungResult(concurrency=concurrency, throughput=2.0 * concurrency, throughput_se=0.0))
            concurrency = ladder.next_concurrency(history)
        self.assertEqual(visited, [1, 2, 4, 8, 16])
        self.assertEqual(ladder.stop_reason, "max_concurrency")

    def test_plateau_stops_with_refinement_midpoint(self) -> None:
        ladder = ConcurrencyLadder()
        history: list[RungResult] = []
        visited: list[int] = []
        concurrency = ladder.next_concurrency(history)
        while concurrency is not None:
            visited.append(concurrency)
            history.append(saturating_rung(concurrency))
            concurrency = ladder.next_concurrency(history)
        self.assertEqual(ladder.stop_reason, "plateau")
        # Geometric ascent, then exactly one non-doubling refinement rung
        # between the top two ladder steps.
        self.assertEqual(visited[:-1], [2**i for i in range(len(visited) - 1)])
        top, second = visited[-2], visited[-3]
        self.assertGreater(visited[-1], second)
        self.assertLess(visited[-1], top)

    def test_plateau_without_refinement(self) -> None:
        ladder = ConcurrencyLadder(config=LadderConfig(refine=False))
        history: list[RungResult] = []
        concurrency = ladder.next_concurrency(history)
        while concurrency is not None:
            history.append(saturating_rung(concurrency))
            concurrency = ladder.next_concurrency(history)
        self.assertEqual(ladder.stop_reason, "plateau")
        self.assertEqual([r.concurrency for r in history], [2**i for i in range(len(history))])

    def test_noise_defers_plateau(self) -> None:
        # A gain of 0.4 on 18.0 is within the 5% threshold, but wide standard
        # errors make it unconfident, so the ladder keeps climbing.
        ladder = ConcurrencyLadder()
        history = [
            RungResult(concurrency=8, throughput=18.0, throughput_se=1.0),
            RungResult(concurrency=16, throughput=18.4, throughput_se=1.0),
        ]
        self.assertEqual(ladder.next_concurrency(history), 32)

    def test_non_stationary_rung_retried_once(self) -> None:
        ladder = ConcurrencyLadder()
        history = [saturating_rung(1), saturating_rung(2), saturating_rung(4, stationary=False)]
        self.assertEqual(ladder.next_concurrency(history), 4)
        history.append(saturating_rung(4, stationary=False))
        self.assertEqual(ladder.next_concurrency(history), 8)

    def test_client_bound_stops_immediately(self) -> None:
        ladder = ConcurrencyLadder()
        history = [saturating_rung(1), saturating_rung(2, signal=SaturationSignal.CLIENT_BOUND)]
        self.assertIsNone(ladder.next_concurrency(history))
        self.assertEqual(ladder.stop_reason, "client_bound")

    def test_stop_is_sticky(self) -> None:
        ladder = ConcurrencyLadder()
        history = [saturating_rung(1, signal=SaturationSignal.CLIENT_BOUND)]
        self.assertIsNone(ladder.next_concurrency(history))
        self.assertIsNone(ladder.next_concurrency([saturating_rung(1)]))
        self.assertEqual(ladder.stop_reason, "client_bound")


if __name__ == "__main__":
    unittest.main()
