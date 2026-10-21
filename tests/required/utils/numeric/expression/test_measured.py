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
"""Tests for measured terms: the window functions a stop condition reads.

Parsing pins each accepted spelling to the window it selects and each
rejected one to its message. Evaluation pins every window kind's membership
and unknown rules, and every aggregate against numpy on the same samples.
"""

import math
import random
from typing import Dict, List, Optional

import numpy as np
import pytest

from inference_perf.apis import InferenceInfo, RequestLifecycleMetric, StreamedResponseMetrics
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.base import compute_request_latency_metrics, summarize_requests
from inference_perf.utils.numeric.expression import SERIES, SIGNALS, Observation, Predicate, PredicateState, Term
from inference_perf.utils.numeric.expression.measured import AGGREGATES, _TimeEvaluator, _Window, extract_terms


# The only term in a condition built as '(t >= 600) | (<term> > 0)'.
def _term(text: str) -> Term:
    terms = list(Predicate(f"(t >= 600) | ({text} > 0)").terms.values())
    assert len(terms) == 1
    return terms[0]


# A PredicateState over '(t >= 600) | (<term> > 0)', so its one term can be read back.
def _state(text: str) -> PredicateState:
    return PredicateState(Predicate(f"(t >= 600) | ({text} > 0)"))


# The single term's value in a state built by _state.
def _value(state: PredicateState, now: float) -> float:
    (value,) = state.values(now).values()
    return value


# One successful request that finished at stage time t with the given series values.
def _ok(t: float, sent_index: Optional[int] = None, **values: float) -> Observation:
    return Observation(t=t, values=values, sent_index=sent_index)


class TestParsing:
    # 'quantile_over_time(0.99, ttft[30s])' is a 30 s time window over ttft with q 0.99, no offset, no '@'.
    def test_time_window(self) -> None:
        assert _term("quantile_over_time(0.99, ttft[30s])") == Term("quantile", "ttft", "time", q=0.99, range_s=30.0)

    # PromQL durations add up their parts: 500ms is 0.5, 1m30s is 90, 2h is 7200.
    @pytest.mark.parametrize("duration, seconds", [("500ms", 0.5), ("30s", 30.0), ("1m30s", 90.0), ("2h", 7200.0)])
    def test_durations(self, duration: str, seconds: float) -> None:
        assert _term(f"avg_over_time(latency[{duration}])").range_s == seconds

    # 'offset 10s' and '@ 30' parse in either order: range 10, offset 10, ends at stage second 30.
    @pytest.mark.parametrize("modifiers", ["offset 10s @ 30", "@ 30 offset 10s"])
    def test_offset_and_at(self, modifiers: str) -> None:
        term = _term(f"avg_over_time(latency[10s] {modifiers})")
        assert (term.range_s, term.offset_s, term.at_s) == (10.0, 10.0, 30.0)

    # 'avg_over_last(latency, 100)' is the last 100 latency samples; quantile takes q first: (0.9, ttft, 50).
    def test_last_window(self) -> None:
        assert _term("avg_over_last(latency, 100)") == Term("avg", "latency", "last", last=100)
        assert _term("quantile_over_last(0.9, ttft, 50)") == Term("quantile", "ttft", "last", q=0.9, last=50)

    # 'avg_over_sent(ttft, 10, 30)' is requests 10 up to 30 in dispatch order (the 11th to 30th sent).
    def test_sent_window(self) -> None:
        assert _term("avg_over_sent(ttft, 10, 30)") == Term("avg", "ttft", "sent", sent_start=10, sent_stop=30)

    # Every aggregate name parses with every window.
    @pytest.mark.parametrize("aggregate", AGGREGATES)
    def test_every_aggregate(self, aggregate: str) -> None:
        q = "0.5, " if aggregate == "quantile" else ""
        assert _term(f"{aggregate}_over_last({q}latency, 5)").aggregate == aggregate

    # The same term written twice, even with different spacing, is one symbol evaluated once.
    def test_repeated_term_is_one_symbol(self) -> None:
        p = Predicate("(t >= 600) | (avg_over_last(ttft, 5) > 2) | (avg_over_last(ttft,5) < 0.1)")
        assert len(p.terms) == 1

    # Each term in '... (avg_over_last(ttft, 20) > 2*avg_over_sent(ttft, 10, 30))' gets its own symbol.
    def test_two_terms_in_one_comparison(self) -> None:
        p = Predicate("(t >= 900) | (avg_over_last(ttft, 20) > 2*avg_over_sent(ttft, 10, 30))")
        assert sorted(term.window for term in p.terms.values()) == ["last", "sent"]

    # A term's source is the text as written, for logs and errors.
    def test_source_is_the_text_as_written(self) -> None:
        assert _term("avg_over_time(latency[10s]  offset 5s)").source == "avg_over_time(latency[10s]  offset 5s)"

    # extract_terms swaps each term for a placeholder SymPy can parse: '(t >= 1) | (_term0 > 2)'.
    def test_extract_terms_rewrites_the_string(self) -> None:
        rest, terms = extract_terms("(t >= 1) | (avg_over_last(ttft, 5) > 2)")
        assert rest == "(t >= 1) | (_term0 > 2)"
        assert list(terms) == ["_term0"]

    # Each malformed term or leftover PromQL syntax raises with a message naming the problem.
    # A condition that does not start with '(t' is wrapped as '(t >= 600) | (<condition>)'.
    @pytest.mark.parametrize(
        "condition, message",
        [
            ("median_over_time(ttft[30s]) > 2", r"For a median use quantile\(0\.5"),
            ("rate_over_time(ttft[30s]) > 2", "unknown aggregate 'rate'"),
            ("avg_over_window(ttft[30s]) > 2", "unknown window 'window'"),
            ("avg_over_time(foo[30s]) > 2", "unknown series 'foo'"),
            ("avg_over_last(itl, 5) > 2", "unknown series 'itl'"),
            ("avg_over_time(time_to_first_token[30s]) > 2", "'time_to_first_token' is 'ttft'"),
            ("avg_over_time(ttft[30s:5s]) > 2", "subqueries"),
            ("avg_over_time(ttft[30]) > 2", "not a duration"),
            ("avg_over_time(ttft[1.5m]) > 2", "not a duration"),
            ("avg_over_time(ttft[0s]) > 2", "longer than zero"),
            ("avg_over_time(ttft[10s] offset 1s offset 2s) > 2", "two offsets"),
            ("avg_over_time(ttft[10s] @ 1 @ 2) > 2", "two '@'"),
            ("avg_over_time(ttft[10s] @ start()) > 2", "stage second the range ends at"),
            ("avg_over_time(ttft[10s] @ 1700000000x) > 2", "stage second the range ends at"),
            ("avg_over_time(ttft[10s] foo) > 2", "expected 'offset <duration>'"),
            ("sum_over_time(error[60s] @ 30) > 2", "start before the stage"),
            ("sum_over_time(error[20s] @ 30 offset 20s) > 2", "start before the stage"),
            ("avg_over_time(ttft) > 2", "needs a range selector"),
            ("avg_over_time(ttft[10s], latency[10s]) > 2", "takes one range selector"),
            ("quantile_over_time(ttft[30s]) > 2", "quantile takes the quantile first"),
            ("quantile_over_time(1.5, ttft[30s]) > 2", "between 0 and 1"),
            ("avg_over_last(ttft, 0) > 2", "positive count"),
            ("avg_over_last(ttft) > 2", "positive count"),
            ("avg_over_sent(ttft, 30, 10) > 2", "two increasing request positions"),
            ("avg_over_last(ttft, Min(5, 6)) > 2", "constants and a series"),
            ("(t >= 600) | avg_over_last(ttft, 5", "no closing parenthesis"),
            ("ttft[30s] > 2", "outside a window function"),
            ("(avg_over_last(ttft, 5) > 2) or (t >= 600)", "logical keyword"),
            ("(t >= 600) | (avg_over_last(ttft, 5) > 2) and (t >= 1)", "logical keyword"),
            ("_term0 > 2", "reserved for window functions"),
        ],
    )
    def test_rejected(self, condition: str, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            Predicate(condition if condition.startswith("(t") or " or " in condition else f"(t >= 600) | ({condition})")

    # A bare series name is not a symbol; the error says to wrap it in a window function.
    def test_bare_series_points_at_a_window_function(self) -> None:
        with pytest.raises(ValueError, match=r"'ttft' is a series: read it through a window function"):
            Predicate("(t >= 600) | (ttft > 2)")

    # A report field name as a bare symbol gets the short-name hint.
    def test_report_name_points_at_the_short_name(self) -> None:
        with pytest.raises(ValueError, match=r"Series use short names: 'request_latency' is 'latency'"):
            Predicate("(t >= 600) | (request_latency > 2)")


class TestProof:
    # '(t >= 600) | (quantile_over_time(0.99, ttft[30s]) > 2)' ends by 600 and can end from 0.
    def test_or_with_a_term(self) -> None:
        p = Predicate("(t >= 600) | (quantile_over_time(0.99, ttft[30s]) > 2)")
        assert (p.boundary, p.earliest) == (600.0, 0.0)

    # '((t >= 60) & (avg_over_last(latency, 50) > 1)) | (t >= 600)' ends by 600 and can end from 60.
    def test_term_gated_by_time(self) -> None:
        p = Predicate("((t >= 60) & (avg_over_last(latency, 50) > 1)) | (t >= 600)")
        assert (p.boundary, p.earliest) == (600.0, 60.0)

    # Signals are measured too: '(t >= 300) | (errors >= 10)' ends by 300 and from 0.
    def test_signal(self) -> None:
        p = Predicate("(t >= 300) | (errors >= 10)")
        assert (p.boundary, p.earliest) == (300.0, 0.0)
        assert p.signals == frozenset({"errors"})

    # A condition that cannot end on time alone is rejected: a term alone, and a term ANDed with time.
    @pytest.mark.parametrize(
        "condition, message",
        [
            ("avg_over_last(ttft, 5) > 2", "must also end on time alone"),
            ("(t >= 60) & (avg_over_last(ttft, 5) > 2)", "never holds on time alone"),
        ],
    )
    def test_must_end_on_time_alone(self, condition: str, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            Predicate(condition)

    # A missing or NaN measured value makes its comparison false; the time part still decides.
    def test_unknown_value_is_false(self) -> None:
        p = Predicate("(t >= 600) | (errors >= 10)")
        assert not p.holds(10.0)
        assert not p.holds(10.0, {"errors": math.nan})
        assert p.holds(10.0, {"errors": 10.0})
        assert p.holds(600.0)


class TestWindow:
    # Random adds and removes; after every step each aggregate equals numpy on the samples left.
    @pytest.mark.parametrize("aggregate", AGGREGATES)
    def test_matches_numpy_under_churn(self, aggregate: str) -> None:
        rng = random.Random(0)
        q = 0.37
        window = _Window(aggregate, q)
        held: List[float] = []
        for _ in range(2000):
            if held and rng.random() < 0.45:
                window.remove(held.pop(rng.randrange(len(held))))
            else:
                value = rng.lognormvariate(0.0, 1.0)
                held.append(value)
                window.add(value)
            arr = np.asarray(held)
            expected = {
                "avg": arr.mean() if held else math.nan,
                "min": arr.min() if held else math.nan,
                "max": arr.max() if held else math.nan,
                "sum": arr.sum() if held else math.nan,
                "count": float(len(held)) if held else math.nan,
                "stddev": arr.std() if held else math.nan,
                "quantile": np.quantile(arr, q) if held else math.nan,
            }[aggregate]
            got = window.value()
            assert (math.isnan(got) and math.isnan(expected)) or math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-9)

    # Equal samples have a stddev of exactly 0, not float noise, because the sums are exact integers.
    def test_constant_samples_have_zero_stddev(self) -> None:
        window = _Window("stddev", None)
        for _ in range(1000):
            window.add(1.1)
        assert window.value() == 0.0

    # Quantile 0 is the minimum and 1 the maximum; 0.5 of [1, 2, 3, 4] interpolates to 2.5, as PromQL and numpy do.
    def test_quantile_ends_and_interpolation(self) -> None:
        values = [4.0, 1.0, 3.0, 2.0]
        for q, expected in ((0.0, 1.0), (1.0, 4.0), (0.5, 2.5)):
            window = _Window("quantile", q)
            for v in values:
                window.add(v)
            assert window.value() == expected

    # Removal is exact: add 1e9 and 1, remove 1e9, and the avg and stddev are exactly 1 and 0.
    def test_removal_is_exact(self) -> None:
        for aggregate, expected in (("avg", 1.0), ("stddev", 0.0)):
            window = _Window(aggregate, None)
            window.add(1e9)
            window.add(1.0)
            window.remove(1e9)
            assert window.value() == expected

    # A non-finite value is dropped: latencies [inf, 2.0] average to 2.0 over a window of 1.
    def test_non_finite_values_are_dropped(self) -> None:
        state = _state("avg_over_last(latency, 1)")
        state.observe(_ok(1.0, latency=math.inf), 1.0)
        assert math.isnan(_value(state, 1.0))
        state.observe(_ok(2.0, latency=2.0), 2.0)
        assert _value(state, 2.0) == 2.0


class TestTimeWindow:
    # 'avg_over_time(latency[10s])' is unknown until t = 10 (the range must lie inside the stage), then averages.
    def test_unknown_until_range_inside_stage(self) -> None:
        state = _state("avg_over_time(latency[10s])")
        state.observe(_ok(2.0, latency=1.0), 2.0)
        assert math.isnan(_value(state, 9.9))
        assert _value(state, 10.0) == 1.0

    # Samples at 1, 5 and 9 s with a 10 s range: at t=10 all three count; at t=12 the one at 1 s has left (range is (2, 12]).
    def test_samples_leave_after_range(self) -> None:
        state = _state("count_over_time(latency[10s])")
        for t in (1.0, 5.0, 9.0):
            state.observe(_ok(t, latency=1.0), t)
        assert _value(state, 10.0) == 3.0
        assert _value(state, 12.0) == 2.0

    # The range is right-closed: a sample that finished exactly at the range's end is in it, with and without offset.
    @pytest.mark.parametrize("modifier, now", [("", 10.0), (" offset 5s", 15.0)])
    def test_range_is_right_closed(self, modifier: str, now: float) -> None:
        state = _state(f"count_over_time(latency[10s]{modifier})")
        state.observe(_ok(10.0, latency=1.0), now)
        assert _value(state, now) == 1.0

    # The range is left-open: a sample exactly range seconds old has left. Sample at 5 s, range 5 s: gone at t=10.
    def test_range_is_left_open(self) -> None:
        state = _state("count_over_time(latency[5s])")
        state.observe(_ok(5.0, latency=1.0), 5.0)
        assert _value(state, 9.99) == 1.0
        assert math.isnan(_value(state, 10.0))

    # 'offset 10s' reads the 10 s before the last 10 s: at t=20, samples in (0, 10] count and the one at 15 s waits.
    def test_offset(self) -> None:
        state = _state("sum_over_time(latency[10s] offset 10s)")
        state.observe(_ok(5.0, latency=1.0), 5.0)
        state.observe(_ok(15.0, latency=100.0), 15.0)
        assert _value(state, 20.0) == 1.0
        assert _value(state, 30.0) == 100.0

    # '@ 30' fixes the range to (20, 30]: unknown before t=30, then fixed; a sample finishing after 30 never enters.
    def test_at_fixes_the_range(self) -> None:
        state = _state("sum_over_time(latency[10s] @ 30)")
        state.observe(_ok(25.0, latency=2.0), 25.0)
        assert math.isnan(_value(state, 29.0))
        assert _value(state, 30.0) == 2.0
        state.observe(_ok(35.0, latency=100.0), 35.0)
        assert _value(state, 500.0) == 2.0

    # A sample that finishes after a fixed '@' end is not kept: 1000 of them leave nothing waiting.
    def test_at_drops_samples_past_its_end(self) -> None:
        state = _state("sum_over_time(latency[10s] @ 30)")
        for i in range(1000):
            state.observe(_ok(31.0 + i, latency=1.0), 31.0 + i)
        (evaluator,) = state._evaluators.values()
        assert isinstance(evaluator, _TimeEvaluator) and not evaluator._pending

    # A sample observed late but finished inside the range still counts: it finished at 8 s, observed at 12 s, range (2, 12].
    def test_late_observation_counts_by_finish_time(self) -> None:
        state = _state("count_over_time(latency[10s])")
        state.observe(_ok(11.0, latency=1.0), 11.0)
        state.observe(_ok(8.0, latency=1.0), 12.0)
        assert _value(state, 12.0) == 2.0

    # A range with no samples yet is unknown, not zero: count over an empty but full range is NaN.
    def test_empty_range_is_unknown(self) -> None:
        state = _state("count_over_time(latency[10s])")
        assert math.isnan(_value(state, 50.0))


class TestLastWindow:
    # 'avg_over_last(latency, 3)' is unknown with 2 samples, then the mean of the newest 3: [1, 2, 3] -> 2, then [2, 3, 10] -> 5.
    def test_slides_over_the_last_n(self) -> None:
        state = _state("avg_over_last(latency, 3)")
        state.observe(_ok(1.0, latency=1.0), 1.0)
        state.observe(_ok(2.0, latency=2.0), 2.0)
        assert math.isnan(_value(state, 2.0))
        state.observe(_ok(3.0, latency=3.0), 3.0)
        assert _value(state, 3.0) == 2.0
        state.observe(_ok(4.0, latency=10.0), 4.0)
        assert _value(state, 4.0) == 5.0

    # A request without the series adds nothing: a failed request has no latency, so 2 successes never fill a window of 3.
    def test_requests_without_the_series_do_not_count(self) -> None:
        state = _state("avg_over_last(latency, 3)")
        state.observe(_ok(1.0, latency=1.0), 1.0)
        state.observe(Observation(t=2.0, failed=True), 2.0)
        state.observe(_ok(3.0, latency=1.0), 3.0)
        assert math.isnan(_value(state, 3.0))

    # error is 1 for a failed request and 0 otherwise: 1 failure in 4 is an error rate of 0.25.
    def test_error_series_follows_failed(self) -> None:
        state = _state("avg_over_last(error, 4)")
        for i, failed in enumerate((False, True, False, False)):
            state.observe(Observation(t=float(i), failed=failed), float(i))
        assert _value(state, 4.0) == 0.25


class TestSentWindow:
    # 'avg_over_sent(latency, 1, 3)' waits for requests 1 and 2 in any finish order, ignoring 0 and 3, then averages them.
    def test_waits_for_every_request_in_range(self) -> None:
        state = _state("avg_over_sent(latency, 1, 3)")
        state.observe(_ok(1.0, sent_index=2, latency=4.0), 1.0)
        state.observe(_ok(2.0, sent_index=0, latency=100.0), 2.0)
        state.observe(_ok(3.0, sent_index=3, latency=100.0), 3.0)
        assert math.isnan(_value(state, 3.0))
        state.observe(_ok(4.0, sent_index=1, latency=2.0), 4.0)
        assert _value(state, 4.0) == 3.0

    # A failed request in range counts as finished but has no latency: requests 0 (2.0) and 1 (failed) give 2.0.
    def test_failed_request_completes_without_a_sample(self) -> None:
        state = _state("avg_over_sent(latency, 0, 2)")
        state.observe(_ok(1.0, sent_index=0, latency=2.0), 1.0)
        state.observe(Observation(t=2.0, failed=True, sent_index=1), 2.0)
        assert _value(state, 2.0) == 2.0

    # The same index twice counts once: request 0 (2.0) seen twice, then request 1 (8.0), averages to 5.0, not 4.0.
    def test_duplicate_index_counts_once(self) -> None:
        state = _state("avg_over_sent(latency, 0, 2)")
        state.observe(_ok(1.0, sent_index=0, latency=2.0), 1.0)
        state.observe(_ok(2.0, sent_index=0, latency=2.0), 2.0)
        assert math.isnan(_value(state, 2.0))
        state.observe(_ok(3.0, sent_index=1, latency=8.0), 3.0)
        assert _value(state, 3.0) == 5.0

    # A request without a sent_index cannot be placed and is ignored by _over_sent terms.
    def test_no_index_is_ignored(self) -> None:
        state = _state("avg_over_sent(latency, 0, 1)")
        state.observe(_ok(1.0, latency=2.0), 1.0)
        assert math.isnan(_value(state, 1.0))


class TestSignals:
    # completed and errors count every finished request: 3 finished, 1 failed.
    def test_counters(self) -> None:
        state = PredicateState(Predicate("(t >= 600) | (errors > completed)"))
        state.observe(_ok(1.0), 1.0)
        state.observe(Observation(t=2.0, failed=True), 2.0)
        state.observe(_ok(3.0), 3.0)
        assert state.values(3.0) == {"completed": 3.0, "errors": 1.0}

    # since_last_completion counts from the stage start before any request finishes, and from the latest finish after.
    def test_since_last_completion(self) -> None:
        state = PredicateState(Predicate("(t >= 600) | (since_last_completion > 30)"))
        assert state.values(31.0) == {"since_last_completion": 31.0}
        state.observe(_ok(40.0), 41.0)
        assert state.values(50.0) == {"since_last_completion": 10.0}

    # in_flight comes from the caller; without it the comparison is unknown and false.
    def test_in_flight_from_caller(self) -> None:
        state = PredicateState(Predicate("(t >= 600) | (in_flight > 5)"))
        assert math.isnan(state.values(1.0)["in_flight"])
        assert not state.holds(1.0)
        assert state.holds(1.0, in_flight=6)

    # Only the signals a condition reads are computed: a condition on errors gets no completed entry.
    def test_only_read_signals(self) -> None:
        state = PredicateState(Predicate("(t >= 600) | (errors > 1)"))
        assert set(state.values(1.0)) == {"errors"}


class TestPredicateState:
    # '(t >= 600) | (avg_over_last(error, 4) > 0.5)' holds on exactly the result that pushes the rate past 0.5.
    def test_holds_on_the_flipping_result(self) -> None:
        state = PredicateState(Predicate("(t >= 600) | (avg_over_last(error, 4) > 0.5)"))
        results = [False, True, True, False, True]
        held = []
        for i, failed in enumerate(results):
            state.observe(Observation(t=float(i), failed=failed), float(i))
            held.append(state.holds(float(i)))
        # Window [F, T, T, F] = 0.5 does not hold; [T, T, F, T] = 0.75 does.
        assert held == [False, False, False, False, True]

    # readings key each value by the term as written, for logs.
    def test_readings_use_source_text(self) -> None:
        state = PredicateState(Predicate("(t >= 600) | (avg_over_last(latency, 1) > 2) | (errors > 3)"))
        state.observe(_ok(1.0, latency=5.0), 1.0)
        assert state.readings(1.0) == {"avg_over_last(latency, 1)": 5.0, "errors": 0.0}

    # The clock never goes backwards: reading at 20 then at 5 keeps the time range at its t=20 position.
    def test_clock_does_not_go_backwards(self) -> None:
        state = _state("count_over_time(latency[10s])")
        state.observe(_ok(1.0, latency=1.0), 1.0)
        state.observe(_ok(15.0, latency=1.0), 15.0)
        assert _value(state, 20.0) == 1.0
        assert _value(state, 5.0) == 1.0

    # A condition over t alone has no terms and no signals, and its state only answers the deadline.
    def test_time_only_condition(self) -> None:
        state = PredicateState(Predicate("t >= 60"))
        assert state.values(1.0) == {}
        assert not state.holds(59.0)
        assert state.holds(60.0)


# Every series that names a report field names one the report computes: request-level fields
# from compute_request_latency_metrics, and schedule_delay from the load summary.
def test_series_report_fields_exist_in_the_report() -> None:
    metric = RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=0.0,
        start_time=1.0,
        end_time=2.0,
        request_data="",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=5)),
            response_metrics=StreamedResponseMetrics(output_tokens=3, output_token_times=[1.2, 1.5, 1.9]),
        ),
        error=None,
    )
    fields = set(compute_request_latency_metrics(metric)) | set(summarize_requests([metric], [50.0]).load_summary)
    named = {s.report_field for s in SERIES.values() if s.report_field is not None}
    assert named <= fields, named - fields


# Series and signals never share a name, so a bare name always means one of them.
def test_series_and_signals_are_disjoint() -> None:
    assert not set(SERIES) & set(SIGNALS)


# Values used by the evaluators are floats whatever the caller passes: an int latency of 2 averages to 2.0.
def test_int_values_are_accepted() -> None:
    state = _state("avg_over_last(latency, 1)")
    values: Dict[str, float] = {"latency": 2}
    state.observe(Observation(t=1.0, values=values), 1.0)
    assert _value(state, 1.0) == 2.0
