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
"""Tests for ``Predicate``, the boolean condition used for stop conditions.

A stop condition is checked at discrete instants, so the only shape that is
safe to accept is one that is false at t=0 and then holds forever from some
boundary time on. These tests pin that acceptance rule, the exact boundary
each accepted shape compiles to, and the message for each rejected shape.
"""

import pytest
from sympy import Interval, oo

from inference_perf.utils.numeric.expression import Expression, Predicate
from inference_perf.utils.numeric.expression import expression as expression_module


class TestAccepted:
    # 't >= 60' compiles to boundary 60.0; false at t=59, true at t=60 and t=61.
    def test_simple_threshold(self) -> None:
        p = Predicate("t >= 60")
        assert p.boundary == 60.0
        assert p.free_symbols == {"t"}
        assert not p.holds(59)
        assert p.holds(60)
        assert p.holds(61)

    # 't > 60' has the same boundary 60.0 but is false AT t=60 and true at t=60.001.
    def test_strict_threshold_excludes_boundary(self) -> None:
        p = Predicate("t > 60")
        assert p.boundary == 60.0
        assert not p.holds(60)
        assert p.holds(60.001)

    # '60 <= t' is the same condition written the other way round; boundary 60.0.
    def test_reversed_comparison(self) -> None:
        assert Predicate("60 <= t").boundary == 60.0

    # '2*t + 1 >= 121' and 't**2 >= 3600' both solve to t >= 60; boundary 60.0 for each.
    def test_polynomial_forms_solve_to_boundary(self) -> None:
        assert Predicate("2*t + 1 >= 121").boundary == 60.0
        assert Predicate("t**2 >= 3600").boundary == 60.0

    # Every allowed non-polynomial node solves to the boundary it should:
    # exp(t) >= 100 -> ln(100), log(t) >= 3 -> e**3, sqrt(t) >= 5 -> 25, 1/t <= 0.1 -> 10,
    # 2**t >= 1024 -> 10, Max(t, 10) >= 60 -> 60, Min(t, 100) >= 60 -> 60.
    @pytest.mark.parametrize(
        "raw, boundary",
        [
            ("exp(t) >= 100", 4.605170185988092),
            ("log(t) >= 3", 20.085536923187668),
            ("sqrt(t) >= 5", 25.0),
            ("1/t <= 0.1", 10.0),
            ("2**t >= 1024", 10.0),
            ("Max(t, 10) >= 60", 60.0),
            ("Min(t, 100) >= 60", 60.0),
        ],
    )
    def test_allowed_non_polynomial_nodes(self, raw: str, boundary: float) -> None:
        assert Predicate(raw).boundary == pytest.approx(boundary)

    # '(t >= 60) | (t >= 30)' holds from whichever comes first: boundary 30.0, true at t=45.
    def test_or_takes_earliest(self) -> None:
        p = Predicate("(t >= 60) | (t >= 30)")
        assert p.boundary == 30.0
        assert p.holds(45)

    # '(t >= 60) & (t >= 30)' holds only once both do: boundary 60.0, false at t=45.
    def test_and_takes_latest(self) -> None:
        p = Predicate("(t >= 60) & (t >= 30)")
        assert p.boundary == 60.0
        assert not p.holds(45)

    # holds() accepts an int and a float alike; 't >= 60' at t=60 (int) and t=60.0 (float) are both True.
    def test_holds_accepts_int_and_float(self) -> None:
        p = Predicate("t >= 60")
        assert p.holds(60) is True
        assert p.holds(60.0) is True

    # repr reproduces the raw string so a rejected config is easy to locate in logs.
    def test_repr(self) -> None:
        assert repr(Predicate("t >= 60")) == "Predicate('t >= 60')"


class TestRejectedShapes:
    # Equality holds at a single instant a discrete check can step over. All five spellings,
    # 't = 60', 't == 60', 't != 60', 'Eq(t, 60)' and 'Ne(t, 60)', get the same "uses equality" error.
    @pytest.mark.parametrize("raw", ["t = 60", "t == 60", "t != 60", "Eq(t, 60)", "Ne(t, 60)"])
    def test_equality_in_any_spelling(self, raw: str) -> None:
        with pytest.raises(ValueError, match="uses equality"):
            Predicate(raw)

    # 't < 60' is already true at t=0 and lapses at 60; rejected as holding only on [0, 60), in interval notation.
    def test_lapsing_condition(self) -> None:
        with pytest.raises(ValueError, match=r"holds only on \[0, 60\);"):
            Predicate("t < 60")

    # '(t >= 60) & (t < 120)' is a window; rejected as holding only on [60, 120).
    def test_window(self) -> None:
        with pytest.raises(ValueError, match=r"holds only on \[60, 120\);"):
            Predicate("(t >= 60) & (t < 120)")

    # 't >= 0', 't > 0' and 't >= -5' all hold from the start; each is rejected as already holding at t=0.
    @pytest.mark.parametrize("raw", ["t >= 0", "t > 0", "t >= -5"])
    def test_already_holding_at_start(self, raw: str) -> None:
        with pytest.raises(ValueError, match="already holds at"):
            Predicate(raw)

    # '(t < 30) | (t >= 60)' is true, lapses, then holds again; rejected naming both pieces: '[0, 30) or [60, infinity)'.
    def test_lapsing_then_holding_again(self) -> None:
        with pytest.raises(ValueError, match=r"holds only on \[0, 30\) or \[60, infinity\);"):
            Predicate("(t < 30) | (t >= 60)")

    # '(t > 60.5) & (t <= 90)' has open and fractional ends; rejected as holding only on (60.5, 90].
    def test_window_with_open_fractional_end(self) -> None:
        with pytest.raises(ValueError, match=r"holds only on \(60\.5, 90\];"):
            Predicate("(t > 60.5) & (t <= 90)")

    # 't < -5' is never true for t >= 0; rejected as never holding.
    def test_never_holds(self) -> None:
        with pytest.raises(ValueError, match="never holds"):
            Predicate("t < -5")

    # 'Min(t, 50) >= 60' caps at 50 so it is never true; rejected as never holding.
    def test_capped_never_holds(self) -> None:
        with pytest.raises(ValueError, match="never holds"):
            Predicate("Min(t, 50) >= 60")

    # A solver that answers wrongly is caught by direct evaluation: with the solved set forced to [30, oo)
    # for 't >= 60', t=45 evaluates False inside the claimed set, so the condition is rejected.
    def test_wrong_solver_answer_is_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(expression_module, "_holding_set", lambda expr: Interval(30, oo))
        with pytest.raises(ValueError, match="disagrees with direct evaluation"):
            Predicate("t >= 60")


class TestRejectedNodes:
    # 'sin(t) > 0' (solver returns only (0, pi)), 'Abs(t - 30) >= 60' (solver wrong on nested Abs) and
    # 'floor(t) >= 60' (unsolvable) use nodes outside the allowlist; each is rejected naming the node.
    @pytest.mark.parametrize("raw, node", [("sin(t) > 0", "sin"), ("Abs(t - 30) >= 60", "Abs"), ("floor(t) >= 60", "floor")])
    def test_node_outside_allowlist(self, raw: str, node: str) -> None:
        with pytest.raises(ValueError, match=f"uses '{node}', which is not allowed"):
            Predicate(raw)

    # 't**t >= 5' has t in both base and exponent; rejected by the power rule. '2**t' (constant base) is accepted above.
    def test_power_with_t_in_base_and_exponent(self) -> None:
        with pytest.raises(ValueError, match="a power needs a constant exponent or a positive constant base"):
            Predicate("t**t >= 5")


class TestRejectedInputs:
    # 't + 60' and '60' are numeric values, not conditions; each is rejected asking for a comparison.
    @pytest.mark.parametrize("raw", ["t + 60", "60"])
    def test_numeric_value_is_not_a_condition(self, raw: str) -> None:
        with pytest.raises(ValueError, match="is not a condition"):
            Predicate(raw)

    # 'Normal(0, 1) >= t' draws a random variable; rejected because a stop condition must be reproducible.
    def test_random_variable(self) -> None:
        with pytest.raises(ValueError, match="random variable"):
            Predicate("Normal(0, 1) >= t")

    # 'x >= 60' uses a symbol the grammar does not know; rejected naming ['x'].
    def test_unknown_symbol(self) -> None:
        with pytest.raises(ValueError, match=r"disallowed symbol\(s\) \['x'\]"):
            Predicate("x >= 60")

    # 'Foo(t) >= 1' calls an unknown function; rejected naming ['Foo'].
    def test_unknown_function(self) -> None:
        with pytest.raises(ValueError, match=r"unknown function\(s\): \['Foo'\]"):
            Predicate("Foo(t) >= 1")

    # 't >= 60 & t < 120' without parentheses parses as 't >= (60 & t) < 120' and fails; the error carries the parenthesise hint.
    def test_unparenthesised_and_gets_hint(self) -> None:
        with pytest.raises(ValueError, match="parenthesise each one"):
            Predicate("t >= 60 & t < 120")

    # A non-string such as 60 or True is a TypeError, not a ValueError: the config layer should never pass one.
    @pytest.mark.parametrize("raw", [60, True])
    def test_non_string_is_type_error(self, raw: object) -> None:
        with pytest.raises(TypeError):
            Predicate(raw)  # type: ignore[arg-type]


class TestExpressionBoundary:
    # Expression('t >= 60') is a condition handed to the numeric grammar; rejected pointing at Predicate.
    def test_expression_rejects_condition(self) -> None:
        with pytest.raises(ValueError, match="use Predicate"):
            Expression("t >= 60")
