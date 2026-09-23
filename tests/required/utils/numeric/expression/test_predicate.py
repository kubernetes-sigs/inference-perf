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

from inference_perf.utils.numeric.expression import Expression, Predicate


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
    # 'Eq(t, 60)' holds only at the single instant t=60, which a discrete check can skip; rejected naming {60}.
    def test_equality_single_instant(self) -> None:
        with pytest.raises(ValueError, match=r"holds only on \{60\}"):
            Predicate("Eq(t, 60)")

    # 't == 60' is Python structural equality and parses to a constant False; rejected with the '==' hint.
    def test_python_double_equals(self) -> None:
        with pytest.raises(ValueError, match="'==' compares structure"):
            Predicate("t == 60")

    # 't < 60' is already true at t=0 and lapses at 60; rejected as holding only on [0, 60).
    def test_lapsing_condition(self) -> None:
        with pytest.raises(ValueError, match="holds only on Interval.Ropen\\(0, 60\\)"):
            Predicate("t < 60")

    # '(t >= 60) & (t < 120)' is a window; rejected as holding only on [60, 120).
    def test_window(self) -> None:
        with pytest.raises(ValueError, match="holds only on Interval.Ropen\\(60, 120\\)"):
            Predicate("(t >= 60) & (t < 120)")

    # 'Ne(t, 60)' holds everywhere except one instant; rejected because the holding set is a Union, not [b, oo).
    def test_not_equal(self) -> None:
        with pytest.raises(ValueError, match="holds only on Union"):
            Predicate("Ne(t, 60)")

    # 't >= 0', 't > 0' and 't >= -5' all hold from the start; each is rejected as already holding at t=0.
    @pytest.mark.parametrize("raw", ["t >= 0", "t > 0", "t >= -5"])
    def test_already_holding_at_start(self, raw: str) -> None:
        with pytest.raises(ValueError, match="already holds at"):
            Predicate(raw)

    # 't < -5' is never true for t >= 0; rejected as never holding.
    def test_never_holds(self) -> None:
        with pytest.raises(ValueError, match="never holds"):
            Predicate("t < -5")

    # 'sin(t) > 0' and 'exp(t) >= 10' are not polynomial in t, so the solver is not trusted; both rejected as unprovable.
    @pytest.mark.parametrize("raw", ["sin(t) > 0", "exp(t) >= 10"])
    def test_non_polynomial_not_proved(self, raw: str) -> None:
        with pytest.raises(ValueError, match="could not be proved"):
            Predicate(raw)


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
