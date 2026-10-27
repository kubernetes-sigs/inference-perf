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
"""Tests for ``Predicate``, the boolean condition a field waits on.

A condition is checked at discrete instants, so it must be false at t=0 and
carry a deadline: a top-level ``t >= N`` alternative that holds from N on
whatever else the condition reads. These tests pin that acceptance rule, the
deadline each accepted shape compiles to, and the message for each rejected
shape.
"""

import pytest

from inference_perf.utils.numeric.expression import Expression, Predicate


class TestAccepted:
    # 't >= 60' has deadline 60.0; false at t=59, true at t=60 and t=61.
    def test_simple_threshold(self) -> None:
        p = Predicate("t >= 60")
        assert p.deadline == 60.0
        assert p.free_symbols == {"t"}
        assert not p.holds(59)
        assert p.holds(60)
        assert p.holds(61)

    # 't > 60' has the same deadline 60.0 but is false AT t=60 and true at t=60.001.
    def test_strict_threshold_excludes_deadline(self) -> None:
        p = Predicate("t > 60")
        assert p.deadline == 60.0
        assert not p.holds(60)
        assert p.holds(60.001)

    # '60 <= t' is the same condition written the other way round; deadline 60.0.
    def test_reversed_comparison(self) -> None:
        assert Predicate("60 <= t").deadline == 60.0

    # The deadline may be any constant expression: 't >= 2*60' -> 120, 't >= sqrt(3600)' -> 60, 't >= log(100)' -> 4.605.
    @pytest.mark.parametrize(
        "raw, deadline", [("t >= 2*60", 120.0), ("t >= sqrt(3600)", 60.0), ("t >= log(100)", 4.605170185988092)]
    )
    def test_constant_expression_deadline(self, raw: str, deadline: float) -> None:
        assert Predicate(raw).deadline == pytest.approx(deadline)

    # '(t >= 60) | (t >= 30)' holds from whichever comes first: deadline 30.0, true at t=45.
    def test_or_takes_earliest(self) -> None:
        p = Predicate("(t >= 60) | (t >= 30)")
        assert p.deadline == 30.0
        assert p.holds(45)

    # The other alternatives can be anything evaluable and only make the condition hold sooner:
    # '(t >= 600) | (sqrt(t) >= 8)' has deadline 600.0, is false at t=63 and true at t=64.
    def test_other_alternatives_hold_sooner(self) -> None:
        p = Predicate("(t >= 600) | (sqrt(t) >= 8)")
        assert p.deadline == 600.0
        assert not p.holds(63)
        assert p.holds(64)

    # An alternative may be a window: '(t >= 600) | ((t >= 60) & (t < 120))' holds at t=100, not at t=130, and from 600.
    def test_window_alternative(self) -> None:
        p = Predicate("(t >= 600) | ((t >= 60) & (t < 120))")
        assert p.deadline == 600.0
        assert p.holds(100)
        assert not p.holds(130)
        assert p.holds(600)

    # Nothing is solved for, so any function the grammar knows is fine in an alternative:
    # '(t >= 600) | (Abs(t - 300) < 10)' is false at t=0, true at t=295; '(t >= 600) | (sin(t) > 0)' true at t=1.
    def test_any_known_function_in_an_alternative(self) -> None:
        p = Predicate("(t >= 600) | (Abs(t - 300) < 10)")
        assert p.deadline == 600.0
        assert not p.holds(0)
        assert p.holds(295)
        assert Predicate("(t >= 600) | (sin(t) > 0)").holds(1)

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

    # Without a top-level 't >= N' alternative there is no deadline, whatever the shape: a lapsing 't < 60',
    # a window '(t >= 60) & (t < 120)', an '&' of thresholds, an implied threshold 'sqrt(t) >= 8' or
    # 't**2 >= 3600', a capped 'Min(t, 50) >= 60' and a never-true 't < -5'. The hint is 't >= 60'.
    @pytest.mark.parametrize(
        "raw",
        [
            "t < 60",
            "(t >= 60) & (t < 120)",
            "(t >= 60) & (t >= 30)",
            "sqrt(t) >= 8",
            "t**2 >= 3600",
            "Min(t, 50) >= 60",
            "t < -5",
        ],
    )
    def test_no_deadline(self, raw: str) -> None:
        with pytest.raises(ValueError, match=r"has no deadline; .* e\.g\. 't >= 60'\."):
            Predicate(raw)

    # 't >= 0', 't > 0' and 't >= -5' have a deadline at or before the start; each is rejected as already holding at t=0.
    @pytest.mark.parametrize("raw", ["t >= 0", "t > 0", "t >= -5"])
    def test_deadline_not_after_start(self, raw: str) -> None:
        with pytest.raises(ValueError, match=r"already holds at \(or immediately after\) t=0"):
            Predicate(raw)

    # '(t >= 600) | (t < 300)' has deadline 600 but another alternative is true at t=0; rejected by direct evaluation.
    def test_another_alternative_holds_at_start(self) -> None:
        with pytest.raises(ValueError, match="already holds at t=0"):
            Predicate("(t >= 600) | (t < 300)")


class TestRejectedInputs:
    # 't + 60', '60', 't' and '1*t' are numeric values, not conditions; each is rejected asking for a comparison.
    # A bare 't' is the trap: sympy's Symbol subclasses Boolean, so a naive check took it for a condition.
    @pytest.mark.parametrize("raw", ["t + 60", "60", "t", "1*t"])
    def test_numeric_value_is_not_a_condition(self, raw: str) -> None:
        with pytest.raises(ValueError, match="is not a condition"):
            Predicate(raw)

    # 'Normal(0, 1) >= t' draws a random variable; rejected because a condition must be the same on every run.
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
