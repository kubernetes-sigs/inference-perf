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

A condition is checked at discrete instants, so the only shape that is
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

    # 'log(t + 1) >= 871' solves to exp(871) - 1, past the largest float; rejected naming the time rather than
    # compiling to an infinite boundary.
    def test_boundary_past_float_range(self) -> None:
        with pytest.raises(ValueError, match=r"holds only from t = -1 \+ exp\(871\), which is too large for a float"):
            Predicate("log(t + 1) >= 871")


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


# 50 clauses joined by '|': 36 'errors >= N', then `planted` as clause 37, then 13 'completed >= N'.
def _fifty_clauses(planted: str) -> str:
    before = [f"(errors >= {n})" for n in range(1, 37)]
    after = [f"(completed >= {n})" for n in range(1, 14)]
    return " | ".join(before + [planted] + after)


class TestErrorsNameTheFault:
    # One faulty clause planted as clause 37 of 50. Each message names the fault, places it in
    # "clause 37 of 50" and stays under 450 characters although the condition is over 800.
    @pytest.mark.parametrize(
        "planted, named",
        [
            ("((t >= 60) & (t < 120))", "Cause: 't < 120' in clause 37 of 50, '((t >= 60) & (t < 120))'."),
            ("(t >= 0)", "Cause: 't >= 0' in clause 37 of 50, '(t >= 0)'."),
            ("(t == 5)", "uses equality in clause 37 of 50, '(t == 5)';"),
            ("(Eq(t, 5))", "uses equality in clause 37 of 50, '(Eq(t, 5))';"),
            ("(Ne(t, 5))", "uses equality in clause 37 of 50, '(Ne(t, 5))';"),
            ("(sin(t) > 0)", "uses 'sin', which is not allowed in a condition"),
            ("(sin(t) > 0)", "Found in clause 37 of 50, '(sin(t) > 0)'."),
            ("(latencyy > 3)", "Found in clause 37 of 50, '(latencyy > 3)'."),
            ("(Foo(t) > 1)", "Found in clause 37 of 50, '(Foo(t) > 1)'."),
            ("(t >= 60 & t < 90)", "Found in clause 37 of 50, '(t >= 60 & t < 90)'."),
        ],
    )
    def test_fault_in_clause_37_of_50(self, planted: str, named: str) -> None:
        raw = _fifty_clauses(planted)
        assert len(raw) > 800
        with pytest.raises(ValueError) as rejected:
            Predicate(raw)
        message = str(rejected.value)
        assert named in message
        assert len(message) < 450

    # A condition over 80 characters is quoted up to its 80th character with its length; clause 50 is not repeated.
    def test_long_condition_is_quoted_up_to_80_characters(self) -> None:
        raw = _fifty_clauses("(t >= 0)")
        with pytest.raises(ValueError) as rejected:
            Predicate(raw)
        message = str(rejected.value)
        assert message.startswith(f"Predicate {raw[:80]!r}... ({len(raw)} characters) already holds at")
        assert "(completed >= 13)" not in message

    # An 80-character condition is quoted whole; at 81 characters it is cut and its length is given.
    @pytest.mark.parametrize("length, quoted_whole", [(80, True), (81, False)])
    def test_quote_limit_is_80_characters(self, length: int, quoted_whole: bool) -> None:
        raw = "t < 60" + " " * (length - 6)
        with pytest.raises(ValueError) as rejected:
            Predicate(raw)
        message = str(rejected.value)
        assert message.startswith(f"Predicate {raw!r} holds only on") is quoted_whole
        assert (f"({length} characters)" in message) is not quoted_whole

    # Predicate('t < 60', name='stop_condition') says "stop_condition 't < 60'"; without a name it says "Predicate 't < 60'".
    def test_name_replaces_predicate_in_the_message(self) -> None:
        with pytest.raises(ValueError, match=r"^stop_condition 't < 60' holds only on"):
            Predicate("t < 60", name="stop_condition")
        with pytest.raises(ValueError, match=r"^Predicate 't < 60' holds only on"):
            Predicate("t < 60")

    # '(t >= 60) & (t < 120)' is one clause, so the cause is the comparison alone with no clause number.
    def test_single_clause_names_the_comparison_only(self) -> None:
        with pytest.raises(ValueError) as rejected:
            Predicate("(t >= 60) & (t < 120)")
        assert str(rejected.value).endswith("Cause: 't < 120'.")

    # '120 > t' is written number-first; the cause keeps that spelling and finds it in clause 2 of 2.
    def test_flipped_comparison_is_found_in_its_clause(self) -> None:
        with pytest.raises(ValueError) as rejected:
            Predicate("(t >= 600) | ((120 > t) & (errors > 3))")
        message = str(rejected.value)
        assert "holds only on [0, 120) or [600, infinity) once its measured values hold;" in message
        assert message.endswith("Cause: '120 > t' in clause 2 of 2, '((120 > t) & (errors > 3))'.")

    # 't < -5' never holds, so it is not a cause: only 't < 120' is named for '(t < -5) | ((t >= 60) & (t < 120))'.
    def test_comparison_that_never_holds_is_not_a_cause(self) -> None:
        with pytest.raises(ValueError) as rejected:
            Predicate("(t < -5) | ((t >= 60) & (t < 120))")
        message = str(rejected.value)
        assert message.endswith("Cause: 't < 120' in clause 2 of 2, '((t >= 60) & (t < 120))'.")
        assert "t < -5'" not in message.split("Cause:")[1]

    # 't**2 - 10*t > -16' holds on [0, 2) and again from 8: one comparison that lapses on its own is named as the cause.
    def test_comparison_holding_on_two_pieces_is_a_cause(self) -> None:
        with pytest.raises(ValueError) as rejected:
            Predicate("t**2 - 10*t > -16")
        message = str(rejected.value)
        assert "holds only on [0, 2) or (8, infinity);" in message
        assert message.endswith("Cause: 't**2 - 10*t > -16'.")

    # Five lapsing clauses 't < 10' .. 't < 50': the first three are named with their clause, then "and 2 more".
    def test_more_than_three_causes_are_counted(self) -> None:
        with pytest.raises(ValueError) as rejected:
            Predicate("(t<10)|(t<20)|(t<30)|(t<40)|(t<50)")
        assert str(rejected.value).endswith(
            "Cause: 't < 10' in clause 1 of 5, '(t<10)'; 't < 20' in clause 2 of 5, '(t<20)'; "
            "'t < 30' in clause 3 of 5, '(t<30)'; and 2 more."
        )

    # 'sin' in clauses 2 and 3 of 3: the first is quoted and the other is counted as "and 1 more".
    def test_fault_in_two_clauses_names_the_first(self) -> None:
        with pytest.raises(ValueError) as rejected:
            Predicate("(t >= 600) | (sin(t) > 0) | (sin(t) > 1)")
        assert str(rejected.value).endswith("Found in clause 2 of 3, '(sin(t) > 0)' and 1 more.")

    # A '|' inside parentheses does not split a clause: '((t < 30) | (t >= 60))' is one clause, so no clause number.
    def test_nested_or_is_one_clause(self) -> None:
        with pytest.raises(ValueError) as rejected:
            Predicate("((t < 30) | (t >= 60))")
        assert str(rejected.value).endswith("Cause: 't < 30'.")

    # No time bound in 3 measured-only-with-time clauses: the message counts the clauses and gives the clause to add.
    def test_no_time_bound_counts_the_clauses(self) -> None:
        with pytest.raises(ValueError) as rejected:
            Predicate("((t >= 60) & (errors > 1)) | ((t >= 60) & (errors > 2)) | ((t >= 60) & (errors > 3))")
        message = str(rejected.value)
        assert (
            "never holds on time alone: it must hold with every measured comparison false, and none of its 3 clauses does."
            in message
        )
        assert message.endswith("Add a time bound as one more clause: '| (t >= 600)'.")

    # 't >= 60 & t < 120' fails inside Python as a TypeError; the message says what is wrong without the Python text.
    def test_unparenthesised_message_has_no_python_error(self) -> None:
        with pytest.raises(ValueError) as rejected:
            Predicate("t >= 60 & t < 120")
        message = str(rejected.value)
        assert "'&' or '|' joins a comparison that has no parentheses." in message
        assert "unsupported operand" not in message

    # '(t >= 60) & (t >' is a syntax error, not a TypeError: the parser's own reason is kept next to the hint.
    def test_syntax_error_keeps_the_parser_reason(self) -> None:
        with pytest.raises(ValueError) as rejected:
            Predicate("(t >= 60) & (t > )")
        message = str(rejected.value)
        assert "joins a comparison that has no parentheses" not in message
        assert "parenthesise each one" in message
