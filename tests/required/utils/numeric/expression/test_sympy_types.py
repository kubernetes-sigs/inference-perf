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
"""Pins the sympy protocols against real sympy objects, so a sympy upgrade that drops an attribute fails here."""

import pytest
import sympy
import sympy.stats

from inference_perf.utils.numeric.expression.sympy_types import SympyExpr, SympyNode, SympyRandomSymbol, SympySet

_T = sympy.Symbol("t")


# Numbers, symbols, expressions, conditions and sets all carry the SympyNode attributes.
@pytest.mark.parametrize(
    "value",
    [sympy.Integer(3), sympy.pi, _T, _T + 1, _T >= 60, (_T >= 60) | (_T < 5), sympy.true, sympy.Interval(0, 1)],
    ids=["integer", "pi", "symbol", "sum", "relational", "or", "true", "interval"],
)
def test_every_sympy_value_is_a_node(value: object) -> None:
    assert isinstance(value, SympyNode)


# 3, pi, t, t + 1 and Min(t, 5) carry the numeric attributes: evalf, float(), negation, >= and <=.
@pytest.mark.parametrize(
    "value",
    [sympy.Integer(3), sympy.pi, _T, _T + 1, sympy.Min(_T, 5)],
    ids=["integer", "pi", "symbol", "sum", "min"],
)
def test_numeric_values_are_exprs(value: object) -> None:
    assert isinstance(value, SympyExpr)


# 't >= 60', '(t >= 60) | (t < 5)' and true are conditions: no float() and no negation, so not a SympyExpr.
@pytest.mark.parametrize("value", [_T >= 60, (_T >= 60) | (_T < 5), sympy.true], ids=["relational", "or", "true"])
def test_conditions_are_not_exprs(value: object) -> None:
    assert not isinstance(value, SympyExpr)


# [0, 1], [0, 1] or [2, 3], and Normal(0, 1)'s support (-oo, oo) all carry inf, sup, intersect and is_subset.
def test_sets_are_sets() -> None:
    support = sympy.stats.Normal("x", 0, 1).pspace.distribution.set
    for value in (sympy.Interval(0, 1), sympy.Interval(0, 1) | sympy.Interval(2, 3), support):
        assert isinstance(value, SympySet)


# Normal(512, 128) is a random symbol whose pspace.distribution.set is a SympySet. t and 5 are not random symbols.
def test_random_symbol() -> None:
    rv = sympy.stats.Normal("x", 512, 128)
    assert isinstance(rv, SympyRandomSymbol)
    assert isinstance(rv.pspace.distribution.set, SympySet)
    assert not isinstance(_T, SympyRandomSymbol)
    assert not isinstance(sympy.Integer(5), SympyRandomSymbol)


# What parse_expr returns for '1, 2', '[1]' and 'True' is a Python value, not a sympy node.
@pytest.mark.parametrize("value", [(1, 2), [1], True], ids=["tuple", "list", "bool"])
def test_python_values_are_not_nodes(value: object) -> None:
    assert not isinstance(value, SympyNode)
