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
"""Structural types for the sympy objects the expression package reads.

sympy ships no type information, so mypy sees every sympy value as ``Any``,
and annotating one as ``sympy.Basic`` checks nothing either. These protocols
name the attributes this package reads, so mypy checks those reads.

A value takes one of these types where sympy hands it back, through an
annotated assignment or a ``TypeGuard``. ``isinstance`` against a sympy class
narrows back to ``Any``, so a branch body under one is unchecked.
``test_sympy_types.py`` pins each protocol against real sympy objects, so a
sympy upgrade that drops an attribute fails there.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Protocol, runtime_checkable


@runtime_checkable
class SympyNode(Protocol):
    """Any sympy tree: a number, a symbol, an expression, a condition or a set."""

    @property
    def args(self) -> tuple[SympyNode, ...]: ...

    @property
    def func(self) -> Callable[..., SympyNode]: ...

    @property
    def free_symbols(self) -> set[SympyNode]: ...

    # sympy's assumptions answer True, False or None for "cannot tell"
    @property
    def is_extended_real(self) -> Optional[bool]: ...

    @property
    def is_number(self) -> bool: ...

    @property
    def is_Number(self) -> bool: ...

    @property
    def is_NumberSymbol(self) -> bool: ...

    @property
    def is_Integer(self) -> bool: ...

    def atoms(self, *types: type) -> set[SympyNode]: ...

    def xreplace(self, rule: Mapping[SympyNode, SympyNode]) -> SympyNode: ...

    def subs(self, old: SympyNode, new: SympyNode) -> SympyNode: ...

    def has(self, *patterns: object) -> bool: ...


@runtime_checkable
class SympyExpr(SympyNode, Protocol):
    """A numeric expression. A condition is a ``SympyNode`` but not one of these: it has no value to negate or evaluate."""

    def evalf(self) -> SympyExpr: ...

    # substituting into an expression gives an expression, or sympy raises
    def subs(self, old: SympyNode, new: SympyNode) -> SympyExpr: ...

    def __float__(self) -> float: ...

    def __neg__(self) -> SympyExpr: ...

    # a comparison is a condition, which only becomes a bool when sympy can decide it
    def __ge__(self, other: object) -> SympyNode: ...

    def __le__(self, other: object) -> SympyNode: ...


@runtime_checkable
class SympySet(SympyNode, Protocol):
    """A set of reals, e.g. the support of a distribution or the times a condition holds."""

    @property
    def inf(self) -> SympyExpr: ...

    @property
    def sup(self) -> SympyExpr: ...

    def intersect(self, other: SympySet) -> SympySet: ...

    def is_subset(self, other: SympySet) -> Optional[bool]: ...


class SympyDistribution(Protocol):
    @property
    def set(self) -> SympySet: ...


class SympyProbabilitySpace(Protocol):
    @property
    def distribution(self) -> SympyDistribution: ...


@runtime_checkable
class SympyRandomSymbol(SympyExpr, Protocol):
    """A ``sympy.stats`` random variable, e.g. what ``Normal(512, 128)`` parses to."""

    @property
    def pspace(self) -> SympyProbabilitySpace: ...
