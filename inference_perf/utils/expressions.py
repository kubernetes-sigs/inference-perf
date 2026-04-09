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

import logging
import inspect
from typing import Any, Optional
import sympy  # type: ignore[import-untyped]
from sympy.parsing.sympy_parser import parse_expr  # type: ignore[import-untyped]
import sympy.stats  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_MAPPING = {}
for name in dir(sympy.stats):
    if name.startswith("_") or not name[0].isupper():
        continue
    obj = getattr(sympy.stats, name)
    try:
        sig = inspect.signature(obj)
        params = list(sig.parameters.keys())
        if params and params[0] == "name":
            _MAPPING[name] = lambda *args, c=obj: c("rv", *args)
    except Exception:
        pass


def evaluate_distribution(expr_str: str) -> Any:
    """Evaluates a distribution expression and returns a SymPy random variable.

    Args:
        expr_str: The string expression to evaluate, e.g., "Normal(512, 200)".

    Returns:
        A SymPy random variable or expression.
    """
    try:
        expr = parse_expr(expr_str, local_dict=_MAPPING)
        return expr
    except Exception as e:
        logger.error(f"Failed to parse distribution expression '{expr_str}': {e}")
        raise


def sample_distribution(expr_str: str | float | int, t: Optional[float] = None) -> float:
    """Samples from a distribution expression.

    Args:
        expr_str: The string expression or number to sample from.
        t: Optional time in seconds since stage started, for time-varying expressions.

    Returns:
        A float sample from the distribution.
    """
    from sympy.stats import sample
    import sympy

    if isinstance(expr_str, (int, float)):
        return float(expr_str)

    try:
        dist = evaluate_distribution(expr_str)
        if t is not None:
            t_sym = sympy.Symbol("t")
            if hasattr(dist, "free_symbols") and t_sym in dist.free_symbols:
                dist = dist.subs(t_sym, t)

        if isinstance(dist, (int, float)):
            return float(dist)
        if hasattr(dist, "is_Number") and dist.is_Number:
            return float(dist)
        # sample returns a generator or a single value depending on arguments.
        return float(sample(dist, library="numpy"))
    except Exception as e:
        logger.error(f"Failed to sample from expression '{expr_str}': {e}")
        raise


def has_random_variables(expr_str: str) -> bool:
    """Checks if a SymPy expression contains random variables.

    Args:
        expr_str: The string expression to check.

    Returns:
        True if the expression contains random variables, False otherwise.
    """
    import sympy
    from sympy.parsing.sympy_parser import parse_expr

    try:
        expr = parse_expr(expr_str, local_dict=_MAPPING)
        if isinstance(expr, (int, float)):
            return False

        if isinstance(expr, sympy.stats.rv.RandomSymbol):
            return True

        if hasattr(expr, "free_symbols"):
            for sym in expr.free_symbols:
                if isinstance(sym, sympy.stats.rv.RandomSymbol):
                    return True
        return False
    except Exception:
        return False


def evaluate_rate(expr_str: str, t: float) -> float:
    """Evaluates a rate expression at time t.

    Args:
        expr_str: The string expression representing the rate, e.g., "10 +
          t/60".
        t: The time in seconds since the stage started.

    Returns:
        The evaluated rate as a float.
    """
    try:
        expr = parse_expr(expr_str)
        if isinstance(expr, (int, float)):
            return float(expr)

        t_sym = sympy.Symbol("t")
        # Check if 't' is actually in the expression
        if t_sym in expr.free_symbols:
            result = expr.subs(t_sym, t)
        else:
            result = expr

        return float(result.evalf())
    except Exception as e:
        logger.error(f"Failed to evaluate rate expression '{expr_str}' at t={t}: {e}")
        raise
