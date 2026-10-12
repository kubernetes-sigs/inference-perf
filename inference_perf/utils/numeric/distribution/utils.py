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
from __future__ import annotations

from math import log, sqrt
from typing import TYPE_CHECKING, Literal, Optional, Union, cast, overload

import numpy as np
from numpy.typing import NDArray

from inference_perf.utils.numeric.expression import Expression

if TYPE_CHECKING:
    from inference_perf.config import Distribution


def distribution_to_expression(config: "Distribution", *, integer: bool = True) -> str:
    """Compile a Distribution config into the equivalent expression string.

    The returned string describes the same distribution the numpy sampler
    previously drew from directly, before the clip-and-round tail that
    :func:`sample_from_distribution` applies on top:

    - ``fixed``: the constant ``int(mean)`` (truncated, not rounded).
    - ``normal``: ``Normal(mean, std_dev)``; ``std_dev == 0`` degenerates to
      the constant ``mean``.
    - ``skew_normal``: the Azzalini (1985) two-normal construction
      ``mean + std_dev*(delta*|N1| + sqrt(1-delta^2)*N2)`` with
      ``delta = skew/sqrt(1+skew^2)``, where ``mean`` is the location
      parameter, not the sample mean. The two normals must be independent,
      which Expression guarantees by naming random variables per occurrence.
    - ``lognormal``: ``LogNormal(mu, sigma)`` moment-matched so the lognormal
      itself (not the underlying normal) has the requested ``mean``/``std_dev``.
    - ``uniform``: ``Uniform(min, max + 1)`` when ``integer``, so ``max`` stays
      fully reachable after rounding rather than a half-weight edge;
      ``Uniform(min, max)`` otherwise.
    - ``poisson``: ``Poisson(mean)``, with nonpositive ``mean`` falling back
      to lambda 1.

    Raises:
        ValueError: For a lognormal with ``mean <= 0``, or an unsupported type.
    """
    from inference_perf.config import DistributionType

    if config.type == DistributionType.FIXED:
        return str(int(config.mean))

    if config.type == DistributionType.NORMAL:
        if config.std_dev <= 0:
            return _num(config.mean)
        return f"Normal({_num(config.mean)}, {_num(config.std_dev)})"

    if config.type == DistributionType.SKEW_NORMAL:
        if config.std_dev <= 0:
            return _num(config.mean)
        delta = config.skew / sqrt(1.0 + config.skew**2)
        tail = sqrt(1.0 - delta**2)
        return f"{_num(config.mean)} + {_num(config.std_dev)}*({delta!r}*Abs(Normal(0, 1)) + {tail!r}*Normal(0, 1))"

    if config.type == DistributionType.LOGNORMAL:
        if config.mean <= 0:
            raise ValueError("Lognormal distribution requires mean > 0.")
        if config.std_dev <= 0:
            return _num(config.mean)
        sigma_sq = log(1.0 + (config.std_dev / config.mean) ** 2)
        mu = log(config.mean) - sigma_sq / 2.0
        return f"LogNormal({mu!r}, {sqrt(sigma_sq)!r})"

    if config.type == DistributionType.UNIFORM:
        return f"Uniform({_num(config.min)}, {_num(config.max + int(integer))})"

    if config.type == DistributionType.POISSON:
        lam = config.mean if config.mean > 0 else 1.0
        return f"Poisson({_num(lam)})"

    raise ValueError(f"Unsupported distribution type: {config.type}")


def _num(value: float) -> str:
    """Render a number the way a config author would write it: ``512`` not ``512.0``."""
    value = float(value)
    return str(int(value)) if value.is_integer() else repr(value)


def _io_distribution_core(min: int, max: int, mean: float, std_dev: float, dist_type: str) -> str:
    """The expression :func:`generate_distribution` draws from, before its clip-and-round tail.

    Its parameterisations differ from :func:`distribution_to_expression`: the
    lognormal moment-matches ``(value - min)`` and shifts by ``min``, the
    uniform draws from ``[min, max]`` inclusive, and "normal" rejects a mean
    outside the bounds.
    """
    if dist_type == "uniform":
        return f"Uniform({_num(min)}, {_num(max)})" if min < max else _num(min)
    if dist_type == "lognormal":
        # Parameterise the underlying normal so the *lognormal* has the
        # requested mean/std_dev, then shift so that ``min`` maps to 0.
        shifted_mean = mean - min
        if shifted_mean <= 0:
            shifted_mean = 1.0
        if std_dev == 0:
            return _num(min + shifted_mean)
        sigma_sq = log(1.0 + (std_dev / shifted_mean) ** 2)
        mu = log(shifted_mean) - sigma_sq / 2.0
        return f"{_num(min)} + LogNormal({mu!r}, {sqrt(sigma_sq)!r})"
    if dist_type == "normal":
        if mean < min or mean > max:
            raise ValueError("Mean cannot be outside min and max range.")
        return f"Normal({_num(mean)}, {_num(std_dev)})" if std_dev > 0 else _num(mean)
    raise ValueError(f"Unknown dist_type {dist_type!r}. Supported types: 'normal', 'lognormal', 'uniform', 'fixed'.")


def compile_legacy_distribution(config: "Distribution", *, integer: bool = True, io: bool = False) -> Union[int, float, str]:
    """Compile a legacy Distribution block into the config value that replaces it.

    This is the rewrite a deprecation warning shows the config author, so it
    carries the block's whole contract, clipping included: a ``fixed`` block
    becomes its (truncated) mean as a bare number, a degenerate block with no
    spread becomes its clamped mean, and every other block becomes its
    distribution wrapped in ``Min``/``Max`` to the block's ``[min, max]``,
    with a wrapper dropped when the distribution's own support already
    respects that side. Rounding is not part of the string; the sampling site
    decides that, as it does for any expression.

    Args:
        config: The legacy block.
        integer: Whether the field holds whole numbers (lengths, counts) or
            fractions (seconds), which decides the uniform's upper edge and
            the type of a constant result.
        io: Compile with :func:`generate_distribution`'s parameterisations,
            which the IO distribution fields of the synthetic and random
            generators used, instead of :func:`sample_from_distribution`'s.
    """
    from inference_perf.config import DistributionType

    def constant(value: float) -> Union[int, float]:
        return int(value) if integer else float(value)

    if config.type == DistributionType.FIXED:
        return constant(config.mean)

    core: Optional[str] = None
    if io and config.type in (DistributionType.NORMAL, DistributionType.LOGNORMAL, DistributionType.UNIFORM):
        try:
            core = _io_distribution_core(config.min, config.max, config.mean, config.std_dev, config.type.value)
        except ValueError:
            # A normal whose mean sits outside [min, max]: generate_distribution
            # refused it, but a dataset type only ever read the bounds, so it
            # has to keep loading. The clamped form keeps those bounds.
            core = None
    if core is None:
        core = distribution_to_expression(config, integer=integer)

    expression = Expression(core, allow_time=False)
    if expression.is_constant:
        return constant(min(max(float(expression.sample()), config.min), config.max))

    bounds = expression.bounds
    lo, hi = bounds if bounds is not None else (float("-inf"), float("inf"))
    if lo < config.min:
        core = f"Max({core}, {_num(config.min)})"
    if hi > config.max:
        core = f"Min({core}, {_num(config.max)})"
    return core


def generate_distribution(
    min: int,
    max: int,
    mean: float,
    std_dev: float,
    total_count: int,
    dist_type: str = "normal",  # one of: "normal", "lognormal", "uniform", "fixed"
    rng: np.random.Generator | None = None,
) -> NDArray[np.int_]:
    """
    Generates an array of lengths in integer adhering to the specified distribution constraints.

    The distribution is compiled to an expression string and drawn through
    :class:`Expression`; this function keeps its legacy contract on top of
    that (its parameterisations differ from :func:`sample_from_distribution`):
    the lognormal moment-matches ``(value - min)`` and shifts by ``min``, the
    uniform draws from ``[min, max]``, "normal" rejects a mean outside the
    bounds, and draws are clipped into ``[min, max]`` and rounded to integers.

    Args:
        min: The minimum allowed length.
        max: The maximum allowed length.
        mean: The target mean of the distribution.
        std_dev: The target standard deviation of the distribution.
        total_count: The total number of lengths to generate.
        dist_type: Distribution type — "normal", "lognormal", "uniform", or "fixed".
        rng: Optional numpy Generator for deterministic output; a fresh
            default Generator is used when *None*.

    Returns:
        A numpy array of integers representing lengths for input prompts or output generations.

    Raises:
        ValueError: If constraints are impossible (e.g., min_val > max_val).
    """
    if min > max:
        raise ValueError("Minimum value cannot be greater than maximum value.")
    if total_count <= 0:
        raise ValueError("Total count must be a positive integer.")
    if std_dev < 0:
        raise ValueError("Standard deviation cannot be negative.")

    if dist_type == "fixed":
        return cast(NDArray[np.int_], np.full(total_count, int(mean), dtype=int))

    raw = _io_distribution_core(min, max, mean, std_dev, dist_type)

    if rng is None:
        rng = np.random.default_rng()

    expression = Expression(raw, allow_time=False)
    generated_numbers = np.atleast_1d(np.asarray(expression.sample(rng=rng, size=total_count), dtype=np.float64))

    clipped_numbers = np.clip(generated_numbers, min, max)
    generated_lengths = np.round(clipped_numbers).astype(int)
    generated_lengths = np.clip(generated_lengths, min, max)

    return cast(NDArray[np.int_], generated_lengths)


def sample_values(
    value: Union[int, float, str],
    count: int,
    rng: Optional[np.random.Generator] = None,
    *,
    integer: bool,
) -> NDArray[np.float64]:
    """Sample from any numeric config value: a number or an expression string.

    - number: the constant, repeated (no random draws are consumed).
    - str: an Expression sampled as written. The expression author owns the
      value range; nothing is clamped on top.

    Args:
        value: The configured value.
        count: Number of values to sample.
        rng: Optional numpy Generator for deterministic seeding. If None, creates a default one.
        integer: Round to whole numbers (counts, token lengths) or keep
            fractions (seconds).

    Returns:
        A float array of ``count`` values, whole-numbered when ``integer``.
    """
    if count <= 0:
        raise ValueError("Count must be a positive integer.")

    if isinstance(value, (int, float)):
        return np.full(count, round(value) if integer else float(value), dtype=np.float64)

    if rng is None:
        rng = np.random.default_rng()
    expression = Expression(value, allow_time=False)
    samples = np.atleast_1d(np.asarray(expression.sample(rng=rng, size=count), dtype=np.float64))
    return np.round(samples) if integer else samples


def sample_lengths(
    value: Union[int, str],
    count: int,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.int_]:
    """Sample integer lengths from any length-field config value; see :func:`sample_values`."""
    return cast(NDArray[np.int_], sample_values(value, count, rng, integer=True).astype(int))


def value_bounds(value: Union[int, float, str]) -> tuple[float, float]:
    """The provable ``(lowest, highest)`` values a numeric config value can produce.

    A number is its own bounds. An expression string's bounds come from
    :attr:`Expression.bounds`, infinite on a side nothing bounds (``Normal(50, 10)``
    on both, ``Poisson(3)`` above) and on both sides when nothing can be proven.
    """
    if isinstance(value, (int, float)):
        return float(value), float(value)
    bounds = Expression(value, allow_time=False).bounds
    return (float("-inf"), float("inf")) if bounds is None else bounds


def value_ceiling(value: Union[int, float, str]) -> Optional[float]:
    """The largest value a numeric config value can produce, for worst-case budget checks.

    A number is its own ceiling. An expression string's ceiling is its
    provable upper bound (:attr:`Expression.bounds`), which is ``inf`` for an
    unbounded expression such as ``Normal(50, 10)``, and ``None`` when nothing
    can be proven.
    """
    if isinstance(value, (int, float)):
        return float(value)
    bounds = Expression(value, allow_time=False).bounds
    return None if bounds is None else bounds[1]


@overload
def sample_from_distribution(
    config: "Distribution",
    count: int,
    rng: Optional[np.random.Generator] = None,
    *,
    integer: Literal[True] = True,
) -> NDArray[np.int_]: ...


@overload
def sample_from_distribution(
    config: "Distribution",
    count: int,
    rng: Optional[np.random.Generator] = None,
    *,
    integer: Literal[False],
) -> NDArray[np.float64]: ...


def sample_from_distribution(
    config: "Distribution",
    count: int,
    rng: Optional[np.random.Generator] = None,
    *,
    integer: bool = True,
) -> NDArray[np.int_] | NDArray[np.float64]:
    """Sample values from a Distribution config.

    The config is compiled to an expression string via
    :func:`distribution_to_expression` and drawn through :class:`Expression`;
    the legacy contract is preserved on top: "fixed" returns ``int(mean)``
    unclamped, every other type has its draws clipped into
    ``[config.min, config.max]`` and rounded to integers.

    Args:
        config: A Distribution specifying the distribution type and parameters.
        count: Number of samples to generate.
        rng: Optional numpy Generator for deterministic seeding. If None, creates a default one.
        integer: Preserve integer count sampling by default. When False, retain fractional
            values and sample uniform distributions over [min, max] without the count-specific +1.

    Returns:
        A numpy array of sampled values. Non-fixed distributions and continuous
        fixed values are clamped to [config.min, config.max].
    """
    from inference_perf.config import DistributionType

    if count <= 0:
        raise ValueError("Count must be a positive integer.")
    if config.min > config.max:
        raise ValueError(f"min ({config.min}) cannot be greater than max ({config.max}).")

    if config.type == DistributionType.FIXED:
        if integer:
            return cast(NDArray[np.int_], np.full(count, int(config.mean), dtype=int))
        return cast(NDArray[np.float64], np.full(count, np.clip(config.mean, config.min, config.max), dtype=np.float64))

    if rng is None:
        rng = np.random.default_rng()

    expression = Expression(distribution_to_expression(config, integer=integer), allow_time=False)
    samples = np.atleast_1d(np.asarray(expression.sample(rng=rng, size=count), dtype=np.float64))

    clipped = np.clip(samples, config.min, config.max)
    if not integer:
        return clipped
    result = np.round(clipped).astype(int)
    result = np.clip(result, config.min, config.max)
    return cast(NDArray[np.int_], result)
