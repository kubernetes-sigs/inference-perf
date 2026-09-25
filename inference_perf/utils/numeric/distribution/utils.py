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
            return repr(float(config.mean))
        return f"Normal({float(config.mean)!r}, {float(config.std_dev)!r})"

    if config.type == DistributionType.SKEW_NORMAL:
        if config.std_dev <= 0:
            return repr(float(config.mean))
        delta = config.skew / sqrt(1.0 + config.skew**2)
        tail = sqrt(1.0 - delta**2)
        return f"{float(config.mean)!r} + {float(config.std_dev)!r}*({delta!r}*Abs(Normal(0, 1)) + {tail!r}*Normal(0, 1))"

    if config.type == DistributionType.LOGNORMAL:
        if config.mean <= 0:
            raise ValueError("Lognormal distribution requires mean > 0.")
        if config.std_dev <= 0:
            return repr(float(config.mean))
        sigma_sq = log(1.0 + (config.std_dev / config.mean) ** 2)
        mu = log(config.mean) - sigma_sq / 2.0
        return f"LogNormal({mu!r}, {sqrt(sigma_sq)!r})"

    if config.type == DistributionType.UNIFORM:
        return f"Uniform({float(config.min)!r}, {float(config.max + int(integer))!r})"

    if config.type == DistributionType.POISSON:
        lam = config.mean if config.mean > 0 else 1.0
        return f"Poisson({float(lam)!r})"

    raise ValueError(f"Unsupported distribution type: {config.type}")


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

    if dist_type == "uniform":
        raw = f"Uniform({float(min)!r}, {float(max)!r})" if min < max else repr(float(min))
    elif dist_type == "lognormal":
        # Parameterise the underlying normal so the *lognormal* has the
        # requested mean/std_dev, then shift so that ``min`` maps to 0.
        shifted_mean = mean - min
        if shifted_mean <= 0:
            shifted_mean = 1.0
        if std_dev == 0:
            raw = repr(float(min + shifted_mean))
        else:
            sigma_sq = log(1.0 + (std_dev / shifted_mean) ** 2)
            mu = log(shifted_mean) - sigma_sq / 2.0
            raw = f"{float(min)!r} + LogNormal({mu!r}, {sqrt(sigma_sq)!r})"
    elif dist_type == "normal":
        if mean < min or mean > max:
            raise ValueError("Mean cannot be outside min and max range.")
        raw = f"Normal({float(mean)!r}, {float(std_dev)!r})" if std_dev > 0 else repr(float(mean))
    else:
        raise ValueError(f"Unknown dist_type {dist_type!r}. Supported types: 'normal', 'lognormal', 'uniform', 'fixed'.")

    if rng is None:
        rng = np.random.default_rng()

    expression = Expression(raw, allow_time=False)
    generated_numbers = np.atleast_1d(np.asarray(expression.sample(rng=rng, size=total_count), dtype=np.float64))

    clipped_numbers = np.clip(generated_numbers, min, max)
    generated_lengths = np.round(clipped_numbers).astype(int)
    generated_lengths = np.clip(generated_lengths, min, max)

    return cast(NDArray[np.int_], generated_lengths)


def sample_values(
    value: Union[int, float, "Distribution", str],
    count: int,
    rng: Optional[np.random.Generator] = None,
    *,
    integer: bool,
) -> NDArray[np.float64]:
    """Sample from any numeric config value: a number, a Distribution, or an expression string.

    - number: the constant, repeated (no random draws are consumed).
    - Distribution: :func:`sample_from_distribution`'s contract (draws clipped
      into the config bounds). A ``fixed`` Distribution is never clipped, in
      either mode: integer mode returns ``int(mean)`` as it always has, and
      fractional mode returns ``mean`` itself.
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
    from inference_perf.config import DistributionType

    if count <= 0:
        raise ValueError("Count must be a positive integer.")

    if isinstance(value, (int, float)):
        return np.full(count, round(value) if integer else float(value), dtype=np.float64)

    if isinstance(value, str):
        if rng is None:
            rng = np.random.default_rng()
        expression = Expression(value, allow_time=False)
        samples = np.atleast_1d(np.asarray(expression.sample(rng=rng, size=count), dtype=np.float64))
        return np.round(samples) if integer else samples

    if not integer and value.type == DistributionType.FIXED:
        return np.full(count, float(value.mean), dtype=np.float64)
    if integer:
        return sample_from_distribution(value, count, rng).astype(np.float64)
    return sample_from_distribution(value, count, rng, integer=False)


def sample_lengths(
    value: Union[int, "Distribution", str],
    count: int,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.int_]:
    """Sample integer lengths from any length-field config value; see :func:`sample_values`."""
    return cast(NDArray[np.int_], sample_values(value, count, rng, integer=True).astype(int))


def value_ceiling(value: Union[int, float, "Distribution", str]) -> Optional[float]:
    """The largest value a numeric config value can produce, for worst-case budget checks.

    A number is its own ceiling. A ``fixed`` Distribution is never clipped, so
    its ceiling is ``mean``; every other Distribution is clipped to ``max``.
    An expression string's ceiling is its provable upper bound
    (:attr:`Expression.bounds`), which is ``inf`` for an unbounded expression
    such as ``Normal(50, 10)``, and ``None`` when nothing can be proven.
    """
    from inference_perf.config import DistributionType

    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        bounds = Expression(value, allow_time=False).bounds
        return None if bounds is None else bounds[1]
    if value.type == DistributionType.FIXED:
        return float(value.mean)
    return float(value.max)


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
