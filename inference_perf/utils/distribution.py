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
from typing import TYPE_CHECKING, Optional, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from inference_perf.config import Distribution


def generate_distribution(min: int, max: int, mean: float, std_dev: float, total_count: int) -> NDArray[np.int_]:
    """
    Generates an array of lengths in integer adhering to the specified distribution constraints.

    Args:
        min: The minimum allowed length.
        max: The maximum allowed length.
        mean: The target mean of the distribution.
        std_dev: The target standard deviation of the distribution.
        total_count: The total number of lengths to generate.

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
    if mean < min or mean > max:
        raise ValueError("Mean cannot be outside min and max range.")

    # Generate floating-point numbers from a normal distribution
    # Use a large enough intermediate pool if std_dev is high relative to range
    # to increase chances of getting values within bounds after generation.
    # This is a heuristic; perfect adherence isn't guaranteed.
    generated_numbers = np.random.normal(loc=mean, scale=std_dev, size=total_count)

    # Clip the numbers to the specified min/max range
    clipped_numbers = np.clip(generated_numbers, min, max)

    # Round to the nearest integer and convert type
    generated_lengths = np.round(clipped_numbers).astype(int)

    # Ensure integer values are strictly within bounds after rounding
    # (e.g., rounding 4.6 when max is 4 could result in 5 without this)
    generated_lengths = np.clip(generated_lengths, min, max)

    return cast(NDArray[np.int_], generated_lengths)


def _sample_skew_normal(rng: np.random.Generator, mean: float, std_dev: float, skew: float, size: int) -> NDArray[np.float64]:
    """Sample from a skew-normal distribution using the Azzalini (1985) two-normal method.

    This avoids a scipy dependency.
    """
    delta = skew / sqrt(1.0 + skew**2)
    u0 = rng.standard_normal(size)
    v = rng.standard_normal(size)
    u1 = delta * u0 + sqrt(1.0 - delta**2) * v
    z = np.where(u0 >= 0, u1, -u1)
    return mean + std_dev * z


def sample_from_distribution(
    config: "Distribution",
    count: int,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.int_]:
    """Sample integer values from a Distribution config.

    Dispatches on config.type to support normal, skew_normal, lognormal,
    uniform, and poisson distributions. Falls back to normal when type is
    not specified (backward compatible).

    Args:
        config: A Distribution specifying the distribution type and parameters.
        count: Number of samples to generate.
        rng: Optional numpy Generator for deterministic seeding. If None, creates a default one.

    Returns:
        A numpy array of integers clamped to [config.min, config.max].
    """
    from inference_perf.config import DistributionType

    if count <= 0:
        raise ValueError("Count must be a positive integer.")
    if config.min > config.max:
        raise ValueError(f"min ({config.min}) cannot be greater than max ({config.max}).")

    if rng is None:
        rng = np.random.default_rng()

    if config.type == DistributionType.NORMAL:
        samples = rng.normal(loc=config.mean, scale=config.std_dev, size=count)

    elif config.type == DistributionType.SKEW_NORMAL:
        samples = _sample_skew_normal(rng, config.mean, config.std_dev, config.skew, count)

    elif config.type == DistributionType.LOGNORMAL:
        # Moment-match: convert desired mean/std_dev to underlying normal mu/sigma.
        if config.mean <= 0:
            raise ValueError("Lognormal distribution requires mean > 0.")
        m = config.mean
        s = config.std_dev
        if s <= 0:
            # Degenerate case: constant value
            samples = np.full(count, m, dtype=np.float64)
        else:
            sigma_sq = log(1.0 + (s / m) ** 2)
            mu = log(m) - sigma_sq / 2.0
            sigma = sqrt(sigma_sq)
            samples = rng.lognormal(mean=mu, sigma=sigma, size=count)

    elif config.type == DistributionType.UNIFORM:
        samples = rng.uniform(low=config.min, high=config.max + 1, size=count)

    elif config.type == DistributionType.POISSON:
        lam = config.mean if config.mean > 0 else 1.0
        samples = rng.poisson(lam=lam, size=count).astype(np.float64)

    else:
        raise ValueError(f"Unsupported distribution type: {config.type}")

    # Clip to bounds and round to integers
    clipped = np.clip(samples, config.min, config.max)
    result = np.round(clipped).astype(int)
    result = np.clip(result, config.min, config.max)
    return cast(NDArray[np.int_], result)
