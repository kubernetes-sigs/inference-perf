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
import numpy as np
from numpy.typing import NDArray
from typing import cast


def generate_distribution(
    min: int,
    max: int,
    mean: float,
    std_dev: float,
    total_count: int,
    dist_type: str = "normal",
    rng: np.random.Generator | None = None,
) -> NDArray[np.int_]:
    """
    Generates an array of lengths in integer adhering to the specified distribution constraints.

    Args:
        min: The minimum allowed length.
        max: The maximum allowed length.
        mean: The target mean of the distribution.
        std_dev: The target standard deviation of the distribution.
        total_count: The total number of lengths to generate.
        dist_type: Distribution type — "normal", "lognormal", "uniform", or "fixed".
        rng: Optional numpy Generator for deterministic output. Falls back to
            legacy ``np.random`` when *None* (preserves existing call-sites).

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
        if rng is not None:
            generated_numbers = rng.uniform(low=min, high=max, size=total_count)
        else:
            generated_numbers = np.random.uniform(low=min, high=max, size=total_count)
    elif dist_type == "lognormal":
        # Parameterise the underlying normal so the *lognormal* has the
        # requested mean/std_dev, then shift so that ``min`` maps to 0.
        shifted_mean = mean - min
        if shifted_mean <= 0:
            shifted_mean = 1.0
        sigma2 = np.log(1 + (std_dev / shifted_mean) ** 2)
        mu = np.log(shifted_mean) - sigma2 / 2
        sigma = np.sqrt(sigma2)
        if rng is not None:
            generated_numbers = rng.lognormal(mean=mu, sigma=sigma, size=total_count) + min
        else:
            generated_numbers = np.random.lognormal(mean=mu, sigma=sigma, size=total_count) + min
    else:  # normal (default)
        if mean < min or mean > max:
            raise ValueError("Mean cannot be outside min and max range.")
        if rng is not None:
            generated_numbers = rng.normal(loc=mean, scale=std_dev, size=total_count)
        else:
            generated_numbers = np.random.normal(loc=mean, scale=std_dev, size=total_count)

    clipped_numbers = np.clip(generated_numbers, min, max)
    generated_lengths = np.round(clipped_numbers).astype(int)
    generated_lengths = np.clip(generated_lengths, min, max)

    return cast(NDArray[np.int_], generated_lengths)
