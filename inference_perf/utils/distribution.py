# Copyright 2025 The Kubernetes Authors.
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

    return generated_lengths
