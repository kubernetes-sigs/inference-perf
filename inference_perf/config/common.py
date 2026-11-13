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
from enum import Enum
from math import sqrt
from typing import Annotated, Optional, Union

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


class StrictBaseModel(BaseModel):
    """Base for every config model: unknown keys are an error, not silently dropped.

    Pydantic's default (`extra="ignore"`) means a renamed or misspelled key is discarded
    without a word, so a stale config keeps "working" while quietly losing the setting it
    was meant to apply. Rejecting extras is what lets the example-config CI gate actually
    detect drift between the YAML in this repo and the schema.
    """

    model_config = ConfigDict(extra="forbid")


def validate_length_expression(value: str, minimum: int = 0) -> str:
    """Validate an expression string for a whole-number field that takes values of at least ``minimum``.

    Constructing the Expression rejects unparseable input, unknown functions or
    symbols, and ``t``, since these never vary with stage time. The expression
    must also be provably integer-valued and provably at least ``minimum``, so
    the config says how a draw is rounded and kept in range instead of the
    sampler deciding silently.
    """
    from inference_perf.utils.numeric.expression import Expression

    expression = Expression(value, allow_time=False)
    whole, within = expression.is_integer_valued, expression.is_within(minimum)
    if not whole and not within:
        raise ValueError(
            f"{value!r} can take values that are not whole numbers or below {minimum}, and this field is a whole number"
            f" of at least {minimum}. Write 'Max(floor({value}), {minimum})', or 'Max(floor({value} + 1/2), {minimum})'"
            " to round to the nearest."
        )
    if not whole:
        raise ValueError(
            f"{value!r} can take values that are not whole numbers, and this field is a whole number."
            f" Write 'floor({value})' to round down, or 'floor({value} + 1/2)' to round to the nearest."
        )
    if not within:
        raise ValueError(
            f"{value!r} is not provably at least {minimum}, the smallest value this field takes. Write 'Max({value}, {minimum})'."
        )
    return value


def validate_seconds_expression(value: str) -> str:
    """Validate an expression string for a duration in seconds, which may be fractional but never negative."""
    from inference_perf.utils.numeric.expression import Expression

    if not Expression(value, allow_time=False).is_within(0):
        raise ValueError(f"{value!r} is not provably at least 0, and a duration cannot be negative. Write 'Max({value}, 0)'.")
    return value


def check_whole_number(value: object, minimum: int) -> None:
    """Check a whole-number field's value, a number or an expression string, against its minimum."""
    if isinstance(value, str):
        validate_length_expression(value, minimum)
    elif isinstance(value, int) and not isinstance(value, bool) and value < minimum:
        raise ValueError(f"{value} is below {minimum}, the smallest value this field takes.")


def check_seconds(value: object) -> None:
    """Check a duration field's value, a number or an expression string: never negative."""
    if isinstance(value, str):
        validate_seconds_expression(value)
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0:
        raise ValueError(f"{value} is below 0, and a duration cannot be negative.")


class DistributionType(str, Enum):
    NORMAL = "normal"
    SKEW_NORMAL = "skew_normal"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    POISSON = "poisson"
    FIXED = "fixed"


# Represents the distribution for input prompts and output generations.
class Distribution(StrictBaseModel):
    min: int = Field(default=10, description="Smallest value the distribution can produce; samples below are clamped.")
    max: int = Field(default=1024, description="Largest value the distribution can produce; samples above are clamped.")
    mean: float = Field(default=512, description="Mean of the distribution.")
    std_dev: float = Field(default=200, description="Standard deviation of the distribution. Exclusive with 'variance'.")
    total_count: Optional[int] = Field(default=None, description="Total number of values to sample from the distribution.")
    type: DistributionType = Field(
        default=DistributionType.NORMAL, description="Shape of the distribution to sample values from."
    )
    variance: Optional[float] = Field(default=None, description="Variance of the distribution. Exclusive with 'std_dev'.")
    skew: float = Field(default=0.0, description="Skewness of the distribution. Only used when type is 'skew_normal'.")

    @model_validator(mode="after")
    def validate_distribution(self) -> "Distribution":
        if self.variance is not None and self.std_dev > 0:
            raise ValueError("Specify either 'std_dev' or 'variance', not both.")
        if self.variance is not None:
            if self.variance < 0:
                raise ValueError("Variance cannot be negative.")
            self.std_dev = sqrt(self.variance)
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) cannot be greater than max ({self.max}).")
        if self.std_dev < 0:
            raise ValueError("std_dev cannot be negative.")
        return self


def _validate_insertion_point(
    value: Optional[Union[float, Distribution, str]],
) -> Optional[Union[float, Distribution, str]]:
    if isinstance(value, str):
        from inference_perf.utils.numeric.expression import Expression

        if not Expression(value, allow_time=False).is_within(0, 1):
            raise ValueError(
                f"insertion_point {value!r} must be provably within [0, 1], e.g. 'Beta(2, 5)' or 'Min(Max({value}, 0), 1)'."
            )
    elif isinstance(value, Distribution):
        if value.min < 0 or value.max > 1:
            raise ValueError("insertion_point distribution min and max must be within [0, 1].")
        if value.type != DistributionType.UNIFORM and not value.min <= value.mean <= value.max:
            raise ValueError("insertion_point distribution mean must be within [min, max].")
    elif value is not None and not 0 <= value <= 1:
        raise ValueError("insertion_point must be within [0, 1].")
    return value


InsertionPoint = Annotated[Optional[Union[float, Distribution, str]], AfterValidator(_validate_insertion_point)]
