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
from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import Annotated, Any, Callable, Optional, Union

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, ValidationInfo, model_validator

logger = logging.getLogger(__name__)


class StrictBaseModel(BaseModel):
    """Base for every config model: unknown keys are an error, not silently dropped.

    Pydantic's default (`extra="ignore"`) means a renamed or misspelled key is discarded
    without a word, so a stale config keeps "working" while quietly losing the setting it
    was meant to apply. Rejecting extras is what lets the example-config CI gate actually
    detect drift between the YAML in this repo and the schema.
    """

    model_config = ConfigDict(extra="forbid")


def validate_length_expression(value: str) -> str:
    """Validate an expression-string length field at config-load time.

    Constructing the Expression is the validation: it raises ValueError for
    unparseable input, unknown functions or symbols, and the time variable
    ``t`` (a length never varies with stage time), which pydantic surfaces as
    a field validation error. The value itself is returned unchanged; sampling
    sites construct their own Expression from it.
    """
    from inference_perf.utils.numeric.expression import Expression

    Expression(value, allow_time=False)
    return value


class DistributionType(str, Enum):
    NORMAL = "normal"
    SKEW_NORMAL = "skew_normal"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    POISSON = "poisson"
    FIXED = "fixed"


# Deprecated: the legacy ``{type, min, max, mean, std_dev}`` block. Every config
# field that took one now takes an expression string (see docs/expressions.md);
# a block still loads, rewritten to that string with a warning, until the type
# is removed.
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


@dataclass(frozen=True)
class LegacyDistributionBlock:
    """Annotated marker: this field still accepts the deprecated distribution block.

    The CLI parser reads it to keep the block's nested flags (``--x.mean 600``)
    working through the deprecation period; the block itself never reaches
    the model, see :func:`_rewrite_legacy_block`.
    """

    integer: bool
    io: bool = False


def _rewrite_legacy_block(
    *, integer: bool, io: bool = False, mean_in_bounds: bool = False
) -> Callable[[Any, ValidationInfo], Any]:
    """A before-validator that turns a legacy distribution block into its expression value.

    A dict (from YAML or the nested CLI flags) or a Distribution instance is
    compiled through :func:`compile_legacy_distribution` and replaced by the
    result, so the model only ever holds a number or an expression string.
    The warning shows the author the exact replacement to paste. With
    ``mean_in_bounds`` a block whose mean falls outside its own ``[min, max]``
    is rejected, as insertion_point always did, instead of compiling to a value
    that would clamp every draw onto one bound.
    """

    def rewrite(value: Any, info: ValidationInfo) -> Any:
        if not isinstance(value, (dict, Distribution)):
            return value
        from inference_perf.utils.numeric.distribution import compile_legacy_distribution

        block = value if isinstance(value, Distribution) else Distribution.model_validate(value)
        if mean_in_bounds and block.type != DistributionType.UNIFORM and not block.min <= block.mean <= block.max:
            raise ValueError(f"{info.field_name} distribution mean must be within [min, max].")
        replacement = compile_legacy_distribution(block, integer=integer, io=io)
        field = info.field_name or "value"
        shown = f'"{replacement}"' if isinstance(replacement, str) else str(replacement)
        logger.warning(
            "%s: the distribution block is deprecated and will be removed in a future release. Replace it with `%s: %s`.",
            field,
            field,
            shown,
        )
        if block.total_count is not None:
            logger.warning(
                "%s.total_count is ignored: the number of values to pre-generate is derived from the load stages.",
                field,
            )
        return replacement

    return rewrite


def _validate_expression_string(value: Any) -> Any:
    if isinstance(value, str):
        validate_length_expression(value)
    return value


# A whole-number config value (token lengths, counts): a bare int or an
# expression string. A legacy distribution block is rewritten on load.
LengthValue = Annotated[
    Union[int, str],
    BeforeValidator(_rewrite_legacy_block(integer=True, io=False)),
    AfterValidator(_validate_expression_string),
    LegacyDistributionBlock(integer=True),
]

# The input/output length fields of DataConfig. Same surface as LengthValue; a
# legacy block compiles with the parameterisations generate_distribution gave
# these two fields, which differ from every other field's.
IOLengthValue = Annotated[
    Union[int, str],
    BeforeValidator(_rewrite_legacy_block(integer=True, io=True)),
    AfterValidator(_validate_expression_string),
    LegacyDistributionBlock(integer=True, io=True),
]

# A fractional config value (seconds): a bare number or an expression string.
ScalarValue = Annotated[
    Union[float, str],
    BeforeValidator(_rewrite_legacy_block(integer=False, io=False)),
    AfterValidator(_validate_expression_string),
    LegacyDistributionBlock(integer=False),
]


def _validate_insertion_point(value: Optional[Union[float, str]]) -> Optional[Union[float, str]]:
    if isinstance(value, str):
        from inference_perf.utils.numeric.expression import Expression

        bounds = Expression(value, allow_time=False).bounds
        if bounds is None or bounds[0] < 0 or bounds[1] > 1:
            raise ValueError(
                f"insertion_point {value!r} must be provably within [0, 1], e.g. 'Beta(2, 5)' or 'Min(Max({value}, 0), 1)'."
            )
    elif value is not None and not 0 <= value <= 1:
        raise ValueError("insertion_point must be within [0, 1].")
    return value


InsertionPoint = Annotated[
    Optional[Union[float, str]],
    BeforeValidator(_rewrite_legacy_block(integer=False, mean_in_bounds=True)),
    AfterValidator(_validate_insertion_point),
    LegacyDistributionBlock(integer=False),
]
