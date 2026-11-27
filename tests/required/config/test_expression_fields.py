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
"""Expression strings as config values for length and IO distribution fields.

The fields that accepted int/Distribution also accept an expression string
(``"floor(Uniform(256, 768))"``), validated at config-load time. Dataset generators
use the IO distribution's min/max as filter bounds, so strings there are a
config error rather than a runtime crash.
"""

import pytest
from pydantic import ValidationError

from inference_perf.config import DataConfig, DataGenType, Distribution, SharedPrefix


class TestSharedPrefixLengthExpressions:
    # Each length field accepts an expression string, and the parsed model
    # stores it verbatim for the datagen to sample from.
    @pytest.mark.parametrize("field", ["system_prompt_len", "question_len", "output_len"])
    def test_expression_string_accepted(self, field: str) -> None:
        sp = SharedPrefix(**{field: "floor(Max(Normal(50, 10), 1))"})
        assert getattr(sp, field) == "floor(Max(Normal(50, 10), 1))"

    # A misspelled distribution name ("Nrml") fails at config-load time with
    # a field validation error, not at first sample.
    def test_unknown_function_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown function"):
            SharedPrefix(question_len="Nrml(50, 10)")

    # Lengths never vary with stage time, so the time variable t is rejected.
    def test_time_variable_rejected(self) -> None:
        with pytest.raises(ValidationError, match="disallowed symbol"):
            SharedPrefix(question_len="50 + t")

    # An expression on question_len plus the legacy question_distribution is
    # ambiguous, exactly like an inline Distribution plus the legacy field.
    def test_expression_conflicts_with_legacy_distribution(self) -> None:
        with pytest.raises(ValidationError, match="one or the other"):
            SharedPrefix(
                question_len="floor(Max(Normal(50, 10), 1))", question_distribution=Distribution(mean=50.0, min=1, max=100)
            )

    # A plain int stays an int. A legacy Distribution block is rewritten on
    # load to the expression string it stood for, clamped to its [min, max] and rounded.
    def test_int_kept_and_legacy_block_rewritten(self) -> None:
        sp = SharedPrefix(question_len=50, output_len=Distribution(mean=25.0, min=1, max=100))
        assert sp.question_len == 50
        assert sp.output_len == "floor(Min(Max(Normal(25, 200), 1), 100) + 1/2)"


class TestDataConfigDistributionExpressions:
    # synthetic and random sample lengths from the field, so an expression
    # string is a valid value for both IO distribution fields.
    @pytest.mark.parametrize("gen_type", [DataGenType.Synthetic, DataGenType.Random])
    def test_expression_string_accepted_for_sampling_types(self, gen_type: DataGenType) -> None:
        config = DataConfig(
            type=gen_type,
            input_distribution="1 + floor(LogNormal(5.0, 0.5))",
            output_distribution="floor(Uniform(10, 100))",
        )
        assert config.input_distribution == "1 + floor(LogNormal(5.0, 0.5))"
        assert config.output_distribution == "floor(Uniform(10, 100))"

    # shareGPT filters its dataset by the value's provable bounds, so an
    # expression with an open side, here the top, is rejected at config-load
    # time and a clamped one is accepted.
    def test_unbounded_expression_rejected_for_dataset_types(self) -> None:
        with pytest.raises(ValidationError, match="provably bounded on both sides"):
            DataConfig(type=DataGenType.ShareGPT, input_distribution="floor(Max(Normal(512, 200), 10))")
        config = DataConfig(type=DataGenType.ShareGPT, input_distribution="floor(Min(Max(Normal(512, 200), 10), 1024))")
        assert config.input_distribution == "floor(Min(Max(Normal(512, 200), 10), 1024))"

    # Invalid expression strings fail field validation before the scope check.
    def test_invalid_expression_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown function"):
            DataConfig(type=DataGenType.Synthetic, input_distribution="Nrml(512, 200)")

    # A legacy Distribution block on an IO field is rewritten on load to the
    # expression generate_distribution drew from, clamped to its [min, max] and rounded.
    def test_legacy_block_rewritten_on_io_fields(self) -> None:
        config = DataConfig(type=DataGenType.Synthetic, input_distribution=Distribution(mean=512.0, min=10, max=1024))
        assert config.input_distribution == "floor(Min(Max(Normal(512, 200), 10), 1024) + 1/2)"
        uniform = DataConfig(type=DataGenType.Random, output_distribution=Distribution(type="uniform", min=10, max=100))
        assert uniform.output_distribution == "floor(Uniform(10, 100) + 1/2)"


class TestWholeNumberExpressions:
    # A continuous expression on a whole-number field is rejected at config load, and the error carries both
    # fixes to paste. Input: question_len 'Uniform(10, 100)', in range but fractional. Expected: the message names
    # 'floor(Uniform(10, 100))' and 'floor(Uniform(10, 100) + 1/2)'.
    def test_continuous_expression_rejected_with_the_floor_form(self) -> None:
        with pytest.raises(ValidationError, match=r"floor\(Uniform\(10, 100\)\)'.*floor\(Uniform\(10, 100\) \+ 1/2\)"):
            SharedPrefix(question_len="Uniform(10, 100)")

    # An unbounded continuous expression misses both rules, and the error gives one form that fixes both.
    # Input: question_len 'Normal(50, 10)'. Expected: the message names 'Max(floor(Normal(50, 10)), 1)'.
    def test_unbounded_continuous_expression_gets_one_fix(self) -> None:
        with pytest.raises(ValidationError, match=r"Max\(floor\(Normal\(50, 10\)\), 1\)"):
            SharedPrefix(question_len="Normal(50, 10)")

    # A discrete distribution needs no wrapper. Input: question_len '1 + Poisson(50)'. Expected: stored as written.
    def test_discrete_expression_accepted(self) -> None:
        assert SharedPrefix(question_len="1 + Poisson(50)").question_len == "1 + Poisson(50)"

    # The IO fields follow the same rule. Input: output_distribution 'Uniform(10, 100)'. Expected: rejected.
    def test_io_field_rejects_continuous_expression(self) -> None:
        with pytest.raises(ValidationError, match="not whole numbers"):
            DataConfig(type=DataGenType.Synthetic, output_distribution="Uniform(10, 100)")
