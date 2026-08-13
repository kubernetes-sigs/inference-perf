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
from typing import Union
from pydantic import Field, model_validator

from inference_perf.config.common import Distribution, StrictBaseModel


class BimodalConfig(StrictBaseModel):
    """Configuration for bimodal data generator."""

    mode_a_system_prompt_len: int = Field(
        0, ge=0, description="Length of shared system prompt prefix (KV cache) for Mode A requests in tokens"
    )
    mode_a_groups: int = Field(1, ge=1, description="Number of KV cache groups for Mode A requests")
    mode_a_user_prompt_len: Union[int, Distribution] = Field(
        10, description="Length or distribution of Mode A user prompt in tokens"
    )
    mode_a_output_len: Union[int, Distribution] = Field(
        10, description="Length or distribution of Mode A output generation in tokens"
    )

    mode_b_system_prompt_len: int = Field(
        0, ge=0, description="Length of shared system prompt prefix (KV cache) for Mode B requests in tokens"
    )
    mode_b_groups: int = Field(1, ge=1, description="Number of KV cache groups for Mode B requests")
    mode_b_user_prompt_len: Union[int, Distribution] = Field(
        1024, description="Length or distribution of Mode B user prompt in tokens"
    )
    mode_b_output_len: Union[int, Distribution] = Field(
        1024, description="Length or distribution of Mode B output generation in tokens"
    )

    mode_a_ratio: float = Field(0.5, ge=0.0, le=1.0, description="Proportion of Mode A requests (0.0 to 1.0)")

    @model_validator(mode="after")
    def validate_bimodal_config(self) -> "BimodalConfig":
        for field_name in [
            "mode_a_user_prompt_len",
            "mode_a_output_len",
            "mode_b_user_prompt_len",
            "mode_b_output_len",
        ]:
            val = getattr(self, field_name)
            if isinstance(val, int) and val < 0:
                raise ValueError(f"{field_name} cannot be negative (got {val})")
            elif isinstance(val, Distribution):
                if val.min < 0:
                    raise ValueError(f"{field_name} distribution min cannot be negative (got {val.min})")
                if val.mean < 0:
                    raise ValueError(f"{field_name} distribution mean cannot be negative (got {val.mean})")
        return self
