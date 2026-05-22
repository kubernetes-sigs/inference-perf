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
from typing import Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from inference_perf.config.common import Distribution
from inference_perf.config.datagen.multimodal import SyntheticMultimodalDatagenConfig
from inference_perf.config.datagen.replay import (
    ConversationReplayConfig,
    OTelTraceReplayConfig,
    TraceConfig,
)


class DataGenType(Enum):
    Mock = "mock"
    ShareGPT = "shareGPT"
    Synthetic = "synthetic"
    Random = "random"
    SharedPrefix = "shared_prefix"
    CNNDailyMail = "cnn_dailymail"
    InfinityInstruct = "infinity_instruct"
    BillsumConversations = "billsum_conversations"
    OTelTraceReplay = "otel_trace_replay"
    ConversationReplay = "conversation_replay"


# Configuration for shared prefix datagen which allows users to specify shared prefixes.
class SharedPrefix(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    num_groups: int = Field(
        10,
        validation_alias=AliasChoices("num_unique_system_prompts", "num_groups"),
        serialization_alias="num_unique_system_prompts",
    )

    num_prompts_per_group: int = Field(
        10,
        validation_alias=AliasChoices("num_users_per_system_prompt", "num_prompts_per_group"),
        serialization_alias="num_users_per_system_prompt",
    )

    system_prompt_len: Union[int, Distribution] = 100
    question_len: Union[int, Distribution] = 50
    output_len: Union[int, Distribution] = 50
    seed: Optional[int] = None

    # Legacy distribution fields — kept for backward compatibility.
    # Prefer using inline distribution syntax on question_len/output_len instead.
    question_distribution: Optional[Distribution] = None
    output_distribution: Optional[Distribution] = None

    enable_multi_turn_chat: bool = False
    multimodal: Optional[SyntheticMultimodalDatagenConfig] = None

    @model_validator(mode="after")
    def validate_no_ambiguous_distributions(self) -> "SharedPrefix":
        if isinstance(self.question_len, Distribution) and self.question_distribution is not None:
            raise ValueError(
                "Cannot specify both inline distribution on 'question_len' and legacy 'question_distribution'."
                " Use one or the other."
            )
        if isinstance(self.output_len, Distribution) and self.output_distribution is not None:
            raise ValueError(
                "Cannot specify both inline distribution on 'output_len' and legacy 'output_distribution'."
                " Use one or the other."
            )
        return self


class DataConfig(BaseModel):
    type: DataGenType = DataGenType.Mock

    # Valid only for shareGPT type at this moment
    path: Optional[str] = None  # path to the downloaded shareGPT dataset

    # Distributions are only supported for synthetic/random dataset at this moment
    input_distribution: Optional[Distribution] = None
    output_distribution: Optional[Distribution] = None
    shared_prefix: Optional[SharedPrefix] = None
    multimodal: Optional[SyntheticMultimodalDatagenConfig] = None

    # Trace file is only supported for random dataset at this moment
    trace: Optional[TraceConfig] = None

    # OTel trace replay configuration
    otel_trace_replay: Optional[OTelTraceReplayConfig] = None

    # Conversation replay configuration
    conversation_replay: Optional[ConversationReplayConfig] = None
