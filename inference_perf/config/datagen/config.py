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
    WekaTraceReplayConfig,
    TraceConfig,
)
from inference_perf.config.datagen.visionarena import VisionArenaConfig


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
    WekaTraceReplay = "weka_trace_replay"
    ConversationReplay = "conversation_replay"
    VisionArena = "visionarena"


# Configuration for shared prefix datagen which allows users to specify shared prefixes.
class SharedPrefix(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    num_groups: int = Field(
        10,
        validation_alias=AliasChoices("num_unique_system_prompts", "num_groups"),
        serialization_alias="num_unique_system_prompts",
        description="Number of unique system prompts (shared prefix groups) to generate.",
    )

    num_prompts_per_group: int = Field(
        10,
        validation_alias=AliasChoices("num_users_per_system_prompt", "num_prompts_per_group"),
        serialization_alias="num_users_per_system_prompt",
        description="Number of prompts generated per shared system prompt.",
    )

    system_prompt_len: Union[int, Distribution] = Field(
        default=100, description="Length of the shared system prompt in tokens: a fixed value or a distribution."
    )
    question_len: Union[int, Distribution] = Field(
        default=50, description="Length of the question part in tokens: a fixed value or a distribution."
    )
    output_len: Union[int, Distribution] = Field(
        default=50, description="Requested output length in tokens: a fixed value or a distribution."
    )
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible prompt generation.")

    # Legacy distribution fields — kept for backward compatibility.
    # Prefer using inline distribution syntax on question_len/output_len instead.
    question_distribution: Optional[Distribution] = Field(
        default=None, description="Legacy question length distribution. Prefer an inline distribution on 'question_len'."
    )
    output_distribution: Optional[Distribution] = Field(
        default=None, description="Legacy output length distribution. Prefer an inline distribution on 'output_len'."
    )

    enable_multi_turn_chat: bool = Field(
        default=False, description="Send each group's prompts as consecutive turns of one chat conversation."
    )
    multimodal: Optional[SyntheticMultimodalDatagenConfig] = Field(
        default=None, description="Attach synthetic multimodal content (images, video, audio) to generated prompts."
    )

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
    type: DataGenType = Field(default=DataGenType.Mock, description="Dataset or generator used to produce prompts.")

    path: Optional[str] = Field(
        default=None, description="Path to the downloaded ShareGPT dataset. Only used by the 'shareGPT' type."
    )
    corpus_file_path: Optional[str] = Field(
        None,
        description="Path to a text file to use as the prompt tokenization corpus instead of the default hardcoded sonnet",
    )

    input_distribution: Optional[Distribution] = Field(
        default=None,
        description="Input (prompt) length distribution in tokens. Only used by the 'synthetic' and 'random' types.",
    )
    output_distribution: Optional[Distribution] = Field(
        default=None,
        description="Output length distribution in tokens. Only used by the 'synthetic' and 'random' types.",
    )
    shared_prefix: Optional[SharedPrefix] = Field(
        default=None, description="Shared prefix generator settings. Only used by the 'shared_prefix' type."
    )
    multimodal: Optional[SyntheticMultimodalDatagenConfig] = Field(
        default=None, description="Attach synthetic multimodal content (images, video, audio) to generated prompts."
    )

    trace: Optional[TraceConfig] = Field(
        default=None, description="Prompt trace file to replay. Only used by the 'random' type."
    )

    otel_trace_replay: Optional[OTelTraceReplayConfig] = Field(
        default=None, description="OTel trace replay settings. Only used by the 'otel_trace_replay' type."
    )

    weka_trace_replay: Optional[WekaTraceReplayConfig] = Field(
        default=None, description="Weka trace replay settings. Only used by the 'weka_trace_replay' type."
    )

    conversation_replay: Optional[ConversationReplayConfig] = Field(
        default=None, description="Synthetic conversation replay settings. Only used by the 'conversation_replay' type."
    )

    visionarena: Optional[VisionArenaConfig] = Field(
        default=None, description="VisionArena-Chat dataset settings. Only used by the 'visionarena' type."
    )
