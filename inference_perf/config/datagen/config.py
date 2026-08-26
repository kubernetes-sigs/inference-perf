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

from pydantic import AliasChoices, ConfigDict, Field, field_validator, model_validator

from inference_perf.config.common import Distribution, StrictBaseModel, validate_length_expression
from inference_perf.config.datagen.multimodal import SyntheticMultimodalDatagenConfig
from inference_perf.config.datagen.replay import (
    ConversationReplayConfig,
    OTelTraceReplayConfig,
    SyntheticAgenticConfig,
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
    SyntheticAgentic = "synthetic_agentic"


# Configuration for shared prefix datagen which allows users to specify shared prefixes.
class SharedPrefix(StrictBaseModel):
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

    system_prompt_len: Union[int, Distribution, str] = Field(
        default=100,
        description="Length of the shared system prompt in tokens: a fixed value, a distribution,"
        " or an expression string like 'Normal(512, 200)'.",
    )
    question_len: Union[int, Distribution, str] = Field(
        default=50,
        description="Length of the question part in tokens: a fixed value, a distribution,"
        " or an expression string like 'Normal(512, 200)'.",
    )
    output_len: Union[int, Distribution, str] = Field(
        default=50,
        description="Requested output length in tokens: a fixed value, a distribution,"
        " or an expression string like 'Normal(512, 200)'.",
    )
    max_model_len: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Maximum model context length in tokens for multi-turn sessions. "
            "Defaults to 225000 when omitted, matching conversation_replay."
        ),
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

    @field_validator("system_prompt_len", "question_len", "output_len", mode="after")
    @classmethod
    def validate_length_expressions(cls, value: Union[int, Distribution, str]) -> Union[int, Distribution, str]:
        if isinstance(value, str):
            validate_length_expression(value)
        return value

    @model_validator(mode="after")
    def validate_no_ambiguous_distributions(self) -> "SharedPrefix":
        # A plain int is the only question_len/output_len form the legacy
        # fields may accompany; an inline distribution or expression string
        # would silently lose to them otherwise.
        if not isinstance(self.question_len, int) and self.question_distribution is not None:
            raise ValueError(
                "Cannot specify both inline distribution on 'question_len' and legacy 'question_distribution'."
                " Use one or the other."
            )
        if not isinstance(self.output_len, int) and self.output_distribution is not None:
            raise ValueError(
                "Cannot specify both inline distribution on 'output_len' and legacy 'output_distribution'."
                " Use one or the other."
            )
        return self


class DataConfig(StrictBaseModel):
    type: DataGenType = Field(default=DataGenType.Mock, description="Dataset or generator used to produce prompts.")

    path: Optional[str] = Field(
        default=None, description="Path to the downloaded ShareGPT dataset. Only used by the 'shareGPT' type."
    )
    corpus_file_path: Optional[str] = Field(
        None,
        description="Path to a text file to use as the prompt tokenization corpus instead of the default hardcoded sonnet",
    )

    input_distribution: Optional[Union[Distribution, str]] = Field(
        default=None,
        description="Input (prompt) length distribution in tokens: a distribution, or (for the 'synthetic' and"
        " 'random' types) an expression string like 'Normal(512, 200)'. Dataset types use the distribution's"
        " min/max as filter bounds.",
    )
    output_distribution: Optional[Union[Distribution, str]] = Field(
        default=None,
        description="Output length distribution in tokens: a distribution, or (for the 'synthetic' and"
        " 'random' types) an expression string like 'Normal(512, 200)'. Dataset types use the distribution's"
        " min/max as filter bounds.",
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

    synthetic_agentic: Optional[SyntheticAgenticConfig] = Field(
        default=None, description="Synthetic agentic sessions settings. Only used by the 'synthetic_agentic' type."
    )

    use_chat_template: bool = Field(
        default=False,
        description=(
            "Wrap each generated prompt in the tokenizer's chat template as a single user turn before sending it"
            " on the completions path, reproducing the request shape of harnesses that benchmark with chat"
            " templating enabled. The input length distribution targets the fully templated prompt, so the"
            " server-side prefill token count still matches the configured length. Only supported by the 'random'"
            " type; setting it with any other type is a config error."
        ),
    )

    @field_validator("input_distribution", "output_distribution", mode="after")
    @classmethod
    def validate_distribution_expressions(
        cls, value: Optional[Union[Distribution, str]]
    ) -> Optional[Union[Distribution, str]]:
        if isinstance(value, str):
            validate_length_expression(value)
        return value

    @model_validator(mode="after")
    def validate_expression_distribution_scope(self) -> "DataConfig":
        # Dataset generators use input/output_distribution min/max as filter
        # bounds, which an expression string does not carry; only the
        # generators that sample lengths from the field can take one.
        if isinstance(self.input_distribution, str) or isinstance(self.output_distribution, str):
            if self.type not in (DataGenType.Synthetic, DataGenType.Random):
                raise ValueError(
                    f"An expression string for input_distribution/output_distribution is only supported by the"
                    f" 'synthetic' and 'random' data generators; type '{self.type.value}' uses the distribution's"
                    f" min/max bounds, which an expression does not define."
                )
        return self

    @model_validator(mode="after")
    def validate_use_chat_template_scope(self) -> "DataConfig":
        if self.use_chat_template and self.type != DataGenType.Random:
            raise ValueError(
                f"data.use_chat_template is only supported by the 'random' data generator and would be"
                f" ignored by type '{self.type.value}'. Unset it or set data.type to 'random'."
            )
        return self

    @model_validator(mode="after")
    def validate_synthetic_agentic_scope(self) -> "DataConfig":
        if self.type == DataGenType.SyntheticAgentic and self.synthetic_agentic is None:
            raise ValueError(f"data.type '{self.type.value}' requires 'data.synthetic_agentic' to be configured.")
        return self
