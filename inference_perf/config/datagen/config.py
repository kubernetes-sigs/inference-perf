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
import math
from enum import Enum
from typing import Optional

from pydantic import AliasChoices, ConfigDict, Field, model_validator

from inference_perf.config.common import IOLengthValue, LengthValue, StrictBaseModel
from inference_perf.config.datagen.multimodal import SyntheticMultimodalDatagenConfig
from inference_perf.config.datagen.replay import (
    ConversationReplayConfig,
    OTelTraceReplayConfig,
    SyntheticAgenticConfig,
    WekaTraceReplayConfig,
    TraceConfig,
)
from inference_perf.config.datagen.visionarena import VisionArenaConfig


logger = logging.getLogger(__name__)


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

    system_prompt_len: LengthValue = Field(
        default=100,
        description="Length of the shared system prompt in tokens: a fixed value or an expression string"
        " like 'Min(Max(Normal(512, 200), 10), 1024)'.",
    )
    question_len: LengthValue = Field(
        default=50,
        description="Length of the question part in tokens: a fixed value or an expression string"
        " like 'Min(Max(Normal(512, 200), 10), 1024)'.",
    )
    output_len: LengthValue = Field(
        default=50,
        description="Requested output length in tokens: a fixed value or an expression string"
        " like 'Min(Max(Normal(512, 200), 10), 1024)'.",
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

    # Deprecated aliases: folded into question_len/output_len on load and
    # left as None, removed together with the distribution block.
    question_distribution: Optional[LengthValue] = Field(
        default=None, description="Deprecated: set 'question_len' to the distribution instead."
    )
    output_distribution: Optional[LengthValue] = Field(
        default=None, description="Deprecated: set 'output_len' to the distribution instead."
    )

    enable_multi_turn_chat: bool = Field(
        default=False,
        description="Send each group's prompts as consecutive turns of one chat conversation. Requires api.type completion.",
    )
    multimodal: Optional[SyntheticMultimodalDatagenConfig] = Field(
        default=None, description="Attach synthetic multimodal content (images, video, audio) to generated prompts."
    )

    @model_validator(mode="after")
    def fold_legacy_distribution_aliases(self) -> "SharedPrefix":
        # A plain int is the only question_len/output_len form the legacy
        # fields may accompany; an expression string would silently lose to
        # them otherwise. The alias then replaces the int and is cleared, so
        # the datagen reads one field.
        for alias, field in (("question_distribution", "question_len"), ("output_distribution", "output_len")):
            value = getattr(self, alias)
            if value is None:
                continue
            if not isinstance(getattr(self, field), int):
                raise ValueError(
                    f"Cannot specify both inline distribution on '{field}' and legacy '{alias}'. Use one or the other."
                )
            shown = f'"{value}"' if isinstance(value, str) else str(value)
            logger.warning(
                "shared_prefix.%s is deprecated and will be removed in a future release. Set `%s: %s` instead.",
                alias,
                field,
                shown,
            )
            setattr(self, field, value)
            setattr(self, alias, None)
        return self


class DataConfig(StrictBaseModel):
    type: DataGenType = Field(default=DataGenType.Mock, description="Dataset or generator used to produce prompts.")

    path: Optional[str] = Field(
        default=None, description="Path to the downloaded ShareGPT dataset. Only used by the 'shareGPT' type."
    )
    load_timeout: Optional[float] = Field(
        default=300.0,
        gt=0,
        allow_inf_nan=False,
        description="Deadline in seconds for Hub dataset initialization, including startup row reads."
        " Does not bound later streaming iteration. Null disables the deadline.",
    )
    corpus_file_path: Optional[str] = Field(
        None,
        description="Path to a text file to use as the prompt tokenization corpus instead of the default hardcoded sonnet",
    )

    input_distribution: Optional[IOLengthValue] = Field(
        default=None,
        description="Input (prompt) length in tokens: a fixed value or an expression string like"
        " 'Min(Max(Normal(512, 200), 10), 1024)'. The 'synthetic' and 'random' types sample prompt lengths"
        " from it; dataset types use its provable bounds as token filter bounds.",
    )
    output_distribution: Optional[IOLengthValue] = Field(
        default=None,
        description="Output length in tokens: a fixed value or an expression string like"
        " 'Min(Max(Normal(512, 200), 10), 1024)'. The 'synthetic' and 'random' types sample output lengths"
        " from it; dataset types use its provable bounds as token filter bounds.",
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

    @model_validator(mode="after")
    def validate_dataset_filter_bounds(self) -> "DataConfig":
        # Dataset generators use input/output_distribution as token filter
        # bounds, so for them the value has to be provably bounded on both
        # sides; only the generators that sample lengths from the field can
        # take an open-ended expression.
        if self.type in (DataGenType.Synthetic, DataGenType.Random):
            return self
        from inference_perf.utils.numeric.distribution import value_bounds

        for name in ("input_distribution", "output_distribution"):
            value = getattr(self, name)
            if value is None:
                continue
            low, high = value_bounds(value)
            if math.isinf(low) or math.isinf(high):
                raise ValueError(
                    f"{name} {value!r} is used by the '{self.type.value}' data generator as token filter bounds, so it"
                    f" must be provably bounded on both sides, e.g. 'Uniform(10, 1024)' or 'Min(Max({value}, 10), 1024)'."
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
