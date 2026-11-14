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
from datetime import datetime
from typing import Any, List, Mapping, Optional

import yaml
from inference_perf.config.common import StrictBaseModel
from pydantic import ConfigDict, Field, model_validator

from inference_perf.config.apis import APIConfig
from inference_perf.config.circuit_breaker import CircuitBreakerConfig
from inference_perf.config.client.filestorage import StorageConfig
from inference_perf.config.client.modelserver import ModelServerClientConfig
from inference_perf.config.datagen import ConversationReplayConfig, DataConfig, DataGenType
from inference_perf.config.loadgen import (
    ConcurrentLoadStage,
    LoadConfig,
    LoadType,
    StandardLoadStage,
    TraceSessionReplayLoadStage,
)
from inference_perf.config.metrics import MetricsClientConfig
from inference_perf.config.observability import ObservabilityConfig
from inference_perf.config.redaction import REDACTED, redact, redacted_credentials
from inference_perf.config.reportgen import ReportConfig
from inference_perf.config.utils import CustomTokenizerConfig

# Generators that replay their corpus for a duration-bounded stage instead of ending when
# it runs out. Mirrors SessionGenerator.supports_corpus_cycling(): weka_trace_replay builds
# every session up front and cannot rebuild a freed slot, so it is absent on purpose.
_CORPUS_CYCLING_DATA_TYPES = frozenset({DataGenType.OTelTraceReplay, DataGenType.SyntheticAgentic})

# Where each session-replay data type keeps its generator settings on DataConfig.
_SESSION_REPLAY_CONFIG_FIELDS = {
    DataGenType.OTelTraceReplay: "otel_trace_replay",
    DataGenType.WekaTraceReplay: "weka_trace_replay",
    DataGenType.SyntheticAgentic: "synthetic_agentic",
}


def _default_conversation_replay_concurrency(config: dict[str, object]) -> dict[str, object]:
    """Fill an omitted ``concurrency_level`` in from ``num_conversations``.

    Every conversation is one session, so the whole pool is the widest a stage can run
    and the only size that needs no explaining: "run all of them". ``load`` cannot work
    that number out itself -- ``num_conversations`` lives under ``data`` -- and a stage
    field cannot carry a default that depends on a sibling section, so it is resolved
    here, on the raw input, before the stage models are built.

    Only raw input is rewritten. A validated ``DataConfig`` or ``LoadConfig``, and any
    dump of one, already carries a ``concurrency_level`` on every stage, so there is
    nothing left to fill.

    The count is copied across as written and the stage field coerces it, so the stage
    and ``data`` read every value the same way: a quoted count still sizes the stage, and
    a count ``data`` rejects is never rounded down into a stage size it would accept.

    Scoped to conversation replay on the concurrent load type, which is the only
    combination that has a pool to size. Every other path is returned untouched, so
    ``concurrency_level`` stays required for other data generators.
    """
    data_config = config.get("data")
    load_config = config.get("load")
    if not isinstance(data_config, Mapping) or not isinstance(load_config, Mapping):
        return config
    if data_config.get("type") != DataGenType.ConversationReplay.value:
        return config
    if load_config.get("type") != LoadType.CONCURRENT.value:
        return config
    stages = load_config.get("stages")
    if not isinstance(stages, list):
        return config

    # Missing conversation_replay settings are rejected by
    # validate_conversation_replay_load, but only after the stages have validated, and a
    # stage without a concurrency_level would fail first with a generic "field required".
    # Starting from the field's own default keeps the specific rejection in charge.
    conversation_config = data_config.get("conversation_replay")
    num_conversations: object = ConversationReplayConfig().num_conversations
    if isinstance(conversation_config, Mapping) and conversation_config.get("num_conversations") is not None:
        num_conversations = conversation_config["num_conversations"]

    defaulted = [
        {**stage, "concurrency_level": num_conversations}
        if isinstance(stage, Mapping) and stage.get("concurrency_level") is None
        else stage
        for stage in stages
    ]
    return {**config, "load": {**load_config, "stages": defaulted}}


class _DataSection(StrictBaseModel):
    """Validates a raw ``data`` section under the path a config file writes it at.

    ``read_config`` turns the stages into models before ``Config`` validates anything, so
    the count copied onto ``concurrency_level`` is reported against that stage field
    alone. Going through ``data`` first puts the error on the field the file actually
    sets, ``data.conversation_replay.num_conversations``, which is also where
    ``Config.model_validate`` reports it.
    """

    data: DataConfig


class Config(StrictBaseModel):
    # A validation error would otherwise quote the input, which holds credentials.
    model_config = ConfigDict(hide_input_in_errors=True)

    api: APIConfig = Field(
        default=APIConfig(), description="API endpoint type and request options used for benchmark requests."
    )
    data: DataConfig = Field(default=DataConfig(), description="Dataset selection and prompt generation settings.")
    load: LoadConfig = Field(default=LoadConfig(), description="Load generation settings: load type, stages, and worker pool.")
    metrics: Optional[MetricsClientConfig] = Field(
        default=None, description="Metrics client settings for collecting server-side metrics."
    )
    report: ReportConfig = Field(default=ReportConfig(), description="Report generation settings for the benchmark results.")
    storage: Optional[StorageConfig] = Field(
        default=StorageConfig(), description="Where generated reports are saved (local path or cloud object storage)."
    )
    server: Optional[ModelServerClientConfig] = Field(
        default=None, description="Model server under test: server type, model name, and base URL."
    )
    tokenizer: Optional[CustomTokenizerConfig] = Field(
        default=None, description="Tokenizer used to count prompt and output tokens. Defaults to the server's model name."
    )
    circuit_breakers: Optional[List[CircuitBreakerConfig]] = Field(
        default=None, description="Circuit breakers that stop the run when observed metrics cross configured thresholds."
    )
    observability: ObservabilityConfig = Field(
        default=ObservabilityConfig(),
        description="Runtime observability of the benchmark run itself: the metrics inference-perf exports.",
    )

    @model_validator(mode="before")
    @classmethod
    def reject_placeholder_credentials(cls, data: Any) -> Any:
        # A saved or dumped config holds a placeholder in place of each credential.
        # read_config validates the merged input, so a credential supplied on the command line counts.
        placeholders = redacted_credentials(data, cls) if isinstance(data, Mapping) else []
        if placeholders:
            raise ValueError(
                f"These credentials hold a masked placeholder, such as {REDACTED}, instead of a value: "
                f"{', '.join(placeholders)}. Supply the real values in the config file or on the command line, "
                "or remove these settings."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def default_conversation_replay_concurrency(cls, data: object) -> object:
        if isinstance(data, dict):
            return _default_conversation_replay_concurrency(data)
        return data

    @model_validator(mode="after")
    def validate_trace_replay_load_type(self) -> "Config":
        """Validate that trace replay data types use trace_session_replay load type."""
        if self.data.type in (DataGenType.OTelTraceReplay, DataGenType.WekaTraceReplay, DataGenType.SyntheticAgentic):
            if self.load.type != LoadType.TRACE_SESSION_REPLAY:
                raise ValueError(
                    f"data.type '{self.data.type.value}' requires load.type 'trace_session_replay', "
                    f"but got '{self.load.type.value}'. Trace replay with dependencies requires "
                    f"session-based load dispatch to properly handle event dependencies and timing."
                )
        return self

    @model_validator(mode="after")
    def validate_duration_bounded_replay_stage(self) -> "Config":
        """Reject datagen settings a duration-bounded stage would silently contradict.

        Such a stage replays the corpus, so duplicate sessions now arise from ``load`` too.
        Two ``data`` settings were written when ``data`` was their only source, and neither
        surface's validator sees both halves — only this one does.
        """
        if self.data.type not in _CORPUS_CYCLING_DATA_TYPES:
            return self
        duration_stages = [
            stage
            for stage in self.load.stages
            if isinstance(stage, TraceSessionReplayLoadStage) and stage.duration is not None
        ]
        if not duration_stages:
            return self

        # Admission needs at least one bound. A replaying corpus is not one: the dispatch
        # loop keeps admitting while the pool has room and the supply holds, and the supply
        # never runs out, so with no concurrency cap and no rate the loop only stops at the
        # deadline — starting sessions as fast as they can be built for the whole window.
        for stage in duration_stages:
            if stage.concurrent_sessions == 0 and stage.session_rate is None:
                raise ValueError(
                    "concurrent_sessions: 0 cannot be combined with a duration-bounded "
                    f"'{self.data.type.value}' stage unless session_rate is set: 0 means start every "
                    "session at once, which has no meaning for a corpus that replays for as long as "
                    "the stage runs - there is no last session to start. Nothing would then limit how "
                    "many sessions the stage opens, so it would keep building them until the deadline "
                    "or until it ran out of memory. Set concurrent_sessions to the pool size you want "
                    "to hold open, or set session_rate to bound how fast sessions start."
                )

        replay_config = getattr(self.data, _SESSION_REPLAY_CONFIG_FIELDS[self.data.type], None)
        if replay_config is None:
            return self

        if getattr(replay_config, "duplicate_sessions_target", None) is not None:
            raise ValueError(
                "duplicate_sessions_target cannot be combined with a duration-bounded "
                f"'{self.data.type.value}' stage: such a stage replays the corpus itself once it is "
                "exhausted, numbering each replay per source session, while "
                "duplicate_sessions_target numbers its copies with a single running counter. The "
                "two independently mint the same '_dupN' session ID, and session state, completion "
                "tracking and cleanup are all keyed by that ID. Remove duplicate_sessions_target: a "
                "duration-bounded stage no longer needs the corpus padded by hand."
            )

        if getattr(replay_config, "disable_output_substitution", False):
            raise ValueError(
                "disable_output_substitution=True cannot be combined with a duration-bounded "
                f"'{self.data.type.value}' stage: such a stage replays the corpus once it is "
                "exhausted, and a replayed session triggers random session-ID injection, which "
                "substitutes live predecessor output into output/shared segments — the opposite of "
                "replaying recorded outputs as-is. Bound the stage by num_sessions to replay "
                "recorded outputs, or set disable_output_substitution=False to allow substitution."
            )

        return self

    @model_validator(mode="after")
    def validate_conversation_replay_load(self) -> "Config":
        if self.data.type != DataGenType.ConversationReplay:
            return self
        if self.load.type != LoadType.CONCURRENT:
            raise ValueError("data.type 'conversation_replay' requires load.type 'concurrent'")

        conversation_config = self.data.conversation_replay
        if conversation_config is None:
            raise ValueError("data.conversation_replay is required when data.type is 'conversation_replay'")
        for stage_index, stage in enumerate(self.load.stages):
            if isinstance(stage, ConcurrentLoadStage) and stage.concurrency_level > conversation_config.num_conversations:
                raise ValueError(
                    f"load.stages[{stage_index}].concurrency_level ({stage.concurrency_level}) cannot exceed "
                    f"data.conversation_replay.num_conversations ({conversation_config.num_conversations})"
                )
        return self


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def read_config(config_file: Optional[str] = None, cli_overrides: Optional[dict[str, Any]] = None) -> Config:
    logger = logging.getLogger(__name__)
    cfg: dict[str, Any] = {}
    if config_file:
        logger.info("Using configuration from: %s", config_file)
        with open(config_file, "r") as stream:
            cfg = yaml.safe_load(stream) or {}

    default_cfg = Config().model_dump(mode="json")
    merged_cfg = deep_merge(default_cfg, cfg)

    if cli_overrides:
        merged_cfg = deep_merge(merged_cfg, cli_overrides)

    # Handle timestamp substitution in storage paths
    if "storage" in merged_cfg and merged_cfg["storage"]:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for storage_type in ["local_storage", "google_cloud_storage", "simple_storage_service"]:
            if (
                storage_type in merged_cfg["storage"]
                and merged_cfg["storage"][storage_type]
                and "path" in merged_cfg["storage"][storage_type]
            ):
                path = merged_cfg["storage"][storage_type]["path"]
                if path and "{timestamp}" in path:
                    merged_cfg["storage"][storage_type]["path"] = path.replace("{timestamp}", timestamp)

    # A conversation replay stage may omit concurrency_level, but the stages below are
    # turned into models before Config validates them, and the field is required there.
    # The helper only rewrites "load", and hands back a copy, so updating with it leaves
    # the rest of the merged config in place.
    merged_cfg.update(_default_conversation_replay_concurrency(merged_cfg))

    # The stages below are turned into models before Config validates them, so the count
    # just copied onto concurrency_level would be reported against that field alone when
    # data rejects it. Validating data first keeps the error on the field the file sets.
    data_section = merged_cfg.get("data")
    if isinstance(data_section, Mapping) and data_section.get("type") == DataGenType.ConversationReplay.value:
        _DataSection.model_validate({"data": data_section})

    # Handle stage type conversion based on load type
    if "load" in merged_cfg and "stages" in merged_cfg["load"] and merged_cfg["load"]["stages"]:
        load_type = merged_cfg["load"].get("type", "constant")
        stages = merged_cfg["load"]["stages"]

        if load_type == "concurrent":
            # Convert to ConcurrentLoadStage objects
            concurrent_stages = []
            for stage in stages:
                concurrent_stages.append(ConcurrentLoadStage(**stage))
            merged_cfg["load"]["stages"] = concurrent_stages
        elif load_type == "trace_session_replay":
            # Convert to TraceSessionReplayLoadStage objects
            trace_session_stages = []
            for stage in stages:
                trace_session_stages.append(TraceSessionReplayLoadStage(**stage))
            merged_cfg["load"]["stages"] = trace_session_stages
        else:
            # Convert to StandardLoadStage objects for constant/poisson/trace_replay
            standard_stages = []
            for stage in stages:
                standard_stages.append(StandardLoadStage(**stage))
            merged_cfg["load"]["stages"] = standard_stages

    # The echoed config is the copy people paste into bug reports, so it goes out
    # with its credentials masked. merged_cfg itself keeps them, since the run
    # needs them.
    logger.info(
        "Benchmarking with the following config:\n\n%s\n",
        yaml.dump(redact(merged_cfg, Config), sort_keys=False, default_flow_style=False),
    )
    return Config(**merged_cfg)
