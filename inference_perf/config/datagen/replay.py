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
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, model_validator

from inference_perf.config.common import Distribution


class TraceFormat(Enum):
    AZURE_PUBLIC_DATASET = "AzurePublicDataset"


class TraceConfig(BaseModel):
    file: str
    format: TraceFormat = TraceFormat.AZURE_PUBLIC_DATASET


class ConversationReplayConfig(BaseModel):
    """Configuration for conversation replay data generator.

    Generates synthetic multi-turn conversations in-memory from configurable
    distributions. Each conversation has a two-part system prompt (shared prefix
    + dynamic per-conversation suffix) and a sequence of user/assistant turns
    with independently sampled input/output token lengths.
    """

    seed: int = Field(42, description="Random seed for deterministic generation")
    num_conversations: int = Field(200, gt=0, description="Number of conversation blueprints to generate")
    shared_system_prompt_len: int = Field(8359, ge=0, description="Fixed shared system prompt length in tokens")
    dynamic_system_prompt_len: Optional[Distribution] = Field(
        None, description="Per-conversation dynamic system prompt length distribution"
    )
    turns_per_conversation: Optional[Distribution] = Field(None, description="Number of turns per conversation distribution")
    input_tokens_per_turn: Optional[Distribution] = Field(None, description="Input tokens per turn distribution")
    output_tokens_per_turn: Optional[Distribution] = Field(None, description="Output tokens per turn distribution")
    tool_call_latency_sec: Optional[Distribution] = Field(
        None,
        description=(
            "Per-turn tool execution latency distribution in seconds. "
            "When set, each turn sleeps for the sampled duration after model "
            "inference completes and before the next turn begins, simulating "
            "tool call round-trips. The sleep holds the session lock so the "
            "GPU is free to serve other concurrent conversations — correctly "
            "modelling offline agentic workloads. Omit for pure GPU throughput "
            "measurement. Values are in seconds; min/max are whole seconds, "
            "mean/std_dev may be fractional."
        ),
    )
    max_model_len: Optional[int] = Field(None, description="Maximum model context length in tokens")


class SessionReplayConfig(BaseModel):
    """Base configuration for session replay data generators."""

    # Model configuration
    use_static_model: bool = Field(False, description="Use a single static model for all requests")
    static_model_name: str = Field("", description="Static model name (required if use_static_model=True)")
    model_mapping: Optional[Dict[str, str]] = Field(None, description="Map recorded model names to target models")

    # Request configuration
    default_max_tokens: int = Field(1000, gt=0, description="Default max_tokens if not specified in trace")

    # KV-cache invalidation
    inject_random_session_id: bool = Field(
        False, description="Inject random string into unique segments to invalidate KV-cache between sessions"
    )

    # Session duplication
    duplicate_sessions_target: Optional[int] = Field(
        None,
        gt=0,
        description="Target number of sessions to reach by duplicating existing sessions. If None, no duplication occurs.",
    )

    # Timing
    max_wait_ms: int = Field(
        15000,
        ge=0,
        description="Maximum inter-event wait time in milliseconds. Caps the delay between predecessor completion and event dispatch to avoid reproducing unusually long tool/agent execution times from the original trace.",
    )

    # Error handling
    include_errors: bool = Field(True, description="Include spans with error status")
    skip_invalid_files: bool = Field(False, description="Skip invalid trace files instead of failing")

    @model_validator(mode="after")
    def validate_static_model(self) -> "SessionReplayConfig":
        # Validate static model configuration
        if self.use_static_model and not self.static_model_name:
            raise ValueError("static_model_name is required when use_static_model=True")
        if not self.use_static_model and self.static_model_name and not self.model_mapping:
            raise ValueError("Either use_static_model must be True or model_mapping must be provided")
        return self


class OTelTraceReplayConfig(SessionReplayConfig):
    """Configuration for OTel trace replay data generator."""

    trace_directory: Optional[str] = Field(None, description="Directory containing OTel JSON trace files")
    trace_files: Optional[List[str]] = Field(None, description="List of paths to specific OTel JSON trace files")
    hf_dataset_path: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        description=(
            "HuggingFace dataset path. Can be:\n"
            "  - String: 'username/dataset-name'\n"
            "  - Dict: {'path': 'username/dataset-name', 'revision': 'main', 'split': 'train'}"
        ),
    )

    @model_validator(mode="after")
    def validate_trace_sources(self) -> "OTelTraceReplayConfig":
        # Validate that exactly one of trace_directory, trace_files, or hf_dataset_path is provided
        sources_provided = sum(
            [
                self.trace_directory is not None,
                self.trace_files is not None,
                self.hf_dataset_path is not None,
            ]
        )

        if sources_provided == 0:
            raise ValueError("Either trace_directory, trace_files, or hf_dataset_path must be provided")
        if sources_provided > 1:
            raise ValueError(
                "Cannot specify multiple trace sources; choose one of: trace_directory, trace_files, or hf_dataset_path"
            )
        return self
