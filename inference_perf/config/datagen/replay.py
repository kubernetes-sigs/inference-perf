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

from pydantic import Field, model_validator

from inference_perf.config.common import Distribution, StrictBaseModel


class TraceFormat(Enum):
    AZURE_PUBLIC_DATASET = "AzurePublicDataset"


class BadToolCallHandling(str, Enum):
    """How to handle a tool-call response whose `function.arguments` is not
    valid JSON.

    Some server-side tool-call parsers (e.g. vLLM's `qwen3_xml`) leak parser
    markers into the `arguments` JSON string at decode time. vLLM still
    returns 200 on the response, but on the *next* turn the chat template's
    `json.loads(arguments)` raises and vLLM returns HTTP 400. Replaying the
    bad bytes verbatim therefore halts the session.

    none
        Default. No mitigation. Bytes propagate; vLLM may HTTP-400 on the
        next turn. Use for benchmarking the upstream bug or for strict
        trace fidelity.

    use_recorded
        When the live model returns malformed `arguments` for a tool_call,
        discard the live response and substitute the recorded assistant
        message at this slot. The recorded `tool_call_id` flows naturally
        into the recorded role:tool successor that follows. The next-turn
        request is structurally identical to a healthy replay: same
        message count, same roles, valid JSON in arguments, matching
        tool_call_id pairs. The next-turn live model never sees its own
        malformed output.

        If the recorded message ALSO has malformed tool_calls (the trace
        was captured from a buggy parser too), the current event is
        hard-failed via _fail_and_notify(); EventFailedError cascades to
        downstream events that await this one's output. Parallel DAG
        branches continue.
    """

    NONE = "none"
    USE_RECORDED = "use_recorded"


class TraceConfig(StrictBaseModel):
    file: str = Field(description="Path to the trace file to replay.")
    format: TraceFormat = Field(default=TraceFormat.AZURE_PUBLIC_DATASET, description="Format of the trace file.")


class ConversationReplayConfig(StrictBaseModel):
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


class SessionReplayConfig(StrictBaseModel):
    """Base configuration for session replay data generators."""

    # Model configuration
    use_static_model: bool = Field(False, description="Use a single static model for all requests")
    static_model_name: str = Field("", description="Static model name (required if use_static_model=True)")
    model_mapping: Optional[Dict[str, str]] = Field(None, description="Map recorded model names to target models")

    # Request configuration
    default_max_tokens: int = Field(1000, gt=0, description="Default max_tokens if not specified in trace")
    override_tool_call_max_tokens: bool = Field(
        True,
        description="Override tool call max_tokens to 4096 instead of using trace recorded length",
    )

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
    predecessor_wait_timeout_sec: float = Field(
        3600.0,
        ge=0,
        description=(
            "Seconds to wait for predecessor events to complete before failing. "
            "0 waits indefinitely; use with care because a genuinely stuck predecessor "
            "will then never time out and successors will wait forever."
        ),
    )

    # Error handling
    include_errors: bool = Field(True, description="Include spans with error status")
    skip_invalid_files: bool = Field(False, description="Skip invalid trace files instead of failing")

    # Client-side mitigation for server-side tool-call parser bugs (e.g.
    # vLLM's `qwen3_xml` leaking closing XML markers into the JSON `arguments`
    # string at decode time). The default `none` preserves upstream behavior
    # (the bug reproduces). `use_recorded` substitutes the recorded
    # assistant message at the affected slot. See BadToolCallHandling.
    bad_tool_call_handling: BadToolCallHandling = Field(
        BadToolCallHandling.NONE,
        description=(
            "How to handle tool_calls whose function.arguments is not valid "
            "JSON. none (default): no mitigation, bytes propagate and vLLM "
            "may return HTTP 400 on the next turn. use_recorded: discard "
            "the live response and substitute the recorded assistant "
            "message at the affected slot; the recorded tool_call_id flows "
            "into the recorded role:tool successor unchanged."
        ),
    )

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
            "  - Dict: {'path': 'username/dataset-name', 'revision': 'main', 'split': 'train'}\n"
            "Any extra keys in the dict are passed as kwargs to datasets.load_dataset()."
        ),
    )
    filter: Optional[str] = Field(
        None,
        description=(
            "Lambda expression to filter trace records. Applied uniformly to all data sources.\n"
            "Example: \"lambda x: x['benchmark'] == 'gsm8k'\" or \"lambda x: 'spans' in x and len(x['spans']) > 5\"\n"
            "Security: Filter expressions use eval() and should only contain trusted input."
        ),
    )
    disable_output_substitution: bool = Field(
        False,
        description=(
            "When True, replay each call with its recorded assistant output "
            "(text and tool calls) instead of substituting the live output from "
            "predecessor calls. Dependency timing (waiting for predecessors) is "
            "still enforced. Default False preserves faithful live-output replay."
        ),
    )
    attribute_to_header_map: Optional[Dict[str, str]] = Field(None, description="Map OTel span attributes to HTTP headers")
    attribute_to_label_map: Optional[Dict[str, str]] = Field(
        None, description="Map OTel span attributes to metrics reporting labels"
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

    @model_validator(mode="after")
    def validate_output_substitution(self) -> "OTelTraceReplayConfig":
        # disable_output_substitution sends recorded assistant outputs verbatim.
        # Random session-ID injection (via inject_random_session_id or session
        # duplication) rewrites 'unique' segments, which runs the substitution
        # pass that also replaces 'output'/'shared' segments with live predecessor
        # output — the exact behavior disable_output_substitution asks to turn off.
        # The two settings contradict, so reject the combination up front rather
        # than silently substituting anyway.
        if self.disable_output_substitution:
            conflicting = []
            if self.inject_random_session_id:
                conflicting.append("inject_random_session_id")
            if self.duplicate_sessions_target is not None:
                conflicting.append("duplicate_sessions_target")
            if conflicting:
                raise ValueError(
                    "disable_output_substitution=True cannot be combined with "
                    f"{' or '.join(conflicting)}: those options trigger random "
                    "session-ID injection, which substitutes live predecessor "
                    "output into output/shared segments — the opposite of replaying "
                    "recorded outputs as-is. Disable "
                    f"{' and '.join(conflicting)} to replay recorded outputs, or "
                    "set disable_output_substitution=False to allow substitution."
                )
        return self


class WekaTraceReplayConfig(SessionReplayConfig):
    """Configuration for Weka trace replay data generator."""

    trace_directory: Optional[str] = Field(None, description="Directory containing Weka JSON trace files")
    trace_files: Optional[List[str]] = Field(None, description="List of paths to specific Weka JSON trace files")
    hf_dataset_path: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        description=(
            "HuggingFace dataset path. Can be:\n"
            "  - String: 'username/dataset-name'\n"
            "  - Dict: {'path': 'username/dataset-name', 'revision': 'main', 'split': 'train'}\n"
            "Any extra keys in the dict are passed as kwargs to datasets.load_dataset()."
        ),
    )
    trace_idle_gap_cap_seconds: float = Field(60.0, description="Cap idle timing gaps between turns in seconds")
    ignore_trace_delays: bool = Field(False, description="Ignore delays/delays from original trace and run back-to-back")
    use_think_time_only: bool = Field(False, description="Only use think_time attribute instead of timestamps")
    default_block_size: int = Field(64, description="Default block size if not specified in trace")
    num_dataset_entries: int = Field(100, description="Max number of dataset traces to load from HuggingFace")

    @model_validator(mode="after")
    def validate_trace_sources(self) -> "WekaTraceReplayConfig":
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


class SyntheticAgentSessionsConfig(SessionReplayConfig):
    """Procedural multi-agent agentic session generation (§8)."""

    # Required — the four workload-shape decisions (no default)
    num_sessions: int = Field(..., gt=0, description="Number of sessions (load volume)")
    rounds_per_session: Distribution = Field(..., description="N principal inputs to the root; N=1 autonomous")
    fanout_probability: float = Field(..., ge=0.0, le=1.0, description="P(an agent execution spawns sub-agents)")
    theme_mix: Dict[str, float] = Field(..., description="theme name -> weight")

    # Defaulted
    seed: int = Field(42, description="Base seed for stable per-session RNG (§2.3a)")
    shared_system_prompt_len: int = Field(0, ge=0, description="Invariant system-prompt head length in tokens")
    tool_turns_per_loop: Optional[Distribution] = Field(None, description="tool-call TURNS per loop (fallback fixed 2)")
    sub_agents_per_spawn: Optional[Distribution] = Field(None, description="K children per spawn (fallback uniform 2-4)")
    max_depth: int = Field(2, ge=0, description="Hard recursion terminator")
    max_events_per_session: int = Field(64, gt=0, description="Self-limiting event budget")
    tool_definitions_per_agent: Optional[Distribution] = Field(None, description="advertised catalog size (fallback fixed 8)")
    parallel_tool_calls_per_turn: Optional[Distribution] = Field(
        None, description="calls per ordinary tool turn (fallback fixed 1)"
    )
    input_tokens_per_turn: Distribution = Field(..., description="per-turn input tokens")
    output_tokens_per_turn: Distribution = Field(..., description="per-turn output tokens (plain-text turns)")
    tool_call_latency_sec: Distribution = Field(..., description="machine/agent wait_ms gaps")
    user_think_time_sec: Optional[Distribution] = Field(None, description="human gap before rounds 2..N")
    max_model_len: Optional[int] = Field(None, description="fail-fast context-length ceiling")

    # Pinned inert / not for synthetic (§2.2a, C3)
    inject_random_session_id: bool = Field(False, frozen=True)
    duplicate_sessions_target: Optional[int] = Field(None, frozen=True)
    override_tool_call_max_tokens: bool = Field(False)

    @model_validator(mode="after")
    def validate_theme_mix(self) -> "SyntheticAgentSessionsConfig":
        if not self.theme_mix:
            raise ValueError("theme_mix must be non-empty (at least one theme name -> weight)")
        if any(w < 0 for w in self.theme_mix.values()):
            raise ValueError(f"theme_mix weights must all be >= 0, got: {self.theme_mix}")
        if sum(self.theme_mix.values()) <= 0:
            raise ValueError(f"theme_mix weights must sum to a positive value (at least one weight > 0), got: {self.theme_mix}")
        return self

    @model_validator(mode="after")
    def validate_max_model_len(self) -> "SyntheticAgentSessionsConfig":
        # Fail-fast context-length ceiling check (§8). Uses the mean of
        # input_tokens_per_turn as the "typical" per-turn input size: mean is
        # always a meaningful, well-defined statistic for every Distribution
        # type (fixed, uniform, normal, ...), unlike max which for some
        # distribution types is just an unconfigured default ceiling rather
        # than a value the sampler is actually likely to produce. A config
        # whose shared invariant system-prompt head plus a *typical* turn
        # already exceeds the model's context window is obviously going to
        # overrun in practice, so we reject it here rather than let it fail
        # deep inside a live benchmark run.
        if self.max_model_len is not None:
            projected = self.shared_system_prompt_len + self.input_tokens_per_turn.mean
            if projected > self.max_model_len:
                raise ValueError(
                    "max_model_len is too small for this configuration: "
                    f"shared_system_prompt_len ({self.shared_system_prompt_len}) + "
                    f"input_tokens_per_turn.mean ({self.input_tokens_per_turn.mean}) = {projected} "
                    f"exceeds max_model_len ({self.max_model_len})"
                )
        return self
