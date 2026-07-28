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

from inference_perf.config.common import Distribution, DistributionType, StrictBaseModel


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


class ContextCompactionConfig(BaseModel):
    """Context compaction policy: model long-horizon agents that COMPACT
    instead of growing the transcript forever.

    When present, and a round's accumulated principal input (message content +
    advertised tool catalog) would cross `trigger_tokens`, the NEXT round starts
    fresh: a `target_tokens`-sized summary block replaces the grown transcript
    instead of re-injecting it — a sharp prefill drop + KV-prefix reset that pure
    growth can't produce. Omit the whole block to never compact (pure growth).
    Both fields are required when the block is present.
    """

    trigger_tokens: Distribution = Field(
        ...,
        description=(
            "When a round's accumulated principal input (message content + advertised tool "
            "catalog) would cross this many tokens, the NEXT round starts fresh with a "
            "summary block replacing the grown transcript. Sampled per session."
        ),
    )
    target_tokens: Distribution = Field(
        ...,
        description=(
            "Size (tokens) of the summary block that replaces the transcript on compaction. "
            "Sampled per session. Set it to the size you want the post-compaction context to "
            "have -- typically a fraction of the trigger (a real compaction reduces a bloated "
            "window to ~20-40% of its size)."
        ),
    )


class SyntheticAgenticConfig(SessionReplayConfig):
    """Procedural multi-agent agentic session generation."""

    # Required: load volume + per-turn token sizing (these drive the load, so the user
    # must choose them; there is no neutral default token profile).
    num_sessions: int = Field(..., gt=0, description="Number of sessions (load volume)")
    input_tokens_per_turn: Distribution = Field(..., description="per-turn input tokens")
    output_tokens_per_turn: Distribution = Field(..., description="per-turn output tokens (plain-text turns)")

    # Structural/content shape: sensible defaults, override to shape the workload.
    turns_per_session: Distribution = Field(
        default_factory=lambda: Distribution(type="fixed", mean=1),
        description="N user turns to the root agent (each triggers one agent run); default 1 = autonomous single-turn",
    )
    fanout_probability: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Probability an agent spawns sub-agents (instead of just answering), rolled fresh for "
            "each of the root's turns and once for each sub-agent. Default 0 = single-agent; 1 = always "
            "spawn (full tree to max_depth)."
        ),
    )
    theme_mix: Dict[str, float] = Field(
        default_factory=lambda: {"generic": 0.25, "db2_latency_incident": 0.25, "research_rag": 0.25, "code_change_task": 0.25},
        description="theme name -> weight; default is an equal mix of the four built-in themes",
    )

    # Defaulted
    seed: int = Field(42, description="Base seed for stable per-session RNG")
    shared_system_prompt_len: int = Field(0, ge=0, description="Invariant system-prompt head length in tokens")
    tool_loop_depth: Optional[Distribution] = Field(
        None,
        description=(
            "How many times an agent goes around its tool loop before answering -- each iteration "
            "is a model call that emits a tool call and gets a result. 0 = answer directly (no tool "
            "loop). Then the agent makes one more model call for its final answer, so total model "
            "calls = this value + 1. Drawn fresh for each of the root's turns, and once for each "
            "sub-agent. Fallback fixed 2."
        ),
    )
    sub_agents_per_spawn: Optional[Distribution] = Field(None, description="K children per spawn (fallback uniform 2-4)")
    max_depth: int = Field(2, ge=0, description="Hard recursion terminator")
    max_events_per_session: int = Field(64, gt=0, description="Self-limiting event budget")
    tool_catalog_size_per_agent: Optional[Distribution] = Field(
        None, description="advertised tool-catalog size per agent (fallback fixed 8)"
    )
    parallel_tool_calls_per_step: Optional[Distribution] = Field(
        None, description="parallel tool calls emitted in one step's tool round (fallback fixed 1)"
    )
    tool_call_latency_sec: Optional[Distribution] = Field(
        None,
        description=(
            "Pause between an agent's steps, in seconds, modelling how long a tool takes "
            "to run (the tool round-trip). Held as an offline wait that frees the GPU. "
            "Omit to use the default (fixed 1s)."
        ),
    )
    user_think_time_sec: Optional[Distribution] = Field(
        None,
        description=(
            "Pause before each follow-up turn (turns 2..N), in seconds, modelling the user's "
            "read/think/reply time. Omit to use the default (fixed 10s)."
        ),
    )
    max_model_len: Optional[int] = Field(
        None,
        description=(
            "Fail-fast context-length ceiling (tokens). When set, a config whose single largest "
            "request -- worst-case inputs (system head + tool catalog + accumulated turns + tool "
            "loop) plus the output to generate -- would exceed this is rejected at load, instead "
            "of 400-ing mid-run. Uses each distribution's clip ceiling (`max`) as the worst case. "
            "Excludes the model's per-message chat-template wrapper (~10-15 tok/msg), so set this "
            "at or a little below your model's true window. Omit to skip the check."
        ),
    )

    # Context compaction: omit to never compact (pure growth). When set,
    # both trigger_tokens and target_tokens are required (enforced by the submodel).
    context_compaction: Optional[ContextCompactionConfig] = Field(
        None,
        description=(
            "Context compaction policy. When set, a round whose accumulated input crosses "
            "trigger_tokens is followed by a fresh round whose grown transcript is replaced "
            "by a target_tokens-sized summary block. Omit for pure growth (no compaction)."
        ),
    )

    # Inherited from SessionReplayConfig but inert for synthetic generation: pinned
    # so a synthetic config can't accidentally enable trace-replay-only behavior.
    inject_random_session_id: bool = Field(False, frozen=True)
    duplicate_sessions_target: Optional[int] = Field(None, frozen=True)
    override_tool_call_max_tokens: bool = Field(False)

    @model_validator(mode="after")
    def validate_theme_mix(self) -> "SyntheticAgenticConfig":
        if not self.theme_mix:
            raise ValueError("theme_mix must be non-empty (at least one theme name -> weight)")
        if any(w < 0 for w in self.theme_mix.values()):
            raise ValueError(f"theme_mix weights must all be >= 0, got: {self.theme_mix}")
        if sum(self.theme_mix.values()) <= 0:
            raise ValueError(f"theme_mix weights must sum to a positive value (at least one weight > 0), got: {self.theme_mix}")
        return self

    @model_validator(mode="after")
    def validate_max_model_len(self) -> "SyntheticAgenticConfig":
        # Fail-fast context-length ceiling. Reject a config whose single
        # LARGEST request would exceed the model's window, so it fails at load
        # instead of 400-ing deep inside a live run. We model the whole final
        # message (all inputs + the output we must still generate), not just a
        # bare turn, because the naive one-turn check happily passes configs
        # (e.g. a 487-tool catalog) that then overrun live.
        #
        # Worst-case sizing per distribution. The sampler HARD-CLIPS every draw
        # to [min, max] (sample_from_distribution), so the realized value never
        # exceeds `max` -- for fixed it is exactly `mean` (max is ignored). We
        # therefore take the worst case as `max` for every distribution EXCEPT
        # fixed (which uses `mean`). This matters because Distribution's unset
        # fields carry stale defaults (`mean` defaults to 512, `max` to 1024),
        # and only the field a given type actually samples from is meaningful:
        # a uniform ignores `mean`, a fixed ignores `max`. Reading the wrong one
        # would inject a bogus 512/1024 into the projection. None-valued optional
        # knobs fall back to the SAME fixed defaults the generator uses (matched
        # by value, not import, to avoid a config -> datagen circular import).
        if self.max_model_len is None:
            return self

        import math

        def _hi(dist: Optional[Distribution], fallback_fixed_mean: float) -> int:
            # Worst-case value a knob can contribute. None => the generator's
            # fixed fallback (its `mean`). fixed => its `mean`. All other types
            # are clipped to `max`, so `max` is the true achievable ceiling.
            if dist is None:
                return int(math.ceil(fallback_fixed_mean))
            if dist.type == DistributionType.FIXED:
                return int(math.ceil(dist.mean))
            return int(math.ceil(dist.max))

        # ~170 tok per advertised tool: real serialized theme tool schemas
        # (name + description + JSON-Schema params) measure ~140-166 tok each
        # across the built-in themes; 170 is the rounded-up ceiling so the
        # estimate over-, never under-, counts the catalog.
        CATALOG_TOKENS_PER_TOOL = 170

        head = self.shared_system_prompt_len
        # input/output tokens are required fields (never None), so their fallback
        # is unreachable -- passed as 0 to make that explicit.
        in_hi = _hi(self.input_tokens_per_turn, 0)
        out_hi = _hi(self.output_tokens_per_turn, 0)
        turns_hi = _hi(self.turns_per_session, 1)
        k_hi = max(0, _hi(self.tool_loop_depth, 2))  # generator _FB_STEPS = fixed 2
        par_hi = max(1, _hi(self.parallel_tool_calls_per_step, 1))  # _FB_PARALLEL = fixed 1
        catalog = _hi(self.tool_catalog_size_per_agent, 8) * CATALOG_TOKENS_PER_TOOL  # _FB_TOOL_DEFS = fixed 8

        # A single agent turn runs a tool loop whose transcript GROWS each
        # iteration: iteration t re-injects the prior turn plus its output-call
        # message and `par` tool results. The deepest step therefore carries the
        # turn's input + k*(one output-call msg + par results). We size a tool
        # result as ~one output turn (results echo the payload-sized call args;
        # there is no separate tool-result-size knob), so the loop adds
        # k*(out + par*out) on top of the turn's input.
        per_turn_loop = in_hi + k_hi * (out_hi + par_hi * out_hi)

        # Peak A -- the ROOT's final turn: the whole accumulated transcript
        # (every prior turn's input AND answer) plus this turn's own loop, then
        # one more output to generate.
        root_peak = head + catalog + turns_hi * per_turn_loop + out_hi
        # Peak B -- a SUB-AGENT's single request: one input turn + its full tool
        # loop + one output (no user turns of its own). Today sub-agents share
        # the root's depth/catalog knobs, so root_peak >= sub_peak whenever
        # turns_hi >= 1; the max() keeps the check correct if sub-agents ever
        # get independent depth/catalog knobs.
        sub_peak = head + catalog + per_turn_loop + out_hi
        projected = max(root_peak, sub_peak)

        if projected > self.max_model_len:
            raise ValueError(
                f"max_model_len ({self.max_model_len}) is too small for this configuration: "
                f"the peak request is ~{projected} tokens and would overrun the model's context "
                f"window (prompt + generated output). Breakdown: system_head({head}) + "
                f"tool_catalog({_hi(self.tool_catalog_size_per_agent, 8)} tools x{CATALOG_TOKENS_PER_TOOL}"
                f"={catalog}) + turns({turns_hi}) x per_turn_loop({per_turn_loop}) + output({out_hi}), "
                f"where per_turn_loop = input({in_hi}) + tool_loop_depth({k_hi}) x "
                f"(output({out_hi}) + parallel({par_hi}) x output). Reduce input/output/catalog/turns/"
                f"tool_loop_depth, or raise max_model_len."
            )
        return self
