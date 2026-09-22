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
"""``aiperf profile`` argv -> inference-perf config, keyed to AIPerf v0.12.0.

The flag surface is the 247 CLI-named fields of AIPerf's ``CLIConfig``
(``src/aiperf/config/flags/cli_config.py`` at the pinned tag), mirrored below
as a flat table so an unknown flag is a refusal rather than a silent drop.
Aliases (``--isl``/``--synthetic-input-tokens-mean``) resolve to one field and
boolean ``--no-<flag>`` negations are honored where AIPerf does not already
define ``--no-...`` as its own flag.

Verified v0.12.0 behavior the rows rely on:

- Synthetic lengths sample a positive normal (ceiling, floor 1, no upper
  clamp); ``stddev <= 0`` returns a true fixed length.
- The completions endpoint sends ``max_tokens`` only when an OSL is set, and
  ``stream`` follows ``--streaming`` (default false; streaming is opt-in,
  unlike vllm bench, which always streams).
- No sampling parameters are injected by the client.
- ``--request-count`` unset is auto-derived for synthetic data as
  ``max(10, concurrency * 2)``; the default load shape is a concurrency
  phase at concurrency 1.
- ``--num-dataset-entries`` (default 100) is a pool of unique entries reused
  across requests, a false friend of vllm's ``--num-prompts``.

Whole subsystems inference-perf does not have (multi-run sweeps and searches,
accuracy evaluation, trace and file datasets, rankings endpoints, agentic
synthesis) refuse with one named reason each; flags whose subsystem is not
active in the given argv are inert in AIPerf too and are dropped with an
annotation instead, since an inert flag cannot change the offered workload.
"""

import math
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union

from inference_perf.config import (
    APIConfig,
    APIType,
    AudioDatagenConfig,
    Config,
    CustomTokenizerConfig,
    DataConfig,
    DataGenType,
    Distribution,
    GoodputConfig,
    ImageDatagenConfig,
    LoadConfig,
    LoadType,
    MetricsClientConfig,
    MetricsClientType,
    ModelServerClientConfig,
    PrometheusClientConfig,
    ReportConfig,
    RequestLifecycleMetricsReportConfig,
    Resolution,
    StandardLoadStage,
    ConcurrentLoadStage,
    StorageConfig,
    StorageConfigBase,
    SyntheticMultimodalDatagenConfig,
    VideoDatagenConfig,
    VideoProfile,
)
from inference_perf.payloads import ImageRepresentation, VideoRepresentation
from inference_perf.tools.convert.model import Conversion, PeerUsageError, fixed_dist, normal_dist
from inference_perf.tools.convert.versions import (
    AIPERF_PROFILE_VERSIONS,
    AiperfProfileFacts,
    resolve_aiperf_profile,
    unverified_version_reason,
)

TOOL_NAME = "aiperf profile"

ParsedValue = Union[bool, str, List[str]]


class _F(NamedTuple):
    field: str
    aliases: Tuple[str, ...]
    kind: str  # "flag" | "value" | "list"
    group: str


# The verified v0.12.0 surface: field name, CLI aliases, arity kind, and the
# CLIParameter group. Generated from the pinned tag's cli_config.py; edits
# here mean the pin moved and every affected table row was re-verified.
_FIELDS: Tuple[_F, ...] = (
    _F(
        "model_names",
        (
            "--model-names",
            "--model",
            "-m",
        ),
        "list",
        "ENDPOINT",
    ),
    _F("model_selection_strategy", ("--model-selection-strategy",), "value", "ENDPOINT"),
    _F(
        "custom_endpoint",
        (
            "--custom-endpoint",
            "--endpoint",
        ),
        "value",
        "ENDPOINT",
    ),
    _F("endpoint_type", ("--endpoint-type",), "value", "ENDPOINT"),
    _F("streaming", ("--streaming",), "flag", "ENDPOINT"),
    _F(
        "urls",
        (
            "--url",
            "-u",
        ),
        "list",
        "ENDPOINT",
    ),
    _F("url_selection_strategy", ("--url-strategy",), "value", "ENDPOINT"),
    _F("timeout_seconds", ("--request-timeout-seconds",), "value", "ENDPOINT"),
    _F("wait_for_model_timeout", ("--wait-for-model-timeout",), "value", "ENDPOINT"),
    _F("wait_for_model_mode", ("--wait-for-model-mode",), "value", "ENDPOINT"),
    _F("wait_for_model_interval", ("--wait-for-model-interval",), "value", "ENDPOINT"),
    _F("api_key", ("--api-key",), "value", "ENDPOINT"),
    _F(
        "transport",
        (
            "--transport",
            "--transport-type",
        ),
        "value",
        "ENDPOINT",
    ),
    _F("use_legacy_max_tokens", ("--use-legacy-max-tokens",), "flag", "ENDPOINT"),
    _F("use_server_token_count", ("--use-server-token-count",), "flag", "ENDPOINT"),
    _F("connection_reuse_strategy", ("--connection-reuse-strategy",), "value", "ENDPOINT"),
    _F("download_video_content", ("--download-video-content",), "flag", "ENDPOINT"),
    _F("request_content_type", ("--request-content-type",), "value", "ENDPOINT"),
    _F("session_header", ("--session-header",), "value", "ENDPOINT"),
    _F("uuid_and_strip", ("--uuid-and-strip",), "flag", "ENDPOINT"),
    _F("tokenizer_name", ("--tokenizer",), "value", "TOKENIZER"),
    _F("tokenizer_revision", ("--tokenizer-revision",), "value", "TOKENIZER"),
    _F("trust_remote_code", ("--tokenizer-trust-remote-code",), "flag", "TOKENIZER"),
    _F("apply_chat_template", ("--apply-chat-template",), "flag", "TOKENIZER"),
    _F("extra_inputs", ("--extra-inputs",), "list", "INPUT"),
    _F(
        "headers",
        (
            "--header",
            "-H",
        ),
        "list",
        "INPUT",
    ),
    _F("input_file", ("--input-file",), "value", "INPUT"),
    _F("public_dataset", ("--public-dataset",), "value", "INPUT"),
    _F("hf_dataset_subset", ("--hf-subset",), "value", "INPUT"),
    _F("hf_weka_dataset", ("--hf-weka-dataset",), "value", "INPUT"),
    _F("dataset_filters", ("--dataset-filter",), "list", "INPUT"),
    _F("custom_dataset_type", ("--custom-dataset-type",), "value", "INPUT"),
    _F("ignore_trace_delays", ("--ignore-trace-delays",), "flag", "INPUT"),
    _F("use_think_time_only", ("--use-think-time-only",), "flag", "INPUT"),
    _F("max_context_length", ("--max-context-length",), "value", "INPUT"),
    _F("dataset_sampling_strategy", ("--dataset-sampling-strategy",), "value", "INPUT"),
    _F("allow_dataset_wrap", ("--allow-dataset-wrap",), "flag", "INPUT"),
    _F("random_seed", ("--random-seed",), "value", "INPUT"),
    _F("trace_session_sample_ratio", ("--trace-session-sample-ratio",), "value", "INPUT"),
    _F(
        "config_file",
        (
            "--config",
            "-f",
        ),
        "value",
        "INPUT",
    ),
    _F("fixed_schedule", ("--fixed-schedule",), "flag", "FIXED_SCHEDULE"),
    _F("disable_auto_fixed_schedule", ("--no-fixed-schedule",), "flag", "FIXED_SCHEDULE"),
    _F("fixed_schedule_auto_offset", ("--fixed-schedule-auto-offset",), "flag", "FIXED_SCHEDULE"),
    _F("fixed_schedule_start_offset", ("--fixed-schedule-start-offset",), "value", "FIXED_SCHEDULE"),
    _F("fixed_schedule_end_offset", ("--fixed-schedule-end-offset",), "value", "FIXED_SCHEDULE"),
    _F("goodput", ("--goodput",), "list", "GOODPUT"),
    _F(
        "conversation_num",
        (
            "--conversation-num",
            "--num-conversations",
            "--num-sessions",
        ),
        "value",
        "CONVERSATION_INPUT",
    ),
    _F(
        "conversation_num_dataset_entries",
        (
            "--num-dataset-entries",
            "--num-prompts",
        ),
        "value",
        "CONVERSATION_INPUT",
    ),
    _F(
        "conversation_turn_mean",
        (
            "--conversation-turn-mean",
            "--session-turns-mean",
        ),
        "value",
        "CONVERSATION_INPUT",
    ),
    _F(
        "conversation_turn_stddev",
        (
            "--conversation-turn-stddev",
            "--session-turns-stddev",
        ),
        "value",
        "CONVERSATION_INPUT",
    ),
    _F(
        "conversation_turn_delay_mean",
        (
            "--conversation-turn-delay-mean",
            "--session-turn-delay-mean",
        ),
        "value",
        "CONVERSATION_INPUT",
    ),
    _F(
        "conversation_turn_delay_stddev",
        (
            "--conversation-turn-delay-stddev",
            "--session-turn-delay-stddev",
        ),
        "value",
        "CONVERSATION_INPUT",
    ),
    _F(
        "conversation_turn_delay_ratio",
        (
            "--conversation-turn-delay-ratio",
            "--session-delay-ratio",
        ),
        "value",
        "CONVERSATION_INPUT",
    ),
    _F("inter_turn_delay_cap_seconds", ("--inter-turn-delay-cap-seconds",), "value", "CONVERSATION_INPUT"),
    _F("max_idle_gap_cap_seconds", ("--max-idle-gap-cap-seconds",), "value", "CONVERSATION_INPUT"),
    _F("replay_speedup", ("--replay-speedup",), "value", "CONVERSATION_INPUT"),
    _F("open_loop_replay", ("--open-loop-replay",), "flag", "CONVERSATION_INPUT"),
    _F("open_loop_strict", ("--open-loop-strict",), "flag", "CONVERSATION_INPUT"),
    _F("omit_kv_hints", ("--omit-kv-hints",), "flag", "CONVERSATION_INPUT"),
    _F("force_min_tokens", ("--force-min-tokens",), "flag", "CONVERSATION_INPUT"),
    _F(
        "prompt_batch_size",
        (
            "--prompt-batch-size",
            "--batch-size-text",
            "--batch-size",
            "-b",
        ),
        "value",
        "PROMPT",
    ),
    _F("prompt_corpus", ("--prompt-corpus",), "value", "PROMPT"),
    _F("cache_bust", ("--cache-bust",), "value", "CACHE_BUST"),
    _F(
        "prompt_prefix_pool_size",
        (
            "--prompt-prefix-pool-size",
            "--prefix-prompt-pool-size",
            "--num-prefix-prompts",
        ),
        "value",
        "PREFIX_PROMPT",
    ),
    _F(
        "prompt_prefix_length",
        (
            "--prompt-prefix-length",
            "--prefix-prompt-length",
        ),
        "value",
        "PREFIX_PROMPT",
    ),
    _F("prompt_prefix_shared_system_length", ("--shared-system-prompt-length",), "value", "PREFIX_PROMPT"),
    _F("prompt_prefix_user_context_length", ("--user-context-prompt-length",), "value", "PREFIX_PROMPT"),
    _F(
        "prompt_input_tokens_mean",
        (
            "--prompt-input-tokens-mean",
            "--synthetic-input-tokens-mean",
            "--isl",
        ),
        "value",
        "ISL",
    ),
    _F(
        "prompt_input_tokens_stddev",
        (
            "--prompt-input-tokens-stddev",
            "--synthetic-input-tokens-stddev",
            "--isl-stddev",
        ),
        "value",
        "ISL",
    ),
    _F(
        "prompt_input_tokens_block_size",
        (
            "--prompt-input-tokens-block-size",
            "--synthetic-input-tokens-block-size",
            "--isl-block-size",
        ),
        "value",
        "ISL",
    ),
    _F(
        "prompt_sequence_distribution",
        (
            "--seq-dist",
            "--sequence-distribution",
        ),
        "value",
        "ISL",
    ),
    _F(
        "prompt_output_tokens_mean",
        (
            "--prompt-output-tokens-mean",
            "--output-tokens-mean",
            "--osl",
        ),
        "value",
        "OSL",
    ),
    _F(
        "prompt_output_tokens_stddev",
        (
            "--prompt-output-tokens-stddev",
            "--output-tokens-stddev",
            "--osl-stddev",
        ),
        "value",
        "OSL",
    ),
    _F(
        "audio_batch_size",
        (
            "--audio-batch-size",
            "--batch-size-audio",
        ),
        "value",
        "AUDIO_INPUT",
    ),
    _F("audio_length_mean", ("--audio-length-mean",), "value", "AUDIO_INPUT"),
    _F("audio_length_stddev", ("--audio-length-stddev",), "value", "AUDIO_INPUT"),
    _F("audio_format", ("--audio-format",), "value", "AUDIO_INPUT"),
    _F("audio_depths", ("--audio-depths",), "list", "AUDIO_INPUT"),
    _F("audio_sample_rates", ("--audio-sample-rates",), "list", "AUDIO_INPUT"),
    _F("audio_num_channels", ("--audio-num-channels",), "value", "AUDIO_INPUT"),
    _F("image_width_mean", ("--image-width-mean",), "value", "IMAGE_INPUT"),
    _F("image_width_stddev", ("--image-width-stddev",), "value", "IMAGE_INPUT"),
    _F("image_height_mean", ("--image-height-mean",), "value", "IMAGE_INPUT"),
    _F("image_height_stddev", ("--image-height-stddev",), "value", "IMAGE_INPUT"),
    _F(
        "image_batch_size",
        (
            "--image-batch-size",
            "--batch-size-image",
        ),
        "value",
        "IMAGE_INPUT",
    ),
    _F("image_format", ("--image-format",), "value", "IMAGE_INPUT"),
    _F("image_source", ("--image-source",), "value", "IMAGE_INPUT"),
    _F("image_source_sampling", ("--image-source-sampling",), "value", "IMAGE_INPUT"),
    _F(
        "video_batch_size",
        (
            "--video-batch-size",
            "--batch-size-video",
        ),
        "value",
        "VIDEO_INPUT",
    ),
    _F("video_duration", ("--video-duration",), "value", "VIDEO_INPUT"),
    _F("video_fps", ("--video-fps",), "value", "VIDEO_INPUT"),
    _F("video_width", ("--video-width",), "value", "VIDEO_INPUT"),
    _F("video_height", ("--video-height",), "value", "VIDEO_INPUT"),
    _F("video_synth_type", ("--video-synth-type",), "value", "VIDEO_INPUT"),
    _F("video_format", ("--video-format",), "value", "VIDEO_INPUT"),
    _F("video_codec", ("--video-codec",), "value", "VIDEO_INPUT"),
    _F("video_audio_sample_rate", ("--video-audio-sample-rate",), "value", "VIDEO_INPUT"),
    _F("video_audio_channels", ("--video-audio-num-channels",), "value", "VIDEO_INPUT"),
    _F("video_audio_codec", ("--video-audio-codec",), "value", "VIDEO_INPUT"),
    _F("video_audio_depth", ("--video-audio-depth",), "value", "VIDEO_INPUT"),
    _F("rankings_passages_mean", ("--rankings-passages-mean",), "value", "RANKINGS"),
    _F("rankings_passages_stddev", ("--rankings-passages-stddev",), "value", "RANKINGS"),
    _F("rankings_passages_prompt_token_mean", ("--rankings-passages-prompt-token-mean",), "value", "RANKINGS"),
    _F("rankings_passages_prompt_token_stddev", ("--rankings-passages-prompt-token-stddev",), "value", "RANKINGS"),
    _F("rankings_query_prompt_token_mean", ("--rankings-query-prompt-token-mean",), "value", "RANKINGS"),
    _F("rankings_query_prompt_token_stddev", ("--rankings-query-prompt-token-stddev",), "value", "RANKINGS"),
    _F("synthesis_speedup_ratio", ("--synthesis-speedup-ratio",), "value", "SYNTHESIS"),
    _F("synthesis_prefix_len_multiplier", ("--synthesis-prefix-len-multiplier",), "value", "SYNTHESIS"),
    _F("synthesis_prefix_root_multiplier", ("--synthesis-prefix-root-multiplier",), "value", "SYNTHESIS"),
    _F("synthesis_prompt_len_multiplier", ("--synthesis-prompt-len-multiplier",), "value", "SYNTHESIS"),
    _F("synthesis_output_len_multiplier", ("--synthesis-output-len-multiplier",), "value", "SYNTHESIS"),
    _F("synthesis_max_isl", ("--synthesis-max-isl",), "value", "SYNTHESIS"),
    _F("synthesis_max_osl", ("--synthesis-max-osl",), "value", "SYNTHESIS"),
    _F("benchmark_duration", ("--benchmark-duration",), "value", "LOAD_GENERATOR"),
    _F("benchmark_grace_period", ("--benchmark-grace-period",), "value", "LOAD_GENERATOR"),
    _F("concurrency", ("--concurrency",), "value", "LOAD_GENERATOR"),
    _F("prefill_concurrency", ("--prefill-concurrency",), "value", "LOAD_GENERATOR"),
    _F("request_rate", ("--request-rate",), "value", "LOAD_GENERATOR"),
    _F(
        "arrival_pattern",
        (
            "--arrival-pattern",
            "--request-rate-mode",
        ),
        "value",
        "LOAD_GENERATOR",
    ),
    _F(
        "arrival_smoothness",
        (
            "--arrival-smoothness",
            "--vllm-burstiness",
        ),
        "value",
        "LOAD_GENERATOR",
    ),
    _F(
        "request_count",
        (
            "--request-count",
            "--num-requests",
        ),
        "value",
        "LOAD_GENERATOR",
    ),
    _F("concurrency_ramp_duration", ("--concurrency-ramp-duration",), "value", "LOAD_GENERATOR"),
    _F("prefill_concurrency_ramp_duration", ("--prefill-concurrency-ramp-duration",), "value", "LOAD_GENERATOR"),
    _F("request_rate_ramp_duration", ("--request-rate-ramp-duration",), "value", "LOAD_GENERATOR"),
    _F("request_rate_series", ("--request-rate-series",), "value", "LOAD_GENERATOR"),
    _F("failed_request_threshold", ("--failed-request-threshold",), "value", "LOAD_GENERATOR"),
    _F("trajectory_start_min_ratio", ("--trajectory-start-min-ratio",), "value", "LOAD_GENERATOR"),
    _F("trajectory_start_max_ratio", ("--trajectory-start-max-ratio",), "value", "LOAD_GENERATOR"),
    _F("burst_phase_starts", ("--burst-phase-starts",), "flag", "LOAD_GENERATOR"),
    _F("trace_idle_gap_cap_seconds", ("--trace-idle-gap-cap-seconds",), "value", "LOAD_GENERATOR"),
    _F("system_idle_gap_cap_seconds", ("--system-idle-gap-cap-seconds",), "value", "LOAD_GENERATOR"),
    _F("scenario", ("--scenario",), "value", "SCENARIO"),
    _F("unsafe_override", ("--unsafe-override",), "flag", "SCENARIO"),
    _F(
        "warmup_request_count",
        (
            "--warmup-request-count",
            "--num-warmup-requests",
        ),
        "value",
        "WARMUP",
    ),
    _F("warmup_duration", ("--warmup-duration",), "value", "WARMUP"),
    _F("agentic_cache_warmup_duration", ("--agentic-cache-warmup-duration",), "value", "WARMUP"),
    _F("agentic_warmup_grace_period", ("--agentic-warmup-grace-period",), "value", "WARMUP"),
    _F("warmup_num_sessions", ("--num-warmup-sessions",), "value", "WARMUP"),
    _F("warmup_concurrency", ("--warmup-concurrency",), "value", "WARMUP"),
    _F("warmup_prefill_concurrency", ("--warmup-prefill-concurrency",), "value", "WARMUP"),
    _F("warmup_request_rate", ("--warmup-request-rate",), "value", "WARMUP"),
    _F("warmup_arrival_pattern", ("--warmup-arrival-pattern",), "value", "WARMUP"),
    _F("warmup_grace_period", ("--warmup-grace-period",), "value", "WARMUP"),
    _F("warmup_concurrency_ramp_duration", ("--warmup-concurrency-ramp-duration",), "value", "WARMUP"),
    _F("warmup_prefill_concurrency_ramp_duration", ("--warmup-prefill-concurrency-ramp-duration",), "value", "WARMUP"),
    _F("warmup_request_rate_ramp_duration", ("--warmup-request-rate-ramp-duration",), "value", "WARMUP"),
    _F("user_centric_rate", ("--user-centric-rate",), "value", "USER_CENTRIC"),
    _F("num_users", ("--num-users",), "value", "USER_CENTRIC"),
    _F("request_cancellation_rate", ("--request-cancellation-rate",), "value", "REQUEST_CANCELLATION"),
    _F("request_cancellation_delay", ("--request-cancellation-delay",), "value", "REQUEST_CANCELLATION"),
    _F(
        "artifact_directory",
        (
            "--output-artifact-dir",
            "--artifact-dir",
        ),
        "value",
        "OUTPUT",
    ),
    _F(
        "profile_export_prefix",
        (
            "--profile-export-prefix",
            "--profile-export-file",
        ),
        "value",
        "OUTPUT",
    ),
    _F(
        "export_level",
        (
            "--export-level",
            "--profile-export-level",
        ),
        "value",
        "OUTPUT",
    ),
    _F("slice_duration", ("--slice-duration",), "value", "OUTPUT"),
    _F("plot_required", ("--plot-required",), "flag", "OUTPUT"),
    _F("export_outputs_json", ("--export-outputs-json",), "flag", "OUTPUT"),
    _F("otel_url", ("--otel-url",), "value", "OUTPUT"),
    _F("stream", ("--stream",), "list", "OUTPUT"),
    _F("otel_resource_attributes", ("--otel-resource-attributes",), "list", "OUTPUT"),
    _F("gen_ai_provider", ("--gen-ai-provider",), "value", "OUTPUT"),
    _F("mlflow_tracking_uri", ("--mlflow-tracking-uri",), "value", "OUTPUT"),
    _F("mlflow_experiment", ("--mlflow-experiment",), "value", "OUTPUT"),
    _F("mlflow_run_name", ("--mlflow-run-name",), "value", "OUTPUT"),
    _F("mlflow_tags", ("--mlflow-tag",), "list", "OUTPUT"),
    _F("mlflow_parent_run_id", ("--mlflow-parent-run-id",), "value", "OUTPUT"),
    _F("mlflow_artifact_globs", ("--mlflow-artifact-glob",), "list", "OUTPUT"),
    _F("wandb_project", ("--wandb-project",), "value", "OUTPUT"),
    _F("wandb_entity", ("--wandb-entity",), "value", "OUTPUT"),
    _F("wandb_run_name", ("--wandb-run-name",), "value", "OUTPUT"),
    _F("wandb_tags", ("--wandb-tag",), "list", "OUTPUT"),
    _F("export_http_trace", ("--export-http-trace",), "flag", "HTTP_TRACE"),
    _F("show_trace_timing", ("--show-trace-timing",), "flag", "HTTP_TRACE"),
    _F("server_metrics", ("--server-metrics",), "list", "SERVER_METRICS"),
    _F("no_server_metrics", ("--no-server-metrics",), "flag", "SERVER_METRICS"),
    _F("server_metrics_formats", ("--server-metrics-formats",), "list", "SERVER_METRICS"),
    _F("network_latency_automatic", ("--network-latency-automatic",), "flag", "NETWORK_LATENCY"),
    _F("network_latency_mean", ("--network-latency-mean",), "value", "NETWORK_LATENCY"),
    _F("network_latency_ping_interval", ("--network-latency-ping-interval",), "value", "NETWORK_LATENCY"),
    _F("gpu_telemetry", ("--gpu-telemetry",), "list", "GPU_TELEMETRY"),
    _F("no_gpu_telemetry", ("--no-gpu-telemetry",), "flag", "GPU_TELEMETRY"),
    _F(
        "ui_type",
        (
            "--ui-type",
            "--ui",
        ),
        "value",
        "UI",
    ),
    _F("num_profile_runs", ("--num-profile-runs",), "value", "MULTI_RUN"),
    _F("profile_run_cooldown_seconds", ("--profile-run-cooldown-seconds",), "value", "MULTI_RUN"),
    _F("confidence_level", ("--confidence-level",), "value", "MULTI_RUN"),
    _F("convergence_metric", ("--convergence-metric",), "value", "MULTI_RUN"),
    _F("convergence_stat", ("--convergence-stat",), "value", "MULTI_RUN"),
    _F("convergence_threshold", ("--convergence-threshold",), "value", "MULTI_RUN"),
    _F("convergence_mode", ("--convergence-mode",), "value", "MULTI_RUN"),
    _F("parameter_sweep_cooldown_seconds", ("--parameter-sweep-cooldown-seconds",), "value", "MULTI_RUN"),
    _F("parameter_sweep_mode", ("--parameter-sweep-mode",), "value", "MULTI_RUN"),
    _F("sweep_type", ("--sweep-type",), "value", "MULTI_RUN"),
    _F("no_sweep_table", ("--no-sweep-table",), "flag", "MULTI_RUN"),
    _F("search_space", ("--search-space",), "list", "MULTI_RUN"),
    _F("search_metric", ("--search-metric",), "value", "MULTI_RUN"),
    _F("search_stat", ("--search-stat",), "value", "MULTI_RUN"),
    _F("search_direction", ("--search-direction",), "value", "MULTI_RUN"),
    _F("search_max_iterations", ("--search-max-iterations",), "value", "MULTI_RUN"),
    _F("search_initial_points", ("--search-initial-points",), "value", "MULTI_RUN"),
    _F("search_random_seed", ("--search-random-seed",), "value", "MULTI_RUN"),
    _F("search_planner", ("--search-planner",), "value", "MULTI_RUN"),
    _F("optuna_sampler", ("--optuna-sampler",), "value", "MULTI_RUN"),
    _F("optuna_acquisition", ("--optuna-acquisition",), "value", "MULTI_RUN"),
    _F("optuna_terminator", ("--optuna-terminator",), "value", "MULTI_RUN"),
    _F("search_percentile_pooling", ("--search-percentile-pooling",), "value", "MULTI_RUN"),
    _F("bo_constraint_mode", ("--bo-constraint-mode",), "value", "MULTI_RUN"),
    _F(
        "sweep_variants",
        (
            "--variant",
            "--sweep-variant",
        ),
        "list",
        "MULTI_RUN",
    ),
    _F("search_sla", ("--search-sla",), "list", "MULTI_RUN"),
    _F("search_sla_tier", ("--search-sla-tier",), "list", "MULTI_RUN"),
    _F("search_recipe", ("--search-recipe",), "value", "MULTI_RUN"),
    _F("ttft_sla_ms", ("--ttft-sla-ms",), "value", "MULTI_RUN"),
    _F("isl_osl_pairs", ("--isl-osl-pairs",), "value", "MULTI_RUN"),
    _F("itl_sla_ms", ("--itl-sla-ms",), "value", "MULTI_RUN"),
    _F("tpot_sla_ms", ("--tpot-sla-ms",), "value", "MULTI_RUN"),
    _F("e2e_sla_ms", ("--e2e-sla-ms",), "value", "MULTI_RUN"),
    _F("error_rate_sla", ("--error-rate-sla",), "value", "MULTI_RUN"),
    _F("slo_attainment_fraction", ("--slo-attainment-fraction",), "value", "MULTI_RUN"),
    _F("search_style", ("--search-style",), "value", "MULTI_RUN"),
    _F("degradation_threshold", ("--degradation-threshold",), "value", "MULTI_RUN"),
    _F("degradation_metric_tag", ("--degradation-metric-tag",), "value", "MULTI_RUN"),
    _F("degradation_stat", ("--degradation-stat",), "value", "MULTI_RUN"),
    _F("isl_min", ("--isl-min",), "value", "MULTI_RUN"),
    _F("isl_max", ("--isl-max",), "value", "MULTI_RUN"),
    _F("isl_steps", ("--isl-steps",), "value", "MULTI_RUN"),
    _F("concurrency_min", ("--concurrency-min",), "value", "MULTI_RUN"),
    _F("concurrency_max", ("--concurrency-max",), "value", "MULTI_RUN"),
    _F("concurrency_steps", ("--concurrency-steps",), "value", "MULTI_RUN"),
    _F("osl_min", ("--osl-min",), "value", "MULTI_RUN"),
    _F("osl_max", ("--osl-max",), "value", "MULTI_RUN"),
    _F("osl_steps", ("--osl-steps",), "value", "MULTI_RUN"),
    _F("accuracy_benchmark", ("--accuracy-benchmark",), "value", "ACCURACY"),
    _F("accuracy_tasks", ("--accuracy-tasks",), "list", "ACCURACY"),
    _F("accuracy_n_shots", ("--accuracy-n-shots",), "value", "ACCURACY"),
    _F("accuracy_enable_cot", ("--accuracy-enable-cot",), "flag", "ACCURACY"),
    _F("accuracy_grader", ("--accuracy-grader",), "value", "ACCURACY"),
    _F("accuracy_system_prompt", ("--accuracy-system-prompt",), "value", "ACCURACY"),
    _F("accuracy_verbose", ("--accuracy-verbose",), "flag", "ACCURACY"),
    _F("log_level", ("--log-level",), "value", "SERVICE"),
    _F(
        "verbose",
        (
            "--verbose",
            "-v",
        ),
        "flag",
        "SERVICE",
    ),
    _F(
        "extra_verbose",
        (
            "--extra-verbose",
            "-vv",
        ),
        "flag",
        "SERVICE",
    ),
    _F(
        "record_processor_service_count",
        (
            "--record-processor-service-count",
            "--record-processors",
        ),
        "value",
        "SERVICE",
    ),
    _F("api_port", ("--api-port",), "value", "SERVICE"),
    _F("api_host", ("--api-host",), "value", "SERVICE"),
    _F("stats_interval", ("--stats-interval",), "value", "SERVICE"),
    _F(
        "workers_max",
        (
            "--workers-max",
            "--max-workers",
        ),
        "value",
        "WORKERS",
    ),
    _F("zmq_tcp_host", ("--zmq-host",), "value", "ZMQ_COMMUNICATION"),
    _F("zmq_ipc_path", ("--zmq-ipc-path",), "value", "ZMQ_COMMUNICATION"),
    _F("zmq_dual_bind", ("--zmq-dual-bind",), "flag", "ZMQ_COMMUNICATION"),
)

_FIELD_BY_NAME: Dict[str, _F] = {f.field: f for f in _FIELDS}

_GROUP_REFUSALS: Dict[str, str] = {
    "FIXED_SCHEDULE": "timestamp replay of trace datasets is not convertible"
    " (different trace formats; offset windowing has no equivalent)",
    "SYNTHESIS": "agentic trace synthesis has no verified mapping (synthetic_agentic is a structurally different generator)",
    "MULTI_RUN": "multi-run, sweep and search flags orchestrate many runs; the converter emits one config"
    " for one run, and load.sweep is a saturation finder with different semantics",
    "ACCURACY": "accuracy evaluation has no inference-perf counterpart",
    "USER_CENTRIC": "per-user gap pacing with blocking turns is a load model outside constant/poisson/concurrent stages",
    "REQUEST_CANCELLATION": "deliberate mid-flight cancellations are part of the offered workload and are not expressible",
    "RANKINGS": "rankings endpoints have no API type",
}

_ANNOTATE_GROUPS: Set[str] = {
    "OUTPUT",
    "HTTP_TRACE",
    "NETWORK_LATENCY",
    "GPU_TELEMETRY",
    "UI",
    "SERVICE",
    "ZMQ_COMMUNICATION",
    "SERVER_METRICS",
}

# Fields with their own handling in convert_aiperf_profile; the group default
# does not apply to them.
_BESPOKE: Set[str] = {
    "model_names",
    "model_selection_strategy",
    "custom_endpoint",
    "endpoint_type",
    "streaming",
    "urls",
    "url_selection_strategy",
    "timeout_seconds",
    "wait_for_model_timeout",
    "wait_for_model_mode",
    "wait_for_model_interval",
    "api_key",
    "transport",
    "use_legacy_max_tokens",
    "use_server_token_count",
    "connection_reuse_strategy",
    "download_video_content",
    "request_content_type",
    "session_header",
    "uuid_and_strip",
    "tokenizer_name",
    "tokenizer_revision",
    "trust_remote_code",
    "apply_chat_template",
    "extra_inputs",
    "headers",
    "input_file",
    "public_dataset",
    "hf_dataset_subset",
    "hf_weka_dataset",
    "dataset_filters",
    "custom_dataset_type",
    "ignore_trace_delays",
    "use_think_time_only",
    "max_context_length",
    "dataset_sampling_strategy",
    "allow_dataset_wrap",
    "random_seed",
    "trace_session_sample_ratio",
    "config_file",
    "goodput",
    "conversation_num",
    "conversation_num_dataset_entries",
    "conversation_turn_mean",
    "conversation_turn_stddev",
    "conversation_turn_delay_mean",
    "conversation_turn_delay_stddev",
    "conversation_turn_delay_ratio",
    "inter_turn_delay_cap_seconds",
    "max_idle_gap_cap_seconds",
    "replay_speedup",
    "open_loop_replay",
    "open_loop_strict",
    "omit_kv_hints",
    "force_min_tokens",
    "prompt_batch_size",
    "prompt_corpus",
    "cache_bust",
    "prompt_prefix_pool_size",
    "prompt_prefix_length",
    "prompt_prefix_shared_system_length",
    "prompt_prefix_user_context_length",
    "prompt_input_tokens_mean",
    "prompt_input_tokens_stddev",
    "prompt_input_tokens_block_size",
    "prompt_sequence_distribution",
    "prompt_output_tokens_mean",
    "prompt_output_tokens_stddev",
    "audio_batch_size",
    "audio_length_mean",
    "audio_length_stddev",
    "audio_format",
    "audio_depths",
    "audio_sample_rates",
    "audio_num_channels",
    "image_width_mean",
    "image_width_stddev",
    "image_height_mean",
    "image_height_stddev",
    "image_batch_size",
    "image_format",
    "image_source",
    "image_source_sampling",
    "video_batch_size",
    "video_duration",
    "video_fps",
    "video_width",
    "video_height",
    "video_synth_type",
    "video_format",
    "video_codec",
    "video_audio_sample_rate",
    "video_audio_channels",
    "video_audio_codec",
    "video_audio_depth",
    "benchmark_duration",
    "benchmark_grace_period",
    "concurrency",
    "prefill_concurrency",
    "request_rate",
    "arrival_pattern",
    "arrival_smoothness",
    "request_count",
    "concurrency_ramp_duration",
    "prefill_concurrency_ramp_duration",
    "request_rate_ramp_duration",
    "request_rate_series",
    "failed_request_threshold",
    "trajectory_start_min_ratio",
    "trajectory_start_max_ratio",
    "burst_phase_starts",
    "trace_idle_gap_cap_seconds",
    "system_idle_gap_cap_seconds",
    "scenario",
    "unsafe_override",
    "workers_max",
    "artifact_directory",
    "server_metrics",
}

_WARMUP_AGENTIC: Set[str] = {"agentic_cache_warmup_duration", "agentic_warmup_grace_period"}

WARMUP_NOTE = (
    "warmup: aiperf excludes its warmup phase from the measured window; inference-perf measures every"
    " request it sends. Workaround per docs/comparability.md: add a short first stage to load.stages and"
    " compare later stages only. The converted run includes cold-start effects the aiperf run excluded"
)

SYNTHETIC_NOTE = (
    "synthetic data: aiperf assembles prompts from its corpus, inference-perf 'random' generates token-id"
    " text to the same target lengths; prompt text never transfers between tools, token lengths define the"
    " workload"
)


class _Parsed:
    """The peer argv resolved against the mirrored surface."""

    def __init__(self) -> None:
        self.values: Dict[str, ParsedValue] = {}
        self.negated: Set[str] = set()

    def present(self, field: str) -> bool:
        return field in self.values

    def flag(self, field: str, default: bool = False) -> bool:
        value = self.values.get(field)
        if value is None:
            return default
        return bool(value)

    def raw(self, field: str) -> Optional[str]:
        value = self.values.get(field)
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, list):
            return value[-1] if value else None
        return value

    def items(self, field: str) -> List[str]:
        value = self.values.get(field)
        if value is None or isinstance(value, bool):
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    def integer(self, field: str, flag: str) -> Optional[int]:
        raw = self.raw(field)
        if raw is None:
            return None
        try:
            return int(raw)
        except ValueError as exc:
            raise PeerUsageError(f"{flag} expects an integer, got '{raw}'") from exc

    def floating(self, field: str, flag: str) -> Optional[float]:
        raw = self.raw(field)
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError as exc:
            raise PeerUsageError(f"{flag} expects a number, got '{raw}'") from exc


def _alias_index() -> Dict[str, Tuple[_F, bool]]:
    """CLI alias -> (field, is_negation). Negations are generated for boolean

    fields the cyclopts way, except where AIPerf defines the ``--no-...``
    spelling as its own field.
    """
    index: Dict[str, Tuple[_F, bool]] = {}
    for spec in _FIELDS:
        for alias in spec.aliases:
            index[alias] = (spec, False)
    for spec in _FIELDS:
        if spec.kind != "flag":
            continue
        for alias in spec.aliases:
            if not alias.startswith("--") or alias.startswith("--no-"):
                continue
            negation = "--no-" + alias[2:]
            if negation not in index:
                index[negation] = (spec, True)
    return index


_ALIASES: Dict[str, Tuple[_F, bool]] = _alias_index()


def _parse(argv: List[str], conversion: Conversion) -> _Parsed:
    parsed = _Parsed()
    i = 0
    while i < len(argv):
        token = argv[i]
        i += 1
        if not token.startswith("-"):
            conversion.refuse(token, f"{TOOL_NAME} takes no positional arguments here")
            continue
        name, eq, inline = token.partition("=")
        entry = _ALIASES.get(name)
        if entry is None:
            conversion.refuse(name, f"not in the verified {TOOL_NAME} v0.12.0 flag surface")
            # Skip a following value token so one unknown flag refuses once.
            if not eq and i < len(argv) and not argv[i].startswith("-"):
                i += 1
            continue
        spec, negation = entry
        if spec.kind == "flag":
            if eq:
                value = inline.strip().lower()
                if value not in ("true", "false", "1", "0"):
                    raise PeerUsageError(f"{name} expects true or false, got '{inline}'")
                state = value in ("true", "1")
            else:
                state = True
            if negation:
                state = not state
                parsed.negated.add(spec.field)
            parsed.values[spec.field] = state
            continue
        if eq:
            value_str = inline
        else:
            if i >= len(argv):
                raise PeerUsageError(f"{name} expects a value")
            value_str = argv[i]
            i += 1
        if spec.kind == "list":
            existing = parsed.values.get(spec.field)
            items: List[str] = list(existing) if isinstance(existing, list) else []
            items.extend(part.strip() for part in value_str.split(",") if part.strip())
            parsed.values[spec.field] = items
        else:
            parsed.values[spec.field] = value_str
    return parsed


def _norm(value: Optional[str]) -> str:
    return (value or "").strip().lower().replace("-", "_")


def _positive_normal(conversion: Conversion, flag: str, mean: float, stddev: float) -> Distribution:
    upper = int(math.ceil(mean + 6 * stddev))
    conversion.assume(
        f"{flag}: aiperf samples a positive normal with no upper clamp (floor 1); the converted"
        f" distribution clamps to [1, {upper}] (mean + 6 sigma), clipping the same far tail"
    )
    return normal_dist(mean=mean, std_dev=stddev, min_value=1, max_value=upper)


def _convert_api_type(conversion: Conversion, parsed: _Parsed, facts: AiperfProfileFacts) -> Optional[APIType]:
    raw = parsed.raw("endpoint_type")
    endpoint = _norm(raw) if raw is not None else "chat"
    if raw is None:
        conversion.assume(
            "no --endpoint-type given: aiperf defaults to chat (unlike vllm bench, whose default is the completions path)"
        )
    mapping = {"completions": APIType.Completion, "chat": APIType.Chat, "messages": APIType.AnthropicMessages}
    if endpoint in mapping:
        return mapping[endpoint]
    if endpoint in facts.endpoint_types:
        conversion.refuse(
            "--endpoint-type",
            f"endpoint type '{endpoint}' has no API type (completions, chat and messages convert)",
        )
    else:
        conversion.refuse("--endpoint-type", f"'{endpoint}' is not in the verified v0.12.0 endpoint registry")
    return None


def _convert_extra_inputs(conversion: Conversion, parsed: _Parsed) -> Optional[bool]:
    """Returns the ignore_eos value carried in --extra-inputs (None: absent)."""
    ignore_eos: Optional[bool] = None
    sampling_keys = {"temperature", "top_p", "top_k", "min_p", "repetition_penalty", "frequency_penalty", "presence_penalty"}
    for entry in parsed.items("extra_inputs"):
        key, sep, value = entry.partition(":")
        key = key.strip()
        if not sep:
            raise PeerUsageError(f"--extra-inputs entry '{entry}' is not key:value")
        if key == "ignore_eos":
            ignore_eos = value.strip().lower() in ("true", "1")
            continue
        if key in sampling_keys:
            conversion.refuse(
                "--extra-inputs " + key,
                "sampling parameters are not configurable for synthetic data; a guessed value would misstate the decode work",
            )
        else:
            conversion.refuse(
                "--extra-inputs " + key,
                "no generic extra-body passthrough exists; silently dropping an injected payload field is"
                " the quiet failure this converter exists to prevent",
            )
    return ignore_eos


def _convert_headers(conversion: Conversion, parsed: _Parsed) -> Optional[Dict[str, str]]:
    entries = parsed.items("headers")
    if not entries:
        return None
    headers: Dict[str, str] = {}
    for entry in entries:
        name, sep, value = entry.partition(":")
        if not sep:
            raise PeerUsageError(f"--header entry '{entry}' is not 'Name: value'")
        name = name.strip()
        if name in headers:
            conversion.refuse("--header", f"duplicate header '{name}'; refusing to pick one instead of last-wins")
            return None
        headers[name] = value.strip()
    return headers


def _convert_goodput(conversion: Conversion, parsed: _Parsed, facts: AiperfProfileFacts) -> Optional[GoodputConfig]:
    entries = parsed.items("goodput")
    if not entries:
        return None
    constraints: Dict[str, float] = {}
    for entry in entries:
        tag, sep, value = entry.partition(":")
        tag = _norm(tag)
        if not sep:
            raise PeerUsageError(f"--goodput entry '{entry}' is not TAG:VALUE")
        target = facts.goodput_tag_map.get(tag)
        if target is None:
            conversion.refuse(
                "--goodput",
                f"goodput tag '{tag}' has no GoodputConfig key"
                " (time_to_first_token, inter_token_latency and request_latency convert)",
            )
            return None
        try:
            millis = float(value)
        except ValueError as exc:
            raise PeerUsageError(f"--goodput value '{value}' is not a number") from exc
        constraints[target] = millis / 1000.0
    conversion.assume("goodput thresholds are converted from aiperf's milliseconds to seconds")
    return GoodputConfig(constraints=constraints)


class _LoadShape(NamedTuple):
    load_type: LoadType
    stage: Union[StandardLoadStage, ConcurrentLoadStage]
    request_count: int


def _convert_load(conversion: Conversion, parsed: _Parsed, facts: AiperfProfileFacts) -> Optional[_LoadShape]:
    refusals_before = len(conversion.refusals)
    concurrency = parsed.integer("concurrency", "--concurrency")
    rate = parsed.floating("request_rate", "--request-rate")
    request_count = parsed.integer("request_count", "--request-count")
    duration = parsed.floating("benchmark_duration", "--benchmark-duration")
    arrival = _norm(parsed.raw("arrival_pattern")) if parsed.present("arrival_pattern") else "poisson"

    for field, flag in (
        ("prefill_concurrency", "--prefill-concurrency"),
        ("concurrency_ramp_duration", "--concurrency-ramp-duration"),
        ("prefill_concurrency_ramp_duration", "--prefill-concurrency-ramp-duration"),
        ("request_rate_ramp_duration", "--request-rate-ramp-duration"),
        ("request_rate_series", "--request-rate-series"),
        ("trace_idle_gap_cap_seconds", "--trace-idle-gap-cap-seconds"),
        ("system_idle_gap_cap_seconds", "--system-idle-gap-cap-seconds"),
    ):
        if parsed.present(field):
            reasons = {
                "--prefill-concurrency": "phase-specific admission control has no equivalent",
                "--request-rate-series": "a piecewise-linear rate trajectory cannot be reproduced by"
                " stepwise-constant stages without fabricating a plausible staircase",
            }
            conversion.refuse(
                flag,
                reasons.get(flag, "a continuous ramp or trace-replay pacing knob is not expressible with stepwise stages"),
            )
    if parsed.flag("burst_phase_starts"):
        conversion.refuse("--burst-phase-starts", "agentic replay pacing has no equivalent")
    if parsed.present("failed_request_threshold"):
        conversion.refuse(
            "--failed-request-threshold",
            "the run-lifetime failure-ratio abort has no equivalent tripwire; configure circuit_breakers manually if needed",
        )
    if parsed.present("arrival_smoothness"):
        conversion.refuse(
            "--arrival-smoothness",
            "the gamma shape knob has no load type; use --arrival-pattern poisson for Poisson arrivals",
        )
    if parsed.present("trajectory_start_min_ratio") or parsed.present("trajectory_start_max_ratio"):
        conversion.drop("--trajectory-start-min-ratio/--trajectory-start-max-ratio (agentic replay only; inert here)")
    if len(conversion.refusals) > refusals_before:
        return None

    if concurrency is not None and rate is not None:
        conversion.refuse(
            "--concurrency with --request-rate",
            "a rate-limited closed loop fits neither constant/poisson stages (no concurrency cap) nor"
            " concurrent stages (no rate)",
        )
        return None

    if rate is not None:
        if arrival == "gamma":
            conversion.refuse("--arrival-pattern", "gamma-spaced arrivals have no load type")
            return None
        load_type = LoadType.CONSTANT if arrival == "constant" else LoadType.POISSON
        if arrival == "constant":
            conversion.assume(
                "aiperf's constant arrival is evenly spaced requests, exactly inference-perf's constant load type"
            )
        if duration is not None and request_count is not None:
            conversion.refuse(
                "--benchmark-duration with --request-count",
                "aiperf stops at whichever bound hits first; a stage has exactly one bound",
            )
            return None
        if duration is not None:
            if duration <= 0 or duration != int(duration):
                conversion.refuse(
                    "--benchmark-duration",
                    f"{duration:g} is not a positive integer number of seconds (stage duration is int seconds)",
                )
                return None
            seconds = int(duration)
            count = int(rate * seconds)
        else:
            if request_count is None:
                request_count = max(facts.auto_request_count_floor, 0)
                conversion.assume(
                    f"--request-count unset: aiperf derives max({facts.auto_request_count_floor},"
                    f" concurrency * {facts.auto_request_count_concurrency_factor}) = {request_count} for"
                    " synthetic data"
                )
            implied = request_count / rate
            if implied <= 0 or implied != int(implied):
                conversion.refuse(
                    "--request-count/--request-rate",
                    f"the implied stage duration {request_count}/{rate:g} is not a positive integer number"
                    " of seconds (stage duration is int seconds)",
                )
                return None
            seconds = int(implied)
            count = request_count
            conversion.assume(
                f"stages[0].duration: {seconds} = --request-count {request_count} / --request-rate {rate:g};"
                " the converted run matches the request count only if the offered rate is achieved"
                ' (docs/comparability.md "Count against duration")'
            )
        return _LoadShape(load_type, StandardLoadStage(rate=rate, duration=seconds), count)

    if duration is not None:
        conversion.refuse(
            "--benchmark-duration",
            "a duration-bounded concurrency phase has no equivalent (concurrent stages are bounded by num_requests only)",
        )
        return None
    level = concurrency
    if level is None:
        level = facts.default_concurrency
        conversion.assume(
            f"neither --concurrency nor --request-rate given: aiperf's default load is a concurrency phase"
            f" at concurrency {facts.default_concurrency}"
        )
    if request_count is None:
        request_count = max(
            facts.auto_request_count_floor,
            facts.auto_request_count_concurrency_factor * level,
        )
        conversion.assume(
            f"--request-count unset: aiperf derives max({facts.auto_request_count_floor},"
            f" concurrency * {facts.auto_request_count_concurrency_factor}) = {request_count} for synthetic data"
        )
    conversion.assume(
        "the worker pool must cover the concurrency level: load.num_workers x load.worker_max_concurrency"
        f" (defaults: cpu count x 100) must be at least {level}"
    )
    return _LoadShape(
        LoadType.CONCURRENT, ConcurrentLoadStage(num_requests=request_count, concurrency_level=level), request_count
    )


def _convert_data(
    conversion: Conversion, parsed: _Parsed, facts: AiperfProfileFacts, api_type: Optional[APIType]
) -> Tuple[Optional[Distribution], Optional[Distribution]]:
    isl_mean = parsed.floating("prompt_input_tokens_mean", "--isl")
    if isl_mean is None:
        isl_mean = float(facts.default_isl_mean)
    isl_stddev = parsed.floating("prompt_input_tokens_stddev", "--isl-stddev") or 0.0
    if parsed.present("prompt_input_tokens_block_size"):
        conversion.refuse("--isl-block-size", "hash-id block synthesis shapes prefix-cache behavior; not expressible")
    if parsed.present("prompt_sequence_distribution"):
        conversion.refuse(
            "--seq-dist",
            "a correlated (ISL, OSL) pair mixture is not expressible with independent input and output distributions",
        )

    osl_mean = parsed.floating("prompt_output_tokens_mean", "--osl")
    osl_stddev = parsed.floating("prompt_output_tokens_stddev", "--osl-stddev") or 0.0
    if osl_mean is None:
        conversion.refuse(
            "--osl",
            "unset --osl means aiperf sends no max_tokens field and the model picks its own stop;"
            " inference-perf always sends max_tokens from data.output_distribution, so unbounded output"
            " length is not expressible",
        )
        return None, None

    conversion.no_equivalent(SYNTHETIC_NOTE)
    if isl_stddev <= 0:
        input_dist = fixed_dist(int(round(isl_mean)))
    else:
        input_dist = _positive_normal(conversion, "--isl-stddev", isl_mean, isl_stddev)
    if osl_stddev <= 0:
        output_dist = fixed_dist(int(round(osl_mean)))
    else:
        output_dist = _positive_normal(conversion, "--osl-stddev", osl_mean, osl_stddev)
    return input_dist, output_dist


def _convert_multimodal(conversion: Conversion, parsed: _Parsed) -> Optional[SyntheticMultimodalDatagenConfig]:
    kwargs: Dict[str, Any] = {}

    audio_mean = parsed.floating("audio_length_mean", "--audio-length-mean") or 0.0
    audio_present = {
        f
        for f in (
            "audio_length_stddev",
            "audio_format",
            "audio_depths",
            "audio_sample_rates",
            "audio_num_channels",
            "audio_batch_size",
        )
        if parsed.present(f)
    }
    if audio_mean > 0:
        refusals_before = len(conversion.refusals)
        stddev = parsed.floating("audio_length_stddev", "--audio-length-stddev") or 0.0
        if stddev > 0:
            conversion.refuse(
                "--audio-length-stddev", "a continuous duration distribution has no equivalent (weighted list only)"
            )
        if parsed.present("audio_format") and _norm(parsed.raw("audio_format")) != "wav":
            conversion.refuse("--audio-format", "the audio wire encoding is not configurable")
        for field, flag in (("audio_depths", "--audio-depths"), ("audio_sample_rates", "--audio-sample-rates")):
            if parsed.present(field):
                conversion.refuse(flag, "no audio bit-depth or sample-rate knob exists")
        if parsed.present("audio_num_channels") and parsed.integer("audio_num_channels", "--audio-num-channels") != 1:
            conversion.refuse("--audio-num-channels", "no audio channel knob exists")
        batch = parsed.integer("audio_batch_size", "--audio-batch-size")
        if batch is not None and batch != 1:
            conversion.refuse("--audio-batch-size", "the per-request audio count mapping is unverified")
        if len(conversion.refusals) == refusals_before:
            conversion.assume("audio durations map to a single fixed duration; generator content equivalence is assumed")
            kwargs["audio"] = AudioDatagenConfig(durations=audio_mean)
    elif audio_present:
        conversion.drop("audio flags (inert: --audio-length-mean is 0, audio input disabled)")

    width = parsed.floating("image_width_mean", "--image-width-mean") or 0.0
    height = parsed.floating("image_height_mean", "--image-height-mean") or 0.0
    image_present = {
        f
        for f in (
            "image_width_mean",
            "image_height_mean",
            "image_width_stddev",
            "image_height_stddev",
            "image_batch_size",
            "image_format",
            "image_source",
            "image_source_sampling",
        )
        if parsed.present(f)
    }
    if width > 0 or height > 0:
        refusals_before = len(conversion.refusals)
        if width <= 0 or height <= 0:
            conversion.refuse("--image-width-mean/--image-height-mean", "image input needs both width and height means")
        for field, flag in (("image_width_stddev", "--image-width-stddev"), ("image_height_stddev", "--image-height-stddev")):
            value = parsed.floating(field, flag) or 0.0
            if value > 0:
                conversion.refuse(flag, "continuous image dimension sampling is not expressible (weighted resolutions only)")
        source = _norm(parsed.raw("image_source")) if parsed.present("image_source") else "noise"
        if source != "noise":
            conversion.refuse("--image-source", "external image pools have no equivalent (synthetic noise converts)")
        elif parsed.present("image_source_sampling"):
            conversion.drop("--image-source-sampling (inert with the noise image source)")
        image_format = _norm(parsed.raw("image_format")) if parsed.present("image_format") else "png"
        if image_format not in ("png", "jpeg"):
            conversion.refuse("--image-format", f"image format '{image_format}' has no representation (png and jpeg exist)")
        if len(conversion.refusals) == refusals_before:
            conversion.assume(
                "aiperf noise images map to synthetic image generation; generator content equivalence is assumed"
            )
            image_kwargs: Dict[str, Any] = {
                "resolutions": Resolution(width=int(width), height=int(height)),
                "representation": ImageRepresentation(image_format),
            }
            batch = parsed.integer("image_batch_size", "--image-batch-size")
            if batch is not None and batch != 1:
                image_kwargs["count"] = fixed_dist(batch)
            kwargs["image"] = ImageDatagenConfig(**image_kwargs)
    elif image_present:
        conversion.drop("image flags (inert: image means are 0, image input disabled)")

    video_width = parsed.integer("video_width", "--video-width")
    video_height = parsed.integer("video_height", "--video-height")
    video_present = {
        f
        for f in (
            "video_batch_size",
            "video_duration",
            "video_fps",
            "video_width",
            "video_height",
            "video_synth_type",
            "video_format",
            "video_codec",
            "video_audio_sample_rate",
            "video_audio_channels",
            "video_audio_codec",
            "video_audio_depth",
        )
        if parsed.present(f)
    }
    if video_width is not None or video_height is not None:
        refusals_before = len(conversion.refusals)
        if video_width is None or video_height is None:
            conversion.refuse("--video-width/--video-height", "video input needs both width and height")
        video_format = _norm(parsed.raw("video_format")) if parsed.present("video_format") else "webm"
        if video_format != "mp4":
            conversion.refuse(
                "--video-format",
                f"video format '{video_format}' (the aiperf default is webm) has no representation;"
                " mp4, png_frames and jpeg_frames exist, and transcoding assumptions are refused",
            )
        if parsed.present("video_synth_type") and _norm(parsed.raw("video_synth_type")) != "moving_shapes":
            conversion.refuse("--video-synth-type", "video generator content semantics are unverified")
        if parsed.present("video_codec") and parsed.raw("video_codec") != "libvpx-vp9":
            conversion.refuse("--video-codec", "no video codec knob exists")
        for field, flag in (
            ("video_audio_sample_rate", "--video-audio-sample-rate"),
            ("video_audio_channels", "--video-audio-num-channels"),
            ("video_audio_codec", "--video-audio-codec"),
            ("video_audio_depth", "--video-audio-depth"),
        ):
            if parsed.present(field):
                conversion.refuse(flag, "generated video has no audio track configuration")
        if len(conversion.refusals) == refusals_before and video_width is not None and video_height is not None:
            duration = parsed.floating("video_duration", "--video-duration")
            fps = parsed.integer("video_fps", "--video-fps")
            duration = 5.0 if duration is None else duration
            fps = 4 if fps is None else fps
            frames = int(duration * fps)
            conversion.assume(f"video profiles.frames = --video-duration {duration:g} x --video-fps {fps} = {frames}")
            video_kwargs: Dict[str, Any] = {
                "profiles": VideoProfile(resolution=Resolution(width=video_width, height=video_height), frames=frames),
                "representation": VideoRepresentation("mp4"),
            }
            batch = parsed.integer("video_batch_size", "--video-batch-size")
            if batch is not None and batch != 1:
                video_kwargs["count"] = fixed_dist(batch)
            kwargs["video"] = VideoDatagenConfig(**video_kwargs)
    elif video_present:
        conversion.drop("video flags (inert: no --video-width/--video-height, video input disabled)")

    if not kwargs:
        return None
    return SyntheticMultimodalDatagenConfig(**kwargs)


def _apply_group_defaults(conversion: Conversion, parsed: _Parsed) -> None:
    for spec in _FIELDS:
        if spec.field in _BESPOKE or not parsed.present(spec.field):
            continue
        flag = spec.aliases[0]
        if spec.group == "WARMUP":
            if spec.field in _WARMUP_AGENTIC:
                conversion.refuse(flag, "agentic replay warmup has no equivalent")
            else:
                conversion.no_equivalent(WARMUP_NOTE)
                conversion.drop(flag)
            continue
        reason = _GROUP_REFUSALS.get(spec.group)
        if reason is not None:
            if spec.kind == "flag" and not parsed.flag(spec.field):
                conversion.drop(f"{flag} (explicitly disabled; inert)")
            else:
                conversion.refuse(flag, reason)
            continue
        if spec.group in _ANNOTATE_GROUPS:
            conversion.drop(flag)
            continue
        raise AssertionError(f"field {spec.field} has neither bespoke handling nor a group default")


def convert_aiperf_profile(argv: List[str], peer_version: str) -> Conversion:
    """Convert one ``aiperf profile`` argv. Never writes a partial config."""
    conversion = Conversion(source_tool=TOOL_NAME, source_version=peer_version, argv=list(argv))
    facts: Optional[AiperfProfileFacts] = resolve_aiperf_profile(peer_version)
    if facts is None:
        conversion.refuse("--peer-version", unverified_version_reason(TOOL_NAME, peer_version, dict(AIPERF_PROFILE_VERSIONS)))
        return conversion
    conversion.source_version = facts.version

    parsed = _parse(argv, conversion)

    # Input and dataset flags that select loaders with no verified twin.
    for field, flag in (
        ("input_file", "--input-file"),
        ("public_dataset", "--public-dataset"),
        ("hf_dataset_subset", "--hf-subset"),
        ("hf_weka_dataset", "--hf-weka-dataset"),
        ("dataset_filters", "--dataset-filter"),
        ("custom_dataset_type", "--custom-dataset-type"),
        ("max_context_length", "--max-context-length"),
        ("trace_session_sample_ratio", "--trace-session-sample-ratio"),
    ):
        if parsed.present(field):
            conversion.refuse(
                flag, "file, public and trace dataset loaders have no verified twin; converting data files is out of scope"
            )
    for field, flag in (("ignore_trace_delays", "--ignore-trace-delays"), ("use_think_time_only", "--use-think-time-only")):
        if parsed.flag(field):
            conversion.refuse(flag, "trace replay pacing knobs ride refused trace datasets")
    if parsed.present("dataset_sampling_strategy"):
        conversion.refuse(
            "--dataset-sampling-strategy",
            "sequential/random/shuffle reuse of a finite entry pool is not expressible: inference-perf"
            " pre-generates at least one unique prompt per request",
        )
    if parsed.flag("allow_dataset_wrap"):
        conversion.refuse("--allow-dataset-wrap", "dataset wrapping reuses entries, which is not expressible")
    if parsed.present("config_file"):
        conversion.refuse("--config", "config-file input is not supported; pass the effective flags")
    if parsed.present("scenario"):
        conversion.refuse("--scenario", "a scenario preset expands into other flags inside aiperf; pass the expanded flags")
    if parsed.flag("unsafe_override"):
        conversion.drop("--unsafe-override")

    # Conversations and sessions.
    if parsed.present("conversation_num"):
        conversion.refuse("--conversation-num", "session-oriented synthetic benchmarking has no verified twin")
    turn_mean = parsed.integer("conversation_turn_mean", "--conversation-turn-mean")
    if turn_mean is not None and turn_mean != 1:
        conversion.refuse("--conversation-turn-mean", "multi-turn synthetic conversations have no verified twin")
    turn_stddev = parsed.integer("conversation_turn_stddev", "--conversation-turn-stddev")
    if turn_stddev is not None and turn_stddev != 0:
        conversion.refuse("--conversation-turn-stddev", "multi-turn synthetic conversations have no verified twin")
    for field, flag in (
        ("conversation_turn_delay_mean", "--conversation-turn-delay-mean"),
        ("conversation_turn_delay_stddev", "--conversation-turn-delay-stddev"),
        ("conversation_turn_delay_ratio", "--conversation-turn-delay-ratio"),
        ("open_loop_replay", "--open-loop-replay"),
        ("open_loop_strict", "--open-loop-strict"),
        ("omit_kv_hints", "--omit-kv-hints"),
        ("force_min_tokens", "--force-min-tokens"),
    ):
        if parsed.present(field):
            conversion.drop(f"{flag} (inert for single-turn synthetic runs)")
    for field, flag in (
        ("inter_turn_delay_cap_seconds", "--inter-turn-delay-cap-seconds"),
        ("max_idle_gap_cap_seconds", "--max-idle-gap-cap-seconds"),
        ("replay_speedup", "--replay-speedup"),
    ):
        if parsed.present(field):
            conversion.refuse(flag, "trace loader pacing knobs ride refused trace datasets")

    # Prompt shape, cache busting and prefix pools.
    batch = parsed.integer("prompt_batch_size", "--prompt-batch-size")
    if batch is not None and batch != 1:
        conversion.drop("--prompt-batch-size (batching applies to embeddings/rankings endpoints; inert here)")
    if parsed.present("prompt_corpus"):
        conversion.drop("--prompt-corpus (corpus choice recorded; prompt text never transfers between tools)")
    if parsed.present("cache_bust") and _norm(parsed.raw("cache_bust")) != "none":
        conversion.refuse(
            "--cache-bust", "cache-bust markers shape prefix-cache behavior; approximating changes what is measured"
        )
    pool = parsed.integer("prompt_prefix_pool_size", "--prompt-prefix-pool-size") or 0
    if pool > 0:
        conversion.refuse(
            "--prompt-prefix-pool-size",
            "a prefix pool sampled with replacement per request is a different cache-hit pattern from"
            " data.shared_prefix's grouped structure; mapping would misstate the cached fraction",
        )
    elif parsed.present("prompt_prefix_length"):
        conversion.drop("--prompt-prefix-length (inert with a prefix pool size of 0)")
    for field, flag in (
        ("prompt_prefix_shared_system_length", "--shared-system-prompt-length"),
        ("prompt_prefix_user_context_length", "--user-context-prompt-length"),
    ):
        if parsed.present(field):
            conversion.refuse(flag, "structured shared or per-session context has no verified equivalent")

    # Endpoint block.
    api_type = _convert_api_type(conversion, parsed, facts)
    if parsed.present("custom_endpoint"):
        conversion.refuse("--custom-endpoint", "a custom endpoint path is not expressible (the client owns its route)")
    models = parsed.items("model_names")
    model_name: Optional[str] = None
    if not models:
        raise PeerUsageError("--model is required")
    if len(models) > 1:
        conversion.refuse("--model-names", "multi-model traffic has no equivalent (a single server.model_name exists)")
    else:
        model_name = models[0]
    if parsed.present("model_selection_strategy"):
        conversion.drop("--model-selection-strategy (inert with a single model)")
    urls = parsed.items("urls")
    base_url = "http://localhost:8000"
    if len(urls) > 1:
        conversion.refuse("--url", "client-side load balancing across servers has no equivalent")
    elif urls:
        base_url = urls[0]
        if "://" not in base_url:
            base_url = "http://" + base_url
            conversion.assume(f"--url without a scheme is normalized to {base_url}, matching aiperf")
    else:
        conversion.assume("no --url given: aiperf defaults to http://localhost:8000")
    if parsed.present("url_selection_strategy"):
        conversion.drop("--url-strategy (inert with a single URL)")
    streaming = parsed.flag("streaming", default=False)
    if not parsed.present("streaming"):
        conversion.assume(
            "api.streaming: false emitted explicitly: aiperf streaming is opt-in (vllm bench cannot even express false)"
        )
    reuse = _norm(parsed.raw("connection_reuse_strategy")) if parsed.present("connection_reuse_strategy") else "pooled"
    if reuse != "pooled":
        conversion.refuse(
            "--connection-reuse-strategy",
            f"'{reuse}' is not expressible; pooled sessions are the only connection model, and connection reuse alters timing",
        )
    if parsed.present("request_content_type") and "multipart" in _norm(parsed.raw("request_content_type")):
        conversion.refuse("--request-content-type", "only JSON request bodies are sent")
    if parsed.flag("download_video_content"):
        conversion.refuse(
            "--download-video-content", "video endpoints are refused and the flag redefines what latency includes"
        )
    if parsed.flag("uuid_and_strip"):
        conversion.refuse("--uuid-and-strip", "the multimodal processor-cache stripping protocol has no equivalent")
    if parsed.present("transport"):
        conversion.drop("--transport (only http is registered at v0.12.0)")
    for field, flag in (
        ("wait_for_model_timeout", "--wait-for-model-timeout"),
        ("wait_for_model_mode", "--wait-for-model-mode"),
        ("wait_for_model_interval", "--wait-for-model-interval"),
    ):
        if parsed.present(field):
            conversion.drop(f"{flag} (pre-benchmark readiness probe, outside the measured window)")

    # Tokenizer block.
    tokenizer_id = parsed.raw("tokenizer_name")
    if tokenizer_id is not None and _norm(tokenizer_id) == "builtin":
        conversion.refuse(
            "--tokenizer",
            "the builtin tiktoken tokenizer counts lengths in a different vocabulary; no tokenizer backend selector exists",
        )
    if parsed.present("tokenizer_revision") and parsed.raw("tokenizer_revision") != "main":
        conversion.refuse(
            "--tokenizer-revision",
            "no tokenizer revision field exists, and a different revision changes synthetic prompt construction",
        )
    if parsed.flag("apply_chat_template"):
        if api_type is not None and api_type != APIType.Completion:
            conversion.refuse(
                "--apply-chat-template",
                "data.use_chat_template is only supported on the completions path with random data",
            )

    ignore_eos = _convert_extra_inputs(conversion, parsed)
    if ignore_eos is None:
        conversion.assume(
            "server.ignore_eos: false is emitted explicitly: inference-perf defaults it to true, while"
            " aiperf sends ignore_eos only when injected via --extra-inputs"
        )
    headers = _convert_headers(conversion, parsed)
    goodput = _convert_goodput(conversion, parsed, facts)

    load_shape = _convert_load(conversion, parsed, facts)
    input_dist, output_dist = _convert_data(conversion, parsed, facts, api_type)
    multimodal = _convert_multimodal(conversion, parsed)

    # The dataset-entry pool must cover the converted request count, since the
    # generated data pool is unique per request and never reused.
    entries = parsed.integer("conversation_num_dataset_entries", "--num-dataset-entries")
    pool_size = entries if entries is not None else facts.default_num_dataset_entries
    if load_shape is not None:
        if pool_size < load_shape.request_count:
            default_note = (
                "" if entries is not None else f" (the untouched default pool is {facts.default_num_dataset_entries})"
            )
            conversion.refuse(
                "--num-dataset-entries",
                f"prompt reuse is not expressible: aiperf samples a pool of {pool_size} unique entries"
                f" across {load_shape.request_count} requests{default_note}, while inference-perf"
                " pre-generates at least one unique prompt per request",
            )
        elif entries is not None:
            conversion.drop("--num-dataset-entries (inert: the pool covers the request count, so no entry is reused)")

    # Session headers ride the multi-turn path; note the default either way.
    if parsed.present("session_header"):
        conversion.assume(
            "--session-header maps to api.session_id_header_key; aiperf's default X-Correlation-ID header"
            " is not replicated when the flag is absent"
        )
    else:
        conversion.no_equivalent(
            "session header: aiperf sends its default X-Correlation-ID header on every request; the"
            " converted run sends no session header"
        )

    if parsed.present("api_key"):
        conversion.assume(
            "--api-key is written into server.api_key in clear text; inference-perf redacts it from logs"
            " and saved copies of the config"
        )
    if parsed.flag("use_legacy_max_tokens") and api_type == APIType.Chat:
        conversion.assume("--use-legacy-max-tokens matches inference-perf's chat payload, which sends legacy max_tokens")
    elif api_type == APIType.Chat:
        conversion.assume(
            "chat path wire fields differ: aiperf sends max_completion_tokens by default while"
            " inference-perf sends legacy max_tokens; OpenAI-compatible servers treat them the same"
        )

    grace = parsed.floating("benchmark_grace_period", "--benchmark-grace-period")
    if grace is not None and math.isinf(grace):
        conversion.refuse("--benchmark-grace-period", "an infinite grace period is not expressible (the field is finite)")
    elif grace is not None:
        conversion.assume(
            "--benchmark-grace-period maps to load.stage_teardown_grace_seconds, but the measurement"
            " windows differ: aiperf includes grace-period responses in metrics, inference-perf excludes"
            " the teardown window from stage metrics"
        )

    workers = parsed.integer("workers_max", "--workers-max")
    if workers is not None:
        conversion.assume(
            "--workers-max maps to load.num_workers: both cap client parallelism, but aiperf workers are"
            " asyncio services and inference-perf workers are processes; the offered load is defined by"
            " the load model provided the pool covers it"
        )

    seed = parsed.integer("random_seed", "--random-seed")
    if seed is not None:
        conversion.assume(
            f"load.base_seed: {seed} from --random-seed; seeds are tool-local, so this pins reproducibility"
            " of the converted run, not identical prompt text"
        )
    else:
        conversion.assume(
            "no --random-seed given: aiperf uses entropy and inference-perf defaults base_seed to the"
            " current time; both runs are unseeded"
        )

    metrics: Optional[MetricsClientConfig] = None
    scrape_urls = parsed.items("server_metrics")
    if len(scrape_urls) == 1:
        conversion.assume("--server-metrics maps to a prometheus metrics client; scrape cadence and format defaults differ")
        metrics = MetricsClientConfig(type=MetricsClientType.PROMETHEUS, prometheus=PrometheusClientConfig(url=scrape_urls[0]))
    elif len(scrape_urls) > 1:
        conversion.drop("--server-metrics (multiple scrape URLs; only a single prometheus url is expressible)")

    storage: Optional[StorageConfig] = None
    if parsed.present("artifact_directory"):
        artifact_dir = parsed.raw("artifact_directory")
        assert artifact_dir is not None
        conversion.assume(
            "--output-artifact-dir maps to storage.local_storage.path; inference-perf writes its own report"
            " file set there, not aiperf's artifact tree"
        )
        storage = StorageConfig(local_storage=StorageConfigBase(path=artifact_dir))

    _apply_group_defaults(conversion, parsed)

    if conversion.refusals:
        return conversion
    assert api_type is not None and load_shape is not None
    assert input_dist is not None and output_dist is not None and model_name is not None

    api_kwargs: Dict[str, Any] = {"type": api_type, "streaming": streaming}
    if headers is not None:
        api_kwargs["headers"] = headers
    if parsed.present("session_header"):
        api_kwargs["session_id_header_key"] = parsed.raw("session_header")

    data_kwargs: Dict[str, Any] = {
        "type": DataGenType.Random,
        "input_distribution": input_dist,
        "output_distribution": output_dist,
    }
    if parsed.flag("apply_chat_template"):
        data_kwargs["use_chat_template"] = True
    if multimodal is not None:
        data_kwargs["multimodal"] = multimodal

    load_kwargs: Dict[str, Any] = {"type": load_shape.load_type, "stages": [load_shape.stage]}
    if seed is not None:
        load_kwargs["base_seed"] = seed
    timeout = parsed.floating("timeout_seconds", "--request-timeout-seconds")
    if timeout is not None:
        load_kwargs["request_timeout"] = timeout
    if grace is not None and not math.isinf(grace):
        load_kwargs["stage_teardown_grace_seconds"] = grace
    if workers is not None:
        load_kwargs["num_workers"] = workers

    server_kwargs: Dict[str, Any] = {
        "base_url": base_url,
        "model_name": model_name,
        "ignore_eos": bool(ignore_eos) if ignore_eos is not None else False,
    }
    if parsed.present("api_key"):
        server_kwargs["api_key"] = parsed.raw("api_key")

    tokenizer_kwargs: Dict[str, Any] = {
        "pretrained_model_name_or_path": tokenizer_id if tokenizer_id is not None else model_name
    }
    if parsed.flag("trust_remote_code"):
        tokenizer_kwargs["trust_remote_code"] = True

    report: Optional[ReportConfig] = None
    report_kwargs: Dict[str, Any] = {}
    if parsed.flag("use_server_token_count"):
        conversion.assume(
            "--use-server-token-count maps to report.request_lifecycle.use_server_output_tokens, the"
            " cross-tool recommendation for comparable token counting"
        )
        report_kwargs["request_lifecycle"] = RequestLifecycleMetricsReportConfig(use_server_output_tokens=True)
    if goodput is not None:
        report_kwargs["goodput"] = goodput
    if report_kwargs:
        report = ReportConfig(**report_kwargs)

    config_kwargs: Dict[str, Any] = {
        "api": APIConfig(**api_kwargs),
        "data": DataConfig(**data_kwargs),
        "load": LoadConfig(**load_kwargs),
        "server": ModelServerClientConfig(**server_kwargs),
        "tokenizer": CustomTokenizerConfig(**tokenizer_kwargs),
    }
    if report is not None:
        config_kwargs["report"] = report
    if storage is not None:
        config_kwargs["storage"] = storage
    if metrics is not None:
        config_kwargs["metrics"] = metrics
    conversion.config = Config(**config_kwargs)
    return conversion
