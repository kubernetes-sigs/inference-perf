# Inference-Perf CLI Flags

These command line flags are automatically generated from the internal `Config` schema. You can override any configuration directly from the CLI without using a yaml configuration file.

| Flag | Type | Description |
| --- | --- | --- |
| `--api.type` | Enum (completion, chat) | API type to exercise on the model server. |
| `--api.streaming` | boolean | Enable streaming so TTFT, ITL, and TPOT can be measured. |
| `--api.headers` | JSON | Custom HTTP headers attached to every request. |
| `--api.slo_unit` | str | Unit for SLO header values (e.g. 'ms', 's'). Defaults to 'ms'. |
| `--api.slo_tpot_header` | str | Header name carrying the per-request TPOT SLO. |
| `--api.slo_ttft_header` | str | Header name carrying the per-request TTFT SLO. |
| `--api.response_format.type` | Enum (json_schema, json_object) | Response-format variant to request. |
| `--api.response_format.name` | str | Schema name embedded in the json_schema payload. |
| `--api.response_format.json_schema` | JSON | JSON Schema describing the required output shape (for json_schema type). |
| `--data.type` | Enum (mock, shareGPT, synthetic, random, shared_prefix, cnn_dailymail, infinity_instruct, billsum_conversations, otel_trace_replay, conversation_replay) | Which data generator to use; the sibling fields below apply only to specific types. |
| `--data.path` | str | Filesystem path to a dataset. Required for shareGPT, cnn_dailymail, billsum_conversations, and infinity_instruct. |
| `--data.input_distribution.min` | int | Minimum sampled value (tokens, items, etc.). |
| `--data.input_distribution.max` | int | Maximum sampled value. |
| `--data.input_distribution.mean` | float | Distribution mean. |
| `--data.input_distribution.std_dev` | float | Standard deviation. Mutually exclusive with `variance`. |
| `--data.input_distribution.total_count` | int | Total number of samples to draw, when the distribution is materialized eagerly. |
| `--data.input_distribution.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Distribution family used to sample values. |
| `--data.input_distribution.variance` | float | Variance. Mutually exclusive with `std_dev`; converted to std_dev when set. |
| `--data.input_distribution.skew` | float | Skew parameter; only used when type=skew_normal. |
| `--data.output_distribution.min` | int | Minimum sampled value (tokens, items, etc.). |
| `--data.output_distribution.max` | int | Maximum sampled value. |
| `--data.output_distribution.mean` | float | Distribution mean. |
| `--data.output_distribution.std_dev` | float | Standard deviation. Mutually exclusive with `variance`. |
| `--data.output_distribution.total_count` | int | Total number of samples to draw, when the distribution is materialized eagerly. |
| `--data.output_distribution.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Distribution family used to sample values. |
| `--data.output_distribution.variance` | float | Variance. Mutually exclusive with `std_dev`; converted to std_dev when set. |
| `--data.output_distribution.skew` | float | Skew parameter; only used when type=skew_normal. |
| `--data.shared_prefix.num_groups` | int | Number of distinct shared-prefix groups (one shared system prompt per group). |
| `--data.shared_prefix.num_prompts_per_group` | int | Number of unique user prompts generated per shared-prefix group. |
| `--data.shared_prefix.system_prompt_len` | string | Shared system-prompt length, as a fixed int or a Distribution. |
| `--data.shared_prefix.question_len` | string | Per-question length, as a fixed int or a Distribution. |
| `--data.shared_prefix.output_len` | string | Per-response output length, as a fixed int or a Distribution. |
| `--data.shared_prefix.seed` | int | Random seed for deterministic prompt generation. |
| `--data.shared_prefix.question_distribution.min` | int | Minimum sampled value (tokens, items, etc.). |
| `--data.shared_prefix.question_distribution.max` | int | Maximum sampled value. |
| `--data.shared_prefix.question_distribution.mean` | float | Distribution mean. |
| `--data.shared_prefix.question_distribution.std_dev` | float | Standard deviation. Mutually exclusive with `variance`. |
| `--data.shared_prefix.question_distribution.total_count` | int | Total number of samples to draw, when the distribution is materialized eagerly. |
| `--data.shared_prefix.question_distribution.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Distribution family used to sample values. |
| `--data.shared_prefix.question_distribution.variance` | float | Variance. Mutually exclusive with `std_dev`; converted to std_dev when set. |
| `--data.shared_prefix.question_distribution.skew` | float | Skew parameter; only used when type=skew_normal. |
| `--data.shared_prefix.output_distribution.min` | int | Minimum sampled value (tokens, items, etc.). |
| `--data.shared_prefix.output_distribution.max` | int | Maximum sampled value. |
| `--data.shared_prefix.output_distribution.mean` | float | Distribution mean. |
| `--data.shared_prefix.output_distribution.std_dev` | float | Standard deviation. Mutually exclusive with `variance`. |
| `--data.shared_prefix.output_distribution.total_count` | int | Total number of samples to draw, when the distribution is materialized eagerly. |
| `--data.shared_prefix.output_distribution.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Distribution family used to sample values. |
| `--data.shared_prefix.output_distribution.variance` | float | Variance. Mutually exclusive with `std_dev`; converted to std_dev when set. |
| `--data.shared_prefix.output_distribution.skew` | float | Skew parameter; only used when type=skew_normal. |
| `--data.shared_prefix.enable_multi_turn_chat` | boolean | When true, prompts within a group form multi-turn chat sessions. |
| `--data.shared_prefix.multimodal.image.count.min` | int | Minimum sampled value (tokens, items, etc.). |
| `--data.shared_prefix.multimodal.image.count.max` | int | Maximum sampled value. |
| `--data.shared_prefix.multimodal.image.count.mean` | float | Distribution mean. |
| `--data.shared_prefix.multimodal.image.count.std_dev` | float | Standard deviation. Mutually exclusive with `variance`. |
| `--data.shared_prefix.multimodal.image.count.total_count` | int | Total number of samples to draw, when the distribution is materialized eagerly. |
| `--data.shared_prefix.multimodal.image.count.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Distribution family used to sample values. |
| `--data.shared_prefix.multimodal.image.count.variance` | float | Variance. Mutually exclusive with `std_dev`; converted to std_dev when set. |
| `--data.shared_prefix.multimodal.image.count.skew` | float | Skew parameter; only used when type=skew_normal. |
| `--data.shared_prefix.multimodal.image.insertion_point` | string | Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from. |
| `--data.shared_prefix.multimodal.image.resolutions` | JSON | Resolution or list of weighted resolutions for generated images. |
| `--data.shared_prefix.multimodal.image.representation` | Enum (png, jpeg) | Wire encoding for emitted image bytes: ``png`` (default, lossless) or ``jpeg`` (lossy, smaller payload). Some VLMs prefer one or the other; consult the model's spec sheet. |
| `--data.shared_prefix.multimodal.video.count.min` | int | Minimum sampled value (tokens, items, etc.). |
| `--data.shared_prefix.multimodal.video.count.max` | int | Maximum sampled value. |
| `--data.shared_prefix.multimodal.video.count.mean` | float | Distribution mean. |
| `--data.shared_prefix.multimodal.video.count.std_dev` | float | Standard deviation. Mutually exclusive with `variance`. |
| `--data.shared_prefix.multimodal.video.count.total_count` | int | Total number of samples to draw, when the distribution is materialized eagerly. |
| `--data.shared_prefix.multimodal.video.count.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Distribution family used to sample values. |
| `--data.shared_prefix.multimodal.video.count.variance` | float | Variance. Mutually exclusive with `std_dev`; converted to std_dev when set. |
| `--data.shared_prefix.multimodal.video.count.skew` | float | Skew parameter; only used when type=skew_normal. |
| `--data.shared_prefix.multimodal.video.insertion_point` | string | Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from. |
| `--data.shared_prefix.multimodal.video.profiles` | JSON | Video profile or list of weighted video profiles for generated videos. |
| `--data.shared_prefix.multimodal.video.representation` | Enum (mp4, png_frames, jpeg_frames) | Wire-format strategy. ``mp4`` sends one ``video_url`` block carrying an MP4 blob (measures full pipeline including server-side decode). ``png_frames`` and ``jpeg_frames`` send ``frames`` × ``image_url`` blocks at one insertion point in the named encoding (no decode dependency, useful for prefix-cache benchmarks and servers that don't accept ``video_url``). |
| `--data.shared_prefix.multimodal.audio.count.min` | int | Minimum sampled value (tokens, items, etc.). |
| `--data.shared_prefix.multimodal.audio.count.max` | int | Maximum sampled value. |
| `--data.shared_prefix.multimodal.audio.count.mean` | float | Distribution mean. |
| `--data.shared_prefix.multimodal.audio.count.std_dev` | float | Standard deviation. Mutually exclusive with `variance`. |
| `--data.shared_prefix.multimodal.audio.count.total_count` | int | Total number of samples to draw, when the distribution is materialized eagerly. |
| `--data.shared_prefix.multimodal.audio.count.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Distribution family used to sample values. |
| `--data.shared_prefix.multimodal.audio.count.variance` | float | Variance. Mutually exclusive with `std_dev`; converted to std_dev when set. |
| `--data.shared_prefix.multimodal.audio.count.skew` | float | Skew parameter; only used when type=skew_normal. |
| `--data.shared_prefix.multimodal.audio.insertion_point` | string | Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from. |
| `--data.shared_prefix.multimodal.audio.durations` | JSON | Duration or list of weighted durations for generated audio clips. |
| `--data.multimodal.image.count.min` | int | Minimum sampled value (tokens, items, etc.). |
| `--data.multimodal.image.count.max` | int | Maximum sampled value. |
| `--data.multimodal.image.count.mean` | float | Distribution mean. |
| `--data.multimodal.image.count.std_dev` | float | Standard deviation. Mutually exclusive with `variance`. |
| `--data.multimodal.image.count.total_count` | int | Total number of samples to draw, when the distribution is materialized eagerly. |
| `--data.multimodal.image.count.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Distribution family used to sample values. |
| `--data.multimodal.image.count.variance` | float | Variance. Mutually exclusive with `std_dev`; converted to std_dev when set. |
| `--data.multimodal.image.count.skew` | float | Skew parameter; only used when type=skew_normal. |
| `--data.multimodal.image.insertion_point` | string | Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from. |
| `--data.multimodal.image.resolutions` | JSON | Resolution or list of weighted resolutions for generated images. |
| `--data.multimodal.image.representation` | Enum (png, jpeg) | Wire encoding for emitted image bytes: ``png`` (default, lossless) or ``jpeg`` (lossy, smaller payload). Some VLMs prefer one or the other; consult the model's spec sheet. |
| `--data.multimodal.video.count.min` | int | Minimum sampled value (tokens, items, etc.). |
| `--data.multimodal.video.count.max` | int | Maximum sampled value. |
| `--data.multimodal.video.count.mean` | float | Distribution mean. |
| `--data.multimodal.video.count.std_dev` | float | Standard deviation. Mutually exclusive with `variance`. |
| `--data.multimodal.video.count.total_count` | int | Total number of samples to draw, when the distribution is materialized eagerly. |
| `--data.multimodal.video.count.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Distribution family used to sample values. |
| `--data.multimodal.video.count.variance` | float | Variance. Mutually exclusive with `std_dev`; converted to std_dev when set. |
| `--data.multimodal.video.count.skew` | float | Skew parameter; only used when type=skew_normal. |
| `--data.multimodal.video.insertion_point` | string | Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from. |
| `--data.multimodal.video.profiles` | JSON | Video profile or list of weighted video profiles for generated videos. |
| `--data.multimodal.video.representation` | Enum (mp4, png_frames, jpeg_frames) | Wire-format strategy. ``mp4`` sends one ``video_url`` block carrying an MP4 blob (measures full pipeline including server-side decode). ``png_frames`` and ``jpeg_frames`` send ``frames`` × ``image_url`` blocks at one insertion point in the named encoding (no decode dependency, useful for prefix-cache benchmarks and servers that don't accept ``video_url``). |
| `--data.multimodal.audio.count.min` | int | Minimum sampled value (tokens, items, etc.). |
| `--data.multimodal.audio.count.max` | int | Maximum sampled value. |
| `--data.multimodal.audio.count.mean` | float | Distribution mean. |
| `--data.multimodal.audio.count.std_dev` | float | Standard deviation. Mutually exclusive with `variance`. |
| `--data.multimodal.audio.count.total_count` | int | Total number of samples to draw, when the distribution is materialized eagerly. |
| `--data.multimodal.audio.count.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Distribution family used to sample values. |
| `--data.multimodal.audio.count.variance` | float | Variance. Mutually exclusive with `std_dev`; converted to std_dev when set. |
| `--data.multimodal.audio.count.skew` | float | Skew parameter; only used when type=skew_normal. |
| `--data.multimodal.audio.insertion_point` | string | Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from. |
| `--data.multimodal.audio.durations` | JSON | Duration or list of weighted durations for generated audio clips. |
| `--data.trace.file` | str | Path to the trace file to replay. |
| `--data.trace.format` | Enum (AzurePublicDataset) | On-disk format of the trace file. |
| `--data.otel_trace_replay.trace_directory` | str | Directory containing OTel JSON trace files |
| `--data.otel_trace_replay.trace_files` | JSON | List of paths to specific OTel JSON trace files |
| `--data.otel_trace_replay.use_static_model` | boolean | Use a single static model for all requests |
| `--data.otel_trace_replay.static_model_name` | str | Static model name (required if use_static_model=True) |
| `--data.otel_trace_replay.model_mapping` | JSON | Map recorded model names to target models |
| `--data.otel_trace_replay.default_max_tokens` | int | Default max_tokens if not specified in trace |
| `--data.otel_trace_replay.include_errors` | boolean | Include spans with error status |
| `--data.otel_trace_replay.skip_invalid_files` | boolean | Skip invalid trace files instead of failing |
| `--data.conversation_replay.seed` | int | Random seed for deterministic generation |
| `--data.conversation_replay.num_conversations` | int | Number of conversation blueprints to generate |
| `--data.conversation_replay.shared_system_prompt_len` | int | Fixed shared system prompt length in tokens |
| `--data.conversation_replay.dynamic_system_prompt_len.min` | int | Minimum sampled value (tokens, items, etc.). |
| `--data.conversation_replay.dynamic_system_prompt_len.max` | int | Maximum sampled value. |
| `--data.conversation_replay.dynamic_system_prompt_len.mean` | float | Distribution mean. |
| `--data.conversation_replay.dynamic_system_prompt_len.std_dev` | float | Standard deviation. Mutually exclusive with `variance`. |
| `--data.conversation_replay.dynamic_system_prompt_len.total_count` | int | Total number of samples to draw, when the distribution is materialized eagerly. |
| `--data.conversation_replay.dynamic_system_prompt_len.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Distribution family used to sample values. |
| `--data.conversation_replay.dynamic_system_prompt_len.variance` | float | Variance. Mutually exclusive with `std_dev`; converted to std_dev when set. |
| `--data.conversation_replay.dynamic_system_prompt_len.skew` | float | Skew parameter; only used when type=skew_normal. |
| `--data.conversation_replay.turns_per_conversation.min` | int | Minimum sampled value (tokens, items, etc.). |
| `--data.conversation_replay.turns_per_conversation.max` | int | Maximum sampled value. |
| `--data.conversation_replay.turns_per_conversation.mean` | float | Distribution mean. |
| `--data.conversation_replay.turns_per_conversation.std_dev` | float | Standard deviation. Mutually exclusive with `variance`. |
| `--data.conversation_replay.turns_per_conversation.total_count` | int | Total number of samples to draw, when the distribution is materialized eagerly. |
| `--data.conversation_replay.turns_per_conversation.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Distribution family used to sample values. |
| `--data.conversation_replay.turns_per_conversation.variance` | float | Variance. Mutually exclusive with `std_dev`; converted to std_dev when set. |
| `--data.conversation_replay.turns_per_conversation.skew` | float | Skew parameter; only used when type=skew_normal. |
| `--data.conversation_replay.input_tokens_per_turn.min` | int | Minimum sampled value (tokens, items, etc.). |
| `--data.conversation_replay.input_tokens_per_turn.max` | int | Maximum sampled value. |
| `--data.conversation_replay.input_tokens_per_turn.mean` | float | Distribution mean. |
| `--data.conversation_replay.input_tokens_per_turn.std_dev` | float | Standard deviation. Mutually exclusive with `variance`. |
| `--data.conversation_replay.input_tokens_per_turn.total_count` | int | Total number of samples to draw, when the distribution is materialized eagerly. |
| `--data.conversation_replay.input_tokens_per_turn.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Distribution family used to sample values. |
| `--data.conversation_replay.input_tokens_per_turn.variance` | float | Variance. Mutually exclusive with `std_dev`; converted to std_dev when set. |
| `--data.conversation_replay.input_tokens_per_turn.skew` | float | Skew parameter; only used when type=skew_normal. |
| `--data.conversation_replay.output_tokens_per_turn.min` | int | Minimum sampled value (tokens, items, etc.). |
| `--data.conversation_replay.output_tokens_per_turn.max` | int | Maximum sampled value. |
| `--data.conversation_replay.output_tokens_per_turn.mean` | float | Distribution mean. |
| `--data.conversation_replay.output_tokens_per_turn.std_dev` | float | Standard deviation. Mutually exclusive with `variance`. |
| `--data.conversation_replay.output_tokens_per_turn.total_count` | int | Total number of samples to draw, when the distribution is materialized eagerly. |
| `--data.conversation_replay.output_tokens_per_turn.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Distribution family used to sample values. |
| `--data.conversation_replay.output_tokens_per_turn.variance` | float | Variance. Mutually exclusive with `std_dev`; converted to std_dev when set. |
| `--data.conversation_replay.output_tokens_per_turn.skew` | float | Skew parameter; only used when type=skew_normal. |
| `--data.conversation_replay.tool_call_latency_sec.min` | int | Minimum sampled value (tokens, items, etc.). |
| `--data.conversation_replay.tool_call_latency_sec.max` | int | Maximum sampled value. |
| `--data.conversation_replay.tool_call_latency_sec.mean` | float | Distribution mean. |
| `--data.conversation_replay.tool_call_latency_sec.std_dev` | float | Standard deviation. Mutually exclusive with `variance`. |
| `--data.conversation_replay.tool_call_latency_sec.total_count` | int | Total number of samples to draw, when the distribution is materialized eagerly. |
| `--data.conversation_replay.tool_call_latency_sec.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Distribution family used to sample values. |
| `--data.conversation_replay.tool_call_latency_sec.variance` | float | Variance. Mutually exclusive with `std_dev`; converted to std_dev when set. |
| `--data.conversation_replay.tool_call_latency_sec.skew` | float | Skew parameter; only used when type=skew_normal. |
| `--load.type` | Enum (constant, poisson, trace_replay, concurrent, trace_session_replay) | Traffic-generation strategy. |
| `--load.interval` | float | Inter-request interval in seconds for load types that pace by interval. |
| `--load.stages` | JSON | Ordered list of load stages; stage subclass must match the chosen load type. |
| `--load.sweep.type` | Enum (geometric, linear) | How rate steps between stages are spaced. |
| `--load.sweep.num_requests` | int | Total requests to issue across the sweep. |
| `--load.sweep.timeout` | float | Per-stage timeout in seconds before saturation is declared. |
| `--load.sweep.num_stages` | int | Number of stages in the sweep. |
| `--load.sweep.stage_duration` | int | Duration of each stage in seconds. |
| `--load.sweep.saturation_percentile` | float | Percentile latency used to detect saturation. |
| `--load.num_workers` | int | Number of worker processes used to drive load. |
| `--load.worker_max_concurrency` | int | Maximum concurrent in-flight requests per worker. |
| `--load.worker_max_tcp_connections` | int | TCP connection pool size per worker. |
| `--load.trace.file` | str | Path to the trace file to replay. |
| `--load.trace.format` | Enum (AzurePublicDataset) | On-disk format of the trace file. |
| `--load.circuit_breakers` | JSON | Names of circuit breakers (defined under `circuit_breakers`) to enable. |
| `--load.request_timeout` | float | Per-request timeout in seconds; None disables the timeout. |
| `--load.lora_traffic_split` | JSON | Optional traffic split across LoRA adapters; splits must sum to 1.0. |
| `--load.base_seed` | int | Base seed shared across workers; defaults to wall-clock time on startup. |
| `--metrics.type` | Enum (prometheus, default) | Metrics backend used to collect server-side metrics. |
| `--metrics.prometheus.scrape_interval` | int | Prometheus scrape interval in seconds; used to align query windows. |
| `--metrics.prometheus.url` | string | Prometheus HTTP endpoint. Mutually exclusive with `google_managed`. |
| `--metrics.prometheus.filters` | JSON | PromQL label filters applied to every query. |
| `--metrics.prometheus.google_managed` | boolean | Use Google Managed Prometheus instead of `url`. Mutually exclusive with `url`. |
| `--report.request_lifecycle.summary` | boolean | Emit an aggregate summary across all stages. |
| `--report.request_lifecycle.per_stage` | boolean | Emit one report per load stage. |
| `--report.request_lifecycle.per_request` | boolean | Emit one row per individual request (large output). |
| `--report.request_lifecycle.per_adapter` | boolean | Emit one report per LoRA adapter. |
| `--report.request_lifecycle.per_adapter_stage` | boolean | Emit one report per (adapter, stage) pair. |
| `--report.request_lifecycle.percentiles` | JSON | Latency percentiles to compute and emit. |
| `--report.prometheus.summary` | boolean | Emit an aggregate Prometheus summary across all stages. |
| `--report.prometheus.per_stage` | boolean | Emit per-stage Prometheus reports. |
| `--report.session_lifecycle.summary` | boolean | Emit aggregate session metrics across all stages. |
| `--report.session_lifecycle.per_stage` | boolean | Emit per-stage session metrics. |
| `--report.session_lifecycle.per_session` | boolean | Emit one row per individual session (large output). |
| `--report.goodput.constraints` | JSON | Mapping of SLO metric name to threshold; requests meeting all thresholds count as good. |
| `--storage.local_storage.path` | str | Directory or object-store path where reports are written. Supports `{timestamp}` substitution. |
| `--storage.local_storage.report_file_prefix` | str | Optional filename prefix for report files written to this destination. |
| `--storage.google_cloud_storage.path` | str | Directory or object-store path where reports are written. Supports `{timestamp}` substitution. |
| `--storage.google_cloud_storage.report_file_prefix` | str | Optional filename prefix for report files written to this destination. |
| `--storage.google_cloud_storage.bucket_name` | str | Target GCS bucket. |
| `--storage.simple_storage_service.path` | str | Directory or object-store path where reports are written. Supports `{timestamp}` substitution. |
| `--storage.simple_storage_service.report_file_prefix` | str | Optional filename prefix for report files written to this destination. |
| `--storage.simple_storage_service.bucket_name` | str | Target S3-compatible bucket. |
| `--storage.simple_storage_service.endpoint_url` | str | Override endpoint URL for S3-compatible services (e.g. MinIO). |
| `--storage.simple_storage_service.region_name` | str | Region for the S3-compatible service. |
| `--storage.simple_storage_service.addressing_style` | string | S3 addressing style (auto, virtual-hosted, or path-style). |
| `--server.type` | Enum (vllm, sglang, tgi, mock) | Model server flavor; controls server-specific metric mappings. |
| `--server.model_name` | str | Model identifier sent on each request. May be auto-detected via /v1/models. |
| `--server.base_url` | str | Base URL of the model server, e.g. http://0.0.0.0:8000. |
| `--server.ignore_eos` | boolean | Ask the server to ignore EOS so output length is governed by max_tokens. |
| `--server.api_key` | str | Bearer token sent as Authorization header. |
| `--server.cert_path` | str | Path to a TLS client certificate (PEM) for mTLS. |
| `--server.key_path` | str | Path to a TLS client private key (PEM) for mTLS. |
| `--tokenizer.pretrained_model_name_or_path` | str | HuggingFace tokenizer name or local path. |
| `--tokenizer.trust_remote_code` | boolean | Forwarded to HuggingFace tokenizer loading; required by some custom tokenizers. |
| `--tokenizer.token` | str | HuggingFace auth token used to download gated tokenizers. |
| `--circuit_breakers` | JSON | Declarative circuit breakers that can stop the run on metric thresholds. |
