# Inference-Perf CLI Flags

These command line flags are automatically generated from the CLI parser. The global flags at the top of the table control the tool itself; every other flag is generated from the internal `Config` schema and overrides that configuration directly from the CLI without using a yaml configuration file.

| Flag | Type | Description |
| --- | --- | --- |
| `-c`, `--config_file` | str | Config File |
| `-a`, `--analyze` | list of str | Path to a report directories to analyze |
| `-u`, `--unified_analysis_dir` | str | Unified analysis directory path |
| `--log-level` | Enum (DEBUG, INFO, WARNING, ERROR, CRITICAL) | Logging level (default: INFO) |
| `--api.type` | Enum (completion, chat, anthropic_messages) | API endpoint to benchmark: text completion or chat completion. |
| `--api.streaming` | boolean | Stream responses instead of waiting for the full response. Enables TTFT and TPOT metrics. |
| `--api.headers` | JSON | Additional HTTP headers to send with every request. |
| `--api.slo_unit` | str | Time unit for SLO header values: 's', 'ms' or 'us'. Defaults to 'ms'. |
| `--api.slo_tpot_header` | str | Request header carrying the per-request TPOT SLO threshold. Defaults to 'x-slo-tpot-<slo_unit>'. |
| `--api.slo_ttft_header` | str | Request header carrying the per-request TTFT SLO threshold. Defaults to 'x-slo-ttft-<slo_unit>'. |
| `--api.response_format.type` | Enum (json_schema, json_object) | Structured output mode: a full JSON schema or any JSON object. |
| `--api.response_format.name` | str | Name given to the JSON schema in the request payload. |
| `--api.response_format.json_schema` | JSON | JSON schema the model output must conform to when type is 'json_schema'. |
| `--api.session_id_header_key` | str | Header used to send the session ID with each request in multi-turn benchmarks. |
| `--api.session_token_header_key` | str | Response header carrying a server-assigned session token, replayed as a request header on later requests of the same session to keep router session affinity. |
| `--data.type` | Enum (mock, shareGPT, synthetic, random, shared_prefix, cnn_dailymail, infinity_instruct, billsum_conversations, otel_trace_replay, weka_trace_replay, conversation_replay, visionarena, synthetic_agentic) | Dataset or generator used to produce prompts. |
| `--data.path` | str | Path to the downloaded ShareGPT dataset. Only used by the 'shareGPT' type. |
| `--data.corpus_file_path` | str | Path to a text file to use as the prompt tokenization corpus instead of the default hardcoded sonnet |
| `--data.input_distribution.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.input_distribution.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.input_distribution.mean` | float | Mean of the distribution. |
| `--data.input_distribution.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.input_distribution.total_count` | int | Total number of values to sample from the distribution. |
| `--data.input_distribution.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.input_distribution.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.input_distribution.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.output_distribution.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.output_distribution.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.output_distribution.mean` | float | Mean of the distribution. |
| `--data.output_distribution.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.output_distribution.total_count` | int | Total number of values to sample from the distribution. |
| `--data.output_distribution.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.output_distribution.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.output_distribution.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.shared_prefix.num_groups` | int | Number of unique system prompts (shared prefix groups) to generate. |
| `--data.shared_prefix.num_prompts_per_group` | int | Number of prompts generated per shared system prompt. |
| `--data.shared_prefix.system_prompt_len` | string | Length of the shared system prompt in tokens: a fixed value or a distribution. |
| `--data.shared_prefix.question_len` | string | Length of the question part in tokens: a fixed value or a distribution. |
| `--data.shared_prefix.output_len` | string | Requested output length in tokens: a fixed value or a distribution. |
| `--data.shared_prefix.seed` | int | Random seed for reproducible prompt generation. |
| `--data.shared_prefix.question_distribution.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.shared_prefix.question_distribution.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.shared_prefix.question_distribution.mean` | float | Mean of the distribution. |
| `--data.shared_prefix.question_distribution.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.shared_prefix.question_distribution.total_count` | int | Total number of values to sample from the distribution. |
| `--data.shared_prefix.question_distribution.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.shared_prefix.question_distribution.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.shared_prefix.question_distribution.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.shared_prefix.output_distribution.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.shared_prefix.output_distribution.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.shared_prefix.output_distribution.mean` | float | Mean of the distribution. |
| `--data.shared_prefix.output_distribution.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.shared_prefix.output_distribution.total_count` | int | Total number of values to sample from the distribution. |
| `--data.shared_prefix.output_distribution.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.shared_prefix.output_distribution.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.shared_prefix.output_distribution.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.shared_prefix.enable_multi_turn_chat` | boolean | Send each group's prompts as consecutive turns of one chat conversation. |
| `--data.shared_prefix.multimodal.image.count.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.shared_prefix.multimodal.image.count.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.shared_prefix.multimodal.image.count.mean` | float | Mean of the distribution. |
| `--data.shared_prefix.multimodal.image.count.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.shared_prefix.multimodal.image.count.total_count` | int | Total number of values to sample from the distribution. |
| `--data.shared_prefix.multimodal.image.count.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.shared_prefix.multimodal.image.count.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.shared_prefix.multimodal.image.count.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.shared_prefix.multimodal.image.insertion_point` | string | Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from. |
| `--data.shared_prefix.multimodal.image.resolutions` | JSON | Resolution or list of weighted resolutions for generated images. |
| `--data.shared_prefix.multimodal.image.representation` | Enum (png, jpeg, webp) | Wire encoding for emitted image bytes: ``png`` (default, lossless) or ``jpeg`` (lossy, smaller payload). Some VLMs prefer one or the other; consult the model's spec sheet. |
| `--data.shared_prefix.multimodal.video.count.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.shared_prefix.multimodal.video.count.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.shared_prefix.multimodal.video.count.mean` | float | Mean of the distribution. |
| `--data.shared_prefix.multimodal.video.count.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.shared_prefix.multimodal.video.count.total_count` | int | Total number of values to sample from the distribution. |
| `--data.shared_prefix.multimodal.video.count.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.shared_prefix.multimodal.video.count.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.shared_prefix.multimodal.video.count.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.shared_prefix.multimodal.video.insertion_point` | string | Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from. |
| `--data.shared_prefix.multimodal.video.profiles` | JSON | Video profile or list of weighted video profiles for generated videos. |
| `--data.shared_prefix.multimodal.video.representation` | Enum (mp4, png_frames, jpeg_frames) | Wire-format strategy. ``mp4`` sends one ``video_url`` block carrying an MP4 blob (measures full pipeline including server-side decode). ``png_frames`` and ``jpeg_frames`` send ``frames`` × ``image_url`` blocks at one insertion point in the named encoding (no decode dependency, useful for prefix-cache benchmarks and servers that don't accept ``video_url``). |
| `--data.shared_prefix.multimodal.audio.count.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.shared_prefix.multimodal.audio.count.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.shared_prefix.multimodal.audio.count.mean` | float | Mean of the distribution. |
| `--data.shared_prefix.multimodal.audio.count.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.shared_prefix.multimodal.audio.count.total_count` | int | Total number of values to sample from the distribution. |
| `--data.shared_prefix.multimodal.audio.count.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.shared_prefix.multimodal.audio.count.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.shared_prefix.multimodal.audio.count.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.shared_prefix.multimodal.audio.insertion_point` | string | Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from. |
| `--data.shared_prefix.multimodal.audio.durations` | JSON | Duration or list of weighted durations for generated audio clips. |
| `--data.multimodal.image.count.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.multimodal.image.count.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.multimodal.image.count.mean` | float | Mean of the distribution. |
| `--data.multimodal.image.count.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.multimodal.image.count.total_count` | int | Total number of values to sample from the distribution. |
| `--data.multimodal.image.count.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.multimodal.image.count.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.multimodal.image.count.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.multimodal.image.insertion_point` | string | Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from. |
| `--data.multimodal.image.resolutions` | JSON | Resolution or list of weighted resolutions for generated images. |
| `--data.multimodal.image.representation` | Enum (png, jpeg, webp) | Wire encoding for emitted image bytes: ``png`` (default, lossless) or ``jpeg`` (lossy, smaller payload). Some VLMs prefer one or the other; consult the model's spec sheet. |
| `--data.multimodal.video.count.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.multimodal.video.count.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.multimodal.video.count.mean` | float | Mean of the distribution. |
| `--data.multimodal.video.count.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.multimodal.video.count.total_count` | int | Total number of values to sample from the distribution. |
| `--data.multimodal.video.count.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.multimodal.video.count.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.multimodal.video.count.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.multimodal.video.insertion_point` | string | Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from. |
| `--data.multimodal.video.profiles` | JSON | Video profile or list of weighted video profiles for generated videos. |
| `--data.multimodal.video.representation` | Enum (mp4, png_frames, jpeg_frames) | Wire-format strategy. ``mp4`` sends one ``video_url`` block carrying an MP4 blob (measures full pipeline including server-side decode). ``png_frames`` and ``jpeg_frames`` send ``frames`` × ``image_url`` blocks at one insertion point in the named encoding (no decode dependency, useful for prefix-cache benchmarks and servers that don't accept ``video_url``). |
| `--data.multimodal.audio.count.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.multimodal.audio.count.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.multimodal.audio.count.mean` | float | Mean of the distribution. |
| `--data.multimodal.audio.count.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.multimodal.audio.count.total_count` | int | Total number of values to sample from the distribution. |
| `--data.multimodal.audio.count.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.multimodal.audio.count.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.multimodal.audio.count.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.multimodal.audio.insertion_point` | string | Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from. |
| `--data.multimodal.audio.durations` | JSON | Duration or list of weighted durations for generated audio clips. |
| `--data.trace.file` | str | Path to the trace file to replay. |
| `--data.trace.format` | Enum (AzurePublicDataset) | Format of the trace file. |
| `--data.otel_trace_replay.use_static_model` | boolean | Use a single static model for all requests |
| `--data.otel_trace_replay.static_model_name` | str | Static model name (required if use_static_model=True) |
| `--data.otel_trace_replay.model_mapping` | JSON | Map recorded model names to target models |
| `--data.otel_trace_replay.default_max_tokens` | int | Default max_tokens if not specified in trace |
| `--data.otel_trace_replay.override_tool_call_max_tokens` | boolean | Override tool call max_tokens to 4096 instead of using trace recorded length |
| `--data.otel_trace_replay.inject_random_session_id` | boolean | Inject random string into unique segments to invalidate KV-cache between sessions |
| `--data.otel_trace_replay.duplicate_sessions_target` | int | Target number of sessions to reach by duplicating existing sessions. If None, no duplication occurs. |
| `--data.otel_trace_replay.max_wait_ms` | int | Maximum inter-event wait time in milliseconds. Caps the delay between predecessor completion and event dispatch to avoid reproducing unusually long tool/agent execution times from the original trace. |
| `--data.otel_trace_replay.predecessor_wait_timeout_sec` | float | Seconds to wait for predecessor events to complete before failing. 0 waits indefinitely; use with care because a genuinely stuck predecessor will then never time out and successors will wait forever. |
| `--data.otel_trace_replay.include_errors` | boolean | Include spans with error status |
| `--data.otel_trace_replay.skip_invalid_files` | boolean | Skip invalid trace files instead of failing |
| `--data.otel_trace_replay.bad_tool_call_handling` | Enum (none, use_recorded) | How to handle tool_calls whose function.arguments is not valid JSON. none (default): no mitigation, bytes propagate and vLLM may return HTTP 400 on the next turn. use_recorded: discard the live response and substitute the recorded assistant message at the affected slot; the recorded tool_call_id flows into the recorded role:tool successor unchanged. |
| `--data.otel_trace_replay.trace_directory` | str | Directory containing OTel JSON trace files |
| `--data.otel_trace_replay.trace_files` | JSON | List of paths to specific OTel JSON trace files |
| `--data.otel_trace_replay.hf_dataset_path` | JSON | HuggingFace dataset path. Can be:
  - String: 'username/dataset-name'
  - Dict: {'path': 'username/dataset-name', 'revision': 'main', 'split': 'train'}
Any extra keys in the dict are passed as kwargs to datasets.load_dataset(). |
| `--data.otel_trace_replay.filter` | str | Lambda expression to filter trace records. Applied uniformly to all data sources.
Example: "lambda x: x['benchmark'] == 'gsm8k'" or "lambda x: 'spans' in x and len(x['spans']) > 5"
Security: Filter expressions use eval() and should only contain trusted input. |
| `--data.otel_trace_replay.disable_output_substitution` | boolean | When True, replay each call with its recorded assistant output (text and tool calls) instead of substituting the live output from predecessor calls. Dependency timing (waiting for predecessors) is still enforced. Default False preserves faithful live-output replay. |
| `--data.otel_trace_replay.attribute_to_header_map` | JSON | Map OTel span attributes to HTTP headers |
| `--data.otel_trace_replay.attribute_to_label_map` | JSON | Map OTel span attributes to metrics reporting labels |
| `--data.weka_trace_replay.use_static_model` | boolean | Use a single static model for all requests |
| `--data.weka_trace_replay.static_model_name` | str | Static model name (required if use_static_model=True) |
| `--data.weka_trace_replay.model_mapping` | JSON | Map recorded model names to target models |
| `--data.weka_trace_replay.default_max_tokens` | int | Default max_tokens if not specified in trace |
| `--data.weka_trace_replay.override_tool_call_max_tokens` | boolean | Override tool call max_tokens to 4096 instead of using trace recorded length |
| `--data.weka_trace_replay.inject_random_session_id` | boolean | Inject random string into unique segments to invalidate KV-cache between sessions |
| `--data.weka_trace_replay.duplicate_sessions_target` | int | Target number of sessions to reach by duplicating existing sessions. If None, no duplication occurs. |
| `--data.weka_trace_replay.max_wait_ms` | int | Maximum inter-event wait time in milliseconds. Caps the delay between predecessor completion and event dispatch to avoid reproducing unusually long tool/agent execution times from the original trace. |
| `--data.weka_trace_replay.predecessor_wait_timeout_sec` | float | Seconds to wait for predecessor events to complete before failing. 0 waits indefinitely; use with care because a genuinely stuck predecessor will then never time out and successors will wait forever. |
| `--data.weka_trace_replay.include_errors` | boolean | Include spans with error status |
| `--data.weka_trace_replay.skip_invalid_files` | boolean | Skip invalid trace files instead of failing |
| `--data.weka_trace_replay.bad_tool_call_handling` | Enum (none, use_recorded) | How to handle tool_calls whose function.arguments is not valid JSON. none (default): no mitigation, bytes propagate and vLLM may return HTTP 400 on the next turn. use_recorded: discard the live response and substitute the recorded assistant message at the affected slot; the recorded tool_call_id flows into the recorded role:tool successor unchanged. |
| `--data.weka_trace_replay.trace_directory` | str | Directory containing Weka JSON trace files |
| `--data.weka_trace_replay.trace_files` | JSON | List of paths to specific Weka JSON trace files |
| `--data.weka_trace_replay.hf_dataset_path` | JSON | HuggingFace dataset path. Can be:
  - String: 'username/dataset-name'
  - Dict: {'path': 'username/dataset-name', 'revision': 'main', 'split': 'train'}
Any extra keys in the dict are passed as kwargs to datasets.load_dataset(). |
| `--data.weka_trace_replay.trace_idle_gap_cap_seconds` | float | Cap idle timing gaps between turns in seconds |
| `--data.weka_trace_replay.ignore_trace_delays` | boolean | Ignore delays/delays from original trace and run back-to-back |
| `--data.weka_trace_replay.use_think_time_only` | boolean | Only use think_time attribute instead of timestamps |
| `--data.weka_trace_replay.default_block_size` | int | Default block size if not specified in trace |
| `--data.weka_trace_replay.num_dataset_entries` | int | Max number of dataset traces to load from HuggingFace |
| `--data.conversation_replay.seed` | int | Random seed for deterministic generation |
| `--data.conversation_replay.num_conversations` | int | Number of conversation blueprints to generate |
| `--data.conversation_replay.shared_system_prompt_len` | int | Fixed shared system prompt length in tokens |
| `--data.conversation_replay.dynamic_system_prompt_len.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.conversation_replay.dynamic_system_prompt_len.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.conversation_replay.dynamic_system_prompt_len.mean` | float | Mean of the distribution. |
| `--data.conversation_replay.dynamic_system_prompt_len.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.conversation_replay.dynamic_system_prompt_len.total_count` | int | Total number of values to sample from the distribution. |
| `--data.conversation_replay.dynamic_system_prompt_len.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.conversation_replay.dynamic_system_prompt_len.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.conversation_replay.dynamic_system_prompt_len.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.conversation_replay.turns_per_conversation.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.conversation_replay.turns_per_conversation.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.conversation_replay.turns_per_conversation.mean` | float | Mean of the distribution. |
| `--data.conversation_replay.turns_per_conversation.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.conversation_replay.turns_per_conversation.total_count` | int | Total number of values to sample from the distribution. |
| `--data.conversation_replay.turns_per_conversation.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.conversation_replay.turns_per_conversation.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.conversation_replay.turns_per_conversation.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.conversation_replay.input_tokens_per_turn.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.conversation_replay.input_tokens_per_turn.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.conversation_replay.input_tokens_per_turn.mean` | float | Mean of the distribution. |
| `--data.conversation_replay.input_tokens_per_turn.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.conversation_replay.input_tokens_per_turn.total_count` | int | Total number of values to sample from the distribution. |
| `--data.conversation_replay.input_tokens_per_turn.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.conversation_replay.input_tokens_per_turn.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.conversation_replay.input_tokens_per_turn.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.conversation_replay.output_tokens_per_turn.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.conversation_replay.output_tokens_per_turn.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.conversation_replay.output_tokens_per_turn.mean` | float | Mean of the distribution. |
| `--data.conversation_replay.output_tokens_per_turn.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.conversation_replay.output_tokens_per_turn.total_count` | int | Total number of values to sample from the distribution. |
| `--data.conversation_replay.output_tokens_per_turn.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.conversation_replay.output_tokens_per_turn.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.conversation_replay.output_tokens_per_turn.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.conversation_replay.tool_call_latency_sec.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.conversation_replay.tool_call_latency_sec.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.conversation_replay.tool_call_latency_sec.mean` | float | Mean of the distribution. |
| `--data.conversation_replay.tool_call_latency_sec.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.conversation_replay.tool_call_latency_sec.total_count` | int | Total number of values to sample from the distribution. |
| `--data.conversation_replay.tool_call_latency_sec.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.conversation_replay.tool_call_latency_sec.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.conversation_replay.tool_call_latency_sec.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.conversation_replay.max_model_len` | int | Maximum model context length in tokens |
| `--data.visionarena.hf_dataset_name` | str | HuggingFace dataset identifier; override only when mirroring the dataset elsewhere. |
| `--data.visionarena.hf_split` | str | HuggingFace split to stream. |
| `--data.visionarena.hf_data_files` | str | Optional ``data_files`` glob forwarded to ``load_dataset``. |
| `--data.visionarena.num_rows` | int | Number of usable rows to stream into the in-memory request pool at startup. Caps memory use; the benchmark cycles through this pool. |
| `--data.visionarena.max_images_per_request` | int | Cap on images attached per request; truncates a row's image list. |
| `--data.visionarena.insertion_point` | string | Placement of the image block(s) within the prompt text. Float in [0.0, 1.0] (0=start, 1=end), or a Distribution to sample per request. |
| `--data.synthetic_agentic.use_static_model` | boolean | Use a single static model for all requests |
| `--data.synthetic_agentic.static_model_name` | str | Static model name (required if use_static_model=True) |
| `--data.synthetic_agentic.model_mapping` | JSON | Map recorded model names to target models |
| `--data.synthetic_agentic.default_max_tokens` | int | Default max_tokens if not specified in trace |
| `--data.synthetic_agentic.override_tool_call_max_tokens` | boolean | Override tool-call max_tokens to 4096 instead of using the generated call's own length. Defaults False here (unlike trace replay) because the generator sizes each tool call itself, so the generated length is already correct for this model. |
| `--data.synthetic_agentic.inject_random_session_id` | boolean | Not applicable to synthetic generation (pinned False): sessions are already generated with distinct content per session index, so there is no recorded session ID to randomize. |
| `--data.synthetic_agentic.duplicate_sessions_target` | int | Not applicable to synthetic generation (pinned None): raise num_sessions to generate more sessions instead of duplicating existing ones. |
| `--data.synthetic_agentic.max_wait_ms` | int | Maximum inter-event wait time in milliseconds. Caps the delay between predecessor completion and event dispatch to avoid reproducing unusually long tool/agent execution times from the original trace. |
| `--data.synthetic_agentic.predecessor_wait_timeout_sec` | float | Seconds to wait for predecessor events to complete before failing. 0 waits indefinitely; use with care because a genuinely stuck predecessor will then never time out and successors will wait forever. |
| `--data.synthetic_agentic.include_errors` | boolean | Include spans with error status |
| `--data.synthetic_agentic.skip_invalid_files` | boolean | Skip invalid trace files instead of failing |
| `--data.synthetic_agentic.bad_tool_call_handling` | Enum (none, use_recorded) | How to handle tool_calls whose function.arguments is not valid JSON. none (default): no mitigation, bytes propagate and vLLM may return HTTP 400 on the next turn. use_recorded: discard the live response and substitute the recorded assistant message at the affected slot; the recorded tool_call_id flows into the recorded role:tool successor unchanged. |
| `--data.synthetic_agentic.num_sessions` | int | Number of sessions (load volume) |
| `--data.synthetic_agentic.input_tokens_per_turn.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.synthetic_agentic.input_tokens_per_turn.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.synthetic_agentic.input_tokens_per_turn.mean` | float | Mean of the distribution. |
| `--data.synthetic_agentic.input_tokens_per_turn.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.synthetic_agentic.input_tokens_per_turn.total_count` | int | Total number of values to sample from the distribution. |
| `--data.synthetic_agentic.input_tokens_per_turn.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.synthetic_agentic.input_tokens_per_turn.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.synthetic_agentic.input_tokens_per_turn.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.synthetic_agentic.output_tokens_per_turn.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.synthetic_agentic.output_tokens_per_turn.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.synthetic_agentic.output_tokens_per_turn.mean` | float | Mean of the distribution. |
| `--data.synthetic_agentic.output_tokens_per_turn.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.synthetic_agentic.output_tokens_per_turn.total_count` | int | Total number of values to sample from the distribution. |
| `--data.synthetic_agentic.output_tokens_per_turn.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.synthetic_agentic.output_tokens_per_turn.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.synthetic_agentic.output_tokens_per_turn.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.synthetic_agentic.turns_per_session.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.synthetic_agentic.turns_per_session.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.synthetic_agentic.turns_per_session.mean` | float | Mean of the distribution. |
| `--data.synthetic_agentic.turns_per_session.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.synthetic_agentic.turns_per_session.total_count` | int | Total number of values to sample from the distribution. |
| `--data.synthetic_agentic.turns_per_session.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.synthetic_agentic.turns_per_session.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.synthetic_agentic.turns_per_session.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.synthetic_agentic.fanout_probability` | float | Probability an agent spawns sub-agents (instead of just answering), rolled fresh for each of the root's turns and once for each sub-agent. Default 0 = single-agent; 1 = always spawn (full tree to max_depth). |
| `--data.synthetic_agentic.theme_mix` | JSON | theme name -> weight. Preferred form is an explicit block, `{name: {weight: W}}`; a bare float `{name: W}` is also accepted for brevity. Default is an equal mix of the four built-in themes. Use theme_weights() to read normalized {name: float}. |
| `--data.synthetic_agentic.seed` | int | Base seed for stable per-session RNG |
| `--data.synthetic_agentic.shared_system_prompt_len` | int | Tokens of a fixed system-prompt head that opens EVERY agent call (the standing 'system head' real agents carry: tool instructions, policies). Defaults to 1000 because virtually every agentic flow ships a non-trivial system prompt; set 0 only for a deliberately head-less baseline. |
| `--data.synthetic_agentic.tool_loop_depth.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.synthetic_agentic.tool_loop_depth.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.synthetic_agentic.tool_loop_depth.mean` | float | Mean of the distribution. |
| `--data.synthetic_agentic.tool_loop_depth.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.synthetic_agentic.tool_loop_depth.total_count` | int | Total number of values to sample from the distribution. |
| `--data.synthetic_agentic.tool_loop_depth.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.synthetic_agentic.tool_loop_depth.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.synthetic_agentic.tool_loop_depth.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.synthetic_agentic.sub_agents_per_spawn.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.synthetic_agentic.sub_agents_per_spawn.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.synthetic_agentic.sub_agents_per_spawn.mean` | float | Mean of the distribution. |
| `--data.synthetic_agentic.sub_agents_per_spawn.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.synthetic_agentic.sub_agents_per_spawn.total_count` | int | Total number of values to sample from the distribution. |
| `--data.synthetic_agentic.sub_agents_per_spawn.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.synthetic_agentic.sub_agents_per_spawn.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.synthetic_agentic.sub_agents_per_spawn.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.synthetic_agentic.max_depth` | int | Hard recursion terminator |
| `--data.synthetic_agentic.max_events_per_session` | int | Self-limiting event budget |
| `--data.synthetic_agentic.tool_catalog_size_per_agent.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.synthetic_agentic.tool_catalog_size_per_agent.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.synthetic_agentic.tool_catalog_size_per_agent.mean` | float | Mean of the distribution. |
| `--data.synthetic_agentic.tool_catalog_size_per_agent.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.synthetic_agentic.tool_catalog_size_per_agent.total_count` | int | Total number of values to sample from the distribution. |
| `--data.synthetic_agentic.tool_catalog_size_per_agent.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.synthetic_agentic.tool_catalog_size_per_agent.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.synthetic_agentic.tool_catalog_size_per_agent.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.synthetic_agentic.parallel_tool_calls_per_step.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.synthetic_agentic.parallel_tool_calls_per_step.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.synthetic_agentic.parallel_tool_calls_per_step.mean` | float | Mean of the distribution. |
| `--data.synthetic_agentic.parallel_tool_calls_per_step.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.synthetic_agentic.parallel_tool_calls_per_step.total_count` | int | Total number of values to sample from the distribution. |
| `--data.synthetic_agentic.parallel_tool_calls_per_step.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.synthetic_agentic.parallel_tool_calls_per_step.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.synthetic_agentic.parallel_tool_calls_per_step.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.synthetic_agentic.tool_call_latency_sec.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.synthetic_agentic.tool_call_latency_sec.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.synthetic_agentic.tool_call_latency_sec.mean` | float | Mean of the distribution. |
| `--data.synthetic_agentic.tool_call_latency_sec.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.synthetic_agentic.tool_call_latency_sec.total_count` | int | Total number of values to sample from the distribution. |
| `--data.synthetic_agentic.tool_call_latency_sec.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.synthetic_agentic.tool_call_latency_sec.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.synthetic_agentic.tool_call_latency_sec.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.synthetic_agentic.user_think_time_sec.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.synthetic_agentic.user_think_time_sec.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.synthetic_agentic.user_think_time_sec.mean` | float | Mean of the distribution. |
| `--data.synthetic_agentic.user_think_time_sec.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.synthetic_agentic.user_think_time_sec.total_count` | int | Total number of values to sample from the distribution. |
| `--data.synthetic_agentic.user_think_time_sec.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.synthetic_agentic.user_think_time_sec.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.synthetic_agentic.user_think_time_sec.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.synthetic_agentic.max_model_len` | int | Fail-fast context-length ceiling (tokens). When set, a config whose single largest request -- worst-case inputs (system head + tool catalog + accumulated turns + tool loop) plus the output to generate -- would exceed this is rejected at load, instead of 400-ing mid-run. Uses each distribution's clip ceiling (`max`) as the worst case. Excludes the model's per-message chat-template wrapper (~10-15 tok/msg), so set this at or a little below your model's true window. Omit to skip the check. |
| `--data.synthetic_agentic.context_compaction.trigger_tokens.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.synthetic_agentic.context_compaction.trigger_tokens.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.synthetic_agentic.context_compaction.trigger_tokens.mean` | float | Mean of the distribution. |
| `--data.synthetic_agentic.context_compaction.trigger_tokens.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.synthetic_agentic.context_compaction.trigger_tokens.total_count` | int | Total number of values to sample from the distribution. |
| `--data.synthetic_agentic.context_compaction.trigger_tokens.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.synthetic_agentic.context_compaction.trigger_tokens.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.synthetic_agentic.context_compaction.trigger_tokens.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.synthetic_agentic.context_compaction.target_tokens.min` | int | Smallest value the distribution can produce; samples below are clamped. |
| `--data.synthetic_agentic.context_compaction.target_tokens.max` | int | Largest value the distribution can produce; samples above are clamped. |
| `--data.synthetic_agentic.context_compaction.target_tokens.mean` | float | Mean of the distribution. |
| `--data.synthetic_agentic.context_compaction.target_tokens.std_dev` | float | Standard deviation of the distribution. Exclusive with 'variance'. |
| `--data.synthetic_agentic.context_compaction.target_tokens.total_count` | int | Total number of values to sample from the distribution. |
| `--data.synthetic_agentic.context_compaction.target_tokens.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Shape of the distribution to sample values from. |
| `--data.synthetic_agentic.context_compaction.target_tokens.variance` | float | Variance of the distribution. Exclusive with 'std_dev'. |
| `--data.synthetic_agentic.context_compaction.target_tokens.skew` | float | Skewness of the distribution. Only used when type is 'skew_normal'. |
| `--data.use_chat_template` | boolean | Wrap each generated prompt in the tokenizer's chat template as a single user turn before sending it on the completions path, reproducing the request shape of harnesses that benchmark with chat templating enabled. The input length distribution targets the fully templated prompt, so the server-side prefill token count still matches the configured length. Only supported by the 'random' type; setting it with any other type is a config error. |
| `--load.type` | Enum (constant, poisson, trace_replay, concurrent, trace_session_replay) | Load pattern used to schedule requests. |
| `--load.interval` | float | Seconds to wait between stages. |
| `--load.stages` | JSON | Load stages to run in sequence. The stage fields depend on the load type. |
| `--load.sweep.type` | Enum (geometric, linear) | How stage rates are spaced up to the saturation rate: 'geometric' or 'linear'. |
| `--load.sweep.num_requests` | int | Number of requests sent in the initial burst used to find the saturation rate. |
| `--load.sweep.timeout` | float | Time limit in seconds for the saturation probe stage. |
| `--load.sweep.num_stages` | int | Number of load stages to generate. |
| `--load.sweep.stage_duration` | int | Duration of each generated stage in seconds. |
| `--load.sweep.saturation_percentile` | float | Percentile of observed request rates taken as the saturation point. |
| `--load.num_workers` | int | Number of worker processes sending requests. Defaults to the CPU count. |
| `--load.worker_max_concurrency` | int | Maximum concurrent in-flight requests per worker. |
| `--load.worker_max_tcp_connections` | int | Maximum TCP connections per worker. |
| `--load.trace.file` | str | Path to the trace file to replay. |
| `--load.trace.format` | Enum (AzurePublicDataset) | Format of the trace file. |
| `--load.circuit_breakers` | JSON | Names of configured circuit breakers to enable for the run. |
| `--load.request_timeout` | float | Per-request timeout in seconds. |
| `--load.lora_traffic_split` | JSON | Traffic split across LoRA adapters. Splits must sum to 1.0. |
| `--load.base_seed` | int | Base random seed for load generation. Defaults to the current time. |
| `--metrics.type` | Enum (prometheus, default) | Metrics client used to collect server-side metrics. |
| `--metrics.prometheus.scrape_interval` | int | Scrape interval of the Prometheus server in seconds. |
| `--metrics.prometheus.url` | string | URL of the Prometheus server to query. |
| `--metrics.prometheus.filters` | JSON | PromQL label matchers (e.g. 'namespace="default"') applied to every metric query. |
| `--metrics.prometheus.google_managed` | boolean | Query Google Cloud Managed Service for Prometheus instead of a self-hosted server. |
| `--report.request_lifecycle.summary` | boolean | Generate a summary report across the whole run. |
| `--report.request_lifecycle.per_stage` | boolean | Generate a report for each load stage. |
| `--report.request_lifecycle.per_request` | boolean | Generate a report with per-request details. |
| `--report.request_lifecycle.per_adapter` | boolean | Generate a report for each LoRA adapter. |
| `--report.request_lifecycle.per_adapter_stage` | boolean | Generate a report for each LoRA adapter within each load stage. |
| `--report.request_lifecycle.percentiles` | JSON | Percentiles reported for each metric. |
| `--report.request_lifecycle.use_server_output_tokens` | boolean | Use the server-reported output token counts in metrics instead of tokenizing the response text. |
| `--report.request_lifecycle.max_error_messages` | int | Cap on the number of distinct example error messages retained per error label in the failure report, and per substitution entry. |
| `--report.prometheus.summary` | boolean | Generate a summary report across the whole run. |
| `--report.prometheus.per_stage` | boolean | Generate a report for each load stage. |
| `--report.session_lifecycle.summary` | boolean | Generate a summary report across the whole run. |
| `--report.session_lifecycle.per_stage` | boolean | Generate a report for each load stage. |
| `--report.session_lifecycle.per_session` | boolean | Generate a report with per-session details. |
| `--report.goodput.constraints` | JSON | SLO thresholds in seconds that a request must meet to count as good. Keys: 'ttft', 'tpot', 'itl', 'ntpot', 'request_latency'. |
| `--storage.local_storage.path` | str | Directory or object key prefix where report files are written. '{timestamp}' is replaced with the run's start time. |
| `--storage.local_storage.report_file_prefix` | str | Prefix added to every report file name. |
| `--storage.google_cloud_storage.path` | str | Directory or object key prefix where report files are written. '{timestamp}' is replaced with the run's start time. |
| `--storage.google_cloud_storage.report_file_prefix` | str | Prefix added to every report file name. |
| `--storage.google_cloud_storage.bucket_name` | str | Name of the Google Cloud Storage bucket where reports are uploaded. |
| `--storage.simple_storage_service.path` | str | Directory or object key prefix where report files are written. '{timestamp}' is replaced with the run's start time. |
| `--storage.simple_storage_service.report_file_prefix` | str | Prefix added to every report file name. |
| `--storage.simple_storage_service.bucket_name` | str | Name of the S3 bucket where reports are uploaded. |
| `--storage.simple_storage_service.endpoint_url` | str | Custom endpoint URL, for S3-compatible object stores. |
| `--storage.simple_storage_service.region_name` | str | AWS region of the bucket. |
| `--storage.simple_storage_service.addressing_style` | string | S3 addressing style: 'auto', 'virtual' (bucket in hostname) or 'path'. |
| `--server.type` | Enum (vllm, sglang, tgi, mock) | Type of model server being benchmarked. |
| `--server.model_name` | str | Model name sent in each request. Auto-detected from the server if unset. |
| `--server.base_url` | str | Base URL of the model server, e.g. 'http://localhost:8000'. |
| `--server.ignore_eos` | boolean | Ask the server to keep generating past the end-of-sequence token so outputs hit the requested length. |
| `--server.api_key` | str | API key sent as a bearer token with each request. |
| `--server.cert_path` | str | Path to a client TLS certificate file. |
| `--server.key_path` | str | Path to the private key for the client TLS certificate. |
| `--tokenizer.pretrained_model_name_or_path` | str | HuggingFace model name or local path of the tokenizer to load. |
| `--tokenizer.trust_remote_code` | boolean | Allow the tokenizer to execute code from its repository when loading. |
| `--tokenizer.token` | str | HuggingFace access token used to download the tokenizer. |
| `--circuit_breakers` | JSON | Circuit breakers that stop the run when observed metrics cross configured thresholds. |
