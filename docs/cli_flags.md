# Inference-Perf CLI Flags

These command line flags are automatically generated from the internal `Config` schema. You can override any configuration directly from the CLI without using a yaml configuration file.

| Flag | Type | Description |
| --- | --- | --- |
| `--api.type` | Enum (completion, chat) | Matches api.type in config |
| `--api.streaming` | boolean | Matches api.streaming in config |
| `--api.headers` | JSON | Matches api.headers in config |
| `--api.slo_unit` | str | Matches api.slo_unit in config |
| `--api.slo_tpot_header` | str | Matches api.slo_tpot_header in config |
| `--api.slo_ttft_header` | str | Matches api.slo_ttft_header in config |
| `--api.response_format.type` | Enum (json_schema, json_object) | Matches api.response_format.type in config |
| `--api.response_format.name` | str | Matches api.response_format.name in config |
| `--api.response_format.json_schema` | JSON | Matches api.response_format.json_schema in config |
| `--data.type` | Enum (mock, shareGPT, synthetic, random, shared_prefix, cnn_dailymail, infinity_instruct, billsum_conversations, otel_trace_replay, conversation_replay) | Matches data.type in config |
| `--data.path` | str | Matches data.path in config |
| `--data.input_distribution.min` | int | Matches data.input_distribution.min in config |
| `--data.input_distribution.max` | int | Matches data.input_distribution.max in config |
| `--data.input_distribution.mean` | float | Matches data.input_distribution.mean in config |
| `--data.input_distribution.std_dev` | float | Matches data.input_distribution.std_dev in config |
| `--data.input_distribution.total_count` | int | Matches data.input_distribution.total_count in config |
| `--data.input_distribution.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Matches data.input_distribution.type in config |
| `--data.input_distribution.variance` | float | Matches data.input_distribution.variance in config |
| `--data.input_distribution.skew` | float | Matches data.input_distribution.skew in config |
| `--data.output_distribution.min` | int | Matches data.output_distribution.min in config |
| `--data.output_distribution.max` | int | Matches data.output_distribution.max in config |
| `--data.output_distribution.mean` | float | Matches data.output_distribution.mean in config |
| `--data.output_distribution.std_dev` | float | Matches data.output_distribution.std_dev in config |
| `--data.output_distribution.total_count` | int | Matches data.output_distribution.total_count in config |
| `--data.output_distribution.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Matches data.output_distribution.type in config |
| `--data.output_distribution.variance` | float | Matches data.output_distribution.variance in config |
| `--data.output_distribution.skew` | float | Matches data.output_distribution.skew in config |
| `--data.shared_prefix.num_groups` | int | Matches data.shared_prefix.num_groups in config |
| `--data.shared_prefix.num_prompts_per_group` | int | Matches data.shared_prefix.num_prompts_per_group in config |
| `--data.shared_prefix.system_prompt_len` | string | Matches data.shared_prefix.system_prompt_len in config |
| `--data.shared_prefix.question_len` | string | Matches data.shared_prefix.question_len in config |
| `--data.shared_prefix.output_len` | string | Matches data.shared_prefix.output_len in config |
| `--data.shared_prefix.seed` | int | Matches data.shared_prefix.seed in config |
| `--data.shared_prefix.question_distribution.min` | int | Matches data.shared_prefix.question_distribution.min in config |
| `--data.shared_prefix.question_distribution.max` | int | Matches data.shared_prefix.question_distribution.max in config |
| `--data.shared_prefix.question_distribution.mean` | float | Matches data.shared_prefix.question_distribution.mean in config |
| `--data.shared_prefix.question_distribution.std_dev` | float | Matches data.shared_prefix.question_distribution.std_dev in config |
| `--data.shared_prefix.question_distribution.total_count` | int | Matches data.shared_prefix.question_distribution.total_count in config |
| `--data.shared_prefix.question_distribution.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Matches data.shared_prefix.question_distribution.type in config |
| `--data.shared_prefix.question_distribution.variance` | float | Matches data.shared_prefix.question_distribution.variance in config |
| `--data.shared_prefix.question_distribution.skew` | float | Matches data.shared_prefix.question_distribution.skew in config |
| `--data.shared_prefix.output_distribution.min` | int | Matches data.shared_prefix.output_distribution.min in config |
| `--data.shared_prefix.output_distribution.max` | int | Matches data.shared_prefix.output_distribution.max in config |
| `--data.shared_prefix.output_distribution.mean` | float | Matches data.shared_prefix.output_distribution.mean in config |
| `--data.shared_prefix.output_distribution.std_dev` | float | Matches data.shared_prefix.output_distribution.std_dev in config |
| `--data.shared_prefix.output_distribution.total_count` | int | Matches data.shared_prefix.output_distribution.total_count in config |
| `--data.shared_prefix.output_distribution.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Matches data.shared_prefix.output_distribution.type in config |
| `--data.shared_prefix.output_distribution.variance` | float | Matches data.shared_prefix.output_distribution.variance in config |
| `--data.shared_prefix.output_distribution.skew` | float | Matches data.shared_prefix.output_distribution.skew in config |
| `--data.shared_prefix.enable_multi_turn_chat` | boolean | Matches data.shared_prefix.enable_multi_turn_chat in config |
| `--data.trace.file` | str | Matches data.trace.file in config |
| `--data.trace.format` | Enum (AzurePublicDataset) | Matches data.trace.format in config |
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
| `--data.conversation_replay.dynamic_system_prompt_len.min` | int | Matches data.conversation_replay.dynamic_system_prompt_len.min in config |
| `--data.conversation_replay.dynamic_system_prompt_len.max` | int | Matches data.conversation_replay.dynamic_system_prompt_len.max in config |
| `--data.conversation_replay.dynamic_system_prompt_len.mean` | float | Matches data.conversation_replay.dynamic_system_prompt_len.mean in config |
| `--data.conversation_replay.dynamic_system_prompt_len.std_dev` | float | Matches data.conversation_replay.dynamic_system_prompt_len.std_dev in config |
| `--data.conversation_replay.dynamic_system_prompt_len.total_count` | int | Matches data.conversation_replay.dynamic_system_prompt_len.total_count in config |
| `--data.conversation_replay.dynamic_system_prompt_len.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Matches data.conversation_replay.dynamic_system_prompt_len.type in config |
| `--data.conversation_replay.dynamic_system_prompt_len.variance` | float | Matches data.conversation_replay.dynamic_system_prompt_len.variance in config |
| `--data.conversation_replay.dynamic_system_prompt_len.skew` | float | Matches data.conversation_replay.dynamic_system_prompt_len.skew in config |
| `--data.conversation_replay.turns_per_conversation.min` | int | Matches data.conversation_replay.turns_per_conversation.min in config |
| `--data.conversation_replay.turns_per_conversation.max` | int | Matches data.conversation_replay.turns_per_conversation.max in config |
| `--data.conversation_replay.turns_per_conversation.mean` | float | Matches data.conversation_replay.turns_per_conversation.mean in config |
| `--data.conversation_replay.turns_per_conversation.std_dev` | float | Matches data.conversation_replay.turns_per_conversation.std_dev in config |
| `--data.conversation_replay.turns_per_conversation.total_count` | int | Matches data.conversation_replay.turns_per_conversation.total_count in config |
| `--data.conversation_replay.turns_per_conversation.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Matches data.conversation_replay.turns_per_conversation.type in config |
| `--data.conversation_replay.turns_per_conversation.variance` | float | Matches data.conversation_replay.turns_per_conversation.variance in config |
| `--data.conversation_replay.turns_per_conversation.skew` | float | Matches data.conversation_replay.turns_per_conversation.skew in config |
| `--data.conversation_replay.input_tokens_per_turn.min` | int | Matches data.conversation_replay.input_tokens_per_turn.min in config |
| `--data.conversation_replay.input_tokens_per_turn.max` | int | Matches data.conversation_replay.input_tokens_per_turn.max in config |
| `--data.conversation_replay.input_tokens_per_turn.mean` | float | Matches data.conversation_replay.input_tokens_per_turn.mean in config |
| `--data.conversation_replay.input_tokens_per_turn.std_dev` | float | Matches data.conversation_replay.input_tokens_per_turn.std_dev in config |
| `--data.conversation_replay.input_tokens_per_turn.total_count` | int | Matches data.conversation_replay.input_tokens_per_turn.total_count in config |
| `--data.conversation_replay.input_tokens_per_turn.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Matches data.conversation_replay.input_tokens_per_turn.type in config |
| `--data.conversation_replay.input_tokens_per_turn.variance` | float | Matches data.conversation_replay.input_tokens_per_turn.variance in config |
| `--data.conversation_replay.input_tokens_per_turn.skew` | float | Matches data.conversation_replay.input_tokens_per_turn.skew in config |
| `--data.conversation_replay.output_tokens_per_turn.min` | int | Matches data.conversation_replay.output_tokens_per_turn.min in config |
| `--data.conversation_replay.output_tokens_per_turn.max` | int | Matches data.conversation_replay.output_tokens_per_turn.max in config |
| `--data.conversation_replay.output_tokens_per_turn.mean` | float | Matches data.conversation_replay.output_tokens_per_turn.mean in config |
| `--data.conversation_replay.output_tokens_per_turn.std_dev` | float | Matches data.conversation_replay.output_tokens_per_turn.std_dev in config |
| `--data.conversation_replay.output_tokens_per_turn.total_count` | int | Matches data.conversation_replay.output_tokens_per_turn.total_count in config |
| `--data.conversation_replay.output_tokens_per_turn.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Matches data.conversation_replay.output_tokens_per_turn.type in config |
| `--data.conversation_replay.output_tokens_per_turn.variance` | float | Matches data.conversation_replay.output_tokens_per_turn.variance in config |
| `--data.conversation_replay.output_tokens_per_turn.skew` | float | Matches data.conversation_replay.output_tokens_per_turn.skew in config |
| `--data.conversation_replay.tool_call_latency_sec.min` | int | Matches data.conversation_replay.tool_call_latency_sec.min in config |
| `--data.conversation_replay.tool_call_latency_sec.max` | int | Matches data.conversation_replay.tool_call_latency_sec.max in config |
| `--data.conversation_replay.tool_call_latency_sec.mean` | float | Matches data.conversation_replay.tool_call_latency_sec.mean in config |
| `--data.conversation_replay.tool_call_latency_sec.std_dev` | float | Matches data.conversation_replay.tool_call_latency_sec.std_dev in config |
| `--data.conversation_replay.tool_call_latency_sec.total_count` | int | Matches data.conversation_replay.tool_call_latency_sec.total_count in config |
| `--data.conversation_replay.tool_call_latency_sec.type` | Enum (normal, skew_normal, lognormal, uniform, poisson, fixed) | Matches data.conversation_replay.tool_call_latency_sec.type in config |
| `--data.conversation_replay.tool_call_latency_sec.variance` | float | Matches data.conversation_replay.tool_call_latency_sec.variance in config |
| `--data.conversation_replay.tool_call_latency_sec.skew` | float | Matches data.conversation_replay.tool_call_latency_sec.skew in config |
| `--load.type` | Enum (constant, poisson, trace_replay, concurrent, trace_session_replay) | Matches load.type in config |
| `--load.interval` | float | Matches load.interval in config |
| `--load.stages` | JSON | Matches load.stages in config |
| `--load.sweep.type` | Enum (geometric, linear) | Matches load.sweep.type in config |
| `--load.sweep.num_requests` | int | Matches load.sweep.num_requests in config |
| `--load.sweep.timeout` | float | Matches load.sweep.timeout in config |
| `--load.sweep.num_stages` | int | Matches load.sweep.num_stages in config |
| `--load.sweep.stage_duration` | int | Matches load.sweep.stage_duration in config |
| `--load.sweep.saturation_percentile` | float | Matches load.sweep.saturation_percentile in config |
| `--load.num_workers` | int | Matches load.num_workers in config |
| `--load.worker_max_concurrency` | int | Matches load.worker_max_concurrency in config |
| `--load.worker_max_tcp_connections` | int | Matches load.worker_max_tcp_connections in config |
| `--load.trace.file` | str | Matches load.trace.file in config |
| `--load.trace.format` | Enum (AzurePublicDataset) | Matches load.trace.format in config |
| `--load.circuit_breakers` | JSON | Matches load.circuit_breakers in config |
| `--load.request_timeout` | float | Matches load.request_timeout in config |
| `--load.lora_traffic_split` | JSON | Matches load.lora_traffic_split in config |
| `--load.base_seed` | int | Matches load.base_seed in config |
| `--metrics.type` | Enum (prometheus, default) | Matches metrics.type in config |
| `--metrics.prometheus.scrape_interval` | int | Matches metrics.prometheus.scrape_interval in config |
| `--metrics.prometheus.url` | string | Matches metrics.prometheus.url in config |
| `--metrics.prometheus.filters` | JSON | Matches metrics.prometheus.filters in config |
| `--metrics.prometheus.google_managed` | boolean | Matches metrics.prometheus.google_managed in config |
| `--report.request_lifecycle.summary` | boolean | Matches report.request_lifecycle.summary in config |
| `--report.request_lifecycle.per_stage` | boolean | Matches report.request_lifecycle.per_stage in config |
| `--report.request_lifecycle.per_request` | boolean | Matches report.request_lifecycle.per_request in config |
| `--report.request_lifecycle.per_adapter` | boolean | Matches report.request_lifecycle.per_adapter in config |
| `--report.request_lifecycle.per_adapter_stage` | boolean | Matches report.request_lifecycle.per_adapter_stage in config |
| `--report.request_lifecycle.percentiles` | JSON | Matches report.request_lifecycle.percentiles in config |
| `--report.prometheus.summary` | boolean | Matches report.prometheus.summary in config |
| `--report.prometheus.per_stage` | boolean | Matches report.prometheus.per_stage in config |
| `--report.session_lifecycle.summary` | boolean | Matches report.session_lifecycle.summary in config |
| `--report.session_lifecycle.per_stage` | boolean | Matches report.session_lifecycle.per_stage in config |
| `--report.session_lifecycle.per_session` | boolean | Matches report.session_lifecycle.per_session in config |
| `--report.goodput.constraints` | JSON | Matches report.goodput.constraints in config |
| `--storage.local_storage.path` | str | Matches storage.local_storage.path in config |
| `--storage.local_storage.report_file_prefix` | str | Matches storage.local_storage.report_file_prefix in config |
| `--storage.google_cloud_storage.path` | str | Matches storage.google_cloud_storage.path in config |
| `--storage.google_cloud_storage.report_file_prefix` | str | Matches storage.google_cloud_storage.report_file_prefix in config |
| `--storage.google_cloud_storage.bucket_name` | str | Matches storage.google_cloud_storage.bucket_name in config |
| `--storage.simple_storage_service.path` | str | Matches storage.simple_storage_service.path in config |
| `--storage.simple_storage_service.report_file_prefix` | str | Matches storage.simple_storage_service.report_file_prefix in config |
| `--storage.simple_storage_service.bucket_name` | str | Matches storage.simple_storage_service.bucket_name in config |
| `--server.type` | Enum (vllm, sglang, tgi, mock) | Matches server.type in config |
| `--server.model_name` | str | Matches server.model_name in config |
| `--server.base_url` | str | Matches server.base_url in config |
| `--server.ignore_eos` | boolean | Matches server.ignore_eos in config |
| `--server.api_key` | str | Matches server.api_key in config |
| `--server.cert_path` | str | Matches server.cert_path in config |
| `--server.key_path` | str | Matches server.key_path in config |
| `--tokenizer.pretrained_model_name_or_path` | str | Matches tokenizer.pretrained_model_name_or_path in config |
| `--tokenizer.trust_remote_code` | boolean | Matches tokenizer.trust_remote_code in config |
| `--tokenizer.token` | str | Matches tokenizer.token in config |
| `--circuit_breakers` | JSON | Matches circuit_breakers in config |
