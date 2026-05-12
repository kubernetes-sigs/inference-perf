# Configuration Reference

<!--
  This file is auto-generated from the Pydantic models in inference_perf/config.py.
  Do not edit by hand. Run `pdm run update:config-docs` after changing the schema.
-->

This reference enumerates every configuration option, generated directly from the
schema. For tutorial-style examples and prose, see [config.md](config.md).

Each section corresponds to one model. Nested models are linked.


## Index

- [Config](#config)
- [APIConfig](#apiconfig)
- [AudioDatagenConfig](#audiodatagenconfig)
- [CircuitBreakerConfig](#circuitbreakerconfig)
- [ConcurrentLoadStage](#concurrentloadstage)
- [ConversationReplayConfig](#conversationreplayconfig)
- [CustomTokenizerConfig](#customtokenizerconfig)
- [DataConfig](#dataconfig)
- [Distribution](#distribution)
- [GoodputConfig](#goodputconfig)
- [GoogleCloudStorageConfig](#googlecloudstorageconfig)
- [ImageDatagenConfig](#imagedatagenconfig)
- [LoadConfig](#loadconfig)
- [MetricsClientConfig](#metricsclientconfig)
- [ModelServerClientConfig](#modelserverclientconfig)
- [MultiLoRAConfig](#multiloraconfig)
- [OTelTraceReplayConfig](#oteltracereplayconfig)
- [PrometheusClientConfig](#prometheusclientconfig)
- [PrometheusMetricsReportConfig](#prometheusmetricsreportconfig)
- [ReportConfig](#reportconfig)
- [RequestLifecycleMetricsReportConfig](#requestlifecyclemetricsreportconfig)
- [Resolution](#resolution)
- [ResponseFormat](#responseformat)
- [SessionLifecycleReportConfig](#sessionlifecyclereportconfig)
- [SharedPrefix](#sharedprefix)
- [SimpleStorageServiceConfig](#simplestorageserviceconfig)
- [StandardLoadStage](#standardloadstage)
- [StorageConfig](#storageconfig)
- [StorageConfigBase](#storageconfigbase)
- [SweepConfig](#sweepconfig)
- [SyntheticMultimodalDatagenConfig](#syntheticmultimodaldatagenconfig)
- [TraceConfig](#traceconfig)
- [TraceSessionReplayLoadStage](#tracesessionreplayloadstage)
- [VideoDatagenConfig](#videodatagenconfig)
- [VideoProfile](#videoprofile)
- [WeightedDuration](#weightedduration)
- [WeightedResolution](#weightedresolution)
- [WeightedVideoProfile](#weightedvideoprofile)
- [MetricsSpec](#metricsspec)
- [TriggerConsecutive](#triggerconsecutive)
- [TriggerRateOverWindow](#triggerrateoverwindow)


## Config

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `api` | [APIConfig](#apiconfig) | _factory_ | API-level request behavior (type, streaming, SLO headers). |
| `data` | [DataConfig](#dataconfig) | _factory_ | Workload data generation configuration. |
| `load` | [LoadConfig](#loadconfig) | _factory_ | Load shaping and worker configuration. |
| `metrics` | [MetricsClientConfig](#metricsclientconfig) (optional) | `None` | Server-side metrics collection (e.g. Prometheus scrape). |
| `report` | [ReportConfig](#reportconfig) | _factory_ | Reporting outputs to emit at the end of a run. |
| `storage` | [StorageConfig](#storageconfig) (optional) | _factory_ | Destinations for written reports. |
| `server` | [ModelServerClientConfig](#modelserverclientconfig) (optional) | `None` | Model server connection details. |
| `tokenizer` | [CustomTokenizerConfig](#customtokenizerconfig) (optional) | `None` | Tokenizer used to compute token-level metrics offline. |
| `circuit_breakers` | List[[CircuitBreakerConfig](#circuitbreakerconfig)] (optional) | `None` | Declarative circuit breakers that can stop the run on metric thresholds. |

## APIConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | APIType ('completion' \| 'chat') | `'completion'` | API type to exercise on the model server. |
| `streaming` | bool | `False` | Enable streaming so TTFT, ITL, and TPOT can be measured. |
| `headers` | Dict[str, str] (optional) | `None` | Custom HTTP headers attached to every request. |
| `slo_unit` | str (optional) | `None` | Unit for SLO header values (e.g. 'ms', 's'). Defaults to 'ms'. |
| `slo_tpot_header` | str (optional) | `None` | Header name carrying the per-request TPOT SLO. |
| `slo_ttft_header` | str (optional) | `None` | Header name carrying the per-request TTFT SLO. |
| `response_format` | [ResponseFormat](#responseformat) (optional) | `None` | Structured-output schema sent via the response_format API parameter. |

## AudioDatagenConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `count` | [Distribution](#distribution) (optional) | `None` | Distribution of the number of media items to generate per request. |
| `insertion_point` | float \| [Distribution](#distribution) (optional) | `None` | Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from. |
| `durations` | float \| List[[WeightedDuration](#weightedduration)] (optional) | `None` | Duration or list of weighted durations for generated audio clips. |

## CircuitBreakerConfig

Declarative breaker configuration.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | str | **required** | Breaker identifier referenced from `load.circuit_breakers`. |
| `metrics` | [MetricsSpec](#metricsspec) | **required** | Selects which metric stream this breaker observes. |
| `triggers` | List[[TriggerConsecutive](#triggerconsecutive) \| [TriggerRateOverWindow](#triggerrateoverwindow)] | **required** | Trigger rules; any matching trigger trips the breaker. |

## ConcurrentLoadStage

Load stage for CONCURRENT load type.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `num_requests` | int | **required** | Number of requests to send |
| `concurrency_level` | int | **required** | Concurrency level |
| `rate` | float (optional) | `None` | Set at runtime for load generation |
| `duration` | int (optional) | `None` | Set at runtime for load generation |

## ConversationReplayConfig

Configuration for conversation replay data generator.

Generates synthetic multi-turn conversations in-memory from configurable
distributions. Each conversation has a two-part system prompt (shared prefix
+ dynamic per-conversation suffix) and a sequence of user/assistant turns
with independently sampled input/output token lengths.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `seed` | int | `42` | Random seed for deterministic generation |
| `num_conversations` | int | `200` | Number of conversation blueprints to generate |
| `shared_system_prompt_len` | int | `8359` | Fixed shared system prompt length in tokens |
| `dynamic_system_prompt_len` | [Distribution](#distribution) (optional) | `None` | Per-conversation dynamic system prompt length distribution |
| `turns_per_conversation` | [Distribution](#distribution) (optional) | `None` | Number of turns per conversation distribution |
| `input_tokens_per_turn` | [Distribution](#distribution) (optional) | `None` | Input tokens per turn distribution |
| `output_tokens_per_turn` | [Distribution](#distribution) (optional) | `None` | Output tokens per turn distribution |
| `tool_call_latency_sec` | [Distribution](#distribution) (optional) | `None` | Per-turn tool execution latency distribution in seconds. When set, each turn sleeps for the sampled duration after model inference completes and before the next turn begins, simulating tool call round-trips. The sleep holds the session lock so the GPU is free to serve other concurrent conversations — correctly modelling offline agentic workloads. Omit for pure GPU throughput measurement. Values are in seconds; min/max are whole seconds, mean/std_dev may be fractional. |

## CustomTokenizerConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `pretrained_model_name_or_path` | str (optional) | `None` | HuggingFace tokenizer name or local path. |
| `trust_remote_code` | bool (optional) | `None` | Forwarded to HuggingFace tokenizer loading; required by some custom tokenizers. |
| `token` | str (optional) | `None` | HuggingFace auth token used to download gated tokenizers. |

## DataConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | DataGenType ('mock' \| 'shareGPT' \| 'synthetic' \| 'random' \| 'shared_prefix' \| 'cnn_dailymail' \| 'infinity_instruct' \| 'billsum_conversations' \| 'otel_trace_replay' \| 'conversation_replay') | `'mock'` | Which data generator to use; the sibling fields below apply only to specific types. |
| `path` | str (optional) | `None` | Filesystem path to a dataset. Required for shareGPT, cnn_dailymail, billsum_conversations, and infinity_instruct. |
| `input_distribution` | [Distribution](#distribution) (optional) | `None` | Input prompt length distribution. Used by synthetic/random datagens. |
| `output_distribution` | [Distribution](#distribution) (optional) | `None` | Output length distribution. Used by synthetic/random datagens. |
| `shared_prefix` | [SharedPrefix](#sharedprefix) (optional) | `None` | Shared-prefix datagen configuration (when type=shared_prefix). |
| `multimodal` | [SyntheticMultimodalDatagenConfig](#syntheticmultimodaldatagenconfig) (optional) | `None` | Multimodal generation block paired with synthetic/shared_prefix text generation. |
| `trace` | [TraceConfig](#traceconfig) (optional) | `None` | Trace file replayed by the random datagen (when type=random). |
| `otel_trace_replay` | [OTelTraceReplayConfig](#oteltracereplayconfig) (optional) | `None` | OTel trace replay configuration (when type=otel_trace_replay). |
| `conversation_replay` | [ConversationReplayConfig](#conversationreplayconfig) (optional) | `None` | Conversation replay configuration (when type=conversation_replay). |

## Distribution

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `min` | int | `10` | Minimum sampled value (tokens, items, etc.). |
| `max` | int | `1024` | Maximum sampled value. |
| `mean` | float | `512` | Distribution mean. |
| `std_dev` | float | `200` | Standard deviation. Mutually exclusive with `variance`. |
| `total_count` | int (optional) | `None` | Total number of samples to draw, when the distribution is materialized eagerly. |
| `type` | DistributionType ('normal' \| 'skew_normal' \| 'lognormal' \| 'uniform' \| 'poisson' \| 'fixed') | `'normal'` | Distribution family used to sample values. |
| `variance` | float (optional) | `None` | Variance. Mutually exclusive with `std_dev`; converted to std_dev when set. |
| `skew` | float | `0.0` | Skew parameter; only used when type=skew_normal. |

## GoodputConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `constraints` | Dict[str, float] | `{}` | Mapping of SLO metric name to threshold; requests meeting all thresholds count as good. |

## GoogleCloudStorageConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `path` | str | _factory_ | Directory or object-store path where reports are written. Supports `{timestamp}` substitution. |
| `report_file_prefix` | str (optional) | `None` | Optional filename prefix for report files written to this destination. |
| `bucket_name` | str | **required** | Target GCS bucket. |

## ImageDatagenConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `count` | [Distribution](#distribution) (optional) | `None` | Distribution of the number of media items to generate per request. |
| `insertion_point` | float \| [Distribution](#distribution) (optional) | `None` | Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from. |
| `resolutions` | ResolutionPreset ('4k' \| '1080p' \| '720p' \| '360p') \| [Resolution](#resolution) \| List[[WeightedResolution](#weightedresolution)] (optional) | `None` | Resolution or list of weighted resolutions for generated images. |
| `representation` | ImageRepresentation ('png' \| 'jpeg') | `'png'` | Wire encoding for emitted image bytes: ``png`` (default, lossless) or ``jpeg`` (lossy, smaller payload). Some VLMs prefer one or the other; consult the model's spec sheet. |

## LoadConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | LoadType ('constant' \| 'poisson' \| 'trace_replay' \| 'concurrent' \| 'trace_session_replay') | `'constant'` | Traffic-generation strategy. |
| `interval` | float | `1.0` | Inter-request interval in seconds for load types that pace by interval. |
| `stages` | List[[StandardLoadStage](#standardloadstage)] \| List[[ConcurrentLoadStage](#concurrentloadstage)] \| List[[TraceSessionReplayLoadStage](#tracesessionreplayloadstage)] | `[]` | Ordered list of load stages; stage subclass must match the chosen load type. |
| `sweep` | [SweepConfig](#sweepconfig) (optional) | `None` | Auto-generated rate sweep (mutually exclusive with concurrent/trace_session_replay). |
| `num_workers` | int | `8` | Number of worker processes used to drive load. |
| `worker_max_concurrency` | int | `100` | Maximum concurrent in-flight requests per worker. |
| `worker_max_tcp_connections` | int | `2500` | TCP connection pool size per worker. |
| `trace` | [TraceConfig](#traceconfig) (optional) | `None` | Trace file replayed for trace-based load types. |
| `circuit_breakers` | List[str] | `[]` | Names of circuit breakers (defined under `circuit_breakers`) to enable. |
| `request_timeout` | float (optional) | `None` | Per-request timeout in seconds; None disables the timeout. |
| `lora_traffic_split` | List[[MultiLoRAConfig](#multiloraconfig)] (optional) | `None` | Optional traffic split across LoRA adapters; splits must sum to 1.0. |
| `base_seed` | int | _factory_ | Base seed shared across workers; defaults to wall-clock time on startup. |

## MetricsClientConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | MetricsClientType ('prometheus' \| 'default') | **required** | Metrics backend used to collect server-side metrics. |
| `prometheus` | [PrometheusClientConfig](#prometheusclientconfig) (optional) | `None` | Prometheus client configuration (when type=prometheus). |

## ModelServerClientConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | ModelServerType ('vllm' \| 'sglang' \| 'tgi' \| 'mock') | `'vllm'` | Model server flavor; controls server-specific metric mappings. |
| `model_name` | str (optional) | `None` | Model identifier sent on each request. May be auto-detected via /v1/models. |
| `base_url` | str | **required** | Base URL of the model server, e.g. http://0.0.0.0:8000. |
| `ignore_eos` | bool | `True` | Ask the server to ignore EOS so output length is governed by max_tokens. |
| `api_key` | str (optional) | `None` | Bearer token sent as Authorization header. |
| `cert_path` | str (optional) | `None` | Path to a TLS client certificate (PEM) for mTLS. |
| `key_path` | str (optional) | `None` | Path to a TLS client private key (PEM) for mTLS. |

## MultiLoRAConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | str | **required** | LoRA adapter name as registered with the model server. |
| `split` | float | **required** | Fraction of traffic routed to this adapter; all splits must sum to 1.0. |

## OTelTraceReplayConfig

Configuration for OTel trace replay data generator.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `trace_directory` | str (optional) | `None` | Directory containing OTel JSON trace files |
| `trace_files` | List[str] (optional) | `None` | List of paths to specific OTel JSON trace files |
| `use_static_model` | bool | `False` | Use a single static model for all requests |
| `static_model_name` | str | `''` | Static model name (required if use_static_model=True) |
| `model_mapping` | Dict[str, str] (optional) | `None` | Map recorded model names to target models |
| `default_max_tokens` | int | `1000` | Default max_tokens if not specified in trace |
| `include_errors` | bool | `True` | Include spans with error status |
| `skip_invalid_files` | bool | `False` | Skip invalid trace files instead of failing |

## PrometheusClientConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `scrape_interval` | int | `15` | Prometheus scrape interval in seconds; used to align query windows. |
| `url` | HttpUrl (optional) | `None` | Prometheus HTTP endpoint. Mutually exclusive with `google_managed`. |
| `filters` | List[str] | `[]` | PromQL label filters applied to every query. |
| `google_managed` | bool | `False` | Use Google Managed Prometheus instead of `url`. Mutually exclusive with `url`. |

## PrometheusMetricsReportConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `summary` | bool (optional) | `True` | Emit an aggregate Prometheus summary across all stages. |
| `per_stage` | bool (optional) | `False` | Emit per-stage Prometheus reports. |

## ReportConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `request_lifecycle` | [RequestLifecycleMetricsReportConfig](#requestlifecyclemetricsreportconfig) | _factory_ | Request-lifecycle (latency/throughput) reporting options. |
| `prometheus` | [PrometheusMetricsReportConfig](#prometheusmetricsreportconfig) (optional) | _factory_ | Prometheus-scraped metrics reporting options. |
| `session_lifecycle` | [SessionLifecycleReportConfig](#sessionlifecyclereportconfig) | _factory_ | Session-lifecycle reporting options (for session-based load types). |
| `goodput` | [GoodputConfig](#goodputconfig) (optional) | `None` | Optional goodput evaluation; counts requests meeting all SLO constraints. |

## RequestLifecycleMetricsReportConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `summary` | bool (optional) | `True` | Emit an aggregate summary across all stages. |
| `per_stage` | bool (optional) | `True` | Emit one report per load stage. |
| `per_request` | bool (optional) | `False` | Emit one row per individual request (large output). |
| `per_adapter` | bool (optional) | `True` | Emit one report per LoRA adapter. |
| `per_adapter_stage` | bool (optional) | `False` | Emit one report per (adapter, stage) pair. |
| `percentiles` | List[float] | `[0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]` | Latency percentiles to compute and emit. |

## Resolution

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `height` | int | **required** | Pixel height (e.g. 1080). |
| `width` | int | **required** | Pixel width (e.g. 1920). |

## ResponseFormat

Configuration for structured output via response_format parameter.

See vLLM docs: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | ResponseFormatType ('json_schema' \| 'json_object') | `'json_schema'` | Response-format variant to request. |
| `name` | str | `'structured_output'` | Schema name embedded in the json_schema payload. |
| `json_schema` | Dict[str, Any] (optional) | `None` | JSON Schema describing the required output shape (for json_schema type). |

## SessionLifecycleReportConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `summary` | bool (optional) | `True` | Emit aggregate session metrics across all stages. |
| `per_stage` | bool (optional) | `True` | Emit per-stage session metrics. |
| `per_session` | bool (optional) | `False` | Emit one row per individual session (large output). |

## SharedPrefix

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `num_groups` | int | `10` | Number of distinct shared-prefix groups (one shared system prompt per group). |
| `num_prompts_per_group` | int | `10` | Number of unique user prompts generated per shared-prefix group. |
| `system_prompt_len` | int \| [Distribution](#distribution) | `100` | Shared system-prompt length, as a fixed int or a Distribution. |
| `question_len` | int \| [Distribution](#distribution) | `50` | Per-question length, as a fixed int or a Distribution. |
| `output_len` | int \| [Distribution](#distribution) | `50` | Per-response output length, as a fixed int or a Distribution. |
| `seed` | int (optional) | `None` | Random seed for deterministic prompt generation. |
| `question_distribution` | [Distribution](#distribution) (optional) | `None` | Legacy: distribution for question lengths. Prefer the inline distribution form on `question_len`. |
| `output_distribution` | [Distribution](#distribution) (optional) | `None` | Legacy: distribution for output lengths. Prefer the inline distribution form on `output_len`. |
| `enable_multi_turn_chat` | bool | `False` | When true, prompts within a group form multi-turn chat sessions. |
| `multimodal` | [SyntheticMultimodalDatagenConfig](#syntheticmultimodaldatagenconfig) (optional) | `None` | Optional multimodal payload generation alongside text prompts. |

## SimpleStorageServiceConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `path` | str | _factory_ | Directory or object-store path where reports are written. Supports `{timestamp}` substitution. |
| `report_file_prefix` | str (optional) | `None` | Optional filename prefix for report files written to this destination. |
| `bucket_name` | str | **required** | Target S3-compatible bucket. |
| `endpoint_url` | str (optional) | `None` | Override endpoint URL for S3-compatible services (e.g. MinIO). |
| `region_name` | str (optional) | `None` | Region for the S3-compatible service. |
| `addressing_style` | Literal['auto', 'virtual', 'path'] (optional) | `None` | S3 addressing style (auto, virtual-hosted, or path-style). |

## StandardLoadStage

Load stage for CONSTANT and POISSON load types.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `rate` | float | **required** | Request rate (QPS) |
| `duration` | int | **required** | Duration in seconds |
| `num_requests` | int (optional) | `None` | Not used for standard load types |
| `concurrency_level` | int (optional) | `None` | Not used for standard load types |

## StorageConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `local_storage` | [StorageConfigBase](#storageconfigbase) | _factory_ | Local filesystem destination for reports. |
| `google_cloud_storage` | [GoogleCloudStorageConfig](#googlecloudstorageconfig) (optional) | `None` | Optional Google Cloud Storage destination. |
| `simple_storage_service` | [SimpleStorageServiceConfig](#simplestorageserviceconfig) (optional) | `None` | Optional S3-compatible storage destination. |

## StorageConfigBase

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `path` | str | _factory_ | Directory or object-store path where reports are written. Supports `{timestamp}` substitution. |
| `report_file_prefix` | str (optional) | `None` | Optional filename prefix for report files written to this destination. |

## SweepConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | StageGenType ('geometric' \| 'linear') | **required** | How rate steps between stages are spaced. |
| `num_requests` | int | `2000` | Total requests to issue across the sweep. |
| `timeout` | float | `60` | Per-stage timeout in seconds before saturation is declared. |
| `num_stages` | int | `5` | Number of stages in the sweep. |
| `stage_duration` | int | `180` | Duration of each stage in seconds. |
| `saturation_percentile` | float | `95` | Percentile latency used to detect saturation. |

## SyntheticMultimodalDatagenConfig

Configuration for standard multimodal data generation.

Caveat: resolutions, video profiles, and audio durations specified here are
sent to the model server as-is. There is no model-aware validation — real
VLMs each impose their own limits (per-request media count via
--limit-mm-per-prompt, vision-encoder pixel caps, video frame budgets,
audio duration caps, max-model-len). Out-of-range payloads typically fail
at the wire (counted in `failures`) or get silently downsized server-side,
which makes per-modality byte/pixel throughput numbers reflect what was
sent rather than what the model processed. Consult your model's spec sheet
when picking values. See docs/config.md ("Multimodal Data Generation").

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `image` | [ImageDatagenConfig](#imagedatagenconfig) (optional) | `None` | Image generation settings (omit to disable images). |
| `video` | [VideoDatagenConfig](#videodatagenconfig) (optional) | `None` | Video generation settings (omit to disable video). |
| `audio` | [AudioDatagenConfig](#audiodatagenconfig) (optional) | `None` | Audio generation settings (omit to disable audio). |

## TraceConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `file` | str | **required** | Path to the trace file to replay. |
| `format` | TraceFormat ('AzurePublicDataset') | `'AzurePublicDataset'` | On-disk format of the trace file. |

## TraceSessionReplayLoadStage

Load stage for TRACE_SESSION_REPLAY load type.

A stage runs exactly ``num_sessions`` sessions (a slice of the corpus) at
``concurrent_sessions`` concurrency.  A session cursor on ``LoadGenerator``
advances across stages so each stage draws the next N sessions — mirroring
how ``get_data()`` advances through data across Standard/Concurrent stages.

Modes:
1. Simple concurrency control: set concurrent_sessions (and optionally num_sessions)
2. Rate-based with concurrency: set concurrent_sessions + session_rate (+ num_sessions)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `concurrent_sessions` | int | **required** | Maximum number of sessions active simultaneously. 0 = all sessions active at once (stress test mode). N > 0 = at most N sessions active; when one completes, next is activated. |
| `session_rate` | float (optional) | `None` | Sessions to start per second (optional, omit for no rate limit) |
| `num_sessions` | int (optional) | `None` | Number of sessions to run in this stage. Draws the next N sessions from the corpus. None = all remaining sessions. |
| `timeout` | float (optional) | `None` | Wall-clock safety limit in seconds. If exceeded, in-flight sessions are cancelled and stage exits as FAILED. Optional. |

## VideoDatagenConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `count` | [Distribution](#distribution) (optional) | `None` | Distribution of the number of media items to generate per request. |
| `insertion_point` | float \| [Distribution](#distribution) (optional) | `None` | Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from. |
| `profiles` | [VideoProfile](#videoprofile) \| List[[WeightedVideoProfile](#weightedvideoprofile)] (optional) | `None` | Video profile or list of weighted video profiles for generated videos. |
| `representation` | VideoRepresentation ('mp4' \| 'png_frames' \| 'jpeg_frames') | `'mp4'` | Wire-format strategy. ``mp4`` sends one ``video_url`` block carrying an MP4 blob (measures full pipeline including server-side decode). ``png_frames`` and ``jpeg_frames`` send ``frames`` × ``image_url`` blocks at one insertion point in the named encoding (no decode dependency, useful for prefix-cache benchmarks and servers that don't accept ``video_url``). |

## VideoProfile

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `resolution` | ResolutionPreset ('4k' \| '1080p' \| '720p' \| '360p') \| [Resolution](#resolution) | **required** | Frame resolution. Preset string or explicit ``Resolution``. |
| `frames` | int | **required** | Number of frames in the video. Required. |

## WeightedDuration

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `duration` | float | **required** | The length of the audio clip in seconds. |
| `weight` | float | `1.0` | Relative frequency of this duration being selected. |

## WeightedResolution

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `resolution` | ResolutionPreset ('4k' \| '1080p' \| '720p' \| '360p') \| [Resolution](#resolution) | **required** | A ``ResolutionPreset`` (e.g. ``"1080p"``) or explicit ``Resolution``. |
| `weight` | float | `1.0` | Relative frequency of this resolution being selected from the list. |

## WeightedVideoProfile

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `profile` | [VideoProfile](#videoprofile) | **required** | The video profile (resolution + frames) to sample. |
| `weight` | float | `1.0` | Relative frequency of this exact video profile being selected. |

## MetricsSpec

Manage matches and rules to select target metrics.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `matches` | List[str] | **required** | Determine data is target metrics or not |
| `rules` | List[str] | `[]` | Determine data is hit or not |

## TriggerConsecutive

Trip after N consecutive hit events.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | Literal['consecutive'] | **required** | Discriminator selecting the consecutive-hits trigger. |
| `threshold` | int | **required** | Number of consecutive hits required to trip. |

## TriggerRateOverWindow

Trip when the hit rate over a sliding window exceeds a threshold.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | Literal['rate_over_window'] | **required** | Discriminator selecting the rate-over-window trigger. |
| `window_sec` | float | **required** | Length of the sliding evaluation window in seconds. |
| `threshold` | float | **required** | Hit-rate threshold in [0, 1] above which the breaker trips. |
| `min_samples` | int | `0` | Minimum samples required in the window before evaluating the rate. |
