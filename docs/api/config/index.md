# Protocol Documentation
<a name="top"></a>

## Table of Contents

- [api/config/v1/config.proto](#api_config_v1_config-proto)
    - [APIConfig](#api-config-v1-APIConfig)
    - [APIConfig.HeadersEntry](#api-config-v1-APIConfig-HeadersEntry)
    - [ConcurrentLoadStage](#api-config-v1-ConcurrentLoadStage)
    - [Config](#api-config-v1-Config)
    - [CustomTokenizerConfig](#api-config-v1-CustomTokenizerConfig)
    - [DataConfig](#api-config-v1-DataConfig)
    - [Distribution](#api-config-v1-Distribution)
    - [GoogleCloudStorageConfig](#api-config-v1-GoogleCloudStorageConfig)
    - [LoadConfig](#api-config-v1-LoadConfig)
    - [MetricsClientConfig](#api-config-v1-MetricsClientConfig)
    - [ModelServerClientConfig](#api-config-v1-ModelServerClientConfig)
    - [MultiLoRAConfig](#api-config-v1-MultiLoRAConfig)
    - [PrometheusClientConfig](#api-config-v1-PrometheusClientConfig)
    - [PrometheusMetricsReportConfig](#api-config-v1-PrometheusMetricsReportConfig)
    - [ReportConfig](#api-config-v1-ReportConfig)
    - [RequestLifecycleMetricsReportConfig](#api-config-v1-RequestLifecycleMetricsReportConfig)
    - [ResponseFormat](#api-config-v1-ResponseFormat)
    - [SharedPrefix](#api-config-v1-SharedPrefix)
    - [SimpleStorageServiceConfig](#api-config-v1-SimpleStorageServiceConfig)
    - [StandardLoadStage](#api-config-v1-StandardLoadStage)
    - [StorageConfig](#api-config-v1-StorageConfig)
    - [StorageConfigBase](#api-config-v1-StorageConfigBase)
    - [SweepConfig](#api-config-v1-SweepConfig)
    - [TraceConfig](#api-config-v1-TraceConfig)
  
    - [APIType](#api-config-v1-APIType)
    - [DataGenType](#api-config-v1-DataGenType)
    - [LoadType](#api-config-v1-LoadType)
    - [MetricsClientType](#api-config-v1-MetricsClientType)
    - [ModelServerType](#api-config-v1-ModelServerType)
    - [ResponseFormatType](#api-config-v1-ResponseFormatType)
    - [StageGenType](#api-config-v1-StageGenType)
    - [TraceFormat](#api-config-v1-TraceFormat)
  
- [Scalar Value Types](#scalar-value-types)



<a name="api_config_v1_config-proto"></a>
<p align="right"><a href="#top">Top</a></p>

## api/config/v1/config.proto



<a name="api-config-v1-APIConfig"></a>

### APIConfig
APIConfig is the configuration for the API endpoint.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| type | [APIType](#api-config-v1-APIType) |  | The type of API (COMPLETION or CHAT). |
| streaming | [bool](#bool) |  | Whether to use streaming responses. |
| headers | [APIConfig.HeadersEntry](#api-config-v1-APIConfig-HeadersEntry) | repeated | Custom HTTP headers to send with requests. |
| slo_unit | [string](#string) |  | Unit for SLO thresholds (e.g., &#34;ms&#34;). |
| slo_tpot_header | [string](#string) |  | Header name for TPOT SLO. |
| slo_ttft_header | [string](#string) |  | Header name for TTFT SLO. |
| response_format | [ResponseFormat](#api-config-v1-ResponseFormat) |  | Structured output configuration. |






<a name="api-config-v1-APIConfig-HeadersEntry"></a>

### APIConfig.HeadersEntry



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| key | [string](#string) |  |  |
| value | [string](#string) |  |  |






<a name="api-config-v1-ConcurrentLoadStage"></a>

### ConcurrentLoadStage
ConcurrentLoadStage defines a load stage with a fixed concurrency level.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| num_requests | [int32](#int32) |  | The total number of requests to send.

validate: gt=0 |
| concurrency_level | [int32](#int32) |  | The fixed concurrency level (number of in-flight requests).

validate: gt=0 |
| rate | [float](#float) | optional | Optional request rate (unused if concurrency is fixed). |
| duration | [int32](#int32) | optional | Optional duration (unused if num_requests is fixed). |






<a name="api-config-v1-Config"></a>

### Config
Config is the root configuration message.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| api | [APIConfig](#api-config-v1-APIConfig) |  | API configuration. |
| data | [DataConfig](#api-config-v1-DataConfig) |  | Data generator configuration. |
| load | [LoadConfig](#api-config-v1-LoadConfig) |  | Load generator configuration. |
| metrics | [MetricsClientConfig](#api-config-v1-MetricsClientConfig) |  | Metrics collection configuration. |
| report | [ReportConfig](#api-config-v1-ReportConfig) |  | Report generation configuration. |
| storage | [StorageConfig](#api-config-v1-StorageConfig) |  | Storage configuration. |
| server | [ModelServerClientConfig](#api-config-v1-ModelServerClientConfig) |  | Model server client configuration. |
| tokenizer | [CustomTokenizerConfig](#api-config-v1-CustomTokenizerConfig) |  | Custom tokenizer configuration. |
| circuit_breakers | [string](#string) | repeated | List of circuit breakers to enable globally. |






<a name="api-config-v1-CustomTokenizerConfig"></a>

### CustomTokenizerConfig
CustomTokenizerConfig defines configuration for a custom tokenizer.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| pretrained_model_name_or_path | [string](#string) | optional | Name or path of the pretrained model/tokenizer. |
| trust_remote_code | [bool](#bool) | optional | Whether to trust remote code when loading tokenizer. |
| token | [string](#string) | optional | Authentication token (optional). |






<a name="api-config-v1-DataConfig"></a>

### DataConfig
DataConfig is the configuration for the dataset or data generator.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| type | [DataGenType](#api-config-v1-DataGenType) |  | The type of data generator to use (e.g., SHAREGPT, RANDOM). |
| path | [string](#string) | optional | Path to the dataset file (required for some types). |
| input_distribution | [Distribution](#api-config-v1-Distribution) |  | Distribution of input lengths (for RANDOM). |
| output_distribution | [Distribution](#api-config-v1-Distribution) |  | Distribution of output lengths (for RANDOM). |
| shared_prefix | [SharedPrefix](#api-config-v1-SharedPrefix) |  | Configuration for shared prefix testing. |
| trace | [TraceConfig](#api-config-v1-TraceConfig) |  | Configuration for trace replay. |






<a name="api-config-v1-Distribution"></a>

### Distribution
Distribution defines a range and statistical parameters for random values.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| min | [int32](#int32) |  | Minimum value.

validate: ge=0 |
| max | [int32](#int32) |  | Maximum value.

validate: ge=0 |
| mean | [float](#float) |  | Mean value.

validate: ge=0 |
| std_dev | [float](#float) |  | Standard deviation.

validate: ge=0 |
| total_count | [int32](#int32) | optional | Total count of values to generate (optional). |






<a name="api-config-v1-GoogleCloudStorageConfig"></a>

### GoogleCloudStorageConfig
GoogleCloudStorageConfig defines configuration for GCS storage.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| path | [string](#string) |  | Path within the bucket. |
| report_file_prefix | [string](#string) | optional | Prefix for report files. |
| bucket_name | [string](#string) |  | Name of the GCS bucket. |






<a name="api-config-v1-LoadConfig"></a>

### LoadConfig
LoadConfig defines parameters for the load generator.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| type | [LoadType](#api-config-v1-LoadType) |  | The type of load to generate (CONSTANT, POISSON, etc.). |
| interval | [float](#float) |  | Interval between requests or stages in seconds.

validate: gt=0 |
| standard_stages | [StandardLoadStage](#api-config-v1-StandardLoadStage) | repeated | List of stages for standard load. |
| concurrent_stages | [ConcurrentLoadStage](#api-config-v1-ConcurrentLoadStage) | repeated | List of stages for concurrent load. |
| sweep | [SweepConfig](#api-config-v1-SweepConfig) |  | Configuration for parameter sweep. |
| num_workers | [int32](#int32) |  | Number of worker processes.

validate: ge=1 |
| worker_max_concurrency | [int32](#int32) |  | Max concurrency per worker. |
| worker_max_tcp_connections | [int32](#int32) |  | Max TCP connections per worker. |
| trace | [TraceConfig](#api-config-v1-TraceConfig) |  | Configuration for trace replay (if applicable). |
| circuit_breakers | [string](#string) | repeated | List of circuit breakers to enable. |
| request_timeout | [float](#float) | optional | Timeout for requests in seconds (optional). |
| lora_traffic_split | [MultiLoRAConfig](#api-config-v1-MultiLoRAConfig) | repeated | Traffic split for Multi-LoRA setups. |
| base_seed | [int64](#int64) | optional | Base seed for random number generation. |






<a name="api-config-v1-MetricsClientConfig"></a>

### MetricsClientConfig
MetricsClientConfig defines configuration for metrics collection.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| type | [MetricsClientType](#api-config-v1-MetricsClientType) |  | The type of metrics client to use. |
| prometheus | [PrometheusClientConfig](#api-config-v1-PrometheusClientConfig) |  | Configuration for Prometheus client (required if type is PROMETHEUS). |






<a name="api-config-v1-ModelServerClientConfig"></a>

### ModelServerClientConfig
ModelServerClientConfig defines configuration for the connection to the model server.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| type | [ModelServerType](#api-config-v1-ModelServerType) |  | The type of model server (VLLM, SGLANG, etc.). |
| model_name | [string](#string) | optional | The name of the model to use. |
| base_url | [string](#string) |  | Base URL of the model server. |
| ignore_eos | [bool](#bool) |  | Whether to ignore EOS tokens (continue generating until max tokens). |
| api_key | [string](#string) | optional | API key for authentication (optional). |
| cert_path | [string](#string) | optional | Path to client certificate (optional). |
| key_path | [string](#string) | optional | Path to client private key (optional). |






<a name="api-config-v1-MultiLoRAConfig"></a>

### MultiLoRAConfig
MultiLoRAConfig defines traffic split for a specific LoRA adapter.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| name | [string](#string) |  | The name of the LoRA adapter. |
| split | [float](#float) |  | The fraction of traffic to send to this adapter (0.0 to 1.0).

validate: ge=0, le=1 |






<a name="api-config-v1-PrometheusClientConfig"></a>

### PrometheusClientConfig
PrometheusClientConfig defines configuration for a Prometheus metrics client.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| scrape_interval | [int32](#int32) |  | Scrape interval in seconds. |
| url | [string](#string) | optional | URL of the Prometheus server (not needed if google_managed is true). |
| filters | [string](#string) | repeated | Filters for metrics. |
| google_managed | [bool](#bool) |  | Whether to use Google Managed Prometheus. |






<a name="api-config-v1-PrometheusMetricsReportConfig"></a>

### PrometheusMetricsReportConfig
PrometheusMetricsReportConfig defines configuration for Prometheus metrics reports.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| summary | [bool](#bool) | optional | Whether to include a summary. |
| per_stage | [bool](#bool) | optional | Whether to include per-stage metrics. |






<a name="api-config-v1-ReportConfig"></a>

### ReportConfig
ReportConfig defines configuration for generated reports.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| request_lifecycle | [RequestLifecycleMetricsReportConfig](#api-config-v1-RequestLifecycleMetricsReportConfig) |  | Configuration for request lifecycle metrics reports. |
| prometheus | [PrometheusMetricsReportConfig](#api-config-v1-PrometheusMetricsReportConfig) |  | Configuration for Prometheus metrics reports. |






<a name="api-config-v1-RequestLifecycleMetricsReportConfig"></a>

### RequestLifecycleMetricsReportConfig
RequestLifecycleMetricsReportConfig defines configuration for request lifecycle metrics reports.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| summary | [bool](#bool) | optional | Whether to include a summary. |
| per_stage | [bool](#bool) | optional | Whether to include per-stage metrics. |
| per_request | [bool](#bool) | optional | Whether to include per-request metrics. |
| per_adapter | [bool](#bool) | optional | Whether to include per-adapter metrics. |
| per_adapter_stage | [bool](#bool) | optional | Whether to include per-adapter-stage metrics. |
| percentiles | [float](#float) | repeated | Percentiles to report (e.g. 50.0, 90.0, 99.0). |






<a name="api-config-v1-ResponseFormat"></a>

### ResponseFormat
ResponseFormat represents structured output options.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| type | [ResponseFormatType](#api-config-v1-ResponseFormatType) |  | The type of response format (e.g. JSON_SCHEMA). |
| name | [string](#string) |  | The name of the format (optional). |
| json_schema | [string](#string) |  | The JSON schema string (required for JSON_SCHEMA). |






<a name="api-config-v1-SharedPrefix"></a>

### SharedPrefix
SharedPrefix defines parameters for testing shared prefix caching.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| num_groups | [int32](#int32) |  | Number of prompt groups. |
| num_prompts_per_group | [int32](#int32) |  | Number of prompts per group. |
| system_prompt_len | [int32](#int32) |  | Length of the common system prompt. |
| question_len | [int32](#int32) |  | Length of the question part. |
| output_len | [int32](#int32) |  | Length of the expected output. |
| question_distribution | [Distribution](#api-config-v1-Distribution) |  | Distribution for question lengths (optional). |
| output_distribution | [Distribution](#api-config-v1-Distribution) |  | Distribution for output lengths (optional). |
| enable_multi_turn_chat | [bool](#bool) |  | Whether to enable multi-turn chat simulation. |






<a name="api-config-v1-SimpleStorageServiceConfig"></a>

### SimpleStorageServiceConfig
SimpleStorageServiceConfig defines configuration for S3 storage.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| path | [string](#string) |  | Path within the bucket. |
| report_file_prefix | [string](#string) | optional | Prefix for report files. |
| bucket_name | [string](#string) |  | Name of the S3 bucket. |






<a name="api-config-v1-StandardLoadStage"></a>

### StandardLoadStage
StandardLoadStage defines a load stage with a fixed rate and duration.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| rate | [float](#float) |  | The request rate (requests per second).

validate: gt=0 |
| duration | [int32](#int32) |  | The duration of the stage in seconds.

validate: gt=0 |
| num_requests | [int32](#int32) | optional | Optional total number of requests (overrides duration if set). |
| concurrency_level | [int32](#int32) | optional | Optional concurrency level (overrides rate if set). |






<a name="api-config-v1-StorageConfig"></a>

### StorageConfig
StorageConfig defines available storage backends.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| local_storage | [StorageConfigBase](#api-config-v1-StorageConfigBase) |  | Local filesystem storage. |
| google_cloud_storage | [GoogleCloudStorageConfig](#api-config-v1-GoogleCloudStorageConfig) |  | Google Cloud Storage. |
| simple_storage_service | [SimpleStorageServiceConfig](#api-config-v1-SimpleStorageServiceConfig) |  | Amazon S3 or compatible storage. |






<a name="api-config-v1-StorageConfigBase"></a>

### StorageConfigBase
StorageConfigBase defines basic storage configuration.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| path | [string](#string) |  | Path to the storage directory or bucket. |
| report_file_prefix | [string](#string) | optional | Prefix for report files. |






<a name="api-config-v1-SweepConfig"></a>

### SweepConfig
SweepConfig defines parameters for a load sweep.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| type | [StageGenType](#api-config-v1-StageGenType) |  | The type of stage generation (GEOMETRIC or LINEAR). |
| num_requests | [int32](#int32) |  | Number of requests per stage. p2p: {&#34;default&#34;: 1, &#34;gt&#34;: 0} |
| timeout | [float](#float) |  | Timeout for each stage in seconds. p2p: {&#34;default&#34;: 1.0, &#34;gt&#34;: 0.0} |
| num_stages | [int32](#int32) |  | Number of stages in the sweep. p2p: {&#34;default&#34;: 1, &#34;gt&#34;: 0} |
| stage_duration | [int32](#int32) |  | Duration of each stage in seconds. p2p: {&#34;default&#34;: 1, &#34;gt&#34;: 0} |
| saturation_percentile | [float](#float) |  | Percentile to use for saturation detection (e.g. 90.0). p2p: {&#34;default&#34;: 0.0, &#34;ge&#34;: 0.0, &#34;le&#34;: 100.0} |






<a name="api-config-v1-TraceConfig"></a>

### TraceConfig
TraceConfig is the configuration for trace replay.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| file | [string](#string) |  | Path to the trace file. |
| format | [TraceFormat](#api-config-v1-TraceFormat) |  | Format of the trace file (e.g. AZURE_PUBLIC_DATASET). |





 


<a name="api-config-v1-APIType"></a>

### APIType
APIType represents the API format type.

| Name | Number | Description |
| ---- | ------ | ----------- |
| API_TYPE_UNSPECIFIED | 0 |  |
| COMPLETION | 1 |  |
| CHAT | 2 |  |



<a name="api-config-v1-DataGenType"></a>

### DataGenType


| Name | Number | Description |
| ---- | ------ | ----------- |
| DATA_GEN_TYPE_UNSPECIFIED | 0 |  |
| MOCK | 1 |  |
| SHAREGPT | 2 |  |
| SYNTHETIC | 3 |  |
| RANDOM | 4 |  |
| SHARED_PREFIX | 5 |  |
| CNN_DAILYMAIL | 6 |  |
| INFINITY_INSTRUCT | 7 |  |
| BILLSUM_CONVERSATIONS | 8 |  |



<a name="api-config-v1-LoadType"></a>

### LoadType


| Name | Number | Description |
| ---- | ------ | ----------- |
| LOAD_TYPE_UNSPECIFIED | 0 |  |
| CONSTANT | 1 |  |
| POISSON | 2 |  |
| TRACE_REPLAY | 3 |  |
| CONCURRENT | 4 |  |



<a name="api-config-v1-MetricsClientType"></a>

### MetricsClientType
MetricsClientType defines the type of metrics client.

| Name | Number | Description |
| ---- | ------ | ----------- |
| METRICS_CLIENT_TYPE_UNSPECIFIED | 0 |  |
| PROMETHEUS | 1 |  |
| DEFAULT | 2 |  |



<a name="api-config-v1-ModelServerType"></a>

### ModelServerType


| Name | Number | Description |
| ---- | ------ | ----------- |
| MODEL_SERVER_TYPE_UNSPECIFIED | 0 |  |
| VLLM | 1 |  |
| SGLANG | 2 |  |
| TGI | 3 |  |
| MOCK_SERVER | 4 |  |



<a name="api-config-v1-ResponseFormatType"></a>

### ResponseFormatType
ResponseFormatType represents structured output options.

| Name | Number | Description |
| ---- | ------ | ----------- |
| RESPONSE_FORMAT_TYPE_UNSPECIFIED | 0 |  |
| JSON_SCHEMA | 1 |  |
| JSON_OBJECT | 2 |  |



<a name="api-config-v1-StageGenType"></a>

### StageGenType
StageGenType defines the type of stage generation for sweeps.

| Name | Number | Description |
| ---- | ------ | ----------- |
| STAGE_GEN_TYPE_UNSPECIFIED | 0 |  |
| GEOMETRIC | 1 |  |
| LINEAR | 2 |  |



<a name="api-config-v1-TraceFormat"></a>

### TraceFormat


| Name | Number | Description |
| ---- | ------ | ----------- |
| TRACE_FORMAT_UNSPECIFIED | 0 |  |
| AZURE_PUBLIC_DATASET | 1 |  |


 

 

 



## Scalar Value Types

| .proto Type | Notes | C++ | Java | Python | Go | C# | PHP | Ruby |
| ----------- | ----- | --- | ---- | ------ | -- | -- | --- | ---- |
| <a name="double" /> double |  | double | double | float | float64 | double | float | Float |
| <a name="float" /> float |  | float | float | float | float32 | float | float | Float |
| <a name="int32" /> int32 | Uses variable-length encoding. Inefficient for encoding negative numbers – if your field is likely to have negative values, use sint32 instead. | int32 | int | int | int32 | int | integer | Bignum or Fixnum (as required) |
| <a name="int64" /> int64 | Uses variable-length encoding. Inefficient for encoding negative numbers – if your field is likely to have negative values, use sint64 instead. | int64 | long | int/long | int64 | long | integer/string | Bignum |
| <a name="uint32" /> uint32 | Uses variable-length encoding. | uint32 | int | int/long | uint32 | uint | integer | Bignum or Fixnum (as required) |
| <a name="uint64" /> uint64 | Uses variable-length encoding. | uint64 | long | int/long | uint64 | ulong | integer/string | Bignum or Fixnum (as required) |
| <a name="sint32" /> sint32 | Uses variable-length encoding. Signed int value. These more efficiently encode negative numbers than regular int32s. | int32 | int | int | int32 | int | integer | Bignum or Fixnum (as required) |
| <a name="sint64" /> sint64 | Uses variable-length encoding. Signed int value. These more efficiently encode negative numbers than regular int64s. | int64 | long | int/long | int64 | long | integer/string | Bignum |
| <a name="fixed32" /> fixed32 | Always four bytes. More efficient than uint32 if values are often greater than 2^28. | uint32 | int | int | uint32 | uint | integer | Bignum or Fixnum (as required) |
| <a name="fixed64" /> fixed64 | Always eight bytes. More efficient than uint64 if values are often greater than 2^56. | uint64 | long | int/long | uint64 | ulong | integer/string | Bignum |
| <a name="sfixed32" /> sfixed32 | Always four bytes. | int32 | int | int | int32 | int | integer | Bignum or Fixnum (as required) |
| <a name="sfixed64" /> sfixed64 | Always eight bytes. | int64 | long | int/long | int64 | long | integer/string | Bignum |
| <a name="bool" /> bool |  | bool | boolean | boolean | bool | bool | boolean | TrueClass/FalseClass |
| <a name="string" /> string | A string must always contain UTF-8 encoded or 7-bit ASCII text. | string | String | str/unicode | string | string | string | String (UTF-8) |
| <a name="bytes" /> bytes | May contain any arbitrary sequence of bytes. | string | ByteString | str | []byte | ByteString | string | String (ASCII-8BIT) |

