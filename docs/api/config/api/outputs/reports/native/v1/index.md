# Protocol Documentation
<a name="top"></a>

## Table of Contents

- [api/outputs/reports/native/v1/native.proto](#api_outputs_reports_native_v1_native-proto)
    - [CombinedSloAttainment](#api-outputs-reports-native-v1-CombinedSloAttainment)
    - [FailuresMetrics](#api-outputs-reports-native-v1-FailuresMetrics)
    - [LatencyMetrics](#api-outputs-reports-native-v1-LatencyMetrics)
    - [LoadSummary](#api-outputs-reports-native-v1-LoadSummary)
    - [NativeReport](#api-outputs-reports-native-v1-NativeReport)
    - [SloAttainment](#api-outputs-reports-native-v1-SloAttainment)
    - [SloMetrics](#api-outputs-reports-native-v1-SloMetrics)
    - [SuccessesMetrics](#api-outputs-reports-native-v1-SuccessesMetrics)
    - [SummaryStats](#api-outputs-reports-native-v1-SummaryStats)
    - [SummaryStats.PercentilesEntry](#api-outputs-reports-native-v1-SummaryStats-PercentilesEntry)
    - [ThroughputMetrics](#api-outputs-reports-native-v1-ThroughputMetrics)
  
- [Scalar Value Types](#scalar-value-types)



<a name="api_outputs_reports_native_v1_native-proto"></a>
<p align="right"><a href="#top">Top</a></p>

## api/outputs/reports/native/v1/native.proto



<a name="api-outputs-reports-native-v1-CombinedSloAttainment"></a>

### CombinedSloAttainment
Combined SLO attainment details.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| attainment_pct | [double](#double) |  | Attainment percentage. |
| requests_met | [int64](#int64) |  | Number of requests meeting SLO. |
| requests_failed | [int64](#int64) |  | Number of requests failing SLO. |
| total_requests | [int64](#int64) |  | Total number of requests. |
| ttft_slo | [double](#double) |  | TTFT SLO threshold. |
| tpot_slo | [double](#double) |  | TPOT SLO threshold. |
| goodput_rate | [double](#double) |  | Goodput rate. |






<a name="api-outputs-reports-native-v1-FailuresMetrics"></a>

### FailuresMetrics
Metrics for failed requests.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| count | [int64](#int64) |  | Total count of failed requests. |
| request_latency | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Request latency statistics. |
| prompt_len | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Prompt length statistics. |






<a name="api-outputs-reports-native-v1-LatencyMetrics"></a>

### LatencyMetrics
Latency metrics for successful requests.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| request_latency | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Request latency (End - Start). |
| normalized_time_per_output_token | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Normalized time per output token. |
| time_per_output_token | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Time per output token. |
| time_to_first_token | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Time to first token. |
| inter_token_latency | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Inter-token latency. |






<a name="api-outputs-reports-native-v1-LoadSummary"></a>

### LoadSummary
Summary of the load applied.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| count | [int64](#int64) |  | Total count of requests. |
| schedule_delay | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Schedule delay statistics. |
| send_duration | [double](#double) |  | Send duration. |
| requested_rate | [double](#double) |  | Requested rate. |
| achieved_rate | [double](#double) |  | Achieved rate. |
| concurrency | [int64](#int64) |  | Concurrency level. |






<a name="api-outputs-reports-native-v1-NativeReport"></a>

### NativeReport
Native report containing all metrics.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| load_summary | [LoadSummary](#api-outputs-reports-native-v1-LoadSummary) |  | Load summary. |
| successes | [SuccessesMetrics](#api-outputs-reports-native-v1-SuccessesMetrics) |  | Metrics for successful requests. |
| failures | [FailuresMetrics](#api-outputs-reports-native-v1-FailuresMetrics) |  | Metrics for failed requests. |






<a name="api-outputs-reports-native-v1-SloAttainment"></a>

### SloAttainment
SLO attainment details.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| attainment_pct | [double](#double) |  | Attainment percentage. |
| requests_met | [int64](#int64) |  | Number of requests meeting SLO. |
| requests_failed | [int64](#int64) |  | Number of requests failing SLO. |
| total_requests | [int64](#int64) |  | Total number of requests. |
| slo | [double](#double) |  | SLO threshold value. |






<a name="api-outputs-reports-native-v1-SloMetrics"></a>

### SloMetrics
SLO metrics summary.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| ttft_slo | [SloAttainment](#api-outputs-reports-native-v1-SloAttainment) |  | TTFT SLO attainment. |
| tpot_slo | [SloAttainment](#api-outputs-reports-native-v1-SloAttainment) |  | TPOT SLO attainment. |
| combined_slo | [CombinedSloAttainment](#api-outputs-reports-native-v1-CombinedSloAttainment) |  | Combined SLO attainment. |






<a name="api-outputs-reports-native-v1-SuccessesMetrics"></a>

### SuccessesMetrics
Metrics for successful requests.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| count | [int64](#int64) |  | Total count of successful requests. |
| latency | [LatencyMetrics](#api-outputs-reports-native-v1-LatencyMetrics) |  | Latency metrics. |
| throughput | [ThroughputMetrics](#api-outputs-reports-native-v1-ThroughputMetrics) |  | Throughput metrics. |
| prompt_len | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Prompt length statistics. |
| output_len | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Output length statistics. |
| slo_metrics | [SloMetrics](#api-outputs-reports-native-v1-SloMetrics) |  | SLO metrics. |
| rate | [double](#double) |  | Prometheus specific or shared metrics Overall rate. |
| queue_len | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Queue length statistics. |
| kv_cache_usage_percentage | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | KV cache usage percentage. |
| num_requests_swapped | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Number of requests swapped. |
| num_preemptions_total | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Number of preemptions total. |
| prefix_cache_hit_percent | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Prefix cache hit percentage. |
| inter_token_latency | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Inter-token latency. |
| num_requests_running | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Number of requests running. |
| request_queue_time | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Request queue time. |
| request_inference_time | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Request inference time. |
| request_prefill_time | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Request prefill time. |
| request_decode_time | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Request decode time. |
| request_prompt_tokens | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Request prompt tokens. |
| request_generation_tokens | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Request generation tokens. |
| request_max_num_generation_tokens | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Request max num generation tokens. |
| request_params_n | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Request params n. |
| request_params_max_tokens | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Request params max tokens. |
| request_success_count | [double](#double) |  | Request success count. |
| iteration_tokens | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Iteration tokens. |
| prompt_tokens_cached | [double](#double) |  | Prompt tokens cached. |
| prompt_tokens_recomputed | [double](#double) |  | Prompt tokens recomputed. |
| external_prefix_cache_hit_percent | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | External prefix cache hit percentage. |
| mm_cache_hit_percent | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | MM cache hit percentage. |
| corrupted_requests | [double](#double) |  | Corrupted requests. |
| request_prefill_kv_computed_tokens | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | Request prefill KV computed tokens. |
| kv_block_idle_before_evict | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | KV block idle before evict. |
| kv_block_lifetime | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | KV block lifetime. |
| kv_block_reuse_gap | [SummaryStats](#api-outputs-reports-native-v1-SummaryStats) |  | KV block reuse gap. |






<a name="api-outputs-reports-native-v1-SummaryStats"></a>

### SummaryStats
Summary statistics for a metric.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| mean | [double](#double) |  | Mean value. |
| min | [double](#double) |  | Minimum value. |
| max | [double](#double) |  | Maximum value. |
| percentiles | [SummaryStats.PercentilesEntry](#api-outputs-reports-native-v1-SummaryStats-PercentilesEntry) | repeated | Percentiles (e.g., &#34;p50&#34;, &#34;p90&#34;). |
| rate | [double](#double) |  | Optional rate, used in some metrics. |






<a name="api-outputs-reports-native-v1-SummaryStats-PercentilesEntry"></a>

### SummaryStats.PercentilesEntry



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| key | [string](#string) |  |  |
| value | [double](#double) |  |  |






<a name="api-outputs-reports-native-v1-ThroughputMetrics"></a>

### ThroughputMetrics
Throughput metrics for successful requests.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| input_tokens_per_sec | [double](#double) |  | Input tokens per second. |
| output_tokens_per_sec | [double](#double) |  | Output tokens per second. |
| total_tokens_per_sec | [double](#double) |  | Total tokens per second. |
| requests_per_sec | [double](#double) |  | Requests per second. |





 

 

 

 



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

