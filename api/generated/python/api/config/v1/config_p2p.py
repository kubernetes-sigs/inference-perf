# This is an automatically generated file, please do not change
# gen by protobuf_to_pydantic[v0.3.3.1](https://github.com/so1n/protobuf_to_pydantic)
# Protobuf Version: 6.33.6
# Pydantic Version: 2.12.5
from enum import IntEnum
from google.protobuf.message import Message  # type: ignore
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
import typing


class APIType(IntEnum):
    """
    APIType represents the API format type.
    """

    API_TYPE_UNSPECIFIED = 0
    COMPLETION = 1
    CHAT = 2


class ResponseFormatType(IntEnum):
    """
    ResponseFormatType represents structured output options.
    """

    RESPONSE_FORMAT_TYPE_UNSPECIFIED = 0
    JSON_SCHEMA = 1
    JSON_OBJECT = 2


class TraceFormat(IntEnum):
    TRACE_FORMAT_UNSPECIFIED = 0
    AZURE_PUBLIC_DATASET = 1


class DataGenType(IntEnum):
    DATA_GEN_TYPE_UNSPECIFIED = 0
    MOCK = 1
    SHAREGPT = 2
    SYNTHETIC = 3
    RANDOM = 4
    SHARED_PREFIX = 5
    CNN_DAILYMAIL = 6
    INFINITY_INSTRUCT = 7
    BILLSUM_CONVERSATIONS = 8


class ModelServerType(IntEnum):
    MODEL_SERVER_TYPE_UNSPECIFIED = 0
    VLLM = 1
    SGLANG = 2
    TGI = 3
    MOCK_SERVER = 4


class LoadType(IntEnum):
    LOAD_TYPE_UNSPECIFIED = 0
    CONSTANT = 1
    POISSON = 2
    TRACE_REPLAY = 3
    CONCURRENT = 4


class StageGenType(IntEnum):
    """
    StageGenType defines the type of stage generation for sweeps.
    """

    STAGE_GEN_TYPE_UNSPECIFIED = 0
    GEOMETRIC = 1
    LINEAR = 2


class MetricsClientType(IntEnum):
    """
    MetricsClientType defines the type of metrics client.
    """

    METRICS_CLIENT_TYPE_UNSPECIFIED = 0
    PROMETHEUS = 1
    DEFAULT = 2


class ResponseFormat(BaseModel):
    """
    ResponseFormat represents structured output options.
    """

    model_config = ConfigDict(validate_default=True)
    # The type of response format (e.g. JSON_SCHEMA).
    type: ResponseFormatType = Field(default=0)
    # The name of the format (optional).
    name: str = Field(default="")
    # The JSON schema string (required for JSON_SCHEMA).
    json_schema: str = Field(default="")


class APIConfig(BaseModel):
    """
    APIConfig is the configuration for the API endpoint.
    """

    model_config = ConfigDict(validate_default=True)
    # The type of API (COMPLETION or CHAT).
    type: APIType = Field(default=0)
    # Whether to use streaming responses.
    streaming: bool = Field(default=False)
    # Custom HTTP headers to send with requests.
    headers: "typing.Dict[str, str]" = Field(default_factory=dict)
    # Unit for SLO thresholds (e.g., "ms").
    slo_unit: str = Field(default="")
    # Header name for TPOT SLO.
    slo_tpot_header: str = Field(default="")
    # Header name for TTFT SLO.
    slo_ttft_header: str = Field(default="")
    # Structured output configuration.
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)


class TraceConfig(BaseModel):
    """
    TraceConfig is the configuration for trace replay.
    """

    model_config = ConfigDict(validate_default=True)
    # Path to the trace file.
    file: str = Field(default="")
    # Format of the trace file (e.g. AZURE_PUBLIC_DATASET).
    format: TraceFormat = Field(default=0)


class Distribution(BaseModel):
    """
    Distribution defines a range and statistical parameters for random values.
    """

    # Minimum value.
    min: int = Field(default=0)  # validate: ge=0
    # Maximum value.
    max: int = Field(default=0)  # validate: ge=0
    # Mean value.
    mean: float = Field(default=0.0)  # validate: ge=0
    # Standard deviation.
    std_dev: float = Field(default=0.0)  # validate: ge=0
    # Total count of values to generate (optional).
    total_count: typing.Optional[int] = Field(default=0)


class SharedPrefix(BaseModel):
    """
    SharedPrefix defines parameters for testing shared prefix caching.
    """

    # Number of prompt groups.
    num_groups: int = Field(default=0)
    # Number of prompts per group.
    num_prompts_per_group: int = Field(default=0)
    # Length of the common system prompt.
    system_prompt_len: int = Field(default=0)
    # Length of the question part.
    question_len: int = Field(default=0)
    # Length of the expected output.
    output_len: int = Field(default=0)
    # Distribution for question lengths (optional).
    question_distribution: Distribution = Field(default_factory=Distribution)
    # Distribution for output lengths (optional).
    output_distribution: Distribution = Field(default_factory=Distribution)
    # Whether to enable multi-turn chat simulation.
    enable_multi_turn_chat: bool = Field(default=False)


class DataConfig(BaseModel):
    """
    DataConfig is the configuration for the dataset or data generator.
    """

    model_config = ConfigDict(validate_default=True)
    # The type of data generator to use (e.g., SHAREGPT, RANDOM).
    type: DataGenType = Field(default=0)
    # Path to the dataset file (required for some types).
    path: typing.Optional[str] = Field(default="")
    # Distribution of input lengths (for RANDOM).
    input_distribution: Distribution = Field(default_factory=Distribution)
    # Distribution of output lengths (for RANDOM).
    output_distribution: Distribution = Field(default_factory=Distribution)
    # Configuration for shared prefix testing.
    shared_prefix: SharedPrefix = Field(default_factory=SharedPrefix)
    # Configuration for trace replay.
    trace: TraceConfig = Field(default_factory=TraceConfig)


class StandardLoadStage(BaseModel):
    """
    StandardLoadStage defines a load stage with a fixed rate and duration.
    """

    # The request rate (requests per second).
    rate: float = Field(default=0.0)  # validate: gt=0
    # The duration of the stage in seconds.
    duration: int = Field(default=0)  # validate: gt=0
    # Optional total number of requests (overrides duration if set).
    num_requests: typing.Optional[int] = Field(default=0)
    # Optional concurrency level (overrides rate if set).
    concurrency_level: typing.Optional[int] = Field(default=0)


class ConcurrentLoadStage(BaseModel):
    """
    ConcurrentLoadStage defines a load stage with a fixed concurrency level.
    """

    # The total number of requests to send.
    num_requests: int = Field(default=0)  # validate: gt=0
    # The fixed concurrency level (number of in-flight requests).
    concurrency_level: int = Field(default=0)  # validate: gt=0
    # Optional request rate (unused if concurrency is fixed).
    rate: typing.Optional[float] = Field(default=0.0)
    # Optional duration (unused if num_requests is fixed).
    duration: typing.Optional[int] = Field(default=0)


class SweepConfig(BaseModel):
    """
    SweepConfig defines parameters for a load sweep.
    """

    model_config = ConfigDict(validate_default=True)
    # The type of stage generation (GEOMETRIC or LINEAR).
    type: StageGenType = Field(default=0)
    # Number of requests per stage.
    num_requests: int = Field(default=1, gt=0)
    # Timeout for each stage in seconds.
    timeout: float = Field(default=1.0, gt=0.0)
    # Number of stages in the sweep.
    num_stages: int = Field(default=1, gt=0)
    # Duration of each stage in seconds.
    stage_duration: int = Field(default=1, gt=0)
    # Percentile to use for saturation detection (e.g. 90.0).
    saturation_percentile: float = Field(default=0.0, ge=0.0, le=100.0)


class MultiLoRAConfig(BaseModel):
    """
    MultiLoRAConfig defines traffic split for a specific LoRA adapter.
    """

    # The name of the LoRA adapter.
    name: str = Field(default="")
    # The fraction of traffic to send to this adapter (0.0 to 1.0).
    split: float = Field(default=0.0)  # validate: ge=0, le=1


class LoadConfig(BaseModel):
    """
    LoadConfig defines parameters for the load generator.
    """

    model_config = ConfigDict(validate_default=True)
    # The type of load to generate (CONSTANT, POISSON, etc.).
    type: LoadType = Field(default=0)
    # Interval between requests or stages in seconds.
    interval: float = Field(default=0.0)  # validate: gt=0
    # List of stages for standard load.
    standard_stages: typing.List[StandardLoadStage] = Field(default_factory=list)
    # List of stages for concurrent load.
    concurrent_stages: typing.List[ConcurrentLoadStage] = Field(default_factory=list)
    # Configuration for parameter sweep.
    sweep: SweepConfig = Field(default_factory=SweepConfig)
    # Number of worker processes.
    num_workers: int = Field(default=0)  # validate: ge=1
    # Max concurrency per worker.
    worker_max_concurrency: int = Field(default=0)
    # Max TCP connections per worker.
    worker_max_tcp_connections: int = Field(default=0)
    # Configuration for trace replay (if applicable).
    trace: TraceConfig = Field(default_factory=TraceConfig)
    # List of circuit breakers to enable.
    circuit_breakers: typing.List[str] = Field(default_factory=list)
    # Timeout for requests in seconds (optional).
    request_timeout: typing.Optional[float] = Field(default=0.0)
    # Traffic split for Multi-LoRA setups.
    lora_traffic_split: typing.List[MultiLoRAConfig] = Field(default_factory=list)
    # Base seed for random number generation.
    base_seed: typing.Optional[int] = Field(default=0)


class StorageConfigBase(BaseModel):
    """
    StorageConfigBase defines basic storage configuration.
    """

    # Path to the storage directory or bucket.
    path: str = Field(default="")
    # Prefix for report files.
    report_file_prefix: typing.Optional[str] = Field(default="")


class GoogleCloudStorageConfig(BaseModel):
    """
    GoogleCloudStorageConfig defines configuration for GCS storage.
    """

    # Path within the bucket.
    path: str = Field(default="")
    # Prefix for report files.
    report_file_prefix: typing.Optional[str] = Field(default="")
    # Name of the GCS bucket.
    bucket_name: str = Field(default="")


class SimpleStorageServiceConfig(BaseModel):
    """
    SimpleStorageServiceConfig defines configuration for S3 storage.
    """

    # Path within the bucket.
    path: str = Field(default="")
    # Prefix for report files.
    report_file_prefix: typing.Optional[str] = Field(default="")
    # Name of the S3 bucket.
    bucket_name: str = Field(default="")


class StorageConfig(BaseModel):
    """
    StorageConfig defines available storage backends.
    """

    # Local filesystem storage.
    local_storage: StorageConfigBase = Field(default_factory=StorageConfigBase)
    # Google Cloud Storage.
    google_cloud_storage: GoogleCloudStorageConfig = Field(
        default_factory=GoogleCloudStorageConfig
    )
    # Amazon S3 or compatible storage.
    simple_storage_service: SimpleStorageServiceConfig = Field(
        default_factory=SimpleStorageServiceConfig
    )


class RequestLifecycleMetricsReportConfig(BaseModel):
    """
    RequestLifecycleMetricsReportConfig defines configuration for request lifecycle metrics reports.
    """

    # Whether to include a summary.
    summary: typing.Optional[bool] = Field(default=False)
    # Whether to include per-stage metrics.
    per_stage: typing.Optional[bool] = Field(default=False)
    # Whether to include per-request metrics.
    per_request: typing.Optional[bool] = Field(default=False)
    # Whether to include per-adapter metrics.
    per_adapter: typing.Optional[bool] = Field(default=False)
    # Whether to include per-adapter-stage metrics.
    per_adapter_stage: typing.Optional[bool] = Field(default=False)
    # Percentiles to report (e.g. 50.0, 90.0, 99.0).
    percentiles: typing.List[float] = Field(default_factory=list)


class PrometheusMetricsReportConfig(BaseModel):
    """
    PrometheusMetricsReportConfig defines configuration for Prometheus metrics reports.
    """

    # Whether to include a summary.
    summary: typing.Optional[bool] = Field(default=False)
    # Whether to include per-stage metrics.
    per_stage: typing.Optional[bool] = Field(default=False)


class ReportConfig(BaseModel):
    """
    ReportConfig defines configuration for generated reports.
    """

    # Configuration for request lifecycle metrics reports.
    request_lifecycle: RequestLifecycleMetricsReportConfig = Field(
        default_factory=RequestLifecycleMetricsReportConfig
    )
    # Configuration for Prometheus metrics reports.
    prometheus: PrometheusMetricsReportConfig = Field(
        default_factory=PrometheusMetricsReportConfig
    )


class PrometheusClientConfig(BaseModel):
    """
    PrometheusClientConfig defines configuration for a Prometheus metrics client.
    """

    # Scrape interval in seconds.
    scrape_interval: int = Field(default=0)
    # URL of the Prometheus server (not needed if google_managed is true).
    url: typing.Optional[str] = Field(default="")
    # Filters for metrics.
    filters: typing.List[str] = Field(default_factory=list)
    # Whether to use Google Managed Prometheus.
    google_managed: bool = Field(default=False)


class MetricsClientConfig(BaseModel):
    """
    MetricsClientConfig defines configuration for metrics collection.
    """

    model_config = ConfigDict(validate_default=True)
    # The type of metrics client to use.
    type: MetricsClientType = Field(default=0)
    # Configuration for Prometheus client (required if type is PROMETHEUS).
    prometheus: PrometheusClientConfig = Field(default_factory=PrometheusClientConfig)


class ModelServerClientConfig(BaseModel):
    """
    ModelServerClientConfig defines configuration for the connection to the model server.
    """

    model_config = ConfigDict(validate_default=True)
    # The type of model server (VLLM, SGLANG, etc.).
    type: ModelServerType = Field(default=0)
    # The name of the model to use.
    model_name: typing.Optional[str] = Field(default="")
    # Base URL of the model server.
    base_url: str = Field(default="")
    # Whether to ignore EOS tokens (continue generating until max tokens).
    ignore_eos: bool = Field(default=False)
    # API key for authentication (optional).
    api_key: typing.Optional[str] = Field(default="")
    # Path to client certificate (optional).
    cert_path: typing.Optional[str] = Field(default="")
    # Path to client private key (optional).
    key_path: typing.Optional[str] = Field(default="")


class CustomTokenizerConfig(BaseModel):
    """
    CustomTokenizerConfig defines configuration for a custom tokenizer.
    """

    # Name or path of the pretrained model/tokenizer.
    pretrained_model_name_or_path: typing.Optional[str] = Field(default="")
    # Whether to trust remote code when loading tokenizer.
    trust_remote_code: typing.Optional[bool] = Field(default=False)
    # Authentication token (optional).
    token: typing.Optional[str] = Field(default="")


class Config(BaseModel):
    """
    Config is the root configuration message.
    """

    # API configuration.
    api: APIConfig = Field(default_factory=APIConfig)
    # Data generator configuration.
    data: DataConfig = Field(default_factory=DataConfig)
    # Load generator configuration.
    load: LoadConfig = Field(default_factory=LoadConfig)
    # Metrics collection configuration.
    metrics: MetricsClientConfig = Field(default_factory=MetricsClientConfig)
    # Report generation configuration.
    report: ReportConfig = Field(default_factory=ReportConfig)
    # Storage configuration.
    storage: StorageConfig = Field(default_factory=StorageConfig)
    # Model server client configuration.
    server: ModelServerClientConfig = Field(default_factory=ModelServerClientConfig)
    # Custom tokenizer configuration.
    tokenizer: CustomTokenizerConfig = Field(default_factory=CustomTokenizerConfig)
    # List of circuit breakers to enable globally.
    circuit_breakers: typing.List[str] = Field(default_factory=list)
