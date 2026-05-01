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
import time
from typing import List, Optional

import pytest

from inference_perf.apis import (
    ErrorResponseInfo,
    InferenceInfo,
    RequestLifecycleMetric,
    StreamedInferenceResponseInfo,
)
from inference_perf.client.metricsclient import PerfRuntimeParameters
from inference_perf.client.metricsclient.base import StageRuntimeInfo, StageStatus
from inference_perf.config import (
    APIConfig,
    BRV02Config,
    BRV02StackComponentOverride,
    Config,
    DataConfig,
    DataGenType,
    LoadConfig,
    LoadType,
    ReportConfig,
    SharedPrefix,
    StandardLoadStage,
)
from inference_perf.reportgen.br.v0_2 import generate_br_v0_2
from inference_perf.reportgen.br.v0_2.collectors import (
    CollectedStackObservability,
    ConfigStackCollector,
    NoopStackCollector,
)
from inference_perf.reportgen.br.v0_2.schema import (
    BenchmarkReportV02,
    Distribution,
    LoadSource,
    Units,
)


def _streaming_metric(start: float, output_tokens: int = 10, itl: float = 0.02, ttft: float = 0.05) -> RequestLifecycleMetric:
    times = [start + ttft + i * itl for i in range(output_tokens)]
    return RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=start - 0.001,
        start_time=start,
        end_time=times[-1] + 0.005,
        request_data="{}",
        response_data="ok",
        info=InferenceInfo(
            input_tokens=128,
            response_info=StreamedInferenceResponseInfo(
                response_chunks=[],
                chunk_times=times,
                output_tokens=output_tokens,
                output_token_times=times,
            ),
        ),
        error=None,
    )


def _shared_prefix_config(stack: Optional[List[BRV02StackComponentOverride]] = None) -> Config:
    return Config(
        api=APIConfig(streaming=True),
        data=DataConfig(
            type=DataGenType.SharedPrefix,
            shared_prefix=SharedPrefix(
                num_groups=2,
                num_prompts_per_group=4,
                system_prompt_len=512,
                question_len=32,
                output_len=64,
            ),
        ),
        load=LoadConfig(
            type=LoadType.CONSTANT,
            stages=[StandardLoadStage(rate=10.0, duration=10)],
            num_workers=2,
        ),
        report=ReportConfig(br_v0_2=BRV02Config(enabled=True, description="test", stack=stack)),
    )


def _runtime_params(start: float, rate: float = 10.0) -> PerfRuntimeParameters:
    return PerfRuntimeParameters(
        start_time=start,
        duration=5.0,
        model_server_metrics={},
        stages={
            0: StageRuntimeInfo(
                stage_id=0,
                rate=rate,
                start_time=start,
                end_time=start + 5,
                status=StageStatus.COMPLETED,
            )
        },
    )


def test_adapter_validates_against_vendored_schema() -> None:
    now = time.time()
    metrics = [_streaming_metric(now + i * 0.1) for i in range(20)]
    report = generate_br_v0_2(
        config=_shared_prefix_config(),
        request_metrics=metrics,
        runtime_parameters=_runtime_params(now),
        server_metrics=None,
        collected=NoopStackCollector().collect(),
    )
    BenchmarkReportV02.model_validate(report)


def test_adapter_run_block() -> None:
    now = time.time()
    metrics = [_streaming_metric(now + i * 0.1) for i in range(5)]
    report = generate_br_v0_2(
        config=_shared_prefix_config(),
        request_metrics=metrics,
        runtime_parameters=_runtime_params(now),
        server_metrics=None,
        collected=None,
    )
    run = report["run"]
    assert run["uid"]
    assert "time" in run
    assert run["description"] == "test"


def test_adapter_aggregate_request_counts_match_inputs() -> None:
    now = time.time()
    metrics: List[RequestLifecycleMetric] = [_streaming_metric(now + i * 0.1) for i in range(30)]
    metrics[2].error = ErrorResponseInfo(error_type="timeout", error_msg="boom")
    metrics[17].error = ErrorResponseInfo(error_type="timeout", error_msg="boom")
    report = generate_br_v0_2(
        config=_shared_prefix_config(),
        request_metrics=metrics,
        runtime_parameters=_runtime_params(now),
        server_metrics=None,
        collected=None,
    )
    requests = report["results"]["request_performance"]["aggregate"]["requests"]
    assert requests["total"] == 30
    assert requests["failures"] == 2


def test_adapter_latency_calculations() -> None:
    now = time.time()
    metrics = [_streaming_metric(now + i * 0.1, output_tokens=11, itl=0.02, ttft=0.05) for i in range(10)]
    report = generate_br_v0_2(
        config=_shared_prefix_config(),
        request_metrics=metrics,
        runtime_parameters=_runtime_params(now),
        server_metrics=None,
        collected=None,
    )
    latency = report["results"]["request_performance"]["aggregate"]["latency"]
    assert latency["time_to_first_token"]["mean"] == pytest.approx(0.05, abs=1e-6)
    # 10 inter-token gaps of 0.02s each, TPOT = (last - first) / (n - 1) = 0.20 / 10 = 0.02
    assert latency["time_per_output_token"]["mean"] == pytest.approx(0.02, abs=1e-6)
    assert latency["inter_token_latency"]["mean"] == pytest.approx(0.02, abs=1e-6)


def test_adapter_uses_shared_prefix_config() -> None:
    now = time.time()
    metrics = [_streaming_metric(now + i * 0.1) for i in range(5)]
    report = generate_br_v0_2(
        config=_shared_prefix_config(),
        request_metrics=metrics,
        runtime_parameters=_runtime_params(now),
        server_metrics=None,
        collected=None,
    )
    standardized = report["scenario"]["load"]["standardized"]
    assert standardized["source"] == LoadSource.RANDOM.value
    assert standardized["prefix"]["num_groups"] == 2
    assert standardized["prefix"]["num_users_per_group"] == 4
    assert standardized["prefix"]["prefix_len"]["distribution"] == Distribution.FIXED.value
    assert standardized["rate_qps"] == 10.0


def test_adapter_aggregate_units() -> None:
    """The schema enforces unit compatibility per metric category. This guards
    that the adapter assigns units the schema validators accept."""
    now = time.time()
    metrics = [_streaming_metric(now + i * 0.1) for i in range(10)]
    report = generate_br_v0_2(
        config=_shared_prefix_config(),
        request_metrics=metrics,
        runtime_parameters=_runtime_params(now),
        server_metrics=None,
        collected=None,
    )
    parsed = BenchmarkReportV02.model_validate(report)
    assert parsed.results.request_performance is not None
    agg = parsed.results.request_performance.aggregate
    assert (
        agg is not None
        and agg.requests is not None
        and agg.requests.input_length is not None
        and agg.latency is not None
        and agg.latency.time_to_first_token is not None
        and agg.latency.time_per_output_token is not None
        and agg.throughput is not None
        and agg.throughput.output_token_rate is not None
        and agg.throughput.request_rate is not None
    )
    assert agg.requests.input_length.units == Units.COUNT
    assert agg.latency.time_to_first_token.units == Units.S
    assert agg.latency.time_per_output_token.units == Units.S_PER_TOKEN
    assert agg.throughput.output_token_rate.units == Units.TOKEN_PER_S
    assert agg.throughput.request_rate.units == Units.QUERY_PER_S


def test_adapter_no_metrics_returns_empty_request_performance() -> None:
    now = time.time()
    report = generate_br_v0_2(
        config=_shared_prefix_config(),
        request_metrics=[],
        runtime_parameters=_runtime_params(now),
        server_metrics=None,
        collected=None,
    )
    BenchmarkReportV02.model_validate(report)
    assert report["results"].get("request_performance") is None
    assert report["run"].get("time") is None


def test_config_stack_collector_passes_through_overrides() -> None:
    override = BRV02StackComponentOverride(
        metadata={"label": "vllm-svc-0", "schema_version": "0.0.1", "cfg_id": "abc"},
        standardized={
            "kind": "inference_engine",
            "tool": "vllm",
            "tool_version": "vllm:0.6.0",
            "role": "replica",
            "replicas": 1,
            "model": {"name": "Qwen/Qwen3-0.6B"},
            "accelerator": {
                "model": "H100",
                "count": 1,
                "parallelism": {"tp": 1, "dp": 1, "dp_local": 1, "workers": 1},
            },
        },
        native={"args": {"--model": "Qwen/Qwen3-0.6B"}, "envars": {}},
    )
    config = _shared_prefix_config(stack=[override])
    assert config.report.br_v0_2 is not None
    collector = ConfigStackCollector(config.report.br_v0_2)
    collected = collector.collect()
    assert isinstance(collected, CollectedStackObservability)
    assert len(collected.stack) == 1
    assert collected.stack[0]["metadata"]["label"] == "vllm-svc-0"


def test_noop_collector_returns_empty() -> None:
    collected = NoopStackCollector().collect()
    assert collected.stack == []
    assert collected.observability is None
    assert collected.component_health == []
