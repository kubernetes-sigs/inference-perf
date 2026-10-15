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
"""Row coverage for the ``aiperf profile`` converter frontend.

Includes the two default traps that make a bare aiperf run refuse loudly: an
unset ``--osl`` (aiperf sends no max_tokens field, inference-perf always
does) and a ``--num-dataset-entries`` pool smaller than the request count
(prompt reuse is not expressible).
"""

from typing import List

import pytest

from inference_perf.config import (
    APIType,
    ConcurrentLoadStage,
    DistributionType,
    LoadType,
    MetricsClientType,
    StandardLoadStage,
)
from inference_perf.tools.convert import convert_aiperf_profile
from inference_perf.tools.convert.model import Conversion

VERSION = "v0.12.0"

# The aiperf leg of the parity cases, mirroring the checked-in vllm args.
CASE_A_ARGV = [
    "--endpoint-type",
    "completions",
    "--streaming",
    "--request-count",
    "600",
    "--request-rate",
    "40",
    "--arrival-pattern",
    "constant",
    "--isl",
    "128",
    "--isl-stddev",
    "0",
    "--osl",
    "64",
    "--osl-stddev",
    "0",
    "--num-dataset-entries",
    "600",
    "--extra-inputs",
    "ignore_eos:true",
]
CASE_B_ARGV = [
    "--endpoint-type",
    "completions",
    "--streaming",
    "--concurrency",
    "8",
    "--request-count",
    "48",
    "--isl",
    "128",
    "--isl-stddev",
    "0",
    "--osl",
    "64",
    "--osl-stddev",
    "0",
    "--num-dataset-entries",
    "48",
    "--extra-inputs",
    "ignore_eos:true",
]
HARNESS_TAIL = ["--model", "google/gemma-3-270m", "--tokenizer", "google/gemma-3-270m", "--url", "http://127.0.0.1:8000"]


def convert(argv: List[str], version: str = VERSION) -> Conversion:
    return convert_aiperf_profile(argv, version)


def test_case_a_fixed_rate_converts() -> None:
    conversion = convert(CASE_A_ARGV + HARNESS_TAIL)
    assert conversion.refusals == []
    config = conversion.config
    assert config is not None
    assert config.api.type == APIType.Completion
    assert config.api.streaming is True
    # aiperf's constant arrival is evenly spaced: exactly load.type constant.
    assert config.load.type == LoadType.CONSTANT
    stage = config.load.stages[0]
    assert isinstance(stage, StandardLoadStage)
    assert stage.rate == 40
    assert stage.duration == 15
    assert config.data.input_distribution is not None
    assert config.data.input_distribution.type == DistributionType.FIXED
    assert config.data.input_distribution.mean == 128
    assert config.data.output_distribution is not None
    assert config.data.output_distribution.mean == 64
    assert config.server is not None
    assert config.server.ignore_eos is True
    assert config.server.base_url == "http://127.0.0.1:8000"
    assert config.server.model_name == "google/gemma-3-270m"


def test_case_b_fixed_concurrency_converts() -> None:
    conversion = convert(CASE_B_ARGV + HARNESS_TAIL)
    assert conversion.refusals == []
    config = conversion.config
    assert config is not None
    assert config.load.type == LoadType.CONCURRENT
    stage = config.load.stages[0]
    assert isinstance(stage, ConcurrentLoadStage)
    assert stage.num_requests == 48
    assert stage.concurrency_level == 8


def test_streaming_default_false_is_emitted_explicitly() -> None:
    conversion = convert(["--model", "m", "--osl", "64", "--num-dataset-entries", "10"])
    assert conversion.refusals == []
    config = conversion.config
    assert config is not None
    assert config.api.streaming is False
    assert config.model_dump(exclude_unset=True)["api"]["streaming"] is False
    # A bare run is a chat run: the default endpoint type differs from vllm.
    assert config.api.type == APIType.Chat


def test_default_load_is_concurrency_one_with_derived_count() -> None:
    conversion = convert(["--model", "m", "--osl", "64"])
    assert conversion.refusals == []
    config = conversion.config
    assert config is not None
    assert config.load.type == LoadType.CONCURRENT
    stage = config.load.stages[0]
    assert isinstance(stage, ConcurrentLoadStage)
    assert stage.concurrency_level == 1
    assert stage.num_requests == 10  # max(10, concurrency * 2)


def test_unset_osl_refuses() -> None:
    conversion = convert(["--model", "m"])
    assert conversion.config is None
    assert any("max_tokens" in refusal for refusal in conversion.refusals)


def test_dataset_entry_pool_smaller_than_request_count_refuses() -> None:
    conversion = convert(["--model", "m", "--osl", "64", "--concurrency", "8", "--request-count", "200"])
    assert conversion.config is None
    assert any("prompt reuse" in refusal and "default pool is 100" in refusal for refusal in conversion.refusals)


def test_poisson_rate_with_count() -> None:
    argv = ["--model", "m", "--osl", "64", "--request-rate", "10", "--request-count", "100", "--num-dataset-entries", "100"]
    conversion = convert(argv)
    assert conversion.refusals == []
    config = conversion.config
    assert config is not None
    assert config.load.type == LoadType.POISSON
    stage = config.load.stages[0]
    assert isinstance(stage, StandardLoadStage)
    assert stage.duration == 10


def test_benchmark_duration_bounds_a_rate_stage() -> None:
    argv = [
        "--model",
        "m",
        "--osl",
        "64",
        "--request-rate",
        "10",
        "--benchmark-duration",
        "30",
        "--num-dataset-entries",
        "300",
    ]
    conversion = convert(argv)
    assert conversion.refusals == []
    config = conversion.config
    assert config is not None
    stage = config.load.stages[0]
    assert isinstance(stage, StandardLoadStage)
    assert stage.duration == 30


def test_normal_isl_gets_wide_clamps_and_assumption() -> None:
    argv = ["--model", "m", "--osl", "64", "--isl", "500", "--isl-stddev", "50", "--num-dataset-entries", "10"]
    conversion = convert(argv)
    assert conversion.refusals == []
    config = conversion.config
    assert config is not None
    dist = config.data.input_distribution
    assert dist is not None
    assert dist.type == DistributionType.NORMAL
    assert dist.std_dev == 50
    assert dist.min == 1
    assert dist.max == 800  # mean + 6 sigma
    assert any("no upper clamp" in item for item in conversion.assumptions)


def test_mapped_reporting_and_server_fields() -> None:
    argv = [
        "--model",
        "m",
        "--osl",
        "64",
        "--use-server-token-count",
        "--random-seed",
        "7",
        "--api-key",
        "sk-test",
        "--header",
        "X-Team: infra",
        "--session-header",
        "X-Session-Id",
        "--goodput",
        "time_to_first_token:200",
        "--goodput",
        "request_latency:1500",
        "--workers-max",
        "4",
        "--request-timeout-seconds",
        "120",
        "--output-artifact-dir",
        "artifacts/run1",
        "--server-metrics",
        "http://localhost:9400/metrics",
    ]
    conversion = convert(argv)
    assert conversion.refusals == []
    config = conversion.config
    assert config is not None
    assert config.report.request_lifecycle.use_server_output_tokens is True
    assert config.report.goodput is not None
    assert config.report.goodput.constraints == {"ttft": 0.2, "request_latency": 1.5}
    assert config.load.base_seed == 7
    assert config.load.num_workers == 4
    assert config.load.request_timeout == 120
    assert config.server is not None
    assert config.server.api_key is not None
    assert config.server.api_key.get_secret_value() == "sk-test"
    assert config.api.headers == {"X-Team": "infra"}
    assert config.api.session_id_header_key == "X-Session-Id"
    assert config.storage is not None
    assert config.storage.local_storage.path == "artifacts/run1"
    assert config.metrics is not None
    assert config.metrics.type == MetricsClientType.PROMETHEUS


def test_warmup_flags_annotate_with_the_stage_workaround() -> None:
    argv = ["--model", "m", "--osl", "64", "--warmup-request-count", "50"]
    conversion = convert(argv)
    assert conversion.refusals == []
    assert any(item.startswith("warmup:") and "first stage" in item for item in conversion.no_equivalents)
    assert any("--warmup-request-count" in item for item in conversion.dropped)


def test_reporting_flags_drop_without_aborting() -> None:
    argv = ["--model", "m", "--osl", "64", "--ui", "dashboard", "--export-level", "records", "--gpu-telemetry", "dcgm"]
    conversion = convert(argv)
    assert conversion.refusals == []
    dropped = "\n".join(conversion.dropped)
    assert "--ui-type" in dropped
    assert "--export-level" in dropped
    assert "--gpu-telemetry" in dropped


@pytest.mark.parametrize(
    ("argv", "reason_fragment"),
    [
        (["--model", "m", "--osl", "64", "--endpoint-type", "embeddings"], "no API type"),
        (["--model", "a,b", "--osl", "64"], "multi-model"),
        (["--model", "m", "--osl", "64", "--url", "http://a:1", "--url", "http://b:2"], "load balancing"),
        (["--model", "m", "--osl", "64", "--custom-endpoint", "/my/path"], "custom endpoint"),
        (["--model", "m", "--osl", "64", "--extra-inputs", "temperature:0.7"], "sampling parameters"),
        (["--model", "m", "--osl", "64", "--extra-inputs", "logit_bias:x"], "passthrough"),
        (["--model", "m", "--osl", "64", "--concurrency", "8", "--request-rate", "5"], "rate-limited closed loop"),
        (["--model", "m", "--osl", "64", "--request-rate", "10", "--arrival-pattern", "gamma"], "gamma"),
        (["--model", "m", "--osl", "64", "--arrival-smoothness", "1.0"], "gamma shape"),
        (["--model", "m", "--osl", "64", "--concurrency", "4", "--benchmark-duration", "60"], "duration-bounded"),
        (
            ["--model", "m", "--osl", "64", "--request-rate", "10", "--request-count", "100", "--benchmark-duration", "10"],
            "whichever bound hits first",
        ),
        (["--model", "m", "--osl", "64", "--request-rate-series", "series.json"], "piecewise-linear"),
        (["--model", "m", "--osl", "64", "--concurrency-ramp-duration", "30"], "ramp"),
        (["--model", "m", "--osl", "64", "--prefill-concurrency", "2"], "admission control"),
        (["--model", "m", "--osl", "64", "--input-file", "prompts.jsonl"], "out of scope"),
        (["--model", "m", "--osl", "64", "--public-dataset", "sharegpt"], "no verified twin"),
        (["--model", "m", "--osl", "64", "--custom-dataset-type", "mooncake_trace"], "out of scope"),
        (["--model", "m", "--osl", "64", "--dataset-sampling-strategy", "shuffle"], "reuse"),
        (["--model", "m", "--osl", "64", "--config", "run.yaml"], "pass the effective flags"),
        (["--model", "m", "--osl", "64", "--scenario", "chat-heavy"], "expanded flags"),
        (["--model", "m", "--osl", "64", "--conversation-num", "20"], "session-oriented"),
        (["--model", "m", "--osl", "64", "--conversation-turn-mean", "3"], "multi-turn"),
        (["--model", "m", "--osl", "64", "--cache-bust", "conversation"], "prefix-cache"),
        (["--model", "m", "--osl", "64", "--prompt-prefix-pool-size", "8", "--prompt-prefix-length", "64"], "cache-hit"),
        (["--model", "m", "--osl", "64", "--shared-system-prompt-length", "200"], "shared or per-session context"),
        (["--model", "m", "--osl", "64", "--isl-block-size", "16"], "prefix-cache"),
        (["--model", "m", "--osl", "64", "--seq-dist", "128,64:0.5;256,32:0.5"], "pair mixture"),
        (["--model", "m", "--osl", "64", "--tokenizer", "builtin"], "different vocabulary"),
        (["--model", "m", "--osl", "64", "--tokenizer-revision", "abc123"], "revision"),
        (["--model", "m", "--osl", "64", "--apply-chat-template"], "completions path"),
        (["--model", "m", "--osl", "64", "--connection-reuse-strategy", "never"], "connection"),
        (["--model", "m", "--osl", "64", "--goodput", "output_token_throughput_per_user:100"], "goodput tag"),
        (["--model", "m", "--osl", "64", "--benchmark-grace-period", "inf"], "infinite grace"),
        (["--model", "m", "--osl", "64", "--fixed-schedule"], "timestamp replay"),
        (["--model", "m", "--osl", "64", "--user-centric-rate", "2"], "load model outside"),
        (["--model", "m", "--osl", "64", "--request-cancellation-rate", "0.1"], "cancellations"),
        (["--model", "m", "--osl", "64", "--sweep-type", "grid"], "one config for one run"),
        (["--model", "m", "--osl", "64", "--num-profile-runs", "5"], "one config for one run"),
        (["--model", "m", "--osl", "64", "--accuracy-benchmark", "mmlu"], "no inference-perf counterpart"),
        (["--model", "m", "--osl", "64", "--synthesis-max-isl", "4096"], "agentic trace synthesis"),
        (["--model", "m", "--osl", "64", "--rankings-passages-mean", "10"], "rankings"),
        (["--model", "m", "--osl", "64", "--video-width", "640", "--video-height", "360"], "webm"),
        (["--model", "m", "--osl", "64", "--audio-length-mean", "5", "--audio-length-stddev", "2"], "weighted list"),
        (["--model", "m", "--osl", "64", "--not-an-aiperf-flag", "x"], "not in the verified"),
    ],
)
def test_refusals_carry_named_reasons(argv: List[str], reason_fragment: str) -> None:
    conversion = convert(argv)
    assert conversion.config is None
    assert conversion.refusals, f"expected a refusal for {argv}"
    assert any(reason_fragment in refusal for refusal in conversion.refusals), conversion.refusals


def test_multimodal_image_maps_when_pointlike() -> None:
    argv = [
        "--model",
        "m",
        "--osl",
        "64",
        "--image-width-mean",
        "512",
        "--image-height-mean",
        "512",
        "--image-format",
        "jpeg",
        "--image-batch-size",
        "2",
    ]
    conversion = convert(argv)
    assert conversion.refusals == []
    config = conversion.config
    assert config is not None
    assert config.data.multimodal is not None
    image = config.data.multimodal.image
    assert image is not None
    assert image.count is not None and image.count.mean == 2
    assert image.resolutions is not None


def test_video_mp4_maps_with_frame_derivation() -> None:
    argv = [
        "--model",
        "m",
        "--osl",
        "64",
        "--video-width",
        "640",
        "--video-height",
        "360",
        "--video-format",
        "mp4",
        "--video-duration",
        "5",
        "--video-fps",
        "4",
    ]
    conversion = convert(argv)
    assert conversion.refusals == []
    config = conversion.config
    assert config is not None
    assert config.data.multimodal is not None
    video = config.data.multimodal.video
    assert video is not None
    assert any("profiles.frames" in item and "20" in item for item in conversion.assumptions)


def test_unverified_version_refuses() -> None:
    conversion = convert(["--model", "m", "--osl", "64"], version="0.11.0")
    assert conversion.config is None
    assert any("not in the verified surface registry" in refusal for refusal in conversion.refusals)
