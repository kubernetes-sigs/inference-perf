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
"""Row coverage for the ``vllm bench serve`` converter frontend.

The mapped rows check emitted config fields; the refusal table pins each
named reason from the "Where the map breaks" list, so a future edit cannot
quietly turn a refusal into a guessed value.
"""

from typing import List

import pytest

from inference_perf.config import (
    APIType,
    ConcurrentLoadStage,
    DataGenType,
    DistributionType,
    LoadType,
    StandardLoadStage,
)
from inference_perf.tools.convert import convert_vllm_bench
from inference_perf.tools.convert.model import Conversion

VERSION = "v0.10.0"

# The case-a fixture argv from the parity harness (e2e/tests/parity), plus the
# tail the harness appends to every case.
CASE_A_ARGV = [
    "--endpoint-type",
    "openai",
    "--endpoint",
    "/v1/completions",
    "--dataset-name",
    "random",
    "--num-prompts",
    "600",
    "--random-input-len",
    "128",
    "--random-output-len",
    "64",
    "--random-range-ratio",
    "0",
    "--request-rate",
    "40",
    "--ignore-eos",
]
HARNESS_TAIL = [
    "--base-url",
    "http://127.0.0.1:8000",
    "--model",
    "google/gemma-3-270m",
    "--tokenizer",
    "google/gemma-3-270m",
    "--save-result",
    "--result-filename",
    "result.json",
]


def convert(argv: List[str], version: str = VERSION) -> Conversion:
    return convert_vllm_bench(argv, version)


def test_case_a_fixed_rate_converts() -> None:
    conversion = convert(CASE_A_ARGV + HARNESS_TAIL)
    assert conversion.refusals == []
    config = conversion.config
    assert config is not None
    assert config.api.type == APIType.Completion
    assert config.api.streaming is True
    assert config.data.type == DataGenType.Random
    assert config.data.input_distribution is not None
    assert config.data.input_distribution.type == DistributionType.FIXED
    assert config.data.input_distribution.mean == 128
    assert config.data.input_distribution.min == config.data.input_distribution.max == 128
    assert config.data.output_distribution is not None
    assert config.data.output_distribution.mean == 64
    # A finite rate at burstiness 1.0 is Poisson; the parity fixture's
    # `constant` is harness-declared arrival, not derived from the args.
    assert config.load.type == LoadType.POISSON
    stage = config.load.stages[0]
    assert isinstance(stage, StandardLoadStage)
    assert stage.rate == 40
    assert stage.duration == 15
    assert config.load.base_seed == 0
    assert config.server is not None
    assert config.server.ignore_eos is True
    assert config.server.base_url == "http://127.0.0.1:8000"
    assert config.server.model_name == "google/gemma-3-270m"
    assert config.tokenizer is not None
    assert config.tokenizer.pretrained_model_name_or_path == "google/gemma-3-270m"
    # Harness-appended reporting flags must never abort a conversion.
    assert any("--save-result" in item for item in conversion.dropped)
    assert any("--result-filename" in item for item in conversion.dropped)


def test_case_b_fixed_concurrency_converts() -> None:
    argv = [
        "--endpoint-type",
        "openai",
        "--endpoint",
        "/v1/completions",
        "--dataset-name",
        "random",
        "--num-prompts",
        "48",
        "--random-input-len",
        "128",
        "--random-output-len",
        "64",
        "--random-range-ratio",
        "0",
        "--max-concurrency",
        "8",
        "--ignore-eos",
    ]
    conversion = convert(argv + HARNESS_TAIL)
    assert conversion.refusals == []
    config = conversion.config
    assert config is not None
    assert config.load.type == LoadType.CONCURRENT
    stage = config.load.stages[0]
    assert isinstance(stage, ConcurrentLoadStage)
    assert stage.num_requests == 48
    assert stage.concurrency_level == 8


def test_standing_annotations_always_present() -> None:
    conversion = convert(["--model", "m"])
    assert any(item.startswith("sampling:") for item in conversion.no_equivalents)
    assert any(item.startswith("warmup:") for item in conversion.no_equivalents)


def test_default_run_is_send_everything_at_once() -> None:
    conversion = convert(["--model", "m"])
    assert conversion.refusals == []
    config = conversion.config
    assert config is not None
    assert config.load.type == LoadType.CONCURRENT
    stage = config.load.stages[0]
    assert isinstance(stage, ConcurrentLoadStage)
    assert stage.num_requests == 1000
    assert stage.concurrency_level == 1000
    # ignore_eos must be emitted explicitly even when the flag is absent:
    # inference-perf defaults it to true, vllm does not send it.
    assert config.server is not None
    assert config.server.ignore_eos is False
    assert "server" in config.model_dump(exclude_unset=True)
    assert "ignore_eos" in config.model_dump(exclude_unset=True)["server"]


def test_host_port_compose_base_url() -> None:
    conversion = convert(["--model", "m", "--host", "10.0.0.5", "--port", "9000"])
    assert conversion.config is not None
    assert conversion.config.server is not None
    assert conversion.config.server.base_url == "http://10.0.0.5:9000"


def test_served_model_name_overrides_payload_model() -> None:
    conversion = convert(["--model", "base", "--served-model-name", "alias"])
    assert conversion.config is not None
    assert conversion.config.server is not None
    assert conversion.config.server.model_name == "alias"
    assert conversion.config.tokenizer is not None
    assert conversion.config.tokenizer.pretrained_model_name_or_path == "base"


def test_range_ratio_fans_out_to_uniform_bounds() -> None:
    argv = ["--model", "m", "--random-input-len", "100", "--random-output-len", "50", "--random-range-ratio", "0.5"]
    conversion = convert(argv)
    assert conversion.config is not None
    input_dist = conversion.config.data.input_distribution
    output_dist = conversion.config.data.output_distribution
    assert input_dist is not None and output_dist is not None
    assert input_dist.type == DistributionType.UNIFORM
    assert (input_dist.min, input_dist.max) == (50, 150)
    assert (output_dist.min, output_dist.max) == (25, 75)


def test_goodput_converts_ms_to_seconds() -> None:
    conversion = convert(["--model", "m", "--goodput", "ttft:200", "tpot:20", "e2el:1500"])
    assert conversion.config is not None
    assert conversion.config.report.goodput is not None
    assert conversion.config.report.goodput.constraints == {"ttft": 0.2, "tpot": 0.02, "request_latency": 1.5}


def test_metric_percentiles_map() -> None:
    conversion = convert(["--model", "m", "--metric-percentiles", "50,90,99"])
    assert conversion.config is not None
    assert conversion.config.report.request_lifecycle.percentiles == [50, 90, 99]


def test_lora_split_that_cannot_sum_to_one_refuses() -> None:
    # Left-fold float addition makes most equal splits sum to exactly 1.0
    # (even 3), so the guard trips only where the validator itself would.
    modules = [f"adapter{i}" for i in range(49)]
    conversion = convert(["--model", "m", "--lora-modules", *modules])
    assert conversion.config is None
    assert any("sum to exactly 1.0" in refusal for refusal in conversion.refusals)


def test_lora_modules_split_equally() -> None:
    conversion = convert(["--model", "m", "--lora-modules", "a", "b"])
    assert conversion.config is not None
    split = conversion.config.load.lora_traffic_split
    assert split is not None
    assert [adapter.name for adapter in split] == ["a", "b"]
    assert [adapter.split for adapter in split] == [0.5, 0.5]


def test_trust_remote_code_and_result_dir_map() -> None:
    conversion = convert(["--model", "m", "--trust-remote-code", "--result-dir", "out/reports"])
    assert conversion.config is not None
    assert conversion.config.tokenizer is not None
    assert conversion.config.tokenizer.trust_remote_code is True
    assert conversion.config.storage is not None
    assert conversion.config.storage.local_storage.path == "out/reports"


def test_inert_flags_drop_instead_of_refusing() -> None:
    argv = ["--model", "m", "--use-beam-search", "--backend", "openai", "--sonnet-input-len", "700", "--burstiness", "2.0"]
    conversion = convert(argv)
    assert conversion.refusals == []
    dropped = "\n".join(conversion.dropped)
    assert "--use-beam-search" in dropped
    assert "--backend" in dropped
    assert "--sonnet-input-len" in dropped  # inert with the random dataset
    assert "--burstiness" in dropped  # inert with an infinite request rate


@pytest.mark.parametrize(
    ("argv", "reason_fragment"),
    [
        (["--model", "m", "--dataset-name", "sharegpt"], "not mapped"),
        (["--model", "m", "--dataset-name", "sonnet"], "not mapped"),
        (["--model", "m", "--dataset-name", "hf"], "not mapped"),
        (["--model", "m", "--random-prefix-len", "100"], "prefix"),
        (["--model", "m", "--request-rate", "40", "--burstiness", "2.0"], "gamma"),
        (["--model", "m", "--request-rate", "40", "--max-concurrency", "8"], "rate-limited closed loop"),
        (["--model", "m", "--num-prompts", "100", "--request-rate", "7"], "not a positive integer"),
        (["--model", "m", "--ramp-up-strategy", "linear"], "ramp"),
        (["--model", "m", "--endpoint-type", "openai-audio"], "audio"),
        (["--model", "m", "--endpoint", "/custom/path"], "custom endpoint path"),
        (["--model", "m", "--logprobs", "5"], "logprobs"),
        (["--model", "m", "--temperature", "0.7"], "sampling parameters"),
        (["--model", "m", "--top-p", "0.9"], "sampling parameters"),
        (["--model", "m", "--profile"], "profiling"),
        (["--model", "m", "--tokenizer-mode", "slow"], "tokenizer-backend"),
        (["--model", "m", "--goodput", "itl:5"], "unknown goodput key"),
        (["--model", "m", "--random-range-ratio", "1.5"], "range ratio"),
        (["--model", "m", "--not-a-vllm-flag", "x"], "not in the verified"),
    ],
)
def test_refusals_carry_named_reasons(argv: List[str], reason_fragment: str) -> None:
    conversion = convert(argv)
    assert conversion.config is None
    assert conversion.refusals, f"expected a refusal for {argv}"
    assert any(reason_fragment in refusal for refusal in conversion.refusals), conversion.refusals


def test_all_refusals_are_collected_not_thrown() -> None:
    argv = ["--model", "m", "--temperature", "0.7", "--top-p", "0.9", "--profile", "--dataset-name", "sonnet"]
    conversion = convert(argv)
    assert len(conversion.refusals) == 4


def test_unverified_version_refuses() -> None:
    conversion = convert(["--model", "m"], version="v0.11.0")
    assert conversion.config is None
    assert any("not in the verified surface registry" in refusal for refusal in conversion.refusals)
    assert any("v0.10.0" in refusal for refusal in conversion.refusals)
