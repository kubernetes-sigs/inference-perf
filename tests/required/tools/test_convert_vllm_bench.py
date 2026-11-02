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

import argparse
from typing import List, Optional, Tuple

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
from inference_perf.tools.convert.model import Conversion, PeerUsageError
from inference_perf.tools.convert.versions import VLLM_BENCH_VERSIONS
from inference_perf.tools.convert.vllm_bench import _Mirror

# The rows below this pin v0.10.0 and stay on it; the tag-difference tests at
# the end of the file run both tags.
VERSION = "v0.10.0"
PIN = "v0.32.0"
SAMPLING_REFUSAL = (
    "sampling parameters are not configurable for synthetic data; a guessed value would misstate the decode work"
)

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


@pytest.mark.parametrize("tail", [["--bogus", "-1"], ["--bogus", "x"], ["--bogus=x"], ["--bogus"]])
def test_unknown_flag_refuses_once(tail: List[str]) -> None:
    # argparse hands an unknown flag and its value back as two tokens; the
    # value, a negative number included, must not refuse a second time.
    conversion = convert(["--model", "m"] + tail)
    assert conversion.config is None
    assert conversion.refusals == ["--bogus: not in the verified vllm bench serve v0.10.0 flag surface"]


def test_stray_positional_refuses_as_positional() -> None:
    conversion = convert(["--model", "m", "stray"])
    assert conversion.refusals == ["stray: vllm bench serve takes no positional arguments"]


def test_unverified_version_refuses() -> None:
    conversion = convert(["--model", "m"], version="v0.11.0")
    assert conversion.config is None
    assert any("not in the verified surface registry" in refusal for refusal in conversion.refusals)
    assert any("v0.10.0" in refusal for refusal in conversion.refusals)


# ---------------------------------------------------------------------------
# Rows whose meaning differs between v0.10.0 and v0.32.0. Each runs both tags.


# Builds a v0.32.0 argv from a v0.10.0 one by swapping --endpoint-type for --backend.
# ["--endpoint-type", "openai", "--num-prompts", "6"] -> ["--backend", "openai", "--num-prompts", "6"].
def _on_pin(argv: List[str]) -> List[str]:
    return ["--backend" if token == "--endpoint-type" else token for token in argv]


# The mirrored parsers hold 55 flags at v0.10.0 and 101 at v0.32.0, the counts read
# from each tag's add_dataset_parser and add_cli_args, and the request-path flag
# offers exactly the endpoint_types registered for that tag.
@pytest.mark.parametrize(("version", "count", "selector"), [("v0.10.0", 55, "endpoint_type"), ("v0.32.0", 101, "backend")])
def test_mirror_surface_matches_the_tag(version: str, count: int, selector: str) -> None:
    mirror = _Mirror(version)
    assert len(mirror.dest_to_flag) == count
    choices = next(action.choices for action in mirror.parser._actions if action.dest == selector)
    assert choices is not None
    assert frozenset(choices) == VLLM_BENCH_VERSIONS[version].endpoint_types


# Case a (600 prompts, 128 in, 64 out, 40/s) converts to the same config on both tags:
# --endpoint-type openai at v0.10.0 and --backend openai at v0.32.0.
def test_case_a_converts_the_same_on_both_tags() -> None:
    old = convert(CASE_A_ARGV + HARNESS_TAIL, "v0.10.0")
    new = convert(_on_pin(CASE_A_ARGV) + HARNESS_TAIL, PIN)
    assert old.refusals == [] and new.refusals == []
    assert old.config is not None and new.config is not None
    assert new.config.model_dump() == old.config.model_dump()


# --endpoint-type picks the request path at v0.10.0, where --backend is dropped and the
# path stays completions. At v0.32.0 --backend picks it and --endpoint-type is unknown.
@pytest.mark.parametrize(
    ("version", "argv", "api_type", "refusal"),
    [
        ("v0.10.0", ["--endpoint-type", "openai-chat", "--endpoint", "/v1/chat/completions"], APIType.Chat, None),
        ("v0.10.0", ["--backend", "openai-chat"], APIType.Completion, None),
        ("v0.32.0", ["--backend", "openai-chat", "--endpoint", "/v1/chat/completions"], APIType.Chat, None),
        ("v0.32.0", ["--backend", "vllm"], APIType.Completion, None),
        ("v0.32.0", ["--endpoint-type", "openai"], None, "--endpoint-type: not in the verified vllm bench serve v0.32.0"),
    ],
)
def test_request_path_flag_per_tag(version: str, argv: List[str], api_type: Optional[APIType], refusal: Optional[str]) -> None:
    conversion = convert(["--model", "m", *argv], version)
    if refusal is not None:
        assert conversion.config is None
        assert any(item.startswith(refusal) for item in conversion.refusals), conversion.refusals
        return
    assert conversion.refusals == []
    assert conversion.config is not None
    assert conversion.config.api.type == api_type


# At v0.32.0 the Responses, audio and pooling backends refuse with --backend as the subject.
# --backend openai-embeddings -> "--backend: pooling backend 'openai-embeddings' is not mapped ...".
@pytest.mark.parametrize(
    ("backend", "fragment"),
    [
        ("openai-responses", "Responses API"),
        ("openai-audio", "openai-audio has no API type"),
        ("openai-embeddings", "pooling backend 'openai-embeddings'"),
        ("vllm-rerank", "pooling backend 'vllm-rerank'"),
    ],
)
def test_pin_backends_without_an_api_type_refuse(backend: str, fragment: str) -> None:
    conversion = convert(["--model", "m", "--backend", backend], PIN)
    assert conversion.config is None
    assert any(item.startswith("--backend:") and fragment in item for item in conversion.refusals), conversion.refusals


# v0.10.0 keeps its exact sampling and warmup lines. At v0.32.0 a default completions run
# notes only repetition_penalty=1.0 and has no warmup line, and a chat run has no sampling line.
def test_standing_notes_per_tag() -> None:
    old = convert(["--model", "m"], "v0.10.0")
    assert old.no_equivalents == [
        "sampling: vllm v0.10.0 sends temperature=0.0 and repetition_penalty=1.0 on every request;"
        " this config takes the server's sampling defaults",
        "warmup: vllm bench always sends 1 untunable test request before the run;"
        " inference-perf measures every request it sends",
    ]
    new = convert(["--model", "m"], PIN)
    assert new.no_equivalents == [
        "sampling: vllm v0.32.0 sends repetition_penalty=1.0 on every completions request and leaves temperature"
        " to the server; this config takes the server's sampling defaults"
    ]
    chat = convert(["--model", "m", "--backend", "openai-chat", "--endpoint", "/v1/chat/completions"], PIN)
    assert chat.no_equivalents == []


# v0.32.0 counts warmups from its flags: --num-warmups 2 is 2 requests, a ready check
# is at least 1 (it retries until the endpoint answers), and both together at least 3.
@pytest.mark.parametrize(
    ("extra", "text"),
    [
        (["--num-warmups", "2"], "sends 2 unmeasured requests before"),
        (["--ready-check-timeout-sec", "60"], "sends at least 1 unmeasured request before"),
        (["--ready-check-timeout-sec", "60", "--num-warmups", "2"], "sends at least 3 unmeasured requests before"),
    ],
)
def test_pin_warmup_count_follows_flags(extra: List[str], text: str) -> None:
    conversion = convert(["--model", "m", *extra], PIN)
    assert conversion.refusals == []
    assert any(item.startswith("warmup:") and text in item for item in conversion.no_equivalents), conversion.no_equivalents


# Without --ignore-eos, v0.10.0 sends no ignore_eos, so the config pins it false. v0.32.0
# sets ignore_eos on every random-dataset request, so the config pins it true.
@pytest.mark.parametrize(("version", "expected"), [("v0.10.0", False), ("v0.32.0", True)])
def test_ignore_eos_without_the_flag_per_tag(version: str, expected: bool) -> None:
    conversion = convert(["--model", "m"], version)
    assert conversion.config is not None
    assert conversion.config.server is not None
    assert conversion.config.server.ignore_eos is expected


# --random-input-len 101 --random-output-len 1 --random-range-ratio 0.5. v0.10.0 truncates:
# input [50, 151], output [0, 1]. v0.32.0 rounds out and floors the output at 1: [50, 152], [1, 2].
@pytest.mark.parametrize(
    ("version", "input_bounds", "output_bounds"),
    [("v0.10.0", (50, 151), (0, 1)), ("v0.32.0", (50, 152), (1, 2))],
)
def test_range_ratio_bounds_per_tag(version: str, input_bounds: Tuple[int, int], output_bounds: Tuple[int, int]) -> None:
    argv = ["--model", "m", "--random-input-len", "101", "--random-output-len", "1", "--random-range-ratio", "0.5"]
    conversion = convert(argv, version)
    assert conversion.config is not None
    input_dist = conversion.config.data.input_distribution
    output_dist = conversion.config.data.output_distribution
    assert input_dist is not None and output_dist is not None
    assert input_dist.type == output_dist.type == DistributionType.UNIFORM
    assert (input_dist.min, input_dist.max) == input_bounds
    assert (output_dist.min, output_dist.max) == output_bounds


# --random-output-len 0 at the default ratio: v0.10.0 maps a fixed 0, v0.32.0 raises the
# output to a fixed 1 and records why.
@pytest.mark.parametrize(("version", "expected"), [("v0.10.0", 0), ("v0.32.0", 1)])
def test_zero_output_len_per_tag(version: str, expected: int) -> None:
    conversion = convert(["--model", "m", "--random-output-len", "0"], version)
    assert conversion.config is not None
    output_dist = conversion.config.data.output_distribution
    assert output_dist is not None
    assert output_dist.type == DistributionType.FIXED
    assert output_dist.min == output_dist.max == expected
    assert any("floor to 1 token" in item for item in conversion.assumptions) is (version == PIN)


# '{"input": 0.5, "output": 0}' with lengths 100 and 50: v0.32.0 gives input uniform
# [50, 150] and a fixed 50 output. v0.10.0 reads the ratio as a float, so it is a usage error.
def test_range_ratio_per_direction_json() -> None:
    argv = ["--model", "m", "--random-input-len", "100", "--random-output-len", "50"]
    ratio = ["--random-range-ratio", '{"input": 0.5, "output": 0}']
    conversion = convert(argv + ratio, PIN)
    assert conversion.config is not None
    input_dist = conversion.config.data.input_distribution
    output_dist = conversion.config.data.output_distribution
    assert input_dist is not None and output_dist is not None
    assert (input_dist.type, input_dist.min, input_dist.max) == (DistributionType.UNIFORM, 50, 150)
    assert (output_dist.type, output_dist.min, output_dist.max) == (DistributionType.FIXED, 50, 50)
    assert any("input ratio to 0.5 and the output ratio to 0" in item for item in conversion.assumptions)
    with pytest.raises(PeerUsageError):
        convert(argv + ratio, "v0.10.0")


# At v0.32.0 a JSON ratio out of [0, 1) refuses like a plain one, and a JSON value that
# is not an object with input and output keys is a usage error, as it is for vllm.
@pytest.mark.parametrize(
    ("ratio", "refuses"),
    [('{"input": 0.2, "output": 1.5}', True), ('{"input": 0.2}', False), ("[0.2, 0.3]", False), ("wide", False)],
)
def test_pin_range_ratio_json_errors(ratio: str, refuses: bool) -> None:
    argv = ["--model", "m", "--random-range-ratio", ratio]
    if refuses:
        conversion = convert(argv, PIN)
        assert conversion.refusals == ["--random-range-ratio: vllm v0.32.0 requires a range ratio in [0, 1), got 1.5"]
        return
    with pytest.raises(PeerUsageError, match="--random-range-ratio"):
        convert(argv, PIN)


# No --model: a usage error at v0.10.0, where it is required. v0.32.0 reads the first
# /v1/models entry, so the config leaves model_name and the tokenizer to auto-detection
# and drops --served-model-name, which vllm never reads without --model.
def test_model_is_optional_only_on_pin() -> None:
    with pytest.raises(PeerUsageError, match="--model"):
        convert(["--dataset-name", "random"], "v0.10.0")
    conversion = convert(["--served-model-name", "alias"], PIN)
    assert conversion.refusals == []
    assert conversion.config is not None
    assert conversion.config.server is not None
    assert conversion.config.server.model_name is None
    assert conversion.config.tokenizer is None
    assert any(item.startswith("--served-model-name (unread without --model") for item in conversion.dropped)
    assert any(item.startswith("no --model:") for item in conversion.assumptions)


# No --model but --tokenizer t at v0.32.0: the payload model is auto-detected and the
# tokenizer is pinned to t.
def test_pin_tokenizer_without_model() -> None:
    conversion = convert(["--tokenizer", "t"], PIN)
    assert conversion.config is not None
    assert conversion.config.tokenizer is not None
    assert conversion.config.tokenizer.pretrained_model_name_or_path == "t"


# --tokenizer-mode auto: v0.10.0 records the fast-tokenizer assumption, v0.32.0 records
# that auto may load mistral_common instead.
@pytest.mark.parametrize(("version", "fragment"), [("v0.10.0", "fast tokenizer is available"), ("v0.32.0", "mistral_common")])
def test_tokenizer_mode_auto_note_per_tag(version: str, fragment: str) -> None:
    conversion = convert(["--model", "m", "--tokenizer-mode", "auto"], version)
    assert conversion.refusals == []
    assert any(item.startswith("--tokenizer-mode auto") and fragment in item for item in conversion.assumptions)


# --host ::1 --port 9000: v0.10.0 composes http://::1:9000 the way it did, v0.32.0
# brackets the IPv6 host to http://[::1]:9000. An IPv4 host is the same on both.
@pytest.mark.parametrize(
    ("version", "host", "url"),
    [
        ("v0.10.0", "::1", "http://::1:9000"),
        ("v0.32.0", "::1", "http://[::1]:9000"),
        ("v0.32.0", "10.0.0.5", "http://10.0.0.5:9000"),
    ],
)
def test_host_port_url_per_tag(version: str, host: str, url: str) -> None:
    conversion = convert(["--model", "m", "--host", host, "--port", "9000"], version)
    assert conversion.config is not None
    assert conversion.config.server is not None
    assert conversion.config.server.base_url == url


# --max-concurrency 0 is a usage error at v0.32.0, where main_async rejects it.
def test_pin_rejects_non_positive_max_concurrency() -> None:
    with pytest.raises(PeerUsageError, match="--max-concurrency must be greater than 0"):
        convert(["--model", "m", "--max-concurrency", "0"], PIN)


# --request-rate 40 --burstiness inf: v0.10.0 refuses it as gamma spacing, v0.32.0 as
# exact 1/rate spacing, which inference-perf's randomized constant load does not produce.
@pytest.mark.parametrize(("version", "fragment"), [("v0.10.0", "gamma-spaced"), ("v0.32.0", "exactly 1/rate apart")])
def test_infinite_burstiness_per_tag(version: str, fragment: str) -> None:
    conversion = convert(["--model", "m", "--request-rate", "40", "--burstiness", "inf"], version)
    assert conversion.config is None
    assert any(item.startswith("--burstiness:") and fragment in item for item in conversion.refusals)


# --dataset-path x.json with the random dataset: dropped as inert at v0.10.0, a usage
# error at v0.32.0. --self-timed, --no-self-timed and --skip-tokenizer-init are usage errors too.
def test_dataset_path_with_random_per_tag() -> None:
    old = convert(["--model", "m", "--dataset-path", "x.json"], "v0.10.0")
    assert old.refusals == []
    assert "--dataset-path (inert with the random dataset at v0.10.0)" in old.dropped
    for extra in (["--dataset-path", "x.json"], ["--self-timed"], ["--no-self-timed"], ["--skip-tokenizer-init"]):
        with pytest.raises(PeerUsageError, match="rejects .* with the random dataset"):
            convert(["--model", "m", *extra], PIN)


# --input-len 64 and --output-len 32 replace --random-input-len 128 and
# --random-output-len 16 at v0.32.0: fixed 64 in, fixed 32 out.
def test_pin_general_lengths_override_random_lengths() -> None:
    argv = [
        "--model",
        "m",
        "--random-input-len",
        "128",
        "--random-output-len",
        "16",
        "--input-len",
        "64",
        "--output-len",
        "32",
    ]
    conversion = convert(argv, PIN)
    assert conversion.config is not None
    input_dist = conversion.config.data.input_distribution
    output_dist = conversion.config.data.output_distribution
    assert input_dist is not None and output_dist is not None
    assert (input_dist.min, input_dist.max) == (64, 64)
    assert (output_dist.min, output_dist.max) == (32, 32)


# v0.32.0 records that it re-cuts random prompts to the server's /tokenize count;
# v0.10.0 has no such step.
@pytest.mark.parametrize(("version", "expected"), [("v0.10.0", False), ("v0.32.0", True)])
def test_server_prompt_alignment_note_per_tag(version: str, expected: bool) -> None:
    conversion = convert(["--model", "m"], version)
    assert any("/tokenize" in item for item in conversion.assumptions) is expected


# Flags that only exist at v0.32.0 and change the workload refuse with a named reason.
# --probe-request-rate 1 -> "--probe-request-rate: single-token probes ...".
@pytest.mark.parametrize(
    ("argv", "reason_fragment"),
    [
        (["--probe-request-rate", "1"], "probes"),
        (["--header", "x-a=1"], "headers"),
        (["--extra-body", '{"top_p": 0.5}'], "body fields"),
        (["--lora-modules", "a", "b", "--lora-assignment", "round-robin"], "round-robin"),
        (["--random-batch-size", "2"], "packed into one request"),
        (["--frequency-penalty", "0.5"], "sampling parameters"),
        (["--presence-penalty", "0.5"], "sampling parameters"),
        (["--repetition-penalty", "1.1"], "sampling parameters"),
        (["--dataset-name", "prefix_repetition"], "not mapped"),
        (["--dataset-name", "timed_trace"], "not mapped"),
        (["--dataset-name", "random-mm"], "not mapped"),
        (["--sonnet-input-len", "10"], "not in the verified vllm bench serve v0.32.0"),
        (["--custom-skip-chat-template"], "not in the verified vllm bench serve v0.32.0"),
    ],
)
def test_pin_refusals_carry_named_reasons(argv: List[str], reason_fragment: str) -> None:
    conversion = convert(["--model", "m", *argv], PIN)
    assert conversion.config is None
    assert any(reason_fragment in item for item in conversion.refusals), conversion.refusals


# --dataset-name sonnet is a choice at v0.10.0 (refused as unmapped) and not one at
# v0.32.0, where vllm itself rejects it, so it is a usage error there.
def test_sonnet_dataset_per_tag() -> None:
    assert any("not mapped" in item for item in convert(["--model", "m", "--dataset-name", "sonnet"], "v0.10.0").refusals)
    with pytest.raises(PeerUsageError, match="invalid choice"):
        convert(["--model", "m", "--dataset-name", "sonnet"], PIN)


# Flags for other datasets, reporting and transport drop at v0.32.0 instead of refusing:
# --hf-name x -> "--hf-name (inert with the random dataset at v0.32.0)", --plot-timeline -> "--plot-timeline".
def test_pin_inert_flags_drop_instead_of_refusing() -> None:
    argv = [
        "--model",
        "m",
        "--hf-name",
        "x",
        "--prefix-repetition-prefix-len",
        "64",
        "--disable-shuffle",
        "--chat-template-kwargs",
        '{"thinking": true}',
        "--random-mm-limit-mm-per-prompt",
        '{"image": 1}',
        "--plot-timeline",
        "--timeline-itl-thresholds",
        "10,20",
        "--insecure",
        "--request-id-prefix",
        "run-",
        "--lora-assignment",
        "round-robin",
    ]
    conversion = convert(argv, PIN)
    assert conversion.refusals == []
    dropped = "\n".join(conversion.dropped)
    for fragment in (
        "--hf-name (inert with the random dataset at v0.32.0)",
        "--prefix-repetition-prefix-len (inert",
        "--disable-shuffle (inert",
        "--chat-template-kwargs (inert",
        "--random-mm-limit-mm-per-prompt (inert",
        "--plot-timeline",
        "--timeline-itl-thresholds",
        "--insecure",
        "--request-id-prefix",
        "--lora-assignment (inert without --lora-modules)",
    ):
        assert fragment in dropped, fragment


# One mirrored flag as text, for pinning a whole surface: --seed parses as
# "--seed Store type=int default=0". Choices, nargs, required and metavar appear only when set.
def _describe(action: argparse.Action) -> str:
    parts = [" ".join(action.option_strings), type(action).__name__.lstrip("_").replace("Action", "")]
    if action.type is not None:
        parts.append(f"type={getattr(action.type, '__name__', action.type)}")
    parts.append(f"default={action.default!r}")
    if action.choices is not None:
        parts.append(f"choices={list(action.choices)}")
    if action.nargs is not None and not isinstance(action, (argparse._StoreTrueAction, argparse.BooleanOptionalAction)):
        parts.append(f"nargs={action.nargs}")
    if action.required:
        parts.append("required")
    if action.metavar is not None:
        parts.append(f"metavar={action.metavar}")
    return " ".join(parts)


# add_dataset_parser then add_cli_args at each tag, read from the vllm source of that
# tag. The v0.32.0 placeholders for flags only other datasets read (--request-id-prefix,
# --random-mm-limit-mm-per-prompt, --random-mm-bucket-config) are the one departure.
SURFACES = {
    "v0.10.0": [
        "--seed Store type=int default=0",
        "--num-prompts Store type=int default=1000",
        "--dataset-name Store type=str default='random' choices=['sharegpt', 'burstgpt', 'sonnet', 'random', 'hf', 'custom']",
        "--no-stream StoreTrue default=False",
        "--dataset-path Store type=str default=None",
        "--custom-output-len Store type=int default=256",
        "--custom-skip-chat-template StoreTrue default=False",
        "--sonnet-input-len Store type=int default=550",
        "--sonnet-output-len Store type=int default=150",
        "--sonnet-prefix-len Store type=int default=200",
        "--sharegpt-output-len Store type=int default=None",
        "--random-input-len Store type=int default=1024",
        "--random-output-len Store type=int default=128",
        "--random-range-ratio Store type=float default=0.0",
        "--random-prefix-len Store type=int default=0",
        "--hf-subset Store type=str default=None",
        "--hf-split Store type=str default=None",
        "--hf-output-len Store type=int default=None",
        "--endpoint-type Store type=str default='openai' choices=['vllm', 'openai', 'openai-chat', 'openai-audio']",
        "--label Store type=str default=None",
        "--backend Store type=str default='vllm' choices=['vllm', 'openai', 'openai-chat', 'openai-audio']",
        "--base-url Store type=str default=None",
        "--host Store type=str default='127.0.0.1'",
        "--port Store type=int default=8000",
        "--endpoint Store type=str default='/v1/completions'",
        "--max-concurrency Store type=int default=None",
        "--model Store type=str default=None required",
        "--tokenizer Store type=str default=None",
        "--use-beam-search StoreTrue default=False",
        "--logprobs Store type=int default=None",
        "--request-rate Store type=float default=inf",
        "--burstiness Store type=float default=1.0",
        "--trust-remote-code StoreTrue default=False",
        "--disable-tqdm StoreTrue default=False",
        "--profile StoreTrue default=False",
        "--save-result StoreTrue default=False",
        "--save-detailed StoreTrue default=False",
        "--append-result StoreTrue default=False",
        "--metadata Store default=None nargs=* metavar=KEY=VALUE",
        "--result-dir Store type=str default=None",
        "--result-filename Store type=str default=None",
        "--ignore-eos StoreTrue default=False",
        "--percentile-metrics Store type=str default='ttft,tpot,itl'",
        "--metric-percentiles Store type=str default='99'",
        "--goodput Store default=None nargs=+",
        "--top-p Store type=float default=None",
        "--top-k Store type=int default=None",
        "--min-p Store type=float default=None",
        "--temperature Store type=float default=None",
        "--tokenizer-mode Store type=str default='auto' choices=['auto', 'slow', 'mistral', 'custom']",
        "--served-model-name Store type=str default=None",
        "--lora-modules Store default=None nargs=+",
        "--ramp-up-strategy Store type=str default=None choices=['linear', 'exponential']",
        "--ramp-up-start-rps Store type=int default=None",
        "--ramp-up-end-rps Store type=int default=None",
    ],
    "v0.32.0": [
        "--trust-remote-code StoreTrue default=False",
        "--seed Store type=int default=0",
        "--num-prompts Store type=int default=1000",
        "--dataset-name Store type=str default='random' choices=['sharegpt', 'burstgpt', 'random', 'random-mm', 'random-rerank', 'hf', 'custom', 'custom_audio', 'custom_image', 'prefix_repetition', 'spec_bench', 'speed_bench', 'timed_trace']",
        "--no-stream StoreTrue default=False",
        "--dataset-path Store type=str default=None",
        "--no-oversample StoreTrue default=False",
        "--skip-chat-template StoreTrue default=False",
        "--enable-multimodal-chat StoreTrue default=False",
        "--disable-shuffle StoreTrue default=False",
        "--custom-output-len Store type=int default=256",
        "--custom-ensure-client-side-data StoreTrue default=False",
        "--spec-bench-output-len Store type=int default=256",
        "--spec-bench-category Store type=str default=None",
        "--sharegpt-output-len Store type=int default=None",
        "--timed-trace-chunk-hash-size Store type=int default=16",
        "--timed-trace-sec-multiplier Store type=float default=1",
        "--timed-trace-label-timestamp Store type=str default='timestamp'",
        "--timed-trace-label-input-length Store type=str default='input_length'",
        "--timed-trace-label-output-length Store type=str default='output_length'",
        "--timed-trace-label-hash-ids Store type=str default='hash_ids'",
        "--blazedit-min-distance Store type=float default=0.0",
        "--blazedit-max-distance Store type=float default=1.0",
        "--asr-max-audio-len-sec Store type=float default=inf",
        "--asr-min-audio-len-sec Store type=float default=0.0",
        "--random-input-len Store type=int default=1024",
        "--random-output-len Store type=int default=128",
        "--random-range-ratio Store type=str default='0.0'",
        "--random-prefix-len Store type=int default=0",
        "--random-batch-size Store type=int default=1",
        "--no-reranker StoreTrue default=False",
        "--random-mm-base-items-per-request Store type=int default=1",
        "--random-mm-num-mm-items-range-ratio Store type=float default=0.0",
        "--random-mm-limit-mm-per-prompt Store type=loads default=None",
        "--random-mm-bucket-config Store type=str default=None",
        "--hf-subset Store type=str default=None",
        "--hf-split Store type=str default=None",
        "--hf-name Store type=str default=None",
        "--hf-output-len Store type=int default=None",
        "--bfcl-categories Store type=str default=None",
        "--prefix-repetition-prefix-len Store type=int default=256",
        "--prefix-repetition-suffix-len Store type=int default=256",
        "--prefix-repetition-num-prefixes Store type=int default=10",
        "--prefix-repetition-output-len Store type=int default=128",
        "--speed-bench-dataset-subset Store type=str default='qualitative' choices=['qualitative', 'throughput_1k', 'throughput_2k', 'throughput_8k', 'throughput_16k', 'throughput_32k']",
        "--speed-bench-output-len Store type=int default=4096",
        "--speed-bench-category Store type=str default=None",
        "--label Store type=str default=None",
        "--backend Store type=str default='openai' choices=['vllm', 'openai', 'openai-chat', 'openai-responses', 'openai-audio', 'openai-embeddings', 'openai-embeddings-chat', 'openai-embeddings-clip', 'openai-embeddings-vlm2vec', 'infinity-embeddings', 'infinity-embeddings-clip', 'vllm-pooling', 'vllm-rerank']",
        "--base-url Store type=str default=None",
        "--host Store type=str default='127.0.0.1'",
        "--port Store type=int default=8000",
        "--endpoint Store type=str default='/v1/completions'",
        "--header Store default=None nargs=* metavar=KEY=VALUE",
        "--max-concurrency Store type=int default=None",
        "--model Store type=str default=None",
        "--input-len Store type=int default=None",
        "--output-len Store type=int default=None",
        "--tokenizer Store type=str default=None",
        "--tokenizer-mode Store type=str default='auto'",
        "--use-beam-search StoreTrue default=False",
        "--logprobs Store type=int default=None",
        "--request-rate Store type=float default=inf",
        "--burstiness Store type=float default=1.0",
        "--probe-request-rate Store type=float default=0.0",
        "--disable-tqdm StoreTrue default=False",
        "--num-warmups Store type=int default=0",
        "--profile StoreTrue default=False",
        "--save-result StoreTrue default=False",
        "--save-detailed StoreTrue default=False",
        "--append-result StoreTrue default=False",
        "--metadata Store default=None nargs=* metavar=KEY=VALUE",
        "--result-dir Store type=str default=None",
        "--result-filename Store type=str default=None",
        "--ignore-eos StoreTrue default=False",
        "--self-timed --no-self-timed BooleanOptional default=None",
        "--percentile-metrics Store type=str default=None",
        "--metric-percentiles Store type=str default='99'",
        "--goodput Store default=None nargs=+",
        "--request-id-prefix Store type=str default=''",
        "--top-p Store type=float default=None",
        "--top-k Store type=int default=None",
        "--min-p Store type=float default=None",
        "--temperature Store type=float default=None",
        "--frequency-penalty Store type=float default=None",
        "--presence-penalty Store type=float default=None",
        "--repetition-penalty Store type=float default=None",
        "--served-model-name Store type=str default=None",
        "--lora-modules Store default=None nargs=+",
        "--lora-assignment Store type=str default='random' choices=['random', 'round-robin']",
        "--ramp-up-strategy Store type=str default=None choices=['linear', 'exponential']",
        "--ramp-up-start-rps Store type=int default=None",
        "--ramp-up-end-rps Store type=int default=None",
        "--ready-check-timeout-sec Store type=int default=0",
        "--chat-template-kwargs Store type=loads default=None",
        "--extra-body Store type=loads default=None",
        "--skip-tokenizer-init StoreTrue default=False",
        "--insecure StoreTrue default=False",
        "--plot-timeline StoreTrue default=False",
        "--timeline-itl-thresholds Store type=str default='25,50'",
        "--plot-dataset-stats StoreTrue default=False",
    ],
}


# Every flag of both mirrored parsers, in source order, with its type, default and
# choices: 55 rows at v0.10.0 and 101 at v0.32.0. A flag renamed or a default changed in
# the mirror fails here, the same way a vllm bump that moves the surface should.
@pytest.mark.parametrize("version", sorted(SURFACES))
def test_mirror_surface_is_pinned(version: str) -> None:
    assert [_describe(action) for action in _Mirror(version).parser._actions] == SURFACES[version]


# A v0.32.0 chat run with every mapped or dropped v0.32.0 row at once: --input-len 64,
# --output-len 32, a JSON ratio, 2 warmups plus a ready check, 2 LoRA adapters,
# --max-concurrency 4 and the reporting flags. Every recorded line is pinned word for word.
def test_pin_mapped_run_records_every_line() -> None:
    argv = [
        "--model",
        "m",
        "--served-model-name",
        "s",
        "--backend",
        "openai-chat",
        "--endpoint",
        "/v1/chat/completions",
        "--input-len",
        "64",
        "--output-len",
        "32",
        "--random-range-ratio",
        '{"input": 0.5, "output": 0.25}',
        "--num-warmups",
        "2",
        "--ready-check-timeout-sec",
        "5",
        "--tokenizer-mode",
        "auto",
        "--lora-modules",
        "a",
        "b",
        "--max-concurrency",
        "4",
        "--use-beam-search",
        "--plot-timeline",
        "--plot-dataset-stats",
        "--insecure",
        "--timeline-itl-thresholds",
        "1,2",
        "--request-id-prefix",
        "p",
        "--label",
        "L",
        "--hf-name",
        "x",
        "--no-stream",
    ]
    conversion = convert(argv, PIN)
    assert conversion.refusals == []
    assert conversion.assumptions == [
        "--tokenizer-mode auto at v0.32.0 loads mistral_common for Mistral models when it is installed and the HF fast"
        " tokenizer otherwise; inference-perf always loads the HF tokenizer",
        "chat path: vllm wraps the raw prompt as a single user message and does not model chat-template overhead,"
        " so prompt-length accounting differs from the completions path",
        "--input-len 64 replaces --random-input-len",
        "--output-len 32 replaces --random-output-len",
        "--random-input-len 64: vllm subtracts the tokenizer's special tokens before sampling, so its realized prompts"
        " run about one token short for BOS-prepending tokenizers; mapped verbatim",
        "vllm v0.32.0 re-cuts its random prompts to the server's /tokenize count when the server's tokenizer disagrees"
        " with the local one; inference-perf trusts the configured tokenizer",
        "--random-range-ratio sets the input ratio to 0.5 and the output ratio to 0.25 (uniform on [len*(1-r),"
        " len*(1+r)]); vllm v0.32.0 rounds the bounds outward (floor, ceil)",
        "--max-concurrency is a plain semaphore at v0.32.0, the same closed-loop meaning as concurrency_level",
        "the worker pool must cover the concurrency level: load.num_workers x load.worker_max_concurrency"
        " (defaults: cpu count x 100) must be at least 4",
        "vllm picks a LoRA module uniformly at random per request; mapped to an equal lora_traffic_split",
        "load.base_seed: 0 from --seed; seeds are tool-local, so this pins reproducibility of the converted run,"
        " not identical prompt text",
        "server.ignore_eos: true: vllm v0.32.0 sets ignore_eos on every random-dataset request, with or without --ignore-eos",
        "--served-model-name sets the payload model name; --model keeps only the tokenizer role",
    ]
    assert conversion.dropped == [
        "--label",
        "--use-beam-search (parsed and never used at v0.32.0)",
        "--plot-timeline",
        "--plot-dataset-stats",
        "--insecure",
        "--timeline-itl-thresholds",
        "--request-id-prefix",
        "--hf-name (inert with the random dataset at v0.32.0)",
        "--no-stream (HF dataset loading mode, not response streaming; inert with the random dataset)",
    ]
    assert conversion.no_equivalents == [
        "warmup: vllm bench sends at least 3 unmeasured requests before the run (the --ready-check-timeout-sec"
        " probe and --num-warmups copies of the first prompt); inference-perf measures every request it sends"
    ]
    config = conversion.config
    assert config is not None
    assert config.api.type == APIType.Chat
    assert config.server is not None
    assert config.server.model_name == "s"
    stage = config.load.stages[0]
    assert isinstance(stage, ConcurrentLoadStage)
    assert (stage.num_requests, stage.concurrency_level) == (1000, 4)
    input_dist = config.data.input_distribution
    output_dist = config.data.output_distribution
    assert input_dist is not None and output_dist is not None
    assert (input_dist.min, input_dist.max) == (32, 96)
    assert (output_dist.min, output_dist.max) == (24, 40)


# A v0.32.0 run with no --model, --random-output-len 1 at ratio 0.5, --host ::1 and three
# inert flags. The output floor, the auto-detected model and every drop are pinned word for word.
def test_pin_bare_run_records_every_line() -> None:
    argv = [
        "--random-output-len",
        "1",
        "--random-range-ratio",
        "0.5",
        "--served-model-name",
        "s",
        "--ramp-up-start-rps",
        "1",
        "--burstiness",
        "2",
        "--lora-assignment",
        "round-robin",
        "--host",
        "::1",
    ]
    conversion = convert(argv, PIN)
    assert conversion.refusals == []
    assert conversion.assumptions == [
        "--random-input-len 1024: vllm subtracts the tokenizer's special tokens before sampling, so its realized"
        " prompts run about one token short for BOS-prepending tokenizers; mapped verbatim",
        "vllm v0.32.0 re-cuts its random prompts to the server's /tokenize count when the server's tokenizer disagrees"
        " with the local one; inference-perf trusts the configured tokenizer",
        "--random-range-ratio 0.5 fans out to both the input and the output distribution (uniform on [len*(1-r),"
        " len*(1+r)]); vllm v0.32.0 rounds the bounds outward (floor, ceil)",
        "vllm v0.32.0 raises the output length floor to 1 token; mapped as 1",
        "vllm offers all prompts at t=0 at the default infinite --request-rate; the nearest equivalent is a concurrent"
        ' stage with concurrency_level equal to num_requests (docs/comparability.md "Where the map breaks")',
        "the worker pool must cover the concurrency level: load.num_workers x load.worker_max_concurrency"
        " (defaults: cpu count x 100) must be at least 1000",
        "load.base_seed: 0 from --seed; seeds are tool-local, so this pins reproducibility of the converted run,"
        " not identical prompt text",
        "server.ignore_eos: true: vllm v0.32.0 sets ignore_eos on every random-dataset request, with or without --ignore-eos",
        "no --model: both tools send the first /v1/models entry's id as the model name; vllm loads its tokenizer"
        " from that entry's root, inference-perf tries the id first, then the root",
    ]
    assert conversion.dropped == [
        "--ramp-up-start-rps/--ramp-up-end-rps (inert without --ramp-up-strategy)",
        "--burstiness (inert with an infinite request rate)",
        "--lora-assignment (inert without --lora-modules)",
        "--served-model-name (unread without --model at v0.32.0)",
    ]
    assert conversion.config is not None
    assert conversion.config.server is not None
    assert conversion.config.server.base_url == "http://[::1]:8000"


# Each v0.32.0 refusal, word for word. --backend vllm-pooling refuses as
# "--backend: pooling backend 'vllm-pooling' is not mapped: ..." and nothing else.
@pytest.mark.parametrize(
    ("argv", "refusals"),
    [
        (
            ["--backend", "openai-responses"],
            ["--backend: the Responses API has no API type (completion, chat and anthropic_messages exist)"],
        ),
        (
            ["--backend", "openai-audio"],
            ["--backend: openai-audio has no API type (completion, chat and anthropic_messages exist)"],
        ),
        (
            ["--backend", "vllm-pooling"],
            [
                "--backend: pooling backend 'vllm-pooling' is not mapped: the embeddings and rerank API types are"
                " unverified against vllm's pooling payloads"
            ],
        ),
        (
            ["--probe-request-rate", "1"],
            [
                "--probe-request-rate: single-token probes sent alongside the run, outside --max-concurrency, add load"
                " no stage produces"
            ],
        ),
        (
            ["--request-rate", "40", "--burstiness", "inf"],
            [
                "--burstiness: burstiness inf spaces requests exactly 1/rate apart at v0.32.0, which no load type"
                " produces (constant scales random gaps to the stage duration)"
            ],
        ),
        (
            ["--random-batch-size", "2"],
            [
                "--random-batch-size: several prompts packed into one request body have no equivalent; vllm applies it"
                " to every backend"
            ],
        ),
        (
            ["--lora-modules", "a", "b", "--lora-assignment", "round-robin"],
            [
                "--lora-assignment: round-robin cycles the adapters in a fixed order, while lora_traffic_split draws one"
                " per request"
            ],
        ),
        (
            ["--header", "x-a=1"],
            ["--header: per-request headers are not mapped; set api.headers on the run host if they matter"],
        ),
        (
            ["--extra-body", '{"a": 1}'],
            ["--extra-body: arbitrary request body fields can change the generation; not expressible"],
        ),
        (
            ["--frequency-penalty", "0.1", "--presence-penalty", "0.1", "--repetition-penalty", "1.1"],
            [
                f"--frequency-penalty: {SAMPLING_REFUSAL}",
                f"--presence-penalty: {SAMPLING_REFUSAL}",
                f"--repetition-penalty: {SAMPLING_REFUSAL}",
            ],
        ),
        (
            ["--random-range-ratio", '{"input": 0.1, "output": 1.0}'],
            ["--random-range-ratio: vllm v0.32.0 requires a range ratio in [0, 1), got 1.0"],
        ),
    ],
)
def test_pin_refusal_text(argv: List[str], refusals: List[str]) -> None:
    assert convert(["--model", "m", *argv], PIN).refusals == refusals


# The v0.32.0 warmup line from its two flags. -1 warmups and no ready check is none at
# all, a 1 second ready check is at least 1 request, 1 warmup alone is exactly 1 request.
@pytest.mark.parametrize(
    ("extra", "note"),
    [
        (["--num-warmups", "-1"], None),
        (
            ["--ready-check-timeout-sec", "1"],
            "warmup: vllm bench sends at least 1 unmeasured request before the run (the --ready-check-timeout-sec"
            " probe and --num-warmups copies of the first prompt); inference-perf measures every request it sends",
        ),
        (
            ["--num-warmups", "1"],
            "warmup: vllm bench sends 1 unmeasured request before the run (the --ready-check-timeout-sec probe and"
            " --num-warmups copies of the first prompt); inference-perf measures every request it sends",
        ),
    ],
)
def test_pin_warmup_line(extra: List[str], note: Optional[str]) -> None:
    notes = [item for item in convert(["--model", "m", *extra], PIN).no_equivalents if item.startswith("warmup:")]
    assert notes == ([] if note is None else [note])


# A JSON ratio whose value is not a number ("wide", null) is a usage error at v0.32.0,
# the ValueError and TypeError vllm's float() raises on it.
@pytest.mark.parametrize("ratio", ['{"input": "wide", "output": 0}', '{"input": 0, "output": null}'])
def test_pin_range_ratio_json_non_numeric(ratio: str) -> None:
    with pytest.raises(PeerUsageError, match="non-numeric ratio"):
        convert(["--model", "m", "--random-range-ratio", ratio], PIN)


# Both mirrors refuse an abbreviated flag (--num-prom for --num-prompts) and --help
# as unknown flags, the way allow_abbrev=False and add_help=False read them.
@pytest.mark.parametrize("version", ["v0.10.0", "v0.32.0"])
def test_mirror_refuses_abbreviations_and_help(version: str) -> None:
    conversion = convert(["--model", "m", "--num-prom", "5", "--help"], version)
    assert conversion.refusals == [
        f"--num-prom: not in the verified vllm bench serve {version} flag surface",
        f"--help: not in the verified vllm bench serve {version} flag surface",
    ]
