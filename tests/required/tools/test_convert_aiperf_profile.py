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

Includes the three default traps that make a bare aiperf run refuse loudly:
the default ``chat`` endpoint (the emitted ``data.type: random`` serves the
completions path only), an unset ``--osl`` (aiperf sends no max_tokens field,
inference-perf always does), and an entry pool smaller than the converted
request count (prompt reuse is not expressible; aiperf sizes the pool to an
explicit ``--request-count``, so only a pool set smaller on purpose or a
``--benchmark-duration`` run past the default pool trips it).
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
from inference_perf.config.redaction import CREDENTIAL_HEADER_FRAGMENTS
from inference_perf.tools.convert import convert_aiperf_profile
from inference_perf.tools.convert.model import Conversion, PeerUsageError

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
    # aiperf's constant arrival is perfectly even spacing; load.type constant
    # matches its total time while jittering individual gaps.
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
    conversion = convert(["--model", "m", "--endpoint-type", "completions", "--osl", "64", "--num-dataset-entries", "10"])
    assert conversion.refusals == []
    config = conversion.config
    assert config is not None
    assert config.api.streaming is False
    assert config.model_dump(exclude_unset=True)["api"]["streaming"] is False
    assert config.api.type == APIType.Completion


def test_default_load_is_concurrency_one_with_derived_count() -> None:
    conversion = convert(["--model", "m", "--endpoint-type", "completions", "--osl", "64"])
    assert conversion.refusals == []
    config = conversion.config
    assert config is not None
    assert config.load.type == LoadType.CONCURRENT
    stage = config.load.stages[0]
    assert isinstance(stage, ConcurrentLoadStage)
    assert stage.concurrency_level == 1
    assert stage.num_requests == 10  # _validate_profiling: prof.setdefault("requests", 10)


@pytest.mark.parametrize("concurrency", ["1", "8", "32"])
def test_unbounded_run_sends_ten_requests_whatever_the_concurrency(concurrency: str) -> None:
    # v0.12.0 _validate_profiling (aiperf/config/flags/_converter_profiling.py):
    # prof.setdefault("requests", 10) when no bound is given, and nothing
    # re-derives it from the concurrency, whatever the --request-count help
    # string says.
    conversion = convert(["--model", "m", "--endpoint-type", "completions", "--osl", "64", "--concurrency", concurrency])
    assert conversion.refusals == []
    config = conversion.config
    assert config is not None
    stage = config.load.stages[0]
    assert isinstance(stage, ConcurrentLoadStage)
    assert stage.concurrency_level == int(concurrency)
    assert stage.num_requests == 10
    assert any('prof.setdefault("requests", 10)' in item for item in conversion.assumptions), conversion.assumptions


def test_unbounded_rate_run_sends_ten_requests() -> None:
    conversion = convert(["--model", "m", "--endpoint-type", "completions", "--osl", "64", "--request-rate", "5"])
    assert conversion.refusals == []
    config = conversion.config
    assert config is not None
    stage = config.load.stages[0]
    assert isinstance(stage, StandardLoadStage)
    assert stage.rate == 5
    assert stage.duration == 2  # 10 requests at 5 per second


@pytest.mark.parametrize(
    ("argv", "flag"),
    [
        (["--model", "m", "--osl", "64", "--request-count", "100,500"], "--request-count"),
        (["--model", "m", "--osl", "64", "--num-requests", "100,500"], "--num-requests"),
        (["--model", "m", "--osl", "64", "--concurrency", "8,16"], "--concurrency"),
        (["--model", "m", "--osl", "64", "--concurrency=8,16"], "--concurrency"),
        (["--model", "m", "--osl", "64", "--request-rate", "10,20"], "--request-rate"),
        (["--model", "m", "--osl", "64", "--request-rate", "10", "--benchmark-duration", "10,20"], "--benchmark-duration"),
        (["--model", "m", "--osl", "64,128"], "--osl"),
        (["--model", "m", "--osl", "64", "--osl-stddev", "0,8"], "--osl-stddev"),
        (["--model", "m", "--osl", "64", "--isl", "100,200"], "--isl"),
        (["--model", "m", "--osl", "64", "--isl-stddev", "0,16"], "--isl-stddev"),
        (["--model", "m", "--osl", "64", "--prefill-concurrency", "2,4"], "--prefill-concurrency"),
        (["--model", "m", "--osl", "64", "--num-users", "2,4"], "--num-users"),
        (["--model", "m", "--osl", "64", "--conversation-num", "10,20"], "--conversation-num"),
        (["--model", "m", "--osl", "64", "--conversation-turn-mean", "1,2"], "--conversation-turn-mean"),
    ],
)
def test_sweep_lists_refuse_with_a_named_reason(argv: List[str], flag: str) -> None:
    # parse_int_or_int_list and parse_float_or_float_list at v0.12.0 turn a
    # comma-separated value into a parameter sweep, one run per value. The
    # peer accepts the argv, so this is a refusal (exit 2), not a usage error.
    conversion = convert(argv)
    assert conversion.config is None
    assert any(refusal.startswith(f"{flag}: ") and "parameter sweep" in refusal for refusal in conversion.refusals), (
        conversion.refusals
    )


def test_trailing_comma_collapses_to_the_single_value() -> None:
    # "8," is the single value 8 at v0.12.0: parse_int_or_int_list drops the
    # empty part, so neither a sweep nor a usage error.
    conversion = convert(["--model", "m", "--endpoint-type", "completions", "--osl", "64", "--concurrency", "8,"])
    assert conversion.refusals == []
    config = conversion.config
    assert config is not None
    stage = config.load.stages[0]
    assert isinstance(stage, ConcurrentLoadStage)
    assert stage.concurrency_level == 8


@pytest.mark.parametrize("tail", [["--bogus", "-1"], ["--bogus", "x"], ["--bogus=x"], ["--bogus"]])
def test_unknown_flag_refuses_once(tail: List[str]) -> None:
    conversion = convert(["--model", "m", "--endpoint-type", "completions", "--osl", "64"] + tail)
    assert conversion.config is None
    assert conversion.refusals == ["--bogus: not in the verified aiperf profile v0.12.0 flag surface"]


# --no-streaming: --streaming exists but declares no negation, and aiperf's
# CLIParameter defaults negative=False, so aiperf rejects the token; a usage
# error, not a refusal.
@pytest.mark.parametrize("flag", ["--no-streaming", "--no-tokenizer-trust-remote-code", "--no-stream"])
def test_undeclared_negation_is_a_usage_error(flag: str) -> None:
    with pytest.raises(PeerUsageError, match="no automatic --no-"):
        convert(["--model", "m", "--endpoint-type", "completions", "--osl", "64", flag])


# --no-foo where --foo is not a flag either: the plain unknown-flag refusal,
# since the token could be a later version's own --no- field.
def test_unknown_negation_refuses_as_an_unknown_flag() -> None:
    conversion = convert(["--model", "m", "--endpoint-type", "completions", "--osl", "64", "--no-foo"])
    assert conversion.config is None
    assert conversion.refusals == ["--no-foo: not in the verified aiperf profile v0.12.0 flag surface"]


# --no-open-loop-replay is one of the nine negations cli_config.py declares:
# it parses, turns the field off, and the flag drops as inert for a
# single-turn synthetic run.
def test_declared_negation_parses_and_drops_as_inert() -> None:
    conversion = convert(["--model", "m", "--endpoint-type", "completions", "--osl", "64", "--no-open-loop-replay"])
    assert conversion.refusals == []
    assert conversion.config is not None
    assert any("--open-loop-replay" in item for item in conversion.dropped), conversion.dropped


# --accuracy-no-enable-cot is the one declared negation without a --no-
# prefix; it reads as accuracy_enable_cot=False, which is inert, so the flag
# drops instead of tripping the ACCURACY group refusal.
def test_accuracy_negation_drops_as_explicitly_disabled() -> None:
    conversion = convert(["--model", "m", "--endpoint-type", "completions", "--osl", "64", "--accuracy-no-enable-cot"])
    assert conversion.refusals == []
    assert conversion.config is not None
    assert any("--accuracy-enable-cot" in item and "disabled" in item for item in conversion.dropped), conversion.dropped


# --auto-plot invokes `aiperf plot` after the run, outside the measured
# window, so it drops with an annotation and the conversion emits.
def test_auto_plot_drops_as_reporting_plumbing() -> None:
    conversion = convert(["--model", "m", "--endpoint-type", "completions", "--osl", "64", "--auto-plot"])
    assert conversion.refusals == []
    assert conversion.config is not None
    assert any("--auto-plot" in item for item in conversion.dropped), conversion.dropped


# --set-consistent-seed rides the multi-run subsystem, so setting it refuses
# with the group reason while its declared --no- spelling is inert and drops.
def test_multi_run_seed_flag_refuses_and_its_negation_drops() -> None:
    base = ["--model", "m", "--endpoint-type", "completions", "--osl", "64"]
    refused = convert(base + ["--set-consistent-seed"])
    assert refused.config is None
    assert any("one config for one run" in refusal for refusal in refused.refusals), refused.refusals
    dropped = convert(base + ["--no-set-consistent-seed"])
    assert dropped.refusals == []
    assert dropped.config is not None
    assert any("--set-consistent-seed" in item and "disabled" in item for item in dropped.dropped), dropped.dropped


# aiperf rejects nonpositive load values in its own validation; the converter
# turns them into usage errors instead of a stage-model traceback (zero or
# negative counts) or a division by zero (zero rate).
@pytest.mark.parametrize(
    ("argv", "flag"),
    [
        (["--concurrency", "0"], "--concurrency"),
        (["--concurrency", "-1"], "--concurrency"),
        (["--request-count", "0"], "--request-count"),
        (["--request-rate", "0"], "--request-rate"),
    ],
)
def test_nonpositive_load_values_are_usage_errors(argv: List[str], flag: str) -> None:
    with pytest.raises(PeerUsageError, match=rf"{flag} expects a positive"):
        convert(["--model", "m", "--endpoint-type", "completions", "--osl", "64"] + argv)


def test_unset_osl_refuses() -> None:
    conversion = convert(["--model", "m"])
    assert conversion.config is None
    assert any("max_tokens" in refusal for refusal in conversion.refusals)


# A bare conversion runs on aiperf's default chat endpoint, and the emitted
# data.type random serves the completions path only, so it refuses and the
# reason names the default and the fix.
def test_default_chat_endpoint_refuses_with_random_data() -> None:
    conversion = convert(["--model", "m", "--osl", "64"])
    assert conversion.config is None
    reason = next(refusal for refusal in conversion.refusals if refusal.startswith("--endpoint-type:"))
    assert "chat" in reason
    assert "default" in reason
    assert "--endpoint-type completions" in reason


# --endpoint-type chat and messages map to API types the random datagen
# refuses to load, so both refuse at conversion instead of emitting a config
# that fails with "Unsupported API type".
@pytest.mark.parametrize("endpoint", ["chat", "messages"])
def test_non_completions_endpoint_refuses_with_random_data(endpoint: str) -> None:
    conversion = convert(["--model", "m", "--osl", "64", "--endpoint-type", endpoint])
    assert conversion.config is None
    assert any(
        refusal.startswith("--endpoint-type:") and "completions path only" in refusal for refusal in conversion.refusals
    ), conversion.refusals


# --request-count 200 with no entry count: aiperf sizes the synthetic pool to
# the explicit request count (_resolve_entries at v0.12.0), so no entry is
# reused and the conversion emits, with the sizing recorded as an assumption.
def test_explicit_request_count_sizes_the_entry_pool() -> None:
    argv = ["--model", "m", "--endpoint-type", "completions", "--osl", "64", "--concurrency", "8", "--request-count", "200"]
    conversion = convert(argv)
    assert conversion.refusals == []
    assert conversion.config is not None
    assert any("sizes aiperf's synthetic entry pool" in item for item in conversion.assumptions), conversion.assumptions


# --num-dataset-entries 100 against --request-count 200: the explicitly set
# pool wins over the request count in aiperf's resolution order, so 200
# requests reuse 100 entries and the conversion refuses.
def test_explicit_entry_pool_smaller_than_request_count_refuses() -> None:
    argv = ["--model", "m", "--endpoint-type", "completions", "--osl", "64", "--request-count", "200"]
    conversion = convert(argv + ["--num-dataset-entries", "100"])
    assert conversion.config is None
    assert any("prompt reuse" in refusal and "pool of 100" in refusal for refusal in conversion.refusals), conversion.refusals


# --request-rate 20 --benchmark-duration 10 derives 200 requests with no
# explicit --request-count, so the pool stays at the 100-entry default and the
# conversion refuses, naming the default.
def test_duration_derived_count_past_the_default_pool_refuses() -> None:
    argv = ["--model", "m", "--endpoint-type", "completions", "--osl", "64", "--request-rate", "20"]
    conversion = convert(argv + ["--benchmark-duration", "10"])
    assert conversion.config is None
    assert any("prompt reuse" in refusal and "default pool is 100" in refusal for refusal in conversion.refusals), (
        conversion.refusals
    )


def test_poisson_rate_with_count() -> None:
    argv = ["--model", "m", "--endpoint-type", "completions", "--osl", "64", "--request-rate", "10", "--request-count", "100"]
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
        "--endpoint-type",
        "completions",
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
    argv = ["--model", "m", "--endpoint-type", "completions", "--osl", "64", "--isl", "500", "--isl-stddev", "50"]
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


# At v0.12.0 --isl is content tokens on the wire and the server's BOS rides on
# top, while inference-perf counts BOS inside the input length; the one-token
# skew is recorded as an assumption, mirroring the vllm converter's
# --random-input-len note.
def test_isl_bos_accounting_assumption_is_recorded() -> None:
    argv = ["--model", "m", "--endpoint-type", "completions", "--osl", "64", "--isl", "128"]
    conversion = convert(argv)
    assert conversion.refusals == []
    assert conversion.config is not None
    assert any("--isl 128" in item and "BOS" in item for item in conversion.assumptions), conversion.assumptions


def test_mapped_reporting_and_server_fields() -> None:
    argv = [
        "--model",
        "m",
        "--endpoint-type",
        "completions",
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
    # The key is never written: the file is meant to be checked in.
    assert config.server.api_key is None
    assert any("set server.api_key" in item for item in conversion.assumptions), conversion.assumptions
    assert config.api.headers == {"X-Team": "infra"}
    assert config.api.session_id_header_key == "X-Session-Id"
    assert config.storage is not None
    assert config.storage.local_storage.path == "artifacts/run1"
    assert config.metrics is not None
    assert config.metrics.type == MetricsClientType.PROMETHEUS


def test_warmup_flags_annotate_with_the_stage_workaround() -> None:
    argv = ["--model", "m", "--endpoint-type", "completions", "--osl", "64", "--warmup-request-count", "50"]
    conversion = convert(argv)
    assert conversion.refusals == []
    assert any(item.startswith("warmup:") and "first stage" in item for item in conversion.no_equivalents)
    assert any("--warmup-request-count" in item for item in conversion.dropped)


def test_reporting_flags_drop_without_aborting() -> None:
    argv = [
        "--model",
        "m",
        "--endpoint-type",
        "completions",
        "--osl",
        "64",
        "--ui",
        "dashboard",
        "--export-level",
        "records",
        "--gpu-telemetry",
        "dcgm",
    ]
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
        "--endpoint-type",
        "completions",
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
        "--endpoint-type",
        "completions",
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


# --header mirrors aiperf v0.12.0 _parse_str_as_tuple_list: a value starting
# with { is a JSON object, anything else is a comma-separated list of
# key:value items split at the first colon. A header whose name carries a
# credential is dropped the way --api-key is.
_HEADER_BASE = ["--model", "m", "--endpoint-type", "completions", "--osl", "64", "--num-dataset-entries", "10"]


def test_credential_header_is_dropped_and_assumed() -> None:
    conversion = convert(_HEADER_BASE + ["-H", "Authorization: Bearer sk-x", "--header", "X-Team: infra"])
    assert conversion.refusals == []
    assert conversion.config is not None
    assert conversion.config.api.headers == {"X-Team": "infra"}
    assert any("set api.headers Authorization" in item for item in conversion.assumptions), conversion.assumptions
    # The recorded argv keeps the header name and loses the value.
    assert "sk-x" not in " ".join(conversion.argv)
    assert "Authorization: <redacted>" in conversion.argv
    assert "X-Team: infra" in conversion.argv


@pytest.mark.parametrize("fragment", sorted(CREDENTIAL_HEADER_FRAGMENTS))
def test_every_credential_fragment_drops_the_header(fragment: str) -> None:
    name = f"X-{fragment.capitalize()}-Id"
    conversion = convert(_HEADER_BASE + ["--header", f"{name}: v-1"])
    assert conversion.refusals == []
    assert conversion.config is not None
    assert conversion.config.api.headers is None
    assert any(f"set api.headers {name}" in item for item in conversion.assumptions), conversion.assumptions
    assert "v-1" not in " ".join(conversion.argv)


def test_json_header_form_is_parsed() -> None:
    conversion = convert(_HEADER_BASE + ["--header", '{"X-C": "3", "X-Team": "infra"}'])
    assert conversion.refusals == []
    assert conversion.config is not None
    assert conversion.config.api.headers == {"X-C": "3", "X-Team": "infra"}
    assert '{"X-C": "3", "X-Team": "infra"}' in conversion.argv  # untouched without a credential


def test_json_header_with_a_credential_is_scrubbed_in_place() -> None:
    conversion = convert(_HEADER_BASE + ["--header", '{"X-Api-Key": "k-1", "X-Team": "infra"}'])
    assert conversion.refusals == []
    assert conversion.config is not None
    assert conversion.config.api.headers == {"X-Team": "infra"}
    assert any("set api.headers X-Api-Key" in item for item in conversion.assumptions), conversion.assumptions
    assert "k-1" not in " ".join(conversion.argv)
    assert '{"X-Api-Key": "<redacted>", "X-Team": "infra"}' in conversion.argv


def test_comma_separated_header_items_are_split_like_aiperf() -> None:
    conversion = convert(_HEADER_BASE + ["--header", "X-A: a, X-B: b", "--header=X-Team:infra"])
    assert conversion.refusals == []
    assert conversion.config is not None
    assert conversion.config.api.headers == {"X-A": "a", "X-B": "b", "X-Team": "infra"}


@pytest.mark.parametrize(
    "entry",
    [
        '{"X-C": ',  # aiperf: must be a valid JSON string
        "X-C",  # aiperf: each item must be in key:value format
        "X-A: a, X-B",  # the second item has no colon
    ],
)
def test_malformed_header_entry_is_a_usage_error(entry: str) -> None:
    with pytest.raises(PeerUsageError, match="--header"):
        convert(_HEADER_BASE + ["--header", entry])


@pytest.mark.parametrize(
    "entry",
    [
        "X-C: 3",  # coerce_value reads 3 as an int
        "X-C: true",  # and true as a bool
        "X-C: null",  # and null as None
        '{"X-C": 3}',  # a JSON number stays a number
    ],
)
def test_header_value_aiperf_would_reject_is_a_usage_error(entry: str) -> None:
    # aiperf's headers map takes strings, so these fail its config validation.
    with pytest.raises(PeerUsageError, match="X-C"):
        convert(_HEADER_BASE + ["--header", entry])


@pytest.mark.parametrize(
    "entry, bad_name",
    [
        (' {"X-C": "3"}', '{"X-C"'),  # a leading space defeats JSON detection, so the name is {"X-C"
        (": v", ""),  # an empty name
        ('{"X C": "3"}', "X C"),  # a JSON key with a space
    ],
)
def test_malformed_header_name_refuses(entry: str, bad_name: str) -> None:
    conversion = convert(_HEADER_BASE + ["--header", entry])
    assert conversion.config is None
    assert any(item.startswith("--header:") and f"'{bad_name}'" in item for item in conversion.refusals), conversion.refusals


# --header X-A:a X-B:b --streaming: one consume_multiple occurrence carries
# both tokens at v0.12.0, so api.headers is {"X-A": "a", "X-B": "b"} and
# --streaming is read as the next flag, not as a third header.
def test_header_consumes_value_tokens_until_the_next_flag() -> None:
    conversion = convert(_HEADER_BASE + ["--header", "X-A:a", "X-B:b", "--streaming"])
    assert conversion.refusals == []
    assert conversion.config is not None
    assert conversion.config.api.headers == {"X-A": "a", "X-B": "b"}
    assert conversion.config.api.streaming is True


# --header=X-A:a X-B:b: the `=` spelling takes its inline value and keeps
# consuming the bare tokens after it, so both headers are read.
def test_header_equals_spelling_keeps_consuming() -> None:
    conversion = convert(_HEADER_BASE + ["--header=X-A:a", "X-B:b"])
    assert conversion.refusals == []
    assert conversion.config is not None
    assert conversion.config.api.headers == {"X-A": "a", "X-B": "b"}


# --header directly followed by the next flag carries no tokens: aiperf leaves
# the field at its default, so the conversion emits no headers and no error.
def test_header_with_no_value_tokens_is_inert() -> None:
    conversion = convert(_HEADER_BASE + ["--header", "--streaming"])
    assert conversion.refusals == []
    assert conversion.config is not None
    assert conversion.config.api.headers is None


# --header X-A:a -1: the negative number is consumed as a header token the way
# aiperf consumes it; "-1" has no colon, so the usage error names occurrence
# #2 and does not echo the token.
def test_negative_number_is_consumed_and_the_error_names_the_occurrence() -> None:
    with pytest.raises(PeerUsageError) as exc_info:
        convert(_HEADER_BASE + ["--header", "X-A:a", "-1"])
    message = str(exc_info.value)
    assert "--header #2" in message
    assert "-1" not in message


# -H 'Authorization=Bearer sk-live-123' (an = typo for :): the key:value usage
# error names --header #1 and the credential never reaches the message.
def test_header_usage_error_does_not_echo_a_credential() -> None:
    with pytest.raises(PeerUsageError) as exc_info:
        convert(_HEADER_BASE + ["-H", "Authorization=Bearer sk-live-123"])
    message = str(exc_info.value)
    assert "--header #1" in message
    assert "sk-live-123" not in message


# --header '{"X-A": ' is unparsable JSON: the usage error names occurrence #1
# instead of echoing the (possibly credential-bearing) entry.
def test_header_json_usage_error_names_the_occurrence() -> None:
    with pytest.raises(PeerUsageError) as exc_info:
        convert(_HEADER_BASE + ["--header", '{"X-A": '])
    message = str(exc_info.value)
    assert "--header #1" in message
    assert "X-A" not in message


# --header X-Team:infra Authorization:tok-9: the credential in the second
# consumed token is dropped from api.headers and scrubbed from the recorded
# argv, exactly as a second --header occurrence would be.
def test_redacted_argv_covers_every_consumed_header_token() -> None:
    conversion = convert(_HEADER_BASE + ["--header", "X-Team:infra", "Authorization:tok-9"])
    assert conversion.refusals == []
    assert conversion.config is not None
    assert conversion.config.api.headers == {"X-Team": "infra"}
    assert "tok-9" not in " ".join(conversion.argv)
    assert "Authorization: <redacted>" in conversion.argv
    assert any("set api.headers Authorization" in item for item in conversion.assumptions), conversion.assumptions


# --url http://a:8000 http://b:8000: both tokens are consumed as URLs, so the
# refusal is the multi-URL one and nothing refuses as a positional argument.
def test_multiple_url_tokens_refuse_as_multiple_urls() -> None:
    conversion = convert(
        ["--model", "m", "--endpoint-type", "completions", "--osl", "64", "--url", "http://a:8000", "http://b:8000"]
    )
    assert conversion.config is None
    assert any(item.startswith("--url:") and "load balancing" in item for item in conversion.refusals), conversion.refusals
    assert not any("positional" in item for item in conversion.refusals), conversion.refusals


# --extra-inputs temperature:0.7 top_p:0.9 (the flag's own help example): both
# tokens are consumed as entries, so both sampling keys refuse by name.
def test_extra_inputs_consumes_multiple_tokens() -> None:
    conversion = convert(["--model", "m", "--osl", "64", "--extra-inputs", "temperature:0.7", "top_p:0.9"])
    assert conversion.config is None
    assert any(item.startswith("--extra-inputs temperature:") for item in conversion.refusals), conversion.refusals
    assert any(item.startswith("--extra-inputs top_p:") for item in conversion.refusals), conversion.refusals


# An --extra-inputs entry with no colon: the usage error names the entry
# position and does not echo the entry, which may carry an injected secret.
def test_extra_inputs_usage_error_names_the_position() -> None:
    with pytest.raises(PeerUsageError) as exc_info:
        convert(["--model", "m", "--osl", "64", "--extra-inputs", "oops-no-colon"])
    message = str(exc_info.value)
    assert "--extra-inputs #1" in message
    assert "oops-no-colon" not in message
