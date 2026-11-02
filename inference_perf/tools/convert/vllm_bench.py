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
"""``vllm bench serve`` argv -> inference-perf config, keyed to vllm v0.10.0 and v0.32.0.

Each verified tag has its own mirrored parser, a transcription of that tag's
argparse surface (v0.10.0: 18 dataset flags + 37 serve flags; v0.32.0: 47
dataset flags + 54 serve flags), so an unknown flag is a refusal rather than a
silent drop: that is how surface drift on newer vllm versions gets caught
instead of ignored. Rows whose meaning moved between tags read the requested
version's ``VllmBenchFacts``.

Verified behavior the rows below rely on, at both tags unless a tag is named:

- The request payload always streams. At v0.10.0 it carries ``temperature
  0.0`` and ``repetition_penalty 1.0`` when the user passed no sampling flags.
  At v0.32.0 only completions requests carry ``repetition_penalty 1.0`` and
  temperature is left to the server.
- v0.10.0 ``benchmark()`` sends one untunable test request before the run.
  v0.32.0 sends none by default: ``--ready-check-timeout-sec`` above 0 probes
  until the endpoint answers and ``--num-warmups`` repeats the first prompt.
- ``--request-rate`` defaults to ``inf`` (all prompts offered at t=0); a
  finite rate with the default ``--burstiness 1.0`` gives Poisson spacing.
- ``RandomDataset`` draws lengths uniform on ``[len*(1-r), len*(1+r)]`` and
  subtracts the tokenizer's special tokens from the requested input length.
  v0.32.0 rounds the bounds outward, keeps the output length at least 1 and
  takes a separate ratio per direction.
- v0.32.0 sets ``ignore_eos`` for the random dataset on every
  OpenAI-compatible backend, with or without ``--ignore-eos``.
"""

import argparse
import ipaddress
import json
import math
from typing import Any, Callable, Dict, List, NoReturn, Optional, Set, Tuple

from inference_perf.config import (
    APIConfig,
    APIType,
    Config,
    DataConfig,
    DataGenType,
    Distribution,
    GoodputConfig,
    LoadConfig,
    LoadType,
    ModelServerClientConfig,
    MultiLoRAConfig,
    ReportConfig,
    RequestLifecycleMetricsReportConfig,
    StandardLoadStage,
    ConcurrentLoadStage,
    CustomTokenizerConfig,
    StorageConfig,
    StorageConfigBase,
)
from inference_perf.tools.convert.model import Conversion, PeerUsageError, fixed_dist, is_value_token, uniform_dist
from inference_perf.tools.convert.versions import (
    VLLM_BENCH_VERSIONS,
    VllmBenchFacts,
    resolve_vllm_bench,
    unverified_version_reason,
)

TOOL_NAME = "vllm bench serve"

_ENDPOINT_PATHS = {
    APIType.Completion: "/v1/completions",
    APIType.Chat: "/v1/chat/completions",
}

# Request functions that build a completions or chat payload; everything else
# in ASYNC_REQUEST_FUNCS refuses in _convert_api.
_COMPLETION_BACKENDS = ("vllm", "openai")
_CHAT_BACKENDS = ("openai-chat",)

_DATASET_REFUSALS = {
    "sharegpt": "vllm filters ShareGPT turns by length rules the shareGPT datagen does not replicate",
    "sonnet": "line-assembled poem prompts with a shared prefix have no equivalent generator",
    "burstgpt": "trace-file dataset with no equivalent",
    "hf": "HF dataset loader transforms are unverified",
    "custom": "arbitrary jsonl dataset with no equivalent",
    "random-mm": "synthetic image and video items have no equivalent generator",
    "random-rerank": "query and document pairs for reranking are not mapped",
    "custom_audio": "audio items from a jsonl dataset have no equivalent",
    "custom_image": "image items from a jsonl dataset have no equivalent",
    "prefix_repetition": "a pool of repeated prefixes is a prefix-cache-sensitive workload with no verified equivalent",
    "spec_bench": "SpecBench question file with no equivalent",
    "speed_bench": "SPEED-Bench question file with no equivalent",
    "timed_trace": "timestamped trace replay with hash-id prefixes is unverified against trace replay",
}

# Flags that only configure a dataset other than `random`. With `random`
# selected they are parsed and never read by vllm at that tag, so they cannot
# change the offered workload and are dropped with an annotation; with their
# own dataset selected the dataset row already refuses.
_OTHER_DATASET_FLAGS: Dict[str, Tuple[str, ...]] = {
    "v0.10.0": (
        "dataset_path",
        "custom_output_len",
        "custom_skip_chat_template",
        "sonnet_input_len",
        "sonnet_output_len",
        "sonnet_prefix_len",
        "sharegpt_output_len",
        "hf_subset",
        "hf_split",
        "hf_output_len",
    ),
    # get_samples hands the random dataset seed, dataset_path, disable_shuffle,
    # num_prompts, the random_* lengths, request_id_prefix, random_batch_size
    # and no_oversample. RandomDataset never reads disable_shuffle or
    # no_oversample, and only the custom and speed_bench loaders read
    # chat_template_kwargs.
    "v0.32.0": (
        "no_oversample",
        "skip_chat_template",
        "enable_multimodal_chat",
        "disable_shuffle",
        "chat_template_kwargs",
        "custom_output_len",
        "custom_ensure_client_side_data",
        "spec_bench_output_len",
        "spec_bench_category",
        "sharegpt_output_len",
        "timed_trace_chunk_hash_size",
        "timed_trace_sec_multiplier",
        "timed_trace_label_timestamp",
        "timed_trace_label_input_length",
        "timed_trace_label_output_length",
        "timed_trace_label_hash_ids",
        "blazedit_min_distance",
        "blazedit_max_distance",
        "asr_max_audio_len_sec",
        "asr_min_audio_len_sec",
        "no_reranker",
        "random_mm_base_items_per_request",
        "random_mm_num_mm_items_range_ratio",
        "random_mm_limit_mm_per_prompt",
        "random_mm_bucket_config",
        "hf_subset",
        "hf_split",
        "hf_name",
        "hf_output_len",
        "bfcl_categories",
        "prefix_repetition_prefix_len",
        "prefix_repetition_suffix_len",
        "prefix_repetition_num_prefixes",
        "prefix_repetition_output_len",
        "speed_bench_dataset_subset",
        "speed_bench_output_len",
        "speed_bench_category",
    ),
}

# Flags vllm itself rejects alongside the random dataset at that tag:
# main_async raises for --dataset-path and --self-timed, and get_samples
# asserts a tokenizer, which --skip-tokenizer-init removes.
_RANDOM_REJECTS: Dict[str, Tuple[str, ...]] = {
    "v0.10.0": (),
    "v0.32.0": ("dataset_path", "self_timed", "skip_tokenizer_init"),
}

_SAMPLING_REFUSAL = (
    "sampling parameters are not configurable for synthetic data; a guessed value would misstate the decode work"
)


class _MirrorParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise PeerUsageError(f"invalid {TOOL_NAME} arguments: {message}")


class _Mirror:
    """One tag's argparse surface plus a flag registry for presence checks."""

    def __init__(self, version: str) -> None:
        self.parser = _MirrorParser(prog=TOOL_NAME, add_help=False, allow_abbrev=False)
        self.alias_to_dest: Dict[str, str] = {}
        self.dest_to_flag: Dict[str, str] = {}
        builders: Dict[str, Callable[[], None]] = {
            "v0.10.0": self._build_v0_10_0,
            "v0.32.0": self._build_v0_32_0,
        }
        builders[version]()

    def _add(self, *names: str, **kwargs: Any) -> None:
        action = self.parser.add_argument(*names, **kwargs)
        # option_strings, not names: BooleanOptionalAction also registers --no-<flag>.
        for name in action.option_strings:
            self.alias_to_dest[name] = action.dest
        self.dest_to_flag[action.dest] = names[0]

    def _build_v0_10_0(self) -> None:
        add = self._add
        # add_dataset_parser (vllm/benchmarks/datasets.py @ v0.10.0)
        add("--seed", type=int, default=0)
        add("--num-prompts", type=int, default=1000)
        add("--dataset-name", type=str, default="random", choices=["sharegpt", "burstgpt", "sonnet", "random", "hf", "custom"])
        add("--no-stream", action="store_true")
        add("--dataset-path", type=str, default=None)
        add("--custom-output-len", type=int, default=256)
        add("--custom-skip-chat-template", action="store_true")
        add("--sonnet-input-len", type=int, default=550)
        add("--sonnet-output-len", type=int, default=150)
        add("--sonnet-prefix-len", type=int, default=200)
        add("--sharegpt-output-len", type=int, default=None)
        add("--random-input-len", type=int, default=1024)
        add("--random-output-len", type=int, default=128)
        add("--random-range-ratio", type=float, default=0.0)
        add("--random-prefix-len", type=int, default=0)
        add("--hf-subset", type=str, default=None)
        add("--hf-split", type=str, default=None)
        add("--hf-output-len", type=int, default=None)
        # add_cli_args (vllm/benchmarks/serve.py @ v0.10.0)
        add("--endpoint-type", type=str, default="openai", choices=["vllm", "openai", "openai-chat", "openai-audio"])
        add("--label", type=str, default=None)
        add("--backend", type=str, default="vllm", choices=["vllm", "openai", "openai-chat", "openai-audio"])
        add("--base-url", type=str, default=None)
        add("--host", type=str, default="127.0.0.1")
        add("--port", type=int, default=8000)
        add("--endpoint", type=str, default="/v1/completions")
        add("--max-concurrency", type=int, default=None)
        add("--model", type=str, required=True)
        add("--tokenizer", type=str)
        add("--use-beam-search", action="store_true")
        add("--logprobs", type=int, default=None)
        add("--request-rate", type=float, default=float("inf"))
        add("--burstiness", type=float, default=1.0)
        add("--trust-remote-code", action="store_true")
        add("--disable-tqdm", action="store_true")
        add("--profile", action="store_true")
        add("--save-result", action="store_true")
        add("--save-detailed", action="store_true")
        add("--append-result", action="store_true")
        add("--metadata", metavar="KEY=VALUE", nargs="*")
        add("--result-dir", type=str, default=None)
        add("--result-filename", type=str, default=None)
        add("--ignore-eos", action="store_true")
        add("--percentile-metrics", type=str, default="ttft,tpot,itl")
        add("--metric-percentiles", type=str, default="99")
        add("--goodput", nargs="+", required=False)
        add("--top-p", type=float, default=None)
        add("--top-k", type=int, default=None)
        add("--min-p", type=float, default=None)
        add("--temperature", type=float, default=None)
        add("--tokenizer-mode", type=str, default="auto", choices=["auto", "slow", "mistral", "custom"])
        add("--served-model-name", type=str, default=None)
        add("--lora-modules", nargs="+", default=None)
        add("--ramp-up-strategy", type=str, default=None, choices=["linear", "exponential"])
        add("--ramp-up-start-rps", type=int, default=None)
        add("--ramp-up-end-rps", type=int, default=None)

    def _build_v0_32_0(self) -> None:
        add = self._add
        # add_dataset_parser (vllm/benchmarks/datasets/datasets.py @ v0.32.0)
        add("--trust-remote-code", action="store_true")
        add("--seed", type=int, default=0)
        add("--num-prompts", type=int, default=1000)
        add(
            "--dataset-name",
            type=str,
            default="random",
            choices=[
                "sharegpt",
                "burstgpt",
                "random",
                "random-mm",
                "random-rerank",
                "hf",
                "custom",
                "custom_audio",
                "custom_image",
                "prefix_repetition",
                "spec_bench",
                "speed_bench",
                "timed_trace",
            ],
        )
        add("--no-stream", action="store_true")
        add("--dataset-path", type=str, default=None)
        add("--no-oversample", action="store_true")
        add("--skip-chat-template", action="store_true")
        add("--enable-multimodal-chat", action="store_true")
        add("--disable-shuffle", action="store_true")
        add("--custom-output-len", type=int, default=256)
        add("--custom-ensure-client-side-data", action="store_true")
        add("--spec-bench-output-len", type=int, default=256)
        add("--spec-bench-category", type=str, default=None)
        add("--sharegpt-output-len", type=int, default=None)
        add("--timed-trace-chunk-hash-size", type=int, default=16)
        add("--timed-trace-sec-multiplier", type=float, default=1)
        add("--timed-trace-label-timestamp", type=str, default="timestamp")
        add("--timed-trace-label-input-length", type=str, default="input_length")
        add("--timed-trace-label-output-length", type=str, default="output_length")
        add("--timed-trace-label-hash-ids", type=str, default="hash_ids")
        add("--blazedit-min-distance", type=float, default=0.0)
        add("--blazedit-max-distance", type=float, default=1.0)
        add("--asr-max-audio-len-sec", type=float, default=float("inf"))
        add("--asr-min-audio-len-sec", type=float, default=0.0)
        # add_random_dataset_base_args
        add("--random-input-len", type=int, default=1024)
        add("--random-output-len", type=int, default=128)
        add("--random-range-ratio", type=str, default="0.0")
        add("--random-prefix-len", type=int, default=0)
        add("--random-batch-size", type=int, default=1)
        add("--no-reranker", action="store_true")
        # add_random_multimodal_dataset_args. The bucket config is a Python
        # literal that only random-mm parses, so its value is kept as text.
        add("--random-mm-base-items-per-request", type=int, default=1)
        add("--random-mm-num-mm-items-range-ratio", type=float, default=0.0)
        add("--random-mm-limit-mm-per-prompt", type=json.loads, default=None)
        add("--random-mm-bucket-config", type=str, default=None)
        add("--hf-subset", type=str, default=None)
        add("--hf-split", type=str, default=None)
        add("--hf-name", type=str, default=None)
        add("--hf-output-len", type=int, default=None)
        add("--bfcl-categories", type=str, default=None)
        add("--prefix-repetition-prefix-len", type=int, default=256)
        add("--prefix-repetition-suffix-len", type=int, default=256)
        add("--prefix-repetition-num-prefixes", type=int, default=10)
        add("--prefix-repetition-output-len", type=int, default=128)
        add(
            "--speed-bench-dataset-subset",
            type=str,
            default="qualitative",
            choices=["qualitative", "throughput_1k", "throughput_2k", "throughput_8k", "throughput_16k", "throughput_32k"],
        )
        add("--speed-bench-output-len", type=int, default=4096)
        add("--speed-bench-category", type=str, default=None)
        # add_cli_args (vllm/benchmarks/serve.py @ v0.32.0)
        add("--label", type=str, default=None)
        add("--backend", type=str, default="openai", choices=list(_V0_32_0_BACKENDS))
        add("--base-url", type=str, default=None)
        add("--host", type=str, default="127.0.0.1")
        add("--port", type=int, default=8000)
        add("--endpoint", type=str, default="/v1/completions")
        add("--header", metavar="KEY=VALUE", nargs="*")
        add("--max-concurrency", type=int, default=None)
        add("--model", type=str, required=False, default=None)
        add("--input-len", type=int, default=None)
        add("--output-len", type=int, default=None)
        add("--tokenizer", type=str)
        add("--tokenizer-mode", type=str, default="auto")
        add("--use-beam-search", action="store_true")
        add("--logprobs", type=int, default=None)
        add("--request-rate", type=float, default=float("inf"))
        add("--burstiness", type=float, default=1.0)
        add("--probe-request-rate", type=float, default=0.0)
        add("--disable-tqdm", action="store_true")
        add("--num-warmups", type=int, default=0)
        add("--profile", action="store_true")
        add("--save-result", action="store_true")
        add("--save-detailed", action="store_true")
        add("--append-result", action="store_true")
        add("--metadata", metavar="KEY=VALUE", nargs="*")
        add("--result-dir", type=str, default=None)
        add("--result-filename", type=str, default=None)
        add("--ignore-eos", action="store_true")
        add("--self-timed", action=argparse.BooleanOptionalAction, default=None)
        add("--percentile-metrics", type=str, default=None)
        add("--metric-percentiles", type=str, default="99")
        add("--goodput", nargs="+", required=False)
        add("--request-id-prefix", type=str, required=False, default="")
        add("--top-p", type=float, default=None)
        add("--top-k", type=int, default=None)
        add("--min-p", type=float, default=None)
        add("--temperature", type=float, default=None)
        add("--frequency-penalty", type=float, default=None)
        add("--presence-penalty", type=float, default=None)
        add("--repetition-penalty", type=float, default=None)
        add("--served-model-name", type=str, default=None)
        add("--lora-modules", nargs="+", default=None)
        add("--lora-assignment", type=str, default="random", choices=["random", "round-robin"])
        add("--ramp-up-strategy", type=str, default=None, choices=["linear", "exponential"])
        add("--ramp-up-start-rps", type=int, default=None)
        add("--ramp-up-end-rps", type=int, default=None)
        add("--ready-check-timeout-sec", type=int, default=0)
        add("--chat-template-kwargs", type=json.loads, default=None)
        add("--extra-body", type=json.loads, default=None)
        add("--skip-tokenizer-init", action="store_true", default=False)
        add("--insecure", action="store_true", default=False)
        add("--plot-timeline", action="store_true")
        add("--timeline-itl-thresholds", type=str, default="25,50")
        add("--plot-dataset-stats", action="store_true")

    def present_dests(self, argv: List[str]) -> Set[str]:
        present: Set[str] = set()
        for token in argv:
            if not token.startswith("--"):
                continue
            flag = token.split("=", 1)[0]
            dest = self.alias_to_dest.get(flag)
            if dest is not None:
                present.add(dest)
        return present


# ASYNC_REQUEST_FUNCS keys (vllm/benchmarks/lib/endpoint_request_func.py @ v0.32.0), in source order.
_V0_32_0_BACKENDS = (
    "vllm",
    "openai",
    "openai-chat",
    "openai-responses",
    "openai-audio",
    "openai-embeddings",
    "openai-embeddings-chat",
    "openai-embeddings-clip",
    "openai-embeddings-vlm2vec",
    "infinity-embeddings",
    "infinity-embeddings-clip",
    "vllm-pooling",
    "vllm-rerank",
)


def _flag(dest: str) -> str:
    return "--" + dest.replace("_", "-")


def _selected_backend(ns: argparse.Namespace, facts: VllmBenchFacts) -> str:
    """The request function key, read from whichever flag picks it at this tag."""
    value: str = getattr(ns, facts.endpoint_type_flag.removeprefix("--").replace("-", "_"))
    return value


def _sampling_note(facts: VllmBenchFacts, backend: str) -> Optional[str]:
    if facts.implicit_temperature is not None:
        return (
            f"sampling: vllm {facts.version} sends temperature={facts.implicit_temperature} and"
            f" repetition_penalty={facts.implicit_repetition_penalty} on every request;"
            " this config takes the server's sampling defaults"
        )
    if backend in _COMPLETION_BACKENDS:
        return (
            f"sampling: vllm {facts.version} sends repetition_penalty={facts.implicit_repetition_penalty} on every"
            " completions request and leaves temperature to the server; this config takes the server's sampling defaults"
        )
    # The chat payload carries no sampling field, so both tools take the server's defaults.
    return None


def _warmup_note(facts: VllmBenchFacts, ns: argparse.Namespace) -> Optional[str]:
    if not facts.warmup_tunable:
        return (
            f"warmup: vllm bench always sends {facts.warmup_requests} untunable test request before the run;"
            " inference-perf measures every request it sends"
        )
    probing = ns.ready_check_timeout_sec > 0
    count = facts.warmup_requests + max(0, ns.num_warmups) + (1 if probing else 0)
    if count == 0:
        return None
    # The ready check retries every 5 seconds until the endpoint answers, so its count is a floor.
    return (
        f"warmup: vllm bench sends {'at least ' if probing else ''}{count} unmeasured"
        f" request{'s' if count != 1 else ''} before the run (the --ready-check-timeout-sec probe and"
        " --num-warmups copies of the first prompt); inference-perf measures every request it sends"
    )


def _convert_load(
    conversion: Conversion,
    ns: argparse.Namespace,
    present: Set[str],
    facts: VllmBenchFacts,
) -> Tuple[Optional[LoadType], Optional[StandardLoadStage], Optional[ConcurrentLoadStage]]:
    num_prompts: int = ns.num_prompts
    rate: float = ns.request_rate
    burstiness: float = ns.burstiness
    max_concurrency: Optional[int] = ns.max_concurrency

    if facts.max_concurrency_positive and max_concurrency is not None and max_concurrency <= 0:
        raise PeerUsageError(f"--max-concurrency must be greater than 0 at vllm {facts.version}, got {max_concurrency}")
    if getattr(ns, "probe_request_rate", 0.0) > 0:
        conversion.refuse(
            "--probe-request-rate",
            "single-token probes sent alongside the run, outside --max-concurrency, add load no stage produces",
        )
        return None, None, None
    if ns.ramp_up_strategy is not None:
        conversion.refuse("--ramp-up-strategy", "a continuous per-request rate ramp is not expressible with stepwise stages")
        return None, None, None
    if "ramp_up_start_rps" in present or "ramp_up_end_rps" in present:
        # Read only when --ramp-up-strategy is set; inert on their own.
        conversion.drop("--ramp-up-start-rps/--ramp-up-end-rps (inert without --ramp-up-strategy)")

    if math.isinf(rate):
        if "burstiness" in present and burstiness != 1.0:
            conversion.drop("--burstiness (inert with an infinite request rate)")
        level = max_concurrency if max_concurrency is not None else num_prompts
        if max_concurrency is None:
            conversion.assume(
                "vllm offers all prompts at t=0 at the default infinite --request-rate; the nearest equivalent is"
                " a concurrent stage with concurrency_level equal to num_requests"
                ' (docs/comparability.md "Where the map breaks")'
            )
        else:
            conversion.assume(
                f"--max-concurrency is a plain semaphore at {facts.version}, the same closed-loop meaning as concurrency_level"
            )
        conversion.assume(
            "the worker pool must cover the concurrency level: load.num_workers x load.worker_max_concurrency"
            f" (defaults: cpu count x 100) must be at least {level}"
        )
        return LoadType.CONCURRENT, None, ConcurrentLoadStage(num_requests=num_prompts, concurrency_level=level)

    if max_concurrency is not None:
        conversion.refuse(
            "--max-concurrency with a finite --request-rate",
            "a rate-limited closed loop fits neither constant/poisson stages (no concurrency cap)"
            " nor concurrent stages (no rate)",
        )
        return None, None, None
    if burstiness != 1.0:
        if math.isinf(burstiness) and facts.burstiness_inf_even:
            conversion.refuse(
                "--burstiness",
                f"burstiness inf spaces requests exactly 1/rate apart at {facts.version}, which no load type"
                " produces (constant scales random gaps to the stage duration)",
            )
            return None, None, None
        conversion.refuse(
            "--burstiness",
            f"burstiness {burstiness} means gamma-spaced arrivals, which no load type produces"
            " (only burstiness 1.0, Poisson, converts)",
        )
        return None, None, None
    duration = num_prompts / rate
    if duration <= 0 or duration != int(duration):
        conversion.refuse(
            "--num-prompts/--request-rate",
            f"the implied stage duration {num_prompts}/{rate} is not a positive integer number of seconds"
            " (stage duration is int seconds)",
        )
        return None, None, None
    conversion.assume(
        f"stages[0].duration: {int(duration)} = --num-prompts {num_prompts} / --request-rate {rate:g};"
        " the converted run matches the request count only if the offered rate is achieved"
        ' (docs/comparability.md "Count against duration")'
    )
    conversion.assume(
        "a finite --request-rate with --burstiness 1.0 is Poisson arrivals, emitted as load.type: poisson;"
        " the parity fixtures declare each tool's arrival spacing in expected.yaml"
    )
    return LoadType.POISSON, StandardLoadStage(rate=rate, duration=int(duration)), None


def _range_ratios(value: object, facts: VllmBenchFacts) -> Tuple[float, float]:
    """(input, output) ratios, read the way get_samples and _resolve_range_ratios read --random-range-ratio."""
    if not facts.range_ratio_per_direction:
        assert isinstance(value, float)
        return value, value
    assert isinstance(value, str)
    try:
        ratio = float(value)
        return ratio, ratio
    except ValueError:
        pass
    try:
        parsed = json.loads(value)
    except ValueError as exc:
        raise PeerUsageError(f"--random-range-ratio '{value}' is neither a number nor JSON") from exc
    if not isinstance(parsed, dict) or "input" not in parsed or "output" not in parsed:
        raise PeerUsageError(f"--random-range-ratio '{value}' must be a number or a JSON object with input and output keys")
    try:
        return float(parsed["input"]), float(parsed["output"])
    except (TypeError, ValueError) as exc:
        raise PeerUsageError(f"--random-range-ratio '{value}' holds a non-numeric ratio") from exc


def _bounded_dist(low: int, high: int) -> Distribution:
    return fixed_dist(low) if low == high else uniform_dist(low, high)


def _convert_data(
    conversion: Conversion, ns: argparse.Namespace, present: Set[str], facts: VllmBenchFacts
) -> Optional[DataConfig]:
    dataset: str = ns.dataset_name
    if dataset != "random":
        conversion.refuse("--dataset-name", f"dataset '{dataset}' is not mapped: {_DATASET_REFUSALS[dataset]}")
        return None
    for dest in _RANDOM_REJECTS[facts.version]:
        if dest in present:
            raise PeerUsageError(f"vllm {facts.version} rejects {_flag(dest)} with the random dataset")
    for dest in _OTHER_DATASET_FLAGS[facts.version]:
        if dest in present:
            conversion.drop(f"{_flag(dest)} (inert with the random dataset at {facts.version})")
    if "no_stream" in present:
        conversion.drop("--no-stream (HF dataset loading mode, not response streaming; inert with the random dataset)")
    if ns.random_prefix_len > 0:
        conversion.refuse(
            "--random-prefix-len",
            "a single random prefix sampled once and shared by every request is a prefix-cache-sensitive"
            " workload with no verified equivalent",
        )
        return None
    if getattr(ns, "random_batch_size", 1) > 1:
        conversion.refuse(
            "--random-batch-size",
            "several prompts packed into one request body have no equivalent; vllm applies it to every backend",
        )
        return None

    input_len: int = ns.random_input_len
    output_len: int = ns.random_output_len
    # main_async copies these general lengths over the random_* ones before sampling.
    if getattr(ns, "input_len", None) is not None:
        input_len = ns.input_len
        conversion.assume(f"--input-len {input_len} replaces --random-input-len")
    if getattr(ns, "output_len", None) is not None:
        output_len = ns.output_len
        conversion.assume(f"--output-len {output_len} replaces --random-output-len")
    input_ratio, output_ratio = _range_ratios(ns.random_range_ratio, facts)
    for ratio in (input_ratio, output_ratio):
        if not 0.0 <= ratio < 1.0:
            conversion.refuse("--random-range-ratio", f"vllm {facts.version} requires a range ratio in [0, 1), got {ratio}")
            return None
    conversion.assume(
        f"--random-input-len {input_len}: vllm subtracts the tokenizer's special tokens before sampling,"
        " so its realized prompts run about one token short for BOS-prepending tokenizers; mapped verbatim"
    )
    if facts.server_prompt_alignment:
        conversion.assume(
            f"vllm {facts.version} re-cuts its random prompts to the server's /tokenize count when the server's"
            " tokenizer disagrees with the local one; inference-perf trusts the configured tokenizer"
        )

    if not facts.range_bounds_floor_ceil:
        ratio = input_ratio
        if ratio == 0.0:
            input_dist = fixed_dist(input_len)
            output_dist = fixed_dist(output_len)
        else:
            conversion.assume(
                f"--random-range-ratio {ratio:g} fans out to both the input and the output distribution"
                " (uniform on [len*(1-r), len*(1+r)]); bounds are integer-truncated like vllm's randint arguments"
            )
            input_dist = uniform_dist(int(input_len * (1 - ratio)), int(input_len * (1 + ratio)))
            output_dist = uniform_dist(int(output_len * (1 - ratio)), int(output_len * (1 + ratio)))
        return DataConfig(type=DataGenType.Random, input_distribution=input_dist, output_distribution=output_dist)

    # get_sampling_params: floor/ceil bounds, and the output never drops below 1 token.
    input_dist = _bounded_dist(math.floor(input_len * (1 - input_ratio)), math.ceil(input_len * (1 + input_ratio)))
    output_low = max(math.floor(output_len * (1 - output_ratio)), 1)
    output_high = max(math.ceil(output_len * (1 + output_ratio)), 1)
    output_dist = _bounded_dist(output_low, output_high)
    if input_ratio > 0.0 or output_ratio > 0.0:
        split = (
            f"{input_ratio:g} fans out to both the input and the output distribution"
            if input_ratio == output_ratio
            else f"sets the input ratio to {input_ratio:g} and the output ratio to {output_ratio:g}"
        )
        conversion.assume(
            f"--random-range-ratio {split} (uniform on [len*(1-r), len*(1+r)]); vllm {facts.version} rounds the"
            " bounds outward (floor, ceil)"
        )
    if output_low > math.floor(output_len * (1 - output_ratio)):
        conversion.assume(f"vllm {facts.version} raises the output length floor to 1 token; mapped as {output_low}")
    return DataConfig(type=DataGenType.Random, input_distribution=input_dist, output_distribution=output_dist)


def _convert_api(conversion: Conversion, ns: argparse.Namespace, facts: VllmBenchFacts) -> Optional[APIConfig]:
    backend = _selected_backend(ns, facts)
    flag = facts.endpoint_type_flag
    if backend == "openai-audio":
        conversion.refuse(flag, "openai-audio has no API type (completion, chat and anthropic_messages exist)")
        return None
    if backend == "openai-responses":
        conversion.refuse(flag, "the Responses API has no API type (completion, chat and anthropic_messages exist)")
        return None
    if backend not in _COMPLETION_BACKENDS + _CHAT_BACKENDS:
        conversion.refuse(
            flag,
            f"pooling backend '{backend}' is not mapped: the embeddings and rerank API types are unverified"
            " against vllm's pooling payloads",
        )
        return None
    api_type = APIType.Chat if backend in _CHAT_BACKENDS else APIType.Completion
    if api_type is APIType.Chat:
        conversion.assume(
            "chat path: vllm wraps the raw prompt as a single user message and does not model chat-template"
            " overhead, so prompt-length accounting differs from the completions path"
        )
    expected_path = _ENDPOINT_PATHS[api_type]
    if ns.endpoint != expected_path:
        conversion.refuse(
            "--endpoint",
            f"custom endpoint path '{ns.endpoint}' is not expressible: the {api_type.value} client owns its"
            f" route '{expected_path}'",
        )
        return None
    # Streaming is hardcoded in every request payload at both tags; non-streaming
    # is not expressible on that CLI, so the wire truth is emitted explicitly.
    return APIConfig(type=api_type, streaming=True)


def _convert_goodput(conversion: Conversion, ns: argparse.Namespace) -> Optional[GoodputConfig]:
    if not ns.goodput:
        return None
    key_map = {"ttft": "ttft", "tpot": "tpot", "e2el": "request_latency"}
    constraints: Dict[str, float] = {}
    entries: List[str] = ns.goodput
    for entry in entries:
        key, sep, value = entry.partition(":")
        if not sep:
            raise PeerUsageError(f"--goodput entry '{entry}' is not KEY:VALUE")
        if key not in key_map:
            conversion.refuse("--goodput", f"unknown goodput key '{key}' (ttft, tpot and e2el convert)")
            return None
        try:
            millis = float(value)
        except ValueError as exc:
            raise PeerUsageError(f"--goodput value '{value}' is not a number") from exc
        constraints[key_map[key]] = millis / 1000.0
    conversion.assume("goodput thresholds are converted from vllm's milliseconds to seconds")
    return GoodputConfig(constraints=constraints)


def _convert_lora(conversion: Conversion, ns: argparse.Namespace, present: Set[str]) -> Optional[List[MultiLoRAConfig]]:
    modules: Optional[List[str]] = ns.lora_modules
    if not modules:
        if "lora_assignment" in present:
            conversion.drop("--lora-assignment (inert without --lora-modules)")
        return None
    if getattr(ns, "lora_assignment", "random") == "round-robin":
        conversion.refuse(
            "--lora-assignment",
            "round-robin cycles the adapters in a fixed order, while lora_traffic_split draws one per request",
        )
        return None
    count = len(modules)
    split = 1.0 / count
    if sum(split for _ in range(count)) != 1.0:
        conversion.refuse(
            "--lora-modules",
            f"an equal split across {count} adapters does not sum to exactly 1.0 in floating point,"
            " which load.lora_traffic_split validation rejects",
        )
        return None
    conversion.assume("vllm picks a LoRA module uniformly at random per request; mapped to an equal lora_traffic_split")
    return [MultiLoRAConfig(name=name, split=split) for name in modules]


def _base_url(ns: argparse.Namespace, facts: VllmBenchFacts) -> str:
    if ns.base_url is not None:
        base_url: str = ns.base_url
        return base_url
    host: str = ns.host
    if facts.ipv6_host_bracketed:
        try:
            ipaddress.IPv6Address(host)
            host = f"[{host}]"
        except ValueError:
            pass
    return f"http://{host}:{ns.port}"


def convert_vllm_bench(argv: List[str], peer_version: str) -> Conversion:
    """Convert one ``vllm bench serve`` argv. Never writes a partial config."""
    conversion = Conversion(source_tool=TOOL_NAME, source_version=peer_version, argv=list(argv))
    facts: Optional[VllmBenchFacts] = resolve_vllm_bench(peer_version)
    if facts is None:
        conversion.refuse("--peer-version", unverified_version_reason(TOOL_NAME, peer_version, dict(VLLM_BENCH_VERSIONS)))
        return conversion
    conversion.source_version = facts.version

    mirror = _Mirror(facts.version)
    ns, unknown = mirror.parser.parse_known_args(argv)
    # argparse hands an unknown flag and its value back as two tokens; consume
    # the value, a negative number included, so one unknown flag refuses once.
    i = 0
    while i < len(unknown):
        token = unknown[i]
        i += 1
        if is_value_token(token):
            conversion.refuse(token, f"{TOOL_NAME} takes no positional arguments")
            continue
        conversion.refuse(token.split("=", 1)[0], f"not in the verified {TOOL_NAME} {facts.version} flag surface")
        if "=" not in token and i < len(unknown) and is_value_token(unknown[i]):
            i += 1
    present = mirror.present_dests(argv)

    # Standing annotations: they hold for every conversion at this tag.
    for note in (_sampling_note(facts, _selected_backend(ns, facts)), _warmup_note(facts, ns)):
        if note is not None:
            conversion.no_equivalent(note)

    # Flags that abort: sampling parameters and the measured-window changers.
    sampling_dests = ("top_p", "top_k", "min_p", "temperature", "frequency_penalty", "presence_penalty", "repetition_penalty")
    if ns.logprobs is not None:
        conversion.refuse("--logprobs", "per-token logprobs change server work and response size per token; not expressible")
    for dest in sampling_dests:
        if getattr(ns, dest, None) is not None:
            conversion.refuse(_flag(dest), _SAMPLING_REFUSAL)
    if ns.profile:
        conversion.refuse("--profile", "server-side torch profiling inside the measured window changes the run")
    if ns.tokenizer_mode != "auto":
        conversion.refuse("--tokenizer-mode", f"'{ns.tokenizer_mode}' has no tokenizer-backend selector (only 'auto' maps)")
    elif "tokenizer_mode" in present:
        if facts.tokenizer_auto_mistral:
            conversion.assume(
                f"--tokenizer-mode auto at {facts.version} loads mistral_common for Mistral models when it is"
                " installed and the HF fast tokenizer otherwise; inference-perf always loads the HF tokenizer"
            )
        else:
            conversion.assume("--tokenizer-mode auto assumes a fast tokenizer is available, matching AutoTokenizer behavior")
    if getattr(ns, "header", None):
        conversion.refuse("--header", "per-request headers are not mapped; set api.headers on the run host if they matter")
    if getattr(ns, "extra_body", None):
        conversion.refuse("--extra-body", "arbitrary request body fields can change the generation; not expressible")

    # Reporting and tool plumbing: workload-neutral, dropped with a record.
    if "label" in present:
        conversion.drop("--label")
    if "backend" in present and facts.endpoint_type_flag != "--backend":
        conversion.drop(f"--backend (no effect at {facts.version}; {facts.endpoint_type_flag} selects the request path)")
    if ns.use_beam_search:
        conversion.drop(f"--use-beam-search (parsed and never used at {facts.version})")
    if ns.disable_tqdm:
        conversion.drop("--disable-tqdm")
    for dest in ("save_result", "save_detailed", "append_result", "plot_timeline", "plot_dataset_stats", "insecure"):
        if getattr(ns, dest, False):
            conversion.drop(_flag(dest))
    for dest in ("result_filename", "timeline_itl_thresholds", "request_id_prefix"):
        if dest in present:
            conversion.drop(_flag(dest))
    if "percentile_metrics" in present:
        conversion.drop("--percentile-metrics (metric selection; inference-perf always reports its full metric set)")
    if ns.metadata:
        conversion.drop("--metadata " + " ".join(ns.metadata))

    api = _convert_api(conversion, ns, facts)
    data = _convert_data(conversion, ns, present, facts)
    load_type, standard_stage, concurrent_stage = _convert_load(conversion, ns, present, facts)
    goodput = _convert_goodput(conversion, ns)
    lora = _convert_lora(conversion, ns, present)

    percentiles: Optional[List[float]] = None
    if "metric_percentiles" in present:
        try:
            percentiles = [float(p) for p in str(ns.metric_percentiles).split(",") if p.strip()]
        except ValueError as exc:
            raise PeerUsageError(
                f"--metric-percentiles '{ns.metric_percentiles}' is not a comma-separated number list"
            ) from exc
        conversion.assume(
            "--metric-percentiles replaces the default percentile list and applies to every reported metric,"
            " not only the --percentile-metrics selection"
        )

    conversion.assume(
        f"load.base_seed: {ns.seed} from --seed; seeds are tool-local, so this pins reproducibility of the"
        " converted run, not identical prompt text"
    )
    # Only the random dataset on a completions or chat backend converts, so every
    # config that comes out at a forcing tag carries the forced reading.
    ignore_eos = bool(ns.ignore_eos) or facts.ignore_eos_forced_for_random
    if facts.ignore_eos_forced_for_random:
        conversion.assume(
            f"server.ignore_eos: true: vllm {facts.version} sets ignore_eos on every random-dataset request,"
            " with or without --ignore-eos"
        )
    elif not ns.ignore_eos:
        conversion.assume(
            "server.ignore_eos: false is emitted explicitly: inference-perf defaults it to true, while vllm"
            " sends ignore_eos only when --ignore-eos is passed"
        )

    if conversion.refusals:
        return conversion
    assert api is not None and data is not None and load_type is not None

    base_url = _base_url(ns, facts)
    model: Optional[str] = ns.model
    model_name: Optional[str]
    if model is None:
        # Reachable only where the mirror leaves --model optional.
        model_name = None
        conversion.assume(
            "no --model: both tools send the first /v1/models entry's id as the model name; vllm loads its"
            " tokenizer from that entry's root, inference-perf tries the id first, then the root"
        )
        if ns.served_model_name is not None:
            conversion.drop(f"--served-model-name (unread without --model at {facts.version})")
    else:
        model_name = ns.served_model_name if ns.served_model_name is not None else model
        if ns.served_model_name is not None:
            conversion.assume("--served-model-name sets the payload model name; --model keeps only the tokenizer role")

    if standard_stage is not None:
        stages: List[Any] = [standard_stage]
    else:
        assert concurrent_stage is not None
        stages = [concurrent_stage]
    load_kwargs: Dict[str, Any] = {"type": load_type, "stages": stages, "base_seed": ns.seed}
    if lora is not None:
        load_kwargs["lora_traffic_split"] = lora
    load = LoadConfig(**load_kwargs)

    server_kwargs: Dict[str, Any] = {"base_url": base_url, "ignore_eos": ignore_eos}
    if model_name is not None:
        server_kwargs["model_name"] = model_name
    server = ModelServerClientConfig(**server_kwargs)
    tokenizer_id: Optional[str] = ns.tokenizer if ns.tokenizer is not None else model
    tokenizer_kwargs: Dict[str, Any] = {}
    if tokenizer_id is not None:
        tokenizer_kwargs["pretrained_model_name_or_path"] = tokenizer_id
    if ns.trust_remote_code:
        tokenizer_kwargs["trust_remote_code"] = True
    tokenizer = CustomTokenizerConfig(**tokenizer_kwargs) if tokenizer_kwargs else None

    report: Optional[ReportConfig] = None
    if goodput is not None or percentiles is not None:
        report_kwargs: Dict[str, Any] = {}
        if percentiles is not None:
            report_kwargs["request_lifecycle"] = RequestLifecycleMetricsReportConfig(percentiles=percentiles)
        if goodput is not None:
            report_kwargs["goodput"] = goodput
        report = ReportConfig(**report_kwargs)

    storage: Optional[StorageConfig] = None
    if ns.result_dir is not None:
        conversion.assume(
            "--result-dir maps to storage.local_storage.path; inference-perf writes its own report file set"
            " there, not vllm's result JSON shape"
        )
        storage = StorageConfig(local_storage=StorageConfigBase(path=ns.result_dir))

    kwargs: Dict[str, Any] = {"api": api, "data": data, "load": load, "server": server}
    if tokenizer is not None:
        kwargs["tokenizer"] = tokenizer
    if report is not None:
        kwargs["report"] = report
    if storage is not None:
        kwargs["storage"] = storage
    conversion.config = Config(**kwargs)
    return conversion
