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
"""``vllm bench serve`` argv -> inference-perf config, keyed to vllm v0.10.0.

The mirrored parser reproduces the pinned version's argparse surface (18
dataset flags + 37 serve flags), so an unknown flag is a refusal rather than a
silent drop: that is how surface drift on newer vllm versions gets caught
instead of ignored.

Verified v0.10.0 behavior the rows below rely on:

- The request payload always streams and always carries ``temperature 0.0``
  and ``repetition_penalty 1.0`` when the user passed no sampling flags.
- ``benchmark()`` sends one untunable test request before the run.
- ``--request-rate`` defaults to ``inf`` (all prompts offered at t=0); a
  finite rate with the default ``--burstiness 1.0`` gives Poisson spacing.
- ``RandomDataset`` draws lengths uniform on ``[len*(1-r), len*(1+r)]`` and
  subtracts the tokenizer's special tokens from the requested input length.
"""

import argparse
import math
from typing import Any, Dict, List, NoReturn, Optional, Set, Tuple

from inference_perf.config import (
    APIConfig,
    APIType,
    Config,
    DataConfig,
    DataGenType,
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
from inference_perf.tools.convert.model import Conversion, PeerUsageError, fixed_dist, uniform_dist
from inference_perf.tools.convert.versions import (
    VLLM_BENCH_VERSIONS,
    VllmBenchFacts,
    resolve_vllm_bench,
    unverified_version_reason,
)

TOOL_NAME = "vllm bench serve"

# The two standing NO EQUIVALENT annotations every conversion carries
# (docs/comparability.md "Where the map breaks").
SAMPLING_NOTE = (
    "sampling: vllm v0.10.0 sends temperature=0.0 and repetition_penalty=1.0 on every request;"
    " this config takes the server's sampling defaults"
)
WARMUP_NOTE = (
    "warmup: vllm bench always sends 1 untunable test request before the run; inference-perf measures every request it sends"
)

_ENDPOINT_PATHS = {
    APIType.Completion: "/v1/completions",
    APIType.Chat: "/v1/chat/completions",
}

_DATASET_REFUSALS = {
    "sharegpt": "vllm filters ShareGPT turns by length rules the shareGPT datagen does not replicate",
    "sonnet": "line-assembled poem prompts with a shared prefix have no equivalent generator",
    "burstgpt": "trace-file dataset with no equivalent",
    "hf": "HF dataset loader transforms are unverified",
    "custom": "arbitrary jsonl dataset with no equivalent",
}

# Flags that only configure a dataset other than `random`. With `random`
# selected they are parsed and never read by vllm at v0.10.0, so they cannot
# change the offered workload and are dropped with an annotation; with their
# own dataset selected the dataset row already refuses.
_OTHER_DATASET_FLAGS = (
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
)


class _MirrorParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise PeerUsageError(f"invalid {TOOL_NAME} arguments: {message}")


class _Mirror:
    """The pinned argparse surface plus a flag registry for presence checks."""

    def __init__(self) -> None:
        self.parser = _MirrorParser(prog=TOOL_NAME, add_help=False, allow_abbrev=False)
        self.alias_to_dest: Dict[str, str] = {}
        self.dest_to_flag: Dict[str, str] = {}
        self._build()

    def _add(self, *names: str, **kwargs: Any) -> None:
        action = self.parser.add_argument(*names, **kwargs)
        for name in names:
            self.alias_to_dest[name] = action.dest
        self.dest_to_flag[action.dest] = names[0]

    def _build(self) -> None:
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


def _convert_load(
    conversion: Conversion,
    ns: argparse.Namespace,
    present: Set[str],
) -> Tuple[Optional[LoadType], Optional[StandardLoadStage], Optional[ConcurrentLoadStage]]:
    num_prompts: int = ns.num_prompts
    rate: float = ns.request_rate
    burstiness: float = ns.burstiness
    max_concurrency: Optional[int] = ns.max_concurrency

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
                "--max-concurrency is a plain semaphore at v0.10.0, the same closed-loop meaning as concurrency_level"
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


def _convert_data(conversion: Conversion, ns: argparse.Namespace, present: Set[str]) -> Optional[DataConfig]:
    dataset: str = ns.dataset_name
    if dataset != "random":
        conversion.refuse("--dataset-name", f"dataset '{dataset}' is not mapped: {_DATASET_REFUSALS[dataset]}")
        return None
    for dest in _OTHER_DATASET_FLAGS:
        if dest in present:
            flag = "--" + dest.replace("_", "-")
            conversion.drop(f"{flag} (inert with the random dataset at v0.10.0)")
    if "no_stream" in present:
        conversion.drop("--no-stream (HF dataset loading mode, not response streaming; inert with the random dataset)")
    if ns.random_prefix_len > 0:
        conversion.refuse(
            "--random-prefix-len",
            "a single random prefix sampled once and shared by every request is a prefix-cache-sensitive"
            " workload with no verified equivalent",
        )
        return None

    input_len: int = ns.random_input_len
    output_len: int = ns.random_output_len
    ratio: float = ns.random_range_ratio
    if not 0.0 <= ratio < 1.0:
        conversion.refuse("--random-range-ratio", f"vllm v0.10.0 requires a range ratio in [0, 1), got {ratio}")
        return None
    conversion.assume(
        f"--random-input-len {input_len}: vllm subtracts the tokenizer's special tokens before sampling,"
        " so its realized prompts run about one token short for BOS-prepending tokenizers; mapped verbatim"
    )
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


def _convert_api(conversion: Conversion, ns: argparse.Namespace) -> Optional[APIConfig]:
    endpoint_type: str = ns.endpoint_type
    if endpoint_type == "openai-audio":
        conversion.refuse("--endpoint-type", "openai-audio has no API type (completion, chat and anthropic_messages exist)")
        return None
    api_type = APIType.Chat if endpoint_type == "openai-chat" else APIType.Completion
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
    # Streaming is hardcoded in every v0.10.0 request payload; non-streaming
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


def _convert_lora(conversion: Conversion, ns: argparse.Namespace) -> Optional[List[MultiLoRAConfig]]:
    modules: Optional[List[str]] = ns.lora_modules
    if not modules:
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


def convert_vllm_bench(argv: List[str], peer_version: str) -> Conversion:
    """Convert one ``vllm bench serve`` argv. Never writes a partial config."""
    conversion = Conversion(source_tool=TOOL_NAME, source_version=peer_version, argv=list(argv))
    facts: Optional[VllmBenchFacts] = resolve_vllm_bench(peer_version)
    if facts is None:
        conversion.refuse("--peer-version", unverified_version_reason(TOOL_NAME, peer_version, dict(VLLM_BENCH_VERSIONS)))
        return conversion
    conversion.source_version = facts.version

    mirror = _Mirror()
    ns, unknown = mirror.parser.parse_known_args(argv)
    for token in unknown:
        if token.startswith("-"):
            conversion.refuse(token.split("=", 1)[0], f"not in the verified {TOOL_NAME} {facts.version} flag surface")
        else:
            conversion.refuse(token, f"{TOOL_NAME} takes no positional arguments")
    present = mirror.present_dests(argv)

    # Standing annotations: they hold for every conversion at this pin.
    conversion.no_equivalent(SAMPLING_NOTE)
    conversion.no_equivalent(WARMUP_NOTE)

    # Flags that abort: sampling parameters and the measured-window changers.
    for dest, reason in (
        ("logprobs", "per-token logprobs change server work and response size per token; not expressible"),
        (
            "top_p",
            "sampling parameters are not configurable for synthetic data; a guessed value would misstate the decode work",
        ),
        (
            "top_k",
            "sampling parameters are not configurable for synthetic data; a guessed value would misstate the decode work",
        ),
        (
            "min_p",
            "sampling parameters are not configurable for synthetic data; a guessed value would misstate the decode work",
        ),
        (
            "temperature",
            "sampling parameters are not configurable for synthetic data; a guessed value would misstate the decode work",
        ),
    ):
        if getattr(ns, dest) is not None:
            conversion.refuse("--" + dest.replace("_", "-"), reason)
    if ns.profile:
        conversion.refuse("--profile", "server-side torch profiling inside the measured window changes the run")
    if ns.tokenizer_mode != "auto":
        conversion.refuse("--tokenizer-mode", f"'{ns.tokenizer_mode}' has no tokenizer-backend selector (only 'auto' maps)")
    elif "tokenizer_mode" in present:
        conversion.assume("--tokenizer-mode auto assumes a fast tokenizer is available, matching AutoTokenizer behavior")

    # Reporting and tool plumbing: workload-neutral, dropped with a record.
    if "label" in present:
        conversion.drop("--label")
    if "backend" in present:
        conversion.drop("--backend (no effect at v0.10.0; --endpoint-type selects the request path)")
    if ns.use_beam_search:
        conversion.drop("--use-beam-search (parsed and never used at v0.10.0)")
    if ns.disable_tqdm:
        conversion.drop("--disable-tqdm")
    for dest in ("save_result", "save_detailed", "append_result"):
        if getattr(ns, dest):
            conversion.drop("--" + dest.replace("_", "-"))
    if "result_filename" in present:
        conversion.drop("--result-filename")
    if "percentile_metrics" in present:
        conversion.drop("--percentile-metrics (metric selection; inference-perf always reports its full metric set)")
    if ns.metadata:
        conversion.drop("--metadata " + " ".join(ns.metadata))

    api = _convert_api(conversion, ns)
    data = _convert_data(conversion, ns, present)
    load_type, standard_stage, concurrent_stage = _convert_load(conversion, ns, present)
    goodput = _convert_goodput(conversion, ns)
    lora = _convert_lora(conversion, ns)

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
    if not ns.ignore_eos:
        conversion.assume(
            "server.ignore_eos: false is emitted explicitly: inference-perf defaults it to true, while vllm"
            " sends ignore_eos only when --ignore-eos is passed"
        )

    if conversion.refusals:
        return conversion
    assert api is not None and data is not None and load_type is not None

    base_url: str = ns.base_url if ns.base_url is not None else f"http://{ns.host}:{ns.port}"
    model_name: str = ns.served_model_name if ns.served_model_name is not None else ns.model
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

    server = ModelServerClientConfig(base_url=base_url, model_name=model_name, ignore_eos=bool(ns.ignore_eos))
    tokenizer_id: str = ns.tokenizer if ns.tokenizer is not None else ns.model
    tokenizer_kwargs: Dict[str, Any] = {"pretrained_model_name_or_path": tokenizer_id}
    if ns.trust_remote_code:
        tokenizer_kwargs["trust_remote_code"] = True
    tokenizer = CustomTokenizerConfig(**tokenizer_kwargs)

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

    kwargs: Dict[str, Any] = {"api": api, "data": data, "load": load, "server": server, "tokenizer": tokenizer}
    if report is not None:
        kwargs["report"] = report
    if storage is not None:
        kwargs["storage"] = storage
    conversion.config = Config(**kwargs)
    return conversion
