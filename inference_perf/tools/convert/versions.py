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
"""Per-tool registry of verified peer versions.

Flag semantics change between peer releases (the pre-v0.8.4 inversion of
vllm's ``--random-range-ratio`` is the origin story of the parity cases), so
the caller supplies the peer version and the converter refuses any version
whose surface has not been verified against source. There is no
nearest-version fallback: a version not in the registry is a refusal, not a
guess.

Each entry records the semantic facts the conversion tables rely on, so a
future version bump is a table re-verification rather than a flag
pass-through. The vllm pin is the same version the parity harness pins in
``e2e/utils/vllm_bench.py`` (``VLLM_PINNED_REF``); if that file exists in the
tree the two constants are asserted equal by the parity test, so the converter
and the parity workloads cannot drift apart silently.
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Tuple


@dataclass(frozen=True)
class VllmBenchFacts:
    """Verified behavior of ``vllm bench serve`` at one tag."""

    version: str
    # Request-function choices (ASYNC_REQUEST_FUNCS keys at this tag).
    endpoint_types: FrozenSet[str] = frozenset({"vllm", "openai", "openai-chat", "openai-audio"})
    # The flag that picks the request function. v0.10.0 also parses --backend
    # and never reads it; vllm removed --endpoint-type in v0.11.1.
    endpoint_type_flag: str = "--endpoint-type"
    # Sampling params sent when the user passes none: temperature on every
    # request (None leaves it to the server), repetition_penalty on
    # completions requests.
    implicit_temperature: Optional[float] = 0.0
    implicit_repetition_penalty: float = 1.0
    # benchmark() sends this many test requests before the run by default.
    warmup_requests: int = 1
    # --ready-check-timeout-sec and --num-warmups exist and change that count.
    warmup_tunable: bool = False
    # --random-range-ratio r draws lengths uniform on [len*(1-r), len*(1+r)];
    # False would mean the pre-v0.8.4 reading (0 = anywhere from 0 to len).
    range_ratio_symmetric: bool = True
    # --random-range-ratio also takes '{"input": a, "output": b}'.
    range_ratio_per_direction: bool = False
    # Bounds are floor/ceil with the output floor at 1, not int() truncation.
    range_bounds_floor_ceil: bool = False
    # Streaming is hardcoded in the request payload and cannot be disabled.
    stream_forced: bool = True
    # main_async sets ignore_eos for the random dataset on OpenAI-compatible
    # backends, with or without --ignore-eos.
    ignore_eos_forced_for_random: bool = False
    # --model may be left out; vllm then reads the first /v1/models entry.
    model_optional: bool = False
    # --tokenizer-mode auto picks mistral_common for Mistral models.
    tokenizer_auto_mistral: bool = False
    # An IPv6 --host is bracketed in the URL (join_host_port).
    ipv6_host_bracketed: bool = False
    # main_async rejects --max-concurrency 0 or below.
    max_concurrency_positive: bool = False
    # --burstiness inf spaces requests exactly 1/rate apart.
    burstiness_inf_even: bool = False
    # Random prompts are re-cut to the server's /tokenize count when the
    # server's tokenizer disagrees with the local one.
    server_prompt_alignment: bool = False


_VLLM_V0_32_0 = VllmBenchFacts(
    version="v0.32.0",
    endpoint_types=frozenset(
        {
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
        }
    ),
    endpoint_type_flag="--backend",
    implicit_temperature=None,
    warmup_requests=0,
    warmup_tunable=True,
    range_ratio_per_direction=True,
    range_bounds_floor_ceil=True,
    ignore_eos_forced_for_random=True,
    model_optional=True,
    tokenizer_auto_mistral=True,
    ipv6_host_bracketed=True,
    max_concurrency_positive=True,
    burstiness_inf_even=True,
    server_prompt_alignment=True,
)


@dataclass(frozen=True)
class AiperfProfileFacts:
    """Verified behavior of ``aiperf profile`` at one tag."""

    version: str
    # Endpoint types registered in the plugin registry at this tag.
    endpoint_types: FrozenSet[str] = frozenset(
        {
            "chat",
            "completions",
            "messages",
            "cohere_rankings",
            "responses",
            "chat_embeddings",
            "embeddings",
            "hf_tei_rankings",
            "huggingface_generate",
            "image_generation",
            "image_edit",
            "video_generation",
            "image_retrieval",
            "nim_embeddings",
            "nim_rankings",
            "solido_rag",
            "raw",
            "template",
        }
    )
    arrival_patterns: FrozenSet[str] = frozenset({"constant", "poisson", "gamma"})
    # An unbounded run (no --request-count, --benchmark-duration or session
    # bound, and not a fixed schedule) sends exactly this many requests:
    # src/aiperf/config/flags/_converter_profiling.py, _validate_profiling,
    # ``prof.setdefault("requests", 10)``. Nothing re-derives it from the
    # concurrency; the ``max(10, concurrency * 2)`` in the --request-count
    # help string is a formula the code does not implement.
    unbounded_request_count: int = 10
    # CLI fields whose BeforeValidator is parse_int_or_int_list or
    # parse_float_or_float_list (cli_config.py): a comma-separated value is a
    # parameter sweep, one run per value, and a trailing comma collapses to
    # the single value. The same names as MAGIC_LIST_FIELDS in sweep/expand.py.
    sweep_fields: FrozenSet[str] = frozenset(
        {
            "benchmark_duration",
            "concurrency",
            "conversation_num",
            "conversation_turn_mean",
            "num_users",
            "prefill_concurrency",
            "prompt_input_tokens_mean",
            "prompt_input_tokens_stddev",
            "prompt_output_tokens_mean",
            "prompt_output_tokens_stddev",
            "request_count",
            "request_rate",
        }
    )
    # The default load shape when neither --concurrency nor --request-rate is
    # given: a concurrency phase at this level.
    default_concurrency: int = 1
    # Defaults that are workload-shaping and differ from a bare vllm run.
    default_num_dataset_entries: int = 100
    default_isl_mean: int = 550
    # Goodput tag -> inference-perf GoodputConfig key. Values arrive in ms.
    goodput_tags: Tuple[Tuple[str, str], ...] = (
        ("time_to_first_token", "ttft"),
        ("inter_token_latency", "itl"),
        ("request_latency", "request_latency"),
    )

    @property
    def goodput_tag_map(self) -> Dict[str, str]:
        return dict(self.goodput_tags)


# The verified vllm pin. e2e/utils/vllm_bench.py pins the same tag for the
# parity harness; tests/required/tools/test_convert_parity.py asserts the two
# stay equal whenever that file is present in the tree.
VLLM_BENCH_PINNED = "v0.32.0"
AIPERF_PROFILE_PINNED = "v0.12.0"

VLLM_BENCH_VERSIONS: Dict[str, VllmBenchFacts] = {
    "v0.10.0": VllmBenchFacts(version="v0.10.0"),
    VLLM_BENCH_PINNED: _VLLM_V0_32_0,
}

AIPERF_PROFILE_VERSIONS: Dict[str, AiperfProfileFacts] = {
    AIPERF_PROFILE_PINNED: AiperfProfileFacts(version=AIPERF_PROFILE_PINNED),
}


def _canonical(version: str) -> str:
    version = version.strip()
    return version if version.startswith("v") else f"v{version}"


def resolve_vllm_bench(version: str) -> Optional[VllmBenchFacts]:
    return VLLM_BENCH_VERSIONS.get(_canonical(version))


def resolve_aiperf_profile(version: str) -> Optional[AiperfProfileFacts]:
    return AIPERF_PROFILE_VERSIONS.get(_canonical(version))


def unverified_version_reason(tool: str, version: str, verified: Dict[str, object]) -> str:
    known = ", ".join(sorted(verified))
    return f"version {version} of {tool} is not in the verified surface registry (verified: {known})"
