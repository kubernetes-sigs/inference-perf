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
    # --endpoint-type choices (ASYNC_REQUEST_FUNCS keys at this tag).
    endpoint_types: FrozenSet[str] = frozenset({"vllm", "openai", "openai-chat", "openai-audio"})
    # Sampling params injected into every request when the user passes none.
    implicit_temperature: float = 0.0
    implicit_repetition_penalty: float = 1.0
    # benchmark() sends this many untunable test requests before the run.
    warmup_requests: int = 1
    # --random-range-ratio r draws lengths uniform on [len*(1-r), len*(1+r)];
    # False would mean the pre-v0.8.4 reading (0 = anywhere from 0 to len).
    range_ratio_symmetric: bool = True
    # Streaming is hardcoded in the request payload and cannot be disabled.
    stream_forced: bool = True


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
    # Request count auto-derivation for synthetic data when --request-count is
    # unset: max(floor, concurrency * factor).
    auto_request_count_floor: int = 10
    auto_request_count_concurrency_factor: int = 2
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
VLLM_BENCH_PINNED = "v0.10.0"
AIPERF_PROFILE_PINNED = "v0.12.0"

VLLM_BENCH_VERSIONS: Dict[str, VllmBenchFacts] = {
    VLLM_BENCH_PINNED: VllmBenchFacts(version=VLLM_BENCH_PINNED),
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
