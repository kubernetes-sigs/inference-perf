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
"""Multimodal payload acceptance against a real vLLM server in CPU mode.

The oracle here is the server: does a real vLLM accept the chat payloads
inference-perf builds for images (data URLs, several per request), MP4
video, PNG frame batches, audio, shared-prefix groups carrying media, and a
real-image dataset, and does every request complete. The sim cannot play
that role: it recognises ``image_url`` / ``input_audio`` / ``video_url``
blocks by type string and never decodes bytes, checks a media format, or
knows which modalities a model supports (that shape-only check is
``test_multimodal_sim.py``). These cases used to be the GPU-only
``tests/optional`` tier against Qwen3-VL-8B / Qwen2-Audio-7B on H100; the
acceptance oracle does not depend on model size or speed, so they run here
against small models on a plain runner instead.

Cases are data: every ``e2e/configs/vllm_cpu_multimodal/*.yaml`` is one
case, and adding a file adds a case. The test overrides ``server`` and
``tokenizer`` with whatever the running server serves and keeps the rest.
The oracle is the one the optional tier had: the run exits cleanly and every
dispatched request succeeds; nothing is asserted about latency.

Two servers are involved because no small model covers image, video and
audio at once: a vision-language model (``VLM_MODEL``, image + video) and a
speech model (``AUDIO_MODEL``). A case runs only when the served model
covers every modality it needs; otherwise it skips with the reason. What the
external server serves is read from ``E2E_VLLM_MODALITIES`` (comma list) when
set, else looked up by served model id in ``MODALITIES_BY_MODEL``, else
treated as text-only, so under the token-oracle job (opt-125m) the
multimodal cases skip and only ``chat`` runs. The ``multimodal-e2e`` CI job
starts each server in turn and fails a pass that skips anything.

Needs vLLM >= 0.27: 0.26.0's CPU backend pins host memory when it batches
more than one multimodal item, killing the engine on the first multi-image
request or on two concurrent image requests (fixed upstream, verified green
on 0.27.1). See utils.vllm_server for provisioning; without a server or a
``vllm`` executable the tests skip.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, FrozenSet, List, Set

import pytest
import yaml

from utils.accuracy import assert_successful_run
from utils.benchmark import run_benchmark_minimal
from utils.net import get_free_port
from utils.testdata import TEST_E2E_DIR
from utils.vllm_server import VLLMServerRunner, served_modalities

# One worker for all vLLM CPU tests: the token-mismatch module's /metrics
# counter deltas need exclusive access to the shared server.
pytestmark = pytest.mark.xdist_group(name="vllm-cpu-server")

CASES_DIR = TEST_E2E_DIR / "configs" / "vllm_cpu_multimodal"

# Smallest ungated models vLLM lists for each modality set that fit a 16 GB
# runner: InternVL3-1B-hf (T+I+V, 1.9 GB) and granite-4.0-1b-speech (T+A,
# 4.6 GB). SmolVLM-256M is image-only in vLLM, so it cannot serve the video
# cases; ultravox-1b pulls a gated Llama base.
VLM_MODEL = "OpenGVLab/InternVL3-1B-hf"
AUDIO_MODEL = "ibm-granite/granite-4.0-1b-speech"

# Media on a CPU VLM: a 64x64 image is a few hundred tokens after tiling, a
# real VisionArena image a couple thousand, so the model needs more context
# than the 2048 the text-only oracle runs with. 4096 is both models'
# max_position_embeddings (vLLM refuses anything higher), and every case
# fits: all nine ran green at 4096.
MAX_MODEL_LEN = 4096

# The slowest case (real-image VisionArena rows) took ~2 min on an 8-core
# machine; a 4-vCPU runner is roughly half that speed.
CASE_TIMEOUT_SEC = 600


@dataclass(frozen=True)
class Case:
    name: str
    config: Dict[str, Any]
    # Modalities the payload carries (image / video / audio); empty for
    # text-only chat.
    modalities: FrozenSet[str]
    # rate * duration of the single constant stage: with open-loop constant
    # load that many requests are dispatched no matter how slow the server
    # is, so it is the exact success count to expect.
    expected_requests: int


# Walks the parsed config and returns which modalities its payload carries:
# every key of every "multimodal" block (the synthetic and shared_prefix
# generators), plus "image" for the visionarena dataset. {"data": {"type":
# "synthetic", "multimodal": {"image": ..., "video": ...}}} -> {"image",
# "video"}; a text-only shared_prefix config -> set().
def payload_modalities(config: Dict[str, Any]) -> Set[str]:
    found: Set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "multimodal" and isinstance(value, dict):
                    found.update(value.keys())
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(config)
    if config.get("data", {}).get("type") == "visionarena":
        found.add("image")
    return found


# Reads one case file into a Case: name from the file stem, modalities from
# the payload, expected requests from the single constant stage. A config
# with rate 1 for 4 s -> expected_requests 4; anything but exactly one
# constant stage is a case-file error, not a server result, so it raises.
def load_case(path: Path) -> Case:
    config = yaml.safe_load(path.read_text())
    stages = config["load"]["stages"]
    assert config["load"]["type"] == "constant" and len(stages) == 1, f"{path.name}: one constant stage per case"
    return Case(
        name=path.stem,
        config=config,
        modalities=frozenset(payload_modalities(config)),
        expected_requests=int(stages[0]["rate"] * stages[0]["duration"]),
    )


CASES: List[Case] = [load_case(p) for p in sorted(CASES_DIR.glob("*.yaml"))]
AUDIO_CASES = [c for c in CASES if "audio" in c.modalities]
VLM_CASES = [c for c in CASES if "audio" not in c.modalities]


@pytest.fixture(scope="module")
async def vlm_server() -> AsyncIterator[VLLMServerRunner]:
    async with VLLMServerRunner(VLM_MODEL, port=get_free_port(), max_model_len=MAX_MODEL_LEN, chat_template=None) as server:
        yield server


@pytest.fixture(scope="module")
async def audio_server() -> AsyncIterator[VLLMServerRunner]:
    async with VLLMServerRunner(AUDIO_MODEL, port=get_free_port(), max_model_len=MAX_MODEL_LEN, chat_template=None) as server:
        yield server


# Runs one case against the server and asserts the optional-tier oracle:
# clean exit, exactly expected_requests successful request entries, zero
# errors. Skips (never fails) when the served model does not cover the
# case's modalities: served {"image","video"} runs images/video/mixed and
# skips audio; served set() (opt-125m) runs only chat.
async def run_case(server: VLLMServerRunner, case: Case) -> None:
    served = served_modalities(server.model)
    missing = case.modalities - served
    if missing:
        pytest.skip(f"{server.model} serves {sorted(served) or 'text only'}; case needs {sorted(missing)}")

    config = dict(case.config)
    config["server"] = {"type": "vllm", "model_name": server.model, "base_url": server.base_url}
    config["tokenizer"] = {"pretrained_model_name_or_path": server.model}
    result = await run_benchmark_minimal(config, timeout_sec=CASE_TIMEOUT_SEC)
    assert_successful_run(result, case.expected_requests)


@pytest.mark.asyncio
@pytest.mark.skipif(not VLLMServerRunner.is_available(), reason="no vLLM server or executable available")
@pytest.mark.parametrize("case", VLM_CASES, ids=[c.name for c in VLM_CASES])
async def test_multimodal_case_vlm(vlm_server: VLLMServerRunner, case: Case) -> None:
    await run_case(vlm_server, case)


@pytest.mark.asyncio
@pytest.mark.skipif(not VLLMServerRunner.is_available(), reason="no vLLM server or executable available")
@pytest.mark.parametrize("case", AUDIO_CASES, ids=[c.name for c in AUDIO_CASES])
async def test_multimodal_case_audio(audio_server: VLLMServerRunner, case: Case) -> None:
    await run_case(audio_server, case)
