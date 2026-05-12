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
"""
Integration tests for multimodal data generation against llm-d-inference-sim.

These exercise inference-perf end-to-end (datagen -> chat-completions wire
payload -> response -> JSON report) for image / video / audio / mixed
configurations without requiring a GPU or a real VLM. The sim accepts the
OpenAI-compatible content-block payloads that the multimodal datagen emits;
we assert that the report carries the expected per-modality summary blocks.

Real-VLM payload-format validation (vLLM / SGLang acceptance) and perf
alignment with vllm-bench / sglang-bench baselines are out of scope here and
require GPU hardware.

If `llm-d-inference-sim` is not on PATH these tests skip automatically.
"""

import os
from typing import Any, Dict

import pytest

from utils.benchmark import run_benchmark_minimal
from utils.llm_d_inference_sim import LLMDInferenceSimRunner
from utils.testdata import extract_tarball

TEST_MODEL_NAME = "google/gemma-3-270m"
TEST_MODEL_TARBALL = "e2e/testdata/models/google_gemma-3-270m.tar.gz"

IMAGE_BLOCK: Dict[str, Any] = {
    "image": {
        "count": {"type": "fixed", "min": 1, "max": 1, "mean": 1},
        "insertion_point": 0.0,
        "resolutions": {"height": 64, "width": 64},
    },
}

VIDEO_BLOCK: Dict[str, Any] = {
    "video": {
        "count": {"type": "fixed", "min": 1, "max": 1, "mean": 1},
        "insertion_point": 0.0,
        "profiles": [
            {
                "profile": {"resolution": {"height": 64, "width": 64}, "frames": 4},
                "weight": 1.0,
            }
        ],
    },
}

AUDIO_BLOCK: Dict[str, Any] = {
    "audio": {
        "count": {"type": "fixed", "min": 1, "max": 1, "mean": 1},
        "insertion_point": 0.0,
        "durations": [{"duration": 1.0, "weight": 1.0}],
    },
}


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
@pytest.mark.parametrize(
    "case_id,multimodal,expected_modalities",
    [
        pytest.param("image_only", IMAGE_BLOCK, ("image",), id="image_only"),
        pytest.param("video_only", VIDEO_BLOCK, ("video",), id="video_only"),
        pytest.param("audio_only", AUDIO_BLOCK, ("audio",), id="audio_only"),
        pytest.param(
            "mixed",
            {**IMAGE_BLOCK, **VIDEO_BLOCK, **AUDIO_BLOCK},
            ("image", "video", "audio"),
            id="mixed_image_video_audio",
        ),
    ],
)
async def test_multimodal_synthetic_against_sim(case_id: str, multimodal: Dict[str, Any], expected_modalities: tuple) -> None:
    """Run a multimodal synthetic config against the sim and verify the JSON
    report exposes the expected per-modality summary blocks and per-modality
    throughput rates."""
    model_name = TEST_MODEL_NAME
    model_path = extract_tarball(TEST_MODEL_TARBALL)

    rate = 1
    duration = 5

    async with LLMDInferenceSimRunner(model_name, port=18001) as sim:
        result = await run_benchmark_minimal(
            {
                "data": {
                    "type": "synthetic",
                    "input_distribution": {"type": "fixed", "min": 32, "max": 32, "mean": 32},
                    "output_distribution": {"type": "fixed", "min": 16, "max": 16, "mean": 16},
                    "multimodal": multimodal,
                },
                "load": {
                    "type": "constant",
                    "stages": [{"rate": rate, "duration": duration}],
                    "num_workers": 1,
                },
                "api": {
                    "type": "chat",
                    "streaming": True,
                },
                "server": {
                    "type": "vllm",
                    "model_name": model_name,
                    "base_url": f"http://{sim.host}:{sim.port}",
                    "ignore_eos": True,
                },
                "tokenizer": {
                    "pretrained_model_name_or_path": str(model_path),
                },
                "report": {
                    "request_lifecycle": {
                        "summary": True,
                        "per_stage": True,
                        "per_request": True,
                    },
                },
            }
        )

    assert result.success, f"Benchmark failed for case {case_id}:\n{result.stdout}"
    assert result.reports, f"No reports generated for case {case_id}"

    summary = result.reports.get("summary_lifecycle_metrics.json")
    assert summary, f"Missing summary_lifecycle_metrics.json for case {case_id}"

    successes = summary.get("successes")
    assert successes and successes.get("count", 0) >= 1, (
        f"No successful requests for case {case_id}; sim may have rejected the payload"
    )

    throughput = successes.get("throughput") or {}
    for modality in expected_modalities:
        rate_key = f"{modality}s_per_sec"
        assert rate_key in throughput, f"missing throughput.{rate_key} in summary for case {case_id}"
        assert throughput[rate_key] is not None and throughput[rate_key] > 0, (
            f"throughput.{rate_key} should be > 0 for case {case_id}, got {throughput[rate_key]!r}"
        )

    for modality in expected_modalities:
        block = successes.get(modality)
        assert block, f"missing per-modality block '{modality}' in summary for case {case_id}"
        count = block.get("count")
        assert count and count.get("mean") is not None and count["mean"] > 0, (
            f"{modality}.count.mean should be populated for case {case_id}, got {count!r}"
        )

    assert successes.get("request_size_bytes"), f"request_size_bytes should be present in successes summary for case {case_id}"

    if "image" in expected_modalities:
        for field in ("pixels", "bytes", "aspect_ratio"):
            assert successes["image"].get(field), f"image.{field} missing for case {case_id}"

    if "video" in expected_modalities:
        for field in ("frames", "pixels", "bytes", "aspect_ratio"):
            assert successes["video"].get(field), f"video.{field} missing for case {case_id}"

    if "audio" in expected_modalities:
        for field in ("seconds", "bytes"):
            assert successes["audio"].get(field), f"audio.{field} missing for case {case_id}"


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
@pytest.mark.skipif(
    not os.environ.get("SHAREGPT4VIDEO_CACHE_DIR"),
    reason="SHAREGPT4VIDEO_CACHE_DIR not set; pre-populated video cache required for sharegpt4video e2e",
)
@pytest.mark.skipif(
    not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
    reason="HF token required (ShareGPT4Video is gated)",
)
async def test_sharegpt4video_against_sim() -> None:
    """End-to-end run of the ShareGPT4Video loader against the sim.

    Skipped by default — requires (1) a pre-populated cache at
    ``$SHAREGPT4VIDEO_CACHE_DIR`` containing extracted MP4s under ``videos/``
    and (2) a HuggingFace token with access to the gated dataset. We disable
    auto-download in this test (``prefetch_fraction=0.0``) to keep the run
    fast and deterministic.
    """
    model_name = TEST_MODEL_NAME
    model_path = extract_tarball(TEST_MODEL_TARBALL)
    cache_dir = os.environ["SHAREGPT4VIDEO_CACHE_DIR"]

    async with LLMDInferenceSimRunner(model_name, port=18002) as sim:
        result = await run_benchmark_minimal(
            {
                "data": {
                    "type": "sharegpt4video",
                    "sharegpt4video": {
                        "cache_dir": cache_dir,
                        "representation": "jpeg_frames",
                        "target_resolution": {"width": 64, "height": 64},
                        "max_frames_per_request": 4,
                        "insertion_point": 0.0,
                    },
                },
                "load": {
                    "type": "constant",
                    "stages": [{"rate": 1, "duration": 5}],
                    "num_workers": 1,
                },
                "api": {"type": "chat", "streaming": True},
                "server": {
                    "type": "vllm",
                    "model_name": model_name,
                    "base_url": f"http://{sim.host}:{sim.port}",
                    "ignore_eos": True,
                },
                "tokenizer": {"pretrained_model_name_or_path": str(model_path)},
                "report": {"request_lifecycle": {"summary": True, "per_stage": True, "per_request": True}},
            }
        )

    assert result.success, f"sharegpt4video benchmark failed:\n{result.stdout}"
    summary = (result.reports or {}).get("summary_lifecycle_metrics.json")
    assert summary, "missing summary_lifecycle_metrics.json"
    successes = summary.get("successes") or {}
    assert successes.get("count", 0) >= 1
    throughput = successes.get("throughput") or {}
    assert throughput.get("videos_per_sec", 0) > 0
    video_block = successes.get("video") or {}
    for field in ("frames", "pixels", "bytes", "aspect_ratio"):
        assert video_block.get(field), f"video.{field} missing"


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMDInferenceSimRunner.is_available(), reason="local environment missing llm-d-inference-sim")
@pytest.mark.skipif(
    not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
    reason="HF token required (MMMU is gated)",
)
async def test_mmmu_against_sim() -> None:
    """End-to-end run of the MMMU loader against the sim.

    Skipped by default — requires a HuggingFace token with access to the gated
    ``MMMU/MMMU`` dataset. No cache pre-population needed: MMMU is small enough
    that the HF datasets library handles caching transparently. We pin to a
    single subject (``Computer_Science``) and aggressively resize images so the
    init load stays fast.
    """
    model_name = TEST_MODEL_NAME
    model_path = extract_tarball(TEST_MODEL_TARBALL)

    async with LLMDInferenceSimRunner(model_name, port=18003) as sim:
        result = await run_benchmark_minimal(
            {
                "data": {
                    "type": "mmmu",
                    "mmmu": {
                        "subjects": ["Computer_Science"],
                        "representation": "jpeg",
                        "target_resolution": {"width": 64, "height": 64},
                        "max_images_per_request": 2,
                        "max_examples": 5,
                        "insertion_point": 0.0,
                        "hf_split": "validation",
                    },
                },
                "load": {
                    "type": "constant",
                    "stages": [{"rate": 1, "duration": 5}],
                    "num_workers": 1,
                },
                "api": {"type": "chat", "streaming": True},
                "server": {
                    "type": "vllm",
                    "model_name": model_name,
                    "base_url": f"http://{sim.host}:{sim.port}",
                    "ignore_eos": True,
                },
                "tokenizer": {"pretrained_model_name_or_path": str(model_path)},
                "report": {"request_lifecycle": {"summary": True, "per_stage": True, "per_request": True}},
            }
        )

    assert result.success, f"mmmu benchmark failed:\n{result.stdout}"
    summary = (result.reports or {}).get("summary_lifecycle_metrics.json")
    assert summary, "missing summary_lifecycle_metrics.json"
    successes = summary.get("successes") or {}
    assert successes.get("count", 0) >= 1
    throughput = successes.get("throughput") or {}
    assert throughput.get("images_per_sec", 0) > 0
    image_block = successes.get("image") or {}
    for field in ("count", "pixels", "bytes", "aspect_ratio"):
        assert image_block.get(field), f"image.{field} missing"
