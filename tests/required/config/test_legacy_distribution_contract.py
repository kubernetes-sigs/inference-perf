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
"""Characterization tests for distribution blocks at the config layer.

A ``{type, min, max, mean, std_dev}`` block is the legacy way to spell a
sampled length, count or latency. These tests pin what such a block does end
to end today, from a config dict through the sampler a datagen would call, so
the migration to expression strings keeps every one of them loading and
sampling the same way. They assert on ranges and determinism, never on exact
draws, because the migration changes which numpy calls back each draw.
"""

import numpy as np
import pytest

from inference_perf.config import DataConfig, DataGenType, ImageDatagenConfig, SharedPrefix, SyntheticAgenticConfig
from inference_perf.datagen.datagen_utils import pregenerate_lengths
from inference_perf.datagen.multimodal_sampling import sample_insertion_point
from inference_perf.utils.numeric.distribution import sample_lengths, sample_values


# random datagen with input {10..100, mean 50, sd 10} and output {10..100, mean 50, sd 10}.
# 100 pre-generated lengths per side, every one an int in [10, 100]; the same seed
# gives the same array twice.
def test_io_distribution_blocks_pregenerate_bounded_ints() -> None:
    cfg = DataConfig(
        type=DataGenType.Random,
        input_distribution={"min": 10, "max": 100, "mean": 50, "std_dev": 10},
        output_distribution={"min": 10, "max": 100, "mean": 50, "std_dev": 10},
    )
    assert cfg.input_distribution is not None and cfg.output_distribution is not None
    for spec in (cfg.input_distribution, cfg.output_distribution):
        first = pregenerate_lengths(spec, 100, np.random.default_rng(0))
        again = pregenerate_lengths(spec, 100, np.random.default_rng(0))
        assert len(first) == 100
        assert np.issubdtype(first.dtype, np.integer)
        assert first.min() >= 10 and first.max() <= 100
        assert np.array_equal(first, again)


# input_distribution {mean: 600} and nothing else, the shape the nested CLI flag
# --data.input_distribution.mean produces. Loads with the block defaults and
# samples ints in the default [10, 1024] range.
def test_partial_block_fills_defaults() -> None:
    cfg = DataConfig(type=DataGenType.Random, input_distribution={"mean": 600}, output_distribution={"mean": 50})
    assert cfg.input_distribution is not None
    lengths = pregenerate_lengths(cfg.input_distribution, 200, np.random.default_rng(1))
    assert lengths.min() >= 10 and lengths.max() <= 1024
    assert 400 < lengths.mean() < 800


# shareGPT with input {min 10, max 100} and output {min 10, max 100} blocks, where
# the block only supplies dataset filter bounds. Loads without error.
def test_dataset_type_accepts_io_distribution_blocks() -> None:
    cfg = DataConfig(
        type=DataGenType.ShareGPT,
        input_distribution={"min": 10, "max": 100},
        output_distribution={"min": 10, "max": 100},
    )
    assert cfg.input_distribution is not None
    assert cfg.output_distribution is not None


# Every legacy type on a shared_prefix length field: normal, skew_normal (skew 2.5),
# lognormal, uniform, poisson, each with min 10 and max 200. 500 sampled lengths
# are all ints in [10, 200] and the same seed reproduces them.
@pytest.mark.parametrize("dist_type", ["normal", "skew_normal", "lognormal", "uniform", "poisson"])
def test_every_block_type_samples_bounded_ints(dist_type: str) -> None:
    block = {"type": dist_type, "min": 10, "max": 200, "mean": 80, "std_dev": 30, "skew": 2.5}
    sp = SharedPrefix(question_len=block)
    first = sample_lengths(sp.question_len, 500, np.random.default_rng(3))
    again = sample_lengths(sp.question_len, 500, np.random.default_rng(3))
    assert np.issubdtype(first.dtype, np.integer)
    assert first.min() >= 10 and first.max() <= 200
    assert np.array_equal(first, again)


# shared_prefix output_len {type: fixed, mean: 99.9}. Every sampled length is 99:
# fixed truncates the mean rather than rounding it, and ignores min and max.
def test_fixed_block_truncates_mean() -> None:
    sp = SharedPrefix(output_len={"type": "fixed", "mean": 99.9, "min": 0, "max": 50})
    assert set(sample_lengths(sp.output_len, 20, np.random.default_rng(0)).tolist()) == {99}


# shared_prefix with question_len 50 and the legacy question_distribution block
# {normal, 10..200, mean 80, sd 30} alongside it. Loads, and the block wins:
# the effective question length samples spread over [10, 200] instead of sitting at 50.
def test_legacy_question_distribution_alias_loads_and_wins() -> None:
    sp = SharedPrefix(question_len=50, question_distribution={"min": 10, "max": 200, "mean": 80, "std_dev": 30})
    spec = sp.question_distribution if sp.question_distribution is not None else sp.question_len
    lengths = sample_lengths(spec, 200, np.random.default_rng(0))
    assert lengths.min() >= 10 and lengths.max() <= 200
    assert len(set(lengths.tolist())) > 1


# synthetic_agentic with turns_per_session {fixed 4}, tool_loop_depth {uniform 0..3}
# and tool_call_latency_sec {fixed 1}. Turns sample to 4 every time, depth stays
# within {0, 1, 2, 3} as ints, and the latency samples to 1.0 seconds as a float.
def test_agentic_knob_blocks_sample_in_range() -> None:
    cfg = SyntheticAgenticConfig(
        num_sessions=1,
        input_tokens_per_turn={"type": "fixed", "mean": 250},
        output_tokens_per_turn={"type": "fixed", "mean": 60},
        turns_per_session={"type": "fixed", "mean": 4},
        tool_loop_depth={"type": "uniform", "min": 0, "max": 3},
        tool_call_latency_sec={"type": "fixed", "mean": 1},
    )
    rng = np.random.default_rng(5)
    assert set(sample_values(cfg.turns_per_session, 10, rng, integer=True).tolist()) == {4.0}
    assert cfg.tool_loop_depth is not None and cfg.tool_call_latency_sec is not None
    depths = sample_values(cfg.tool_loop_depth, 200, rng, integer=True)
    assert set(depths.tolist()) <= {0.0, 1.0, 2.0, 3.0}
    assert set(sample_values(cfg.tool_call_latency_sec, 10, rng, integer=False).tolist()) == {1.0}


# image config with count {uniform 1..3} and insertion_point {normal, 0..1, mean 0.5, sd 0.3}.
# Counts sample to ints within {1, 2, 3}; insertion points are floats in [0, 1] that
# are not all identical, so the block is being sampled rather than fixed.
def test_media_count_and_insertion_point_blocks_sample_in_range() -> None:
    cfg = ImageDatagenConfig(
        count={"type": "uniform", "min": 1, "max": 3, "mean": 2},
        insertion_point={"type": "normal", "min": 0, "max": 1, "mean": 0.5, "std_dev": 0.3},
    )
    rng = np.random.default_rng(7)
    assert cfg.count is not None
    assert set(sample_values(cfg.count, 200, rng, integer=True).tolist()) <= {1.0, 2.0, 3.0}
    points = [sample_insertion_point(cfg.insertion_point, rng) for _ in range(50)]
    assert all(0.0 <= p <= 1.0 for p in points)
    assert len(set(points)) > 1
