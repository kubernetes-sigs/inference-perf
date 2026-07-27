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
"""Validity rules for the ``DataConfig`` / ``SharedPrefix`` surface.

These exercise the ``data:`` block of the config (``Config.data``).
"""

import os
import tempfile

import pytest
import yaml

from inference_perf.config import (
    Config,
    DataGenType,
    Distribution,
    DistributionType,
    VideoProfile,
    read_config,
)


# --- SharedPrefix aliases / distribution syntax --------------------------


def test_shared_prefix_short_names() -> None:
    config = Config.model_validate(
        {
            "data": {
                "type": DataGenType.SharedPrefix,
                "shared_prefix": {"num_groups": 5, "num_prompts_per_group": 20},
            }
        }
    )
    assert config.data.shared_prefix is not None
    assert config.data.shared_prefix.num_groups == 5
    assert config.data.shared_prefix.num_prompts_per_group == 20


def test_shared_prefix_alias_names() -> None:
    config = Config.model_validate(
        {
            "data": {
                "type": DataGenType.SharedPrefix,
                "shared_prefix": {"num_unique_system_prompts": 7, "num_users_per_system_prompt": 15},
            }
        }
    )
    assert config.data.shared_prefix is not None
    assert config.data.shared_prefix.num_groups == 7
    assert config.data.shared_prefix.num_prompts_per_group == 15


def test_shared_prefix_serializes_with_aliases() -> None:
    config = Config.model_validate(
        {
            "data": {
                "type": DataGenType.SharedPrefix,
                "shared_prefix": {"num_unique_system_prompts": 7, "num_users_per_system_prompt": 15},
            }
        }
    )
    dumped = config.model_dump(mode="json", by_alias=True)
    shared_prefix_dump = dumped["data"]["shared_prefix"]
    assert shared_prefix_dump["num_unique_system_prompts"] == 7
    assert shared_prefix_dump["num_users_per_system_prompt"] == 15


def test_shared_prefix_inline_distribution() -> None:
    config = Config.model_validate(
        {
            "data": {
                "type": DataGenType.SharedPrefix,
                "shared_prefix": {
                    "num_groups": 5,
                    "question_len": {
                        "type": "skew_normal",
                        "mean": 200,
                        "std_dev": 80,
                        "skew": 2.5,
                        "min": 10,
                        "max": 2000,
                    },
                },
            }
        }
    )
    sp = config.data.shared_prefix
    assert sp is not None
    assert isinstance(sp.question_len, Distribution)
    assert sp.question_len.type == DistributionType.SKEW_NORMAL
    assert sp.question_len.mean == 200
    assert sp.question_len.skew == 2.5


def test_shared_prefix_fixed_int_unchanged() -> None:
    config = Config.model_validate(
        {
            "data": {
                "type": DataGenType.SharedPrefix,
                "shared_prefix": {"question_len": 75},
            }
        }
    )
    sp = config.data.shared_prefix
    assert sp is not None
    assert sp.question_len == 75
    assert isinstance(sp.question_len, int)


def test_shared_prefix_ambiguous_distribution_is_error() -> None:
    with pytest.raises(Exception, match="Cannot specify both"):
        Config.model_validate(
            {
                "data": {
                    "type": DataGenType.SharedPrefix,
                    "shared_prefix": {
                        "question_len": {
                            "type": "normal",
                            "mean": 200,
                            "min": 10,
                            "max": 2000,
                            "std_dev": 50,
                        },
                        "question_distribution": {
                            "min": 10,
                            "max": 1024,
                            "mean": 512,
                            "std_dev": 200,
                        },
                    },
                }
            }
        )


def test_shared_prefix_legacy_distribution_compat() -> None:
    config = Config.model_validate(
        {
            "data": {
                "type": DataGenType.SharedPrefix,
                "shared_prefix": {
                    "question_len": 50,
                    "question_distribution": {
                        "min": 10,
                        "max": 1024,
                        "mean": 512,
                        "std_dev": 200,
                    },
                },
            }
        }
    )
    sp = config.data.shared_prefix
    assert sp is not None
    assert sp.question_len == 50
    assert sp.question_distribution is not None
    assert sp.question_distribution.mean == 512


def test_shared_prefix_seed_field() -> None:
    config = Config.model_validate(
        {
            "data": {
                "type": DataGenType.SharedPrefix,
                "shared_prefix": {"seed": 42},
            }
        }
    )
    assert config.data.shared_prefix is not None
    assert config.data.shared_prefix.seed == 42


# --- multimodal block ----------------------------------------------------


def test_multimodal_config_parsing() -> None:
    config_content = {
        "data": {
            "type": "synthetic",
            "multimodal": {
                "image": {
                    "count": {"type": "uniform", "min": 3, "max": 3, "mean": 3},
                    "insertion_point": 0.5,
                },
                "video": {
                    "count": {"type": "uniform", "min": 1, "max": 1, "mean": 1},
                    "profiles": {"resolution": "1080p", "frames": 30},
                },
            },
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(config_content, tmp)
        tmp_path = tmp.name

    try:
        config = read_config(tmp_path)
        assert config.data.multimodal is not None
        assert config.data.multimodal.image is not None
        assert config.data.multimodal.image.insertion_point == 0.5
        assert config.data.multimodal.image.count is not None
        assert config.data.multimodal.image.count.mean == 3
        assert config.data.multimodal.video is not None
        assert config.data.multimodal.video.count is not None
        assert config.data.multimodal.video.count.mean == 1
        assert config.data.multimodal.video.profiles is not None
        assert isinstance(config.data.multimodal.video.profiles, VideoProfile)
        assert config.data.multimodal.video.profiles.resolution == "1080p"
        assert config.data.multimodal.video.profiles.frames == 30
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
