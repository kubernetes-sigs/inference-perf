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
import argparse

from inference_perf.config import Config
from inference_perf.utils.cli_parser import add_global_args, add_pydantic_args


def test_add_global_args_documents_every_flag() -> None:
    parser = argparse.ArgumentParser()
    docs: list[str] = []
    base_args = add_global_args(parser, docs=docs)

    assert base_args == {"config_file", "analyze", "unified_analysis_dir", "log_level"}
    # One doc row per global flag, so a flag added through this helper cannot go undocumented.
    assert len(docs) == len(base_args)
    for dest in base_args:
        assert any(dest in row or dest.replace("_", "-") in row for row in docs)


def test_global_doc_rows_precede_config_rows() -> None:
    parser = argparse.ArgumentParser()
    docs: list[str] = []
    add_global_args(parser, docs=docs)
    global_rows = list(docs)
    add_pydantic_args(parser, Config, docs=docs)

    assert len(docs) > len(global_rows)
    # Config rows are appended after the global rows, which survive as an unchanged prefix.
    assert docs[: len(global_rows)] == global_rows


def test_base_args_filter_separates_config_overrides() -> None:
    parser = argparse.ArgumentParser()
    base_args = add_global_args(parser)
    add_pydantic_args(parser, Config)

    args = parser.parse_args(["-c", "cfg.yaml", "--log-level", "DEBUG", "--api.streaming", "true"])
    overrides = {k: v for k, v in vars(args).items() if k not in base_args}

    assert overrides == {"api.streaming": True}
    assert args.config_file == "cfg.yaml"
    assert args.log_level == "DEBUG"


# A config field that is a Distribution OR an expression string keeps both CLI forms. Inputs: the real Config
# parsed with '--data.input_distribution.mean 600' and, separately, '--data.input_distribution "Normal(512, 128)"'.
# Expected: the nested flag exists and lands at data.input_distribution.mean as the float 600.0; the whole-value
# flag lands as a string.
def test_model_or_string_field_keeps_nested_flags() -> None:
    from inference_perf.utils.cli_parser import unflatten_dict

    parser = argparse.ArgumentParser()
    add_pydantic_args(parser, Config)
    nested = unflatten_dict(vars(parser.parse_args(["--data.input_distribution.mean", "600"])))
    assert nested == {"data": {"input_distribution": {"mean": 600.0}}}
    whole = unflatten_dict(vars(parser.parse_args(["--data.input_distribution", "Normal(512, 128)"])))
    assert whole == {"data": {"input_distribution": "Normal(512, 128)"}}


# Setting the whole value and one of its fields in the same command is ambiguous. Input: both
# '--data.input_distribution "Normal(512, 128)"' and '--data.input_distribution.mean 600'. Expected: a ValueError
# naming the conflict, in either key order.
def test_whole_value_and_field_flags_conflict() -> None:
    import pytest

    from inference_perf.utils.cli_parser import unflatten_dict

    for flat in (
        {"data.input_distribution": "Normal(512, 128)", "data.input_distribution.mean": "600"},
        {"data.input_distribution.mean": "600", "data.input_distribution": "Normal(512, 128)"},
    ):
        with pytest.raises(ValueError, match="set the whole value or its fields, not both"):
            unflatten_dict(flat)
