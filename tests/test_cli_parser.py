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
import sys

import pytest

from inference_perf import main
from inference_perf.config import Config
from inference_perf.utils.cli_parser import add_global_args, add_pydantic_args


def test_add_global_args_documents_every_flag() -> None:
    parser = argparse.ArgumentParser()
    docs: list[str] = []
    base_args = add_global_args(parser, docs=docs)

    assert base_args == {
        "config_file",
        "analyze",
        "unified_analysis_dir",
        "analyze_labels",
        "analyze_rate",
        "analyze_latency_stat",
        "log_level",
    }
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


# `--analyze run` alone: no labels, rate "achieved" and latency stat "mean", the
# values analyze_reports used before these flags existed.
def test_analysis_flags_default_to_previous_behavior() -> None:
    parser = argparse.ArgumentParser()
    add_global_args(parser)

    args = parser.parse_args(["--analyze", "run"])

    assert args.analyze_labels is None
    assert args.analyze_rate == "achieved"
    assert args.analyze_latency_stat == "mean"


# `--analyze-labels vLLM SGLang --analyze-rate requested --analyze-latency-stat p90`
# parses to ["vLLM", "SGLang"], "requested" and "p90". An unknown rate exits with status 2.
def test_analysis_flags_parse() -> None:
    parser = argparse.ArgumentParser()
    add_global_args(parser)

    args = parser.parse_args(
        "--analyze a b --analyze-labels vLLM SGLang --analyze-rate requested --analyze-latency-stat p90".split()
    )

    assert args.analyze == ["a", "b"]
    assert args.analyze_labels == ["vLLM", "SGLang"]
    assert args.analyze_rate == "requested"
    assert args.analyze_latency_stat == "p90"
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["--analyze", "a", "--analyze-rate", "configured"])
    assert exit_info.value.code == 2


# The doc rows for the new flags: --analyze-labels is a list of str, --analyze-rate an
# Enum (achieved, requested) with default achieved, --analyze-latency-stat a str with default mean.
def test_analysis_flag_doc_rows() -> None:
    docs: list[str] = []
    add_global_args(argparse.ArgumentParser(), docs=docs)

    rows = {row.split(" | ")[0]: row for row in docs}
    assert "| list of str |" in rows["| `--analyze-labels`"]
    assert "| Enum (achieved, requested) |" in rows["| `--analyze-rate`"]
    assert rows["| `--analyze-rate`"].endswith("(default: achieved) |")
    assert "| str |" in rows["| `--analyze-latency-stat`"]
    assert rows["| `--analyze-latency-stat`"].endswith("(default: mean) |")


# Replaces setup_logging in main so a main_cli test leaves the root logger alone.
@pytest.fixture
def quiet_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "setup_logging", lambda level: None)


# `inference-perf --analyze a b -u out --analyze-labels x y --analyze-rate requested --analyze-latency-stat p99`
# calls analyze_reports(["a", "b"], "out", labels=["x", "y"], rate="requested", latency_stat="p99").
def test_main_cli_passes_analysis_flags(monkeypatch: pytest.MonkeyPatch, quiet_main: None) -> None:
    calls: list[tuple[list[str], str | None, dict[str, object]]] = []

    def fake_analyze_reports(report_dirs: list[str], analysis_dir: str | None = None, **kwargs: object) -> None:
        calls.append((report_dirs, analysis_dir, kwargs))

    monkeypatch.setattr(main, "analyze_reports", fake_analyze_reports)
    monkeypatch.setattr(
        sys,
        "argv",
        ["inference-perf", "--analyze", "a", "b", "-u", "out", "--analyze-labels", "x", "y"]
        + ["--analyze-rate", "requested", "--analyze-latency-stat", "p99"],
    )

    main.main_cli()

    assert calls == [(["a", "b"], "out", {"labels": ["x", "y"], "rate": "requested", "latency_stat": "p99"})]


# 2 report directories with 1 label: main_cli exits with status 2 and prints the
# label count error as a usage error, without a traceback.
def test_main_cli_reports_bad_analysis_options_as_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], quiet_main: None
) -> None:
    monkeypatch.setattr(sys, "argv", ["inference-perf", "--analyze", "a", "b", "--analyze-labels", "x"])

    with pytest.raises(SystemExit) as exit_info:
        main.main_cli()

    assert exit_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "error: The number of labels (1) is not equal to the number of report directories (2)" in stderr
    assert "Traceback" not in stderr


# A ValueError that is not an option error (here from a fake analyze_reports, as matplotlib
# or numpy could raise while plotting) leaves main_cli as that ValueError, not as exit 2.
def test_main_cli_keeps_other_value_errors(monkeypatch: pytest.MonkeyPatch, quiet_main: None) -> None:
    def failing_analyze_reports(report_dirs: list[str], analysis_dir: str | None = None, **kwargs: object) -> None:
        raise ValueError("bad value while plotting")

    monkeypatch.setattr(main, "analyze_reports", failing_analyze_reports)
    monkeypatch.setattr(sys, "argv", ["inference-perf", "--analyze", "a"])

    with pytest.raises(ValueError, match="bad value while plotting"):
        main.main_cli()


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
