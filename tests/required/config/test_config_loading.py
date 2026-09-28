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
"""Top-level Config assembly: read_config, deep_merge, and cross-field validators.

Per-surface schema rules live in the sibling modules that mirror the
``inference_perf.config`` package (``loadgen/``, ``datagen/``, ``client/``,
etc.). This file covers only the glue: loading YAML into a ``Config``,
merging overrides, timestamp substitution, and the whole-config validators
defined on ``Config`` itself.
"""

import os
import tempfile
from typing import Any

import pytest
import yaml

from inference_perf.config import (
    APIType,
    Config,
    DataGenType,
    LoadType,
    MetricsClientType,
    deep_merge,
    read_config,
)
from inference_perf.config.loadgen import TraceSessionReplayLoadStage

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_read_config() -> None:
    config = read_config(os.path.join(REPO_ROOT, "config.yml"))

    assert isinstance(config, Config)
    assert config.api.type == APIType.Completion
    assert config.data.type == DataGenType.ShareGPT
    assert config.load.type == LoadType.CONSTANT
    if config.metrics:
        assert config.metrics.type == MetricsClientType.PROMETHEUS
    assert config.report.request_lifecycle.summary is True


def test_read_config_empty_yaml_uses_defaults() -> None:
    """An empty config file is valid; the resulting Config is all defaults."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp.write("")
        tmp_path = tmp.name
    try:
        config = read_config(tmp_path)
        assert config == Config()
    finally:
        os.remove(tmp_path)


def test_read_config_no_file_returns_defaults() -> None:
    assert read_config() == Config()


def test_deep_merge() -> None:
    base = {
        "api": APIType.Chat,
        "data": {"type": DataGenType.ShareGPT},
        "load": {"type": LoadType.CONSTANT},
        "metrics": {"type": MetricsClientType.PROMETHEUS},
    }
    override = {
        "data": {"type": DataGenType.Mock},
        "load": {"type": LoadType.POISSON},
    }
    merged = deep_merge(base, override)

    assert merged["api"] == APIType.Chat
    assert merged["data"]["type"] == DataGenType.Mock
    assert merged["load"]["type"] == LoadType.POISSON
    assert merged["metrics"]["type"] == MetricsClientType.PROMETHEUS


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"data": {"type": "mock", "path": "keep"}}
    override = {"data": {"type": "synthetic"}}

    merged = deep_merge(base, override)

    assert merged["data"] == {"type": "synthetic", "path": "keep"}
    # Originals untouched.
    assert base == {"data": {"type": "mock", "path": "keep"}}
    assert override == {"data": {"type": "synthetic"}}


def test_read_config_cli_overrides_win() -> None:
    config_content = {"data": {"type": "shareGPT"}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(config_content, tmp)
        tmp_path = tmp.name
    try:
        config = read_config(tmp_path, cli_overrides={"data": {"type": "mock"}})
        assert config.data.type == DataGenType.Mock
    finally:
        os.remove(tmp_path)


def test_read_config_timestamp_substitution() -> None:
    # Create a minimalistic config with {timestamp} in the storage path
    config_content = {
        "storage": {
            "local_storage": {"path": "reports-{timestamp}"},
            "google_cloud_storage": {"bucket_name": "my-bucket", "path": "gcs-reports-{timestamp}"},
            "simple_storage_service": {"bucket_name": "my-bucket", "path": "s3-reports-{timestamp}"},
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(config_content, tmp)
        tmp_path = tmp.name

    try:
        config = read_config(tmp_path)
        # Verify substitution happened
        assert config.storage is not None
        assert "{timestamp}" not in config.storage.local_storage.path
        assert config.storage.local_storage.path.startswith("reports-")

        assert config.storage.google_cloud_storage is not None
        assert "{timestamp}" not in config.storage.google_cloud_storage.path
        assert config.storage.google_cloud_storage.path.startswith("gcs-reports-")

        assert config.storage.simple_storage_service is not None
        assert "{timestamp}" not in config.storage.simple_storage_service.path
        assert config.storage.simple_storage_service.path.startswith("s3-reports-")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_otel_trace_replay_requires_trace_session_replay_load() -> None:
    """Config-level validator: otel_trace_replay data demands session-replay load."""
    with pytest.raises(ValueError, match="requires load.type 'trace_session_replay'"):
        Config.model_validate(
            {
                "data": {"type": "otel_trace_replay"},
                "load": {"type": "constant"},
            }
        )


def test_otel_trace_replay_with_session_replay_load_ok() -> None:
    config = Config.model_validate(
        {
            "data": {"type": "otel_trace_replay"},
            "load": {"type": "trace_session_replay"},
        }
    )
    assert config.data.type == DataGenType.OTelTraceReplay
    assert config.load.type == LoadType.TRACE_SESSION_REPLAY


# --- duration-bounded stages vs datagen settings -------------------------
#
# A duration-bounded session-replay stage replays the corpus once it runs out, which
# turns two datagen settings into silent contradictions. Both settings live under
# ``data`` and ``duration`` lives under ``load``, so neither surface's own validator
# can see the conflict — it has to be caught here.

_OTEL_SOURCE = {"hf_dataset_path": "org/dataset"}


def _replay_config(
    datagen_overrides: dict[str, Any], stage: dict[str, Any], data_type: str = "otel_trace_replay"
) -> dict[str, Any]:
    """A minimal session-replay config, parameterized on the bits under test."""
    source = _OTEL_SOURCE if data_type == "otel_trace_replay" else {"trace_directory": "/traces"}
    return {
        "data": {"type": data_type, data_type: {**source, **datagen_overrides}},
        "load": {"type": "trace_session_replay", "stages": [stage]},
    }


def test_duration_stage_rejects_duplicate_sessions_target_for_otel_replay() -> None:
    """Padding the corpus and replaying it mint colliding ``_dupN`` session IDs.

    ``duplicate_sessions_target`` numbers its copies with one global counter, while
    corpus cycling numbers per source trace, so the two independently produce the same
    ID — and session state, completion tracking and cleanup are all keyed by ID.
    Cycling makes the padding redundant, so the fix is to drop it.
    """
    with pytest.raises(ValueError, match="duplicate_sessions_target cannot be combined with"):
        Config.model_validate(_replay_config({"duplicate_sessions_target": 100}, {"concurrent_sessions": 4, "duration": 1800}))


def test_duration_stage_rejects_disable_output_substitution() -> None:
    """A replayed session triggers the substitution this setting asks to turn off.

    ``OTelTraceReplayConfig`` already rejects ``disable_output_substitution`` alongside
    the two other ways of producing duplicate sessions. Corpus cycling is a third way,
    and it is switched on from ``load``, where that validator cannot see it.
    """
    with pytest.raises(ValueError, match="disable_output_substitution=True cannot be combined with"):
        Config.model_validate(
            _replay_config({"disable_output_substitution": True}, {"concurrent_sessions": 4, "duration": 1800})
        )


def test_count_bounded_stage_still_allows_duplicate_sessions_target() -> None:
    """Guard: the rejection is about cycling, not about the setting itself."""
    config = Config.model_validate(
        _replay_config({"duplicate_sessions_target": 100}, {"concurrent_sessions": 4, "num_sessions": 50})
    )
    assert config.data.otel_trace_replay is not None
    assert config.data.otel_trace_replay.duplicate_sessions_target == 100


def test_duration_stage_allows_duplicate_sessions_target_for_weka_replay() -> None:
    """Guard: weka_trace_replay cannot cycle, so padding is how duration covers a window.

    Weka builds every session up front (the eager path), so it has no way to resolve an
    index past its corpus. A duration-bounded weka stage ends when its corpus does, which
    makes ``duplicate_sessions_target`` the only way to fill the window — the opposite of
    the otel case above.
    """
    config = Config.model_validate(
        _replay_config(
            {"duplicate_sessions_target": 100},
            {"concurrent_sessions": 4, "duration": 1800},
            data_type="weka_trace_replay",
        )
    )
    assert config.data.weka_trace_replay is not None
    assert config.data.weka_trace_replay.duplicate_sessions_target == 100


def test_disable_output_substitution_without_duration_still_allowed() -> None:
    """Guard: verbatim replay is fine on its own; only cycling contradicts it."""
    config = Config.model_validate(
        _replay_config({"disable_output_substitution": True}, {"concurrent_sessions": 4, "num_sessions": 50})
    )
    assert config.data.otel_trace_replay is not None
    assert config.data.otel_trace_replay.disable_output_substitution is True


def test_duration_stage_rejects_unlimited_concurrency_without_a_rate() -> None:
    """Nothing would bound admission: an unlimited pool drawing from a replaying corpus.

    ``concurrent_sessions: 0`` means "start everything at once", which is well defined
    for a fixed corpus and meaningless for one that replays -- "everything" is unbounded.
    The dispatch loop admits sessions until something says stop, and with no concurrency
    cap and no rate the only thing left is the deadline, so a single loop iteration starts
    sessions as fast as the process can build them for the whole window. Measured at ~120k
    sessions/second against a scripted generator, each one permanently growing the
    generator's index tables: a 30-minute stage would try for hundreds of millions.
    """
    with pytest.raises(ValueError, match="concurrent_sessions: 0 cannot be combined with"):
        Config.model_validate(_replay_config({}, {"concurrent_sessions": 0, "duration": 1800}))


def test_duration_stage_allows_unlimited_concurrency_with_a_rate() -> None:
    """Guard: session_rate bounds admission on its own, so the pool need not.

    This is a legitimate open-loop shape -- offer N sessions per second for the window and
    let the pool grow to whatever that implies -- so it must keep working.
    """
    config = Config.model_validate(_replay_config({}, {"concurrent_sessions": 0, "duration": 1800, "session_rate": 10}))
    stage = config.load.stages[0]
    assert isinstance(stage, TraceSessionReplayLoadStage)
    assert stage.concurrent_sessions == 0


def test_count_bounded_stage_still_allows_unlimited_concurrency() -> None:
    """Guard: the documented stress mode is untouched. A fixed corpus bounds itself."""
    config = Config.model_validate(_replay_config({}, {"concurrent_sessions": 0, "num_sessions": 50}))
    stage = config.load.stages[0]
    assert isinstance(stage, TraceSessionReplayLoadStage)
    assert stage.concurrent_sessions == 0


def test_duration_stage_allows_unlimited_concurrency_for_weka_replay() -> None:
    """Guard: weka cannot replay, so its corpus still bounds admission by itself."""
    config = Config.model_validate(
        _replay_config({}, {"concurrent_sessions": 0, "duration": 1800}, data_type="weka_trace_replay")
    )
    assert config.data.weka_trace_replay is not None
