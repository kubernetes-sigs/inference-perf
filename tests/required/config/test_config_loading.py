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
from inference_perf.config.loadgen import ConcurrentLoadStage, TraceSessionReplayLoadStage

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


# --- conversation replay load-type and concurrency validation ------------


def test_conversation_replay_requires_concurrent_load() -> None:
    with pytest.raises(ValueError, match="requires load.type 'concurrent'"):
        Config.model_validate(
            {
                "data": {"type": "conversation_replay", "conversation_replay": {"num_conversations": 4}},
                "load": {"type": "constant"},
            }
        )


def test_conversation_replay_rejects_concurrency_above_conversation_count() -> None:
    with pytest.raises(ValueError, match=r"concurrency_level \(5\) cannot exceed.*num_conversations \(4\)"):
        Config.model_validate(
            {
                "data": {"type": "conversation_replay", "conversation_replay": {"num_conversations": 4}},
                "load": {
                    "type": "concurrent",
                    "stages": [{"num_requests": 10, "concurrency_level": 5}],
                },
            }
        )


def test_conversation_replay_accepts_concurrency_at_or_below_conversation_count() -> None:
    config = Config.model_validate(
        {
            "data": {"type": "conversation_replay", "conversation_replay": {"num_conversations": 4}},
            "load": {
                "type": "concurrent",
                "stages": [
                    {"num_requests": 10, "concurrency_level": 2},
                    {"num_requests": 10, "concurrency_level": 4},
                ],
            },
        }
    )

    assert config.data.type == DataGenType.ConversationReplay
    assert config.load.type == LoadType.CONCURRENT


# --- conversation replay concurrency default ------------------------------
#
# `concurrency_level` is required on every concurrent stage, but conversation replay has
# a pool size in `data.conversation_replay.num_conversations` that a stage can be sized
# from, so leaving it out means "run the whole pool". These tests pin that default to
# conversation replay alone.


# Builds the smallest conversation replay config: `num_conversations` blueprints and the
# given concurrent stages, with no other settings.
def _conversation_replay_config(
    stages: list[dict[str, object]], num_conversations: int | float | str = 4
) -> dict[str, object]:
    return {
        "data": {"type": "conversation_replay", "conversation_replay": {"num_conversations": num_conversations}},
        "load": {"type": "concurrent", "stages": stages},
    }


# An omitted concurrency_level on a conversation replay stage resolves to the pool size:
# num_conversations 4 with one bare stage {num_requests: 10} yields concurrency_level 4.
def test_conversation_replay_omitted_concurrency_defaults_to_num_conversations() -> None:
    config = Config.model_validate(_conversation_replay_config([{"num_requests": 10}]))

    stage = config.load.stages[0]
    assert isinstance(stage, ConcurrentLoadStage)
    assert stage.concurrency_level == 4


# An explicit value is never replaced by the default: a stage of 2 out of 4 conversations
# keeps concurrency_level 2, and a sibling that omits it still gets 4.
def test_conversation_replay_explicit_concurrency_is_kept() -> None:
    config = Config.model_validate(_conversation_replay_config([{"num_requests": 10, "concurrency_level": 2}]))
    defaulted = Config.model_validate(
        _conversation_replay_config([{"num_requests": 10, "concurrency_level": 2}, {"num_requests": 10}])
    )

    explicit_stage = config.load.stages[0]
    assert isinstance(explicit_stage, ConcurrentLoadStage)
    assert explicit_stage.concurrency_level == 2
    sibling_stage = defaulted.load.stages[1]
    assert isinstance(sibling_stage, ConcurrentLoadStage)
    assert sibling_stage.concurrency_level == 4


# The default is conversation replay only: with any other data generator an omitted
# concurrency_level is still a validation error naming the missing field.
def test_non_conversation_replay_still_requires_concurrency() -> None:
    with pytest.raises(ValueError, match="concurrency_level"):
        Config.model_validate({"data": {"type": "mock"}, "load": {"type": "concurrent", "stages": [{"num_requests": 10}]}})


# Guard: the unchanged path still accepts an explicit concurrency_level, so the test above
# fails because the field is required rather than because plain concurrent load broke.
def test_non_conversation_replay_explicit_concurrency_still_loads() -> None:
    config = Config.model_validate(
        {"data": {"type": "mock"}, "load": {"type": "concurrent", "stages": [{"num_requests": 10, "concurrency_level": 3}]}}
    )

    stage = config.load.stages[0]
    assert isinstance(stage, ConcurrentLoadStage)
    assert stage.concurrency_level == 3


# Defaulting fills gaps, it does not relax the ceiling: with one stage defaulted and a
# sibling asking for 5 of 4 conversations, the num_conversations check still fires.
def test_conversation_replay_still_rejects_concurrency_above_num_conversations() -> None:
    with pytest.raises(ValueError, match=r"concurrency_level \(5\) cannot exceed.*num_conversations \(4\)"):
        Config.model_validate(
            _conversation_replay_config([{"num_requests": 10}, {"num_requests": 10, "concurrency_level": 5}])
        )


# read_config turns stages into models before Config validates them, so the default has to
# be filled in there too: a YAML file with a bare stage loads and ends up at 7.
def test_read_config_defaults_conversation_replay_concurrency() -> None:
    config_content = _conversation_replay_config([{"num_requests": 10}], num_conversations=7)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(config_content, tmp)
        tmp_path = tmp.name
    try:
        config = read_config(tmp_path)
    finally:
        os.remove(tmp_path)

    stage = config.load.stages[0]
    assert isinstance(stage, ConcurrentLoadStage)
    assert stage.concurrency_level == 7


# Writes a conversation replay config to a temporary YAML file and returns the path; the
# caller removes it.
def _write_conversation_replay_config(stages: list[dict[str, object]], num_conversations: int | float | str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(_conversation_replay_config(stages, num_conversations), tmp)
        return tmp.name


# read_config builds the stage models before Config validates anything, so a fractional count
# has to be reported against the field the file sets: num_conversations 4.7 names
# data.conversation_replay.num_conversations rather than only the copied concurrency_level.
def test_read_config_reports_a_fractional_num_conversations_by_its_data_field() -> None:
    path = _write_conversation_replay_config([{"num_requests": 10}], 4.7)
    try:
        with pytest.raises(ValueError, match=r"data\.conversation_replay\.num_conversations"):
            read_config(path)
    finally:
        os.remove(path)


# A zero count is rejected by num_conversations and by the stage size copied from it, so the
# error has to name the field the file writes instead of the stage field it was copied to.
def test_read_config_reports_a_zero_num_conversations_by_its_data_field() -> None:
    path = _write_conversation_replay_config([{"num_requests": 10}], 0)
    try:
        with pytest.raises(ValueError, match=r"data\.conversation_replay\.num_conversations"):
            read_config(path)
    finally:
        os.remove(path)


# A quoted count reaches the stage as written and the field coerces it, so num_conversations
# "4" sizes the stage at 4 rather than at the field's own default, which the ceiling check
# would then reject against the real count.
def test_conversation_replay_default_follows_a_quoted_num_conversations() -> None:
    config = Config.model_validate(_conversation_replay_config([{"num_requests": 10}], num_conversations="4"))

    stage = config.load.stages[0]
    assert isinstance(stage, ConcurrentLoadStage)
    assert stage.concurrency_level == 4


# A fractional count is rejected with an error that names the field the user wrote, not a
# rounded stage size: the message points at data.conversation_replay.num_conversations.
def test_conversation_replay_fractional_num_conversations_is_rejected_by_data() -> None:
    with pytest.raises(ValueError, match=r"data\.conversation_replay\.num_conversations"):
        Config.model_validate(_conversation_replay_config([{"num_requests": 10}], num_conversations=4.7))


# The same for a count that is not a number at all: the error names
# data.conversation_replay.num_conversations, so the stage default never hides it.
def test_conversation_replay_non_integer_num_conversations_is_reported_by_data() -> None:
    with pytest.raises(ValueError, match=r"data\.conversation_replay\.num_conversations"):
        Config.model_validate(_conversation_replay_config([{"num_requests": 10}], num_conversations="four"))


# A missing conversation_replay block keeps the specific message instead of a stage error:
# the stage is filled in with the field's own default, so the data validator runs and
# reports that data.conversation_replay is what is missing.
def test_conversation_replay_without_the_block_keeps_the_specific_error() -> None:
    with pytest.raises(ValueError, match=r"data\.conversation_replay is required when data\.type is 'conversation_replay'"):
        Config.model_validate(
            {"data": {"type": "conversation_replay"}, "load": {"type": "concurrent", "stages": [{"num_requests": 10}]}}
        )


# An empty conversation_replay block is valid, so the stage follows the field's own default
# and the two agree: an empty block still yields concurrency_level == num_conversations.
def test_conversation_replay_default_follows_the_num_conversations_field_default() -> None:
    config = Config.model_validate(
        {
            "data": {"type": "conversation_replay", "conversation_replay": {}},
            "load": {"type": "concurrent", "stages": [{"num_requests": 10}]},
        }
    )

    stage = config.load.stages[0]
    conversation_config = config.data.conversation_replay
    assert isinstance(stage, ConcurrentLoadStage)
    assert conversation_config is not None
    assert stage.concurrency_level == conversation_config.num_conversations


# A validated config is dumped with concurrency_level already on every stage, in JSON mode
# ("concurrent") and in python mode (LoadType.CONCURRENT), so reloading either dump leaves
# the resolved size alone instead of needing the null-safe rewrite to run again.
def test_conversation_replay_default_survives_a_round_trip() -> None:
    config = Config.model_validate(_conversation_replay_config([{"num_requests": 10}]))

    for reloaded in (Config.model_validate(config.model_dump(mode="json")), Config.model_validate(config.model_dump())):
        stage = reloaded.load.stages[0]
        assert isinstance(stage, ConcurrentLoadStage)
        assert stage.concurrency_level == 4
