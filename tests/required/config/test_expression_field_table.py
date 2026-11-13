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
"""One row per config field that takes an expression string, and a guard that no such field goes without one."""

import types
import typing
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Set, Tuple

import pytest
from pydantic import BaseModel, ValidationError

from inference_perf.config import Config, DataConfig, DataGenType, Distribution, SharedPrefix
from inference_perf.config.datagen.multimodal import AudioDatagenConfig, ImageDatagenConfig, VideoDatagenConfig
from inference_perf.config.datagen.replay import ContextCompactionConfig, ConversationReplayConfig, SyntheticAgenticConfig
from inference_perf.config.datagen.visionarena import VisionArenaConfig

_AGENTIC_REQUIRED: Dict[str, object] = {
    "num_sessions": 1,
    "fanout_probability": 0.0,
    "theme_mix": {"generic": 1.0},
    "input_tokens_per_turn": "100",
    "output_tokens_per_turn": "50",
    "turns_per_session": "3",
}
_COMPACTION_REQUIRED: Dict[str, object] = {"trigger_tokens": "700", "target_tokens": "100"}


@dataclass(frozen=True)
class Kind:
    accepted: Tuple[str, ...]
    rejected: Tuple[Tuple[str, str], ...]  # (expression, error the field must give)


# a whole-number field takes a discrete or floored expression and refuses a bare continuous one
WHOLE = Kind(
    accepted=("Poisson(3)", "floor(Uniform(10, 20))", "Min(Poisson(30), 57)"),
    rejected=(("Uniform(10, 20)", "not whole numbers"), ("10 + t", "disallowed symbol")),
)
SECONDS = Kind(accepted=("Uniform(0.1, 0.4)", "2"), rejected=(("0.1 + t", "disallowed symbol"),))
UNIT_INTERVAL = Kind(
    accepted=("Beta(2, 5)", "Min(Max(Normal(0.5, 0.1), 0), 1)"), rejected=(("Uniform(0, 2)", "provably within"),)
)


@dataclass(frozen=True)
class Row:
    model: type[BaseModel]
    field: str
    kind: Kind
    build: Callable[[str, str], object]


def _alone(model: type[BaseModel], required: Dict[str, object]) -> Callable[[str, str], object]:
    return lambda field, raw: model(**{**required, field: raw})


def _io(field: str, raw: str) -> object:
    return DataConfig(type=DataGenType.Synthetic, **{field: raw})


ROWS: List[Row] = [
    Row(DataConfig, "input_distribution", WHOLE, _io),
    Row(DataConfig, "output_distribution", WHOLE, _io),
    *(Row(SharedPrefix, f, WHOLE, _alone(SharedPrefix, {})) for f in ("system_prompt_len", "question_len", "output_len")),
    *(Row(m, "count", WHOLE, _alone(m, {})) for m in (ImageDatagenConfig, VideoDatagenConfig, AudioDatagenConfig)),
    *(
        Row(m, "insertion_point", UNIT_INTERVAL, _alone(m, {}))
        for m in (ImageDatagenConfig, VideoDatagenConfig, AudioDatagenConfig, VisionArenaConfig)
    ),
    *(
        Row(ConversationReplayConfig, f, WHOLE, _alone(ConversationReplayConfig, {}))
        for f in ("dynamic_system_prompt_len", "turns_per_conversation", "input_tokens_per_turn", "output_tokens_per_turn")
    ),
    Row(ConversationReplayConfig, "tool_call_latency_sec", SECONDS, _alone(ConversationReplayConfig, {})),
    *(
        Row(SyntheticAgenticConfig, f, WHOLE, _alone(SyntheticAgenticConfig, _AGENTIC_REQUIRED))
        for f in (
            "input_tokens_per_turn",
            "output_tokens_per_turn",
            "turns_per_session",
            "tool_loop_depth",
            "sub_agents_per_spawn",
            "tool_catalog_size_per_agent",
            "parallel_tool_calls_per_step",
        )
    ),
    *(
        Row(SyntheticAgenticConfig, f, SECONDS, _alone(SyntheticAgenticConfig, _AGENTIC_REQUIRED))
        for f in ("tool_call_latency_sec", "user_think_time_sec")
    ),
    *(
        Row(ContextCompactionConfig, f, WHOLE, _alone(ContextCompactionConfig, _COMPACTION_REQUIRED))
        for f in ("trigger_tokens", "target_tokens")
    ),
]


def _members(annotation: object) -> List[object]:
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _members(typing.get_args(annotation)[0])
    if origin in (typing.Union, types.UnionType):
        return [m for arg in typing.get_args(annotation) for m in _members(arg)]
    return [annotation]


def _nested_models(annotation: object) -> List[type[BaseModel]]:
    found: List[type[BaseModel]] = []
    for member in _members(annotation):
        if isinstance(member, type) and issubclass(member, BaseModel):
            found.append(member)
        found.extend(m for arg in typing.get_args(member) for m in _nested_models(arg))
    return found


# a field takes an expression when str sits in the same union as Distribution
def _expression_fields() -> Set[Tuple[str, str]]:
    seen: Set[type[BaseModel]] = set()
    fields: Set[Tuple[str, str]] = set()
    pending: List[type[BaseModel]] = [Config]
    while pending:
        model = pending.pop()
        if model in seen:
            continue
        seen.add(model)
        for name, info in model.model_fields.items():
            members = _members(info.annotation)
            if str in members and Distribution in members:
                fields.add((model.__name__, name))
            pending.extend(_nested_models(info.annotation))
    return fields


def _cases(pick: Callable[[Row], Sequence[object]]) -> List[object]:
    return [pytest.param(row, item, id=f"{row.model.__name__}.{row.field}={item!r}") for row in ROWS for item in pick(row)]


# Walks the whole Config tree. Expected: exactly the fields in ROWS take an expression, so a new
# expression field fails here until it gets a row.
def test_every_expression_field_has_a_row() -> None:
    assert _expression_fields() == {(row.model.__name__, row.field) for row in ROWS}


# Each accepted example loads and is stored as written. Example: SharedPrefix.question_len='Poisson(3)'.
@pytest.mark.parametrize(("row", "raw"), _cases(lambda row: row.kind.accepted))
def test_field_accepts(row: Row, raw: str) -> None:
    assert getattr(row.build(row.field, raw), row.field) == raw


# Each rejected example fails at config load with its field's reason. Example: question_len='Uniform(10, 20)'
# -> 'not whole numbers'; tool_call_latency_sec='0.1 + t' -> 'disallowed symbol'.
@pytest.mark.parametrize(("row", "case"), _cases(lambda row: row.kind.rejected))
def test_field_rejects(row: Row, case: Tuple[str, str]) -> None:
    raw, reason = case
    with pytest.raises(ValidationError, match=reason):
        row.build(row.field, raw)
