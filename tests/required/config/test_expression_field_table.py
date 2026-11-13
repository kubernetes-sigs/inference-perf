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

from inference_perf.config import Config, DataConfig, DataGenType, SharedPrefix
from inference_perf.config.common import LegacyDistributionBlock
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
    # "{m}" in any of these strings is the row's minimum
    accepted: Tuple[str, ...]
    rejected: Tuple[Tuple[str, str], ...]  # (expression, error the field must give)
    range_text: str  # the sentence each field's description states
    bad_number: Callable[[int], float]  # a plain number outside the range
    bad_number_reason: str


# a whole-number field takes a discrete or floored expression, provably at least its minimum
WHOLE = Kind(
    accepted=("{m} + Poisson(3)", "floor(Uniform({m}, {m} + 20))", "Min({m} + Poisson(30), 57)"),
    rejected=(
        ("Uniform(10, 20)", "not whole numbers"),
        ("10 + t", "disallowed symbol"),
        ("{m} - 1 + Poisson(3)", "not provably at least {m}"),
        ("Normal(50, 10)", r"Max\(floor\(Normal\(50, 10\)\), {m}\)"),
        # whole-numbered, but the bounds walk has no rule for Piecewise, so the minimum is unproven
        ("Piecewise(({m} + 1, Uniform(0, 1) < 1/2), ({m} + 2, True))", "not provably at least {m}"),
    ),
    range_text="At least {m}.",
    bad_number=lambda m: m - 1,
    bad_number_reason="below {m}",
)
SECONDS = Kind(
    accepted=("Uniform(0.1, 0.4)", "2"),
    rejected=(("0.1 + t", "disallowed symbol"), ("Uniform(-1, 1)", "not provably at least 0")),
    range_text="At least 0.",
    bad_number=lambda m: -0.5,
    bad_number_reason="below 0",
)
UNIT_INTERVAL = Kind(
    accepted=("Beta(2, 5)", "Min(Max(Normal(0.5, 0.1), 0), 1)"),
    rejected=(("Uniform(0, 2)", "provably within"),),
    range_text="Within [0, 1].",
    bad_number=lambda m: 1.5,
    bad_number_reason="within",
)


@dataclass(frozen=True)
class Row:
    model: type[BaseModel]
    field: str
    kind: Kind
    build: Callable[[str, object], object]
    minimum: int = 0
    stored_as: str = ""  # an alias folds into another field on load


def _alone(model: type[BaseModel], required: Dict[str, object]) -> Callable[[str, object], object]:
    return lambda field, raw: model(**{**required, field: raw})


def _io(field: str, raw: object) -> object:
    return DataConfig(type=DataGenType.Synthetic, **{field: raw})


_REPLAY = _alone(ConversationReplayConfig, {})
_AGENTIC = _alone(SyntheticAgenticConfig, _AGENTIC_REQUIRED)

ROWS: List[Row] = [
    Row(DataConfig, "input_distribution", WHOLE, _io, minimum=1),
    Row(DataConfig, "output_distribution", WHOLE, _io, minimum=1),
    Row(SharedPrefix, "system_prompt_len", WHOLE, _alone(SharedPrefix, {}), minimum=0),
    *(Row(SharedPrefix, f, WHOLE, _alone(SharedPrefix, {}), minimum=1) for f in ("question_len", "output_len")),
    Row(SharedPrefix, "question_distribution", WHOLE, _alone(SharedPrefix, {}), minimum=1, stored_as="question_len"),
    Row(SharedPrefix, "output_distribution", WHOLE, _alone(SharedPrefix, {}), minimum=1, stored_as="output_len"),
    *(Row(m, "count", WHOLE, _alone(m, {})) for m in (ImageDatagenConfig, VideoDatagenConfig, AudioDatagenConfig)),
    *(
        Row(m, "insertion_point", UNIT_INTERVAL, _alone(m, {}))
        for m in (ImageDatagenConfig, VideoDatagenConfig, AudioDatagenConfig, VisionArenaConfig)
    ),
    Row(ConversationReplayConfig, "dynamic_system_prompt_len", WHOLE, _REPLAY, minimum=0),
    *(
        Row(ConversationReplayConfig, f, WHOLE, _REPLAY, minimum=1)
        for f in ("turns_per_conversation", "input_tokens_per_turn", "output_tokens_per_turn")
    ),
    Row(ConversationReplayConfig, "tool_call_latency_sec", SECONDS, _REPLAY),
    *(
        Row(SyntheticAgenticConfig, f, WHOLE, _AGENTIC, minimum=1)
        for f in ("input_tokens_per_turn", "output_tokens_per_turn", "turns_per_session")
    ),
    *(
        Row(SyntheticAgenticConfig, f, WHOLE, _AGENTIC, minimum=0)
        for f in ("tool_loop_depth", "sub_agents_per_spawn", "tool_catalog_size_per_agent", "parallel_tool_calls_per_step")
    ),
    *(Row(SyntheticAgenticConfig, f, SECONDS, _AGENTIC) for f in ("tool_call_latency_sec", "user_think_time_sec")),
    *(
        Row(ContextCompactionConfig, f, WHOLE, _alone(ContextCompactionConfig, _COMPACTION_REQUIRED), minimum=1)
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


# every expression field type carries the LegacyDistributionBlock marker, on the field or on an Annotated member
def _takes_expression(annotation: object, metadata: List[object]) -> bool:
    if any(isinstance(item, LegacyDistributionBlock) for item in metadata):
        return True
    if typing.get_origin(annotation) is typing.Annotated:
        inner, *extras = typing.get_args(annotation)
        return _takes_expression(inner, list(extras))
    return any(_takes_expression(arg, []) for arg in typing.get_args(annotation))


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
            if _takes_expression(info.annotation, list(info.metadata)):
                fields.add((model.__name__, name))
            pending.extend(_nested_models(info.annotation))
    return fields


def _cases(pick: Callable[[Row], Sequence[object]]) -> List[object]:
    return [pytest.param(row, item, id=f"{row.model.__name__}.{row.field}={item!r}") for row in ROWS for item in pick(row)]


def _takes_number(row: Row) -> bool:
    members = _members(row.model.model_fields[row.field].annotation)
    return int in members or float in members


def _rows(numbers_only: bool = False) -> List[object]:
    return [
        pytest.param(row, id=f"{row.model.__name__}.{row.field}") for row in ROWS if not numbers_only or _takes_number(row)
    ]


# Walks the whole Config tree. Expected: exactly the fields in ROWS take an expression, so a new
# expression field fails here until it gets a row.
def test_every_expression_field_has_a_row() -> None:
    assert _expression_fields() == {(row.model.__name__, row.field) for row in ROWS}


# Each accepted example, built from the field's minimum, loads and is stored as written, on the field an
# alias folds into. Example: SharedPrefix.question_len (at least 1) takes '1 + Poisson(3)';
# question_distribution='1 + Poisson(3)' lands on question_len.
@pytest.mark.parametrize(("row", "raw"), _cases(lambda row: row.kind.accepted))
def test_field_accepts(row: Row, raw: str) -> None:
    raw = raw.format(m=row.minimum)
    assert getattr(row.build(row.field, raw), row.stored_as or row.field) == raw


# Each rejected example fails at config load with its field's reason. Example: question_len='Uniform(10, 20)'
# -> 'not whole numbers'; question_len='0 + Poisson(3)' -> 'not provably at least 1'.
@pytest.mark.parametrize(("row", "case"), _cases(lambda row: row.kind.rejected))
def test_field_rejects(row: Row, case: Tuple[str, str]) -> None:
    raw, reason = case
    with pytest.raises(ValidationError, match=reason.format(m=row.minimum)):
        row.build(row.field, raw.format(m=row.minimum))


# On a field that takes a plain number, one outside the range is rejected too. Example: question_len=0 ->
# 'below 1'; insertion_point=1.5 -> 'within'.
@pytest.mark.parametrize("row", _rows(numbers_only=True))
def test_number_outside_range_rejected(row: Row) -> None:
    with pytest.raises(ValidationError, match=row.kind.bad_number_reason.format(m=row.minimum)):
        row.build(row.field, row.kind.bad_number(row.minimum))


# Each field's description states its range, so a user reading the docs or the CLI flags sees it.
# Example: question_len's description contains 'At least 1.'.
@pytest.mark.parametrize("row", _rows())
def test_description_states_range(row: Row) -> None:
    description = row.model.model_fields[row.field].description or ""
    assert row.kind.range_text.format(m=row.minimum) in description, description
