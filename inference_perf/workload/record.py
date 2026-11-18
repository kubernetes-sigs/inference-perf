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
"""The record: one request, stated as data, and the delivery of a session's
records: when and in what order they went out.

The two halves are kept apart. A record says what one request carries and
nothing about when it was sent. A delivery holds a schedule per record: when
it went out, how long it took, which records it waited for. A session joins
the two. A replay honours the delivery and a paced run drops it, so the same
records serve both. A request that belongs to no session is a session of one.

A record's content is messages, and a message's content is parts. There is
one part per kind of content: text, image, audio, video, a tool call, a tool
result. Text and media are stated one of two ways: as recorded, or as a spec
to build at run time because the trace kept sizes and not content. Which way
a part points is not on the part. What a request sends is in `messages`,
what the recording got back is in `reply`, and a reply that a later request
carries as history is the same part in both places.

Every class and every field carries a one line description, and a test fails
when one is missing. All of it is data, not behaviour: nothing here knows how
to tokenize, schedule or send, and every type survives a round trip through
its own dump.
"""

from __future__ import annotations

import math
from typing import Annotated, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Bumped when a field changes meaning or a required field is added. Additive
# optional fields do not bump it.
SCHEMA_VERSION = "v1alpha3"


class StrictModel(BaseModel):
    """A key the schema does not define is an error, not silently dropped."""

    model_config = ConfigDict(extra="forbid")


# ---- records: what one request carries


class TextSpec(StrictModel):
    """Text to build at run time, for a trace that kept lengths and not content."""

    num_tokens: int = Field(ge=0, description="Length of the text to build, before any chat template.")
    block_ids: List[int] = Field(
        default=[],
        description="Leading prefix blocks. Two texts with the same leading ids share that prefix.",
    )
    block_size: int = Field(gt=0, description="Tokens per block.")
    scope: Literal["trace", "session"] = Field(
        default="trace",
        description="Where a block id means the same text: the whole trace or one session.",
    )

    @model_validator(mode="after")
    def _blocks_fit(self) -> TextSpec:
        max_blocks = math.ceil(self.num_tokens / self.block_size)
        if len(self.block_ids) > max_blocks:
            raise ValueError(
                f"{len(self.block_ids)} block ids of {self.block_size} tokens cannot fit in {self.num_tokens} tokens"
            )
        return self

    @property
    def prefix_tokens(self) -> int:
        """Tokens covered by the named blocks, capped at the text's own length."""
        return min(len(self.block_ids) * self.block_size, self.num_tokens)


class ImageSpec(StrictModel):
    """An image to build at run time."""

    width: int = Field(gt=0, description="Width in pixels.")
    height: int = Field(gt=0, description="Height in pixels.")


class AudioSpec(StrictModel):
    """Audio to build at run time."""

    seconds: float = Field(gt=0, description="How long it plays.")


class VideoSpec(StrictModel):
    """A video to build at run time."""

    width: int = Field(gt=0, description="Frame width in pixels.")
    height: int = Field(gt=0, description="Frame height in pixels.")
    frames: int = Field(gt=0, description="How many frames it holds.")


def _one_of(part: StrictModel, recorded: object, spec: object, recorded_name: str) -> None:
    if (recorded is None) == (spec is None):
        raise ValueError(f"{type(part).__name__} takes exactly one of `{recorded_name}` and `spec`")


class TextPart(StrictModel):
    """Text in a message. Exactly one of `text` and `spec` is set."""

    type: Literal["text"] = "text"
    text: Optional[str] = Field(default=None, description="The text as recorded.")
    spec: Optional[TextSpec] = Field(default=None, description="The text to build, when it was not recorded.")

    @model_validator(mode="after")
    def _recorded_or_built(self) -> TextPart:
        _one_of(self, self.text, self.spec, "text")
        return self


class ImagePart(StrictModel):
    """An image in a message. Exactly one of `ref` and `spec` is set."""

    type: Literal["image"] = "image"
    ref: Optional[str] = Field(default=None, description="Where the recorded image is: a path, URL or data URI.")
    spec: Optional[ImageSpec] = Field(default=None, description="The image to build, when it was not recorded.")

    @model_validator(mode="after")
    def _recorded_or_built(self) -> ImagePart:
        _one_of(self, self.ref, self.spec, "ref")
        return self


class AudioPart(StrictModel):
    """Audio in a message. Exactly one of `ref` and `spec` is set."""

    type: Literal["audio"] = "audio"
    ref: Optional[str] = Field(default=None, description="Where the recorded audio is: a path, URL or data URI.")
    spec: Optional[AudioSpec] = Field(default=None, description="The audio to build, when it was not recorded.")

    @model_validator(mode="after")
    def _recorded_or_built(self) -> AudioPart:
        _one_of(self, self.ref, self.spec, "ref")
        return self


class VideoPart(StrictModel):
    """A video in a message. Exactly one of `ref` and `spec` is set."""

    type: Literal["video"] = "video"
    ref: Optional[str] = Field(default=None, description="Where the recorded video is: a path, URL or data URI.")
    spec: Optional[VideoSpec] = Field(default=None, description="The video to build, when it was not recorded.")

    @model_validator(mode="after")
    def _recorded_or_built(self) -> VideoPart:
        _one_of(self, self.ref, self.spec, "ref")
        return self


class ToolCallPart(StrictModel):
    """The model asking for a tool to run."""

    type: Literal["tool_call"] = "tool_call"
    id: Optional[str] = Field(default=None, description="The call's id, when recorded.")
    name: Optional[str] = Field(default=None, description="The tool asked for, when recorded.")
    arguments: Optional[TextPart] = Field(default=None, description="The arguments, as text.")


# what a tool result or the model's thinking can hold: any content, but not a call or another wrapper
Content = Annotated[Union[TextPart, ImagePart, AudioPart, VideoPart], Field(discriminator="type")]


class ToolResultPart(StrictModel):
    """What a tool returned."""

    type: Literal["tool_result"] = "tool_result"
    call_id: Optional[str] = Field(default=None, description="The call it answers, when recorded.")
    content: List[Content] = Field(default=[], description="What came back, in order.")


class ReasoningPart(StrictModel):
    """The model's thinking, kept apart from its answer."""

    type: Literal["reasoning"] = "reasoning"
    content: List[Content] = Field(default=[], description="What it thought, in order: text for most models, media for some.")


Part = Annotated[
    Union[TextPart, ImagePart, AudioPart, VideoPart, ToolCallPart, ToolResultPart, ReasoningPart], Field(discriminator="type")
]


class Message(StrictModel):
    """One message of a conversation."""

    role: Literal["system", "user", "assistant"] = Field(description="Who it is from.")
    parts: List[Part] = Field(default=[], description="Its content, in order.")


class Tool(StrictModel):
    """A tool the request offers the model."""

    name: str = Field(description="What the model calls it.")
    description: Optional[str] = Field(default=None, description="What it does, as told to the model.")
    parameters: Optional[Dict[str, object]] = Field(default=None, description="The arguments it takes, as JSON Schema.")


class Record(StrictModel):
    """One request, and nothing about when it is sent."""

    id: str = Field(description="Its name, unique within its session.")
    messages: List[Message] = Field(min_length=1, description="The conversation the request sends.")
    tools: List[Tool] = Field(default=[], description="The tools it offers the model.")
    model: Optional[str] = Field(default=None, description="The model it was sent to, when recorded.")
    output_tokens: Optional[int] = Field(default=None, ge=0, description="Reply length to ask for.")
    reply: List[Part] = Field(default=[], description="What the recording got back, when it kept it.")
    agent: Optional[str] = Field(default=None, description="Which agent of the session sent it, when there are several.")
    call_id: Optional[str] = Field(
        default=None, description="The tool call it was sent to carry out, when a subagent sent it."
    )
    settings: Dict[str, object] = Field(
        default={}, description="Request settings as recorded, under the recording API's own names, for an API to apply."
    )
    metadata: Dict[str, object] = Field(default={}, description="Labels a consumer may ignore.")

    @model_validator(mode="after")
    def _results_answer_earlier_calls(self) -> Record:
        calls: set[str] = set()
        for parts in [*(message.parts for message in self.messages), self.reply]:
            for part in parts:
                if isinstance(part, ToolCallPart) and part.id is not None:
                    calls.add(part.id)
                elif isinstance(part, ToolResultPart) and part.call_id is not None and part.call_id not in calls:
                    raise ValueError(f"record {self.id!r}: a tool result answers {part.call_id!r}, which no earlier call made")
        return self


# ---- deliveries: when and in what order requests go out


class Schedule(StrictModel):
    """When one record went out.

    Timing is stated two ways and a trace may record either or both.
    `start_ms` and `duration_ms` place the record on a timeline. `after` and
    `wait_ms` place it behind the records it waited for.
    """

    start_ms: Optional[int] = Field(default=None, ge=0, description="When it was sent, on the trace's clock.")
    duration_ms: Optional[int] = Field(default=None, ge=0, description="How long it took.")
    after: List[str] = Field(default=[], description="The records it waited for, by id.")
    wait_ms: int = Field(default=0, ge=0, description="How long after the last of those finished it was sent.")


class Delivery(StrictModel):
    """When and in what order a session's records went out."""

    schedules: Dict[str, Schedule] = Field(default={}, description="One schedule per record id.")

    def of(self, record_id: str) -> Schedule:
        """The schedule of one record. A record without one has no recorded time and waited for nothing."""
        return self.schedules.get(record_id) or Schedule()


# ---- the join


class Session(StrictModel):
    """Requests that belong together, and their delivery."""

    id: str = Field(description="Its name, unique within the trace.")
    records: List[Record] = Field(min_length=1, description="Its requests, in the order they were first sent.")
    delivery: Delivery = Field(default=Delivery(), description="When and in what order they went out.")

    @model_validator(mode="after")
    def _carried_out_calls_were_made(self) -> Session:
        # a subagent's requests name the call that started it, which is in another record's reply
        for record in self.records:
            if record.call_id is None:
                continue
            made_elsewhere = any(
                isinstance(part, ToolCallPart) and part.id == record.call_id
                for other in self.records
                if other.id != record.id
                for part in other.reply
            )
            if not made_elsewhere:
                raise ValueError(
                    f"session {self.id!r}: record {record.id!r} carries out {record.call_id!r}, "
                    "which no other record's reply made"
                )
        return self

    @model_validator(mode="after")
    def _waits_can_be_honoured(self) -> Session:
        ids = [record.id for record in self.records]
        if len(set(ids)) != len(ids):
            raise ValueError(f"session {self.id!r}: record ids must be unique")
        known = set(ids)
        for record_id in self.delivery.schedules:
            if record_id not in known:
                raise ValueError(f"session {self.id!r}: the delivery schedules {record_id!r}, which is not in the session")
        for record in self.records:
            for other in self.delivery.of(record.id).after:
                if other not in known:
                    raise ValueError(
                        f"session {self.id!r}: record {record.id!r} waits for {other!r}, which is not in the session"
                    )
        # a record in a cycle never becomes runnable, so the replay would stall on
        # it instead of failing here.
        waiting = {record.id: set(self.delivery.of(record.id).after) for record in self.records}
        dependents: Dict[str, List[str]] = {record_id: [] for record_id in waiting}
        for record_id, others in waiting.items():
            for other in others:
                dependents[other].append(record_id)
        ready = [record_id for record_id, others in waiting.items() if not others]
        while ready:
            done = ready.pop()
            del waiting[done]
            for child in dependents[done]:
                waiting[child].discard(done)
                if not waiting[child]:
                    ready.append(child)
        if waiting:
            raise ValueError(f"session {self.id!r}: waits form a cycle, so these records can never run: {sorted(waiting)}")
        return self
