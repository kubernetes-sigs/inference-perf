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
"""The record: one conversation's worth of content, with no timing attached.

A record is what every dataset and trace format reduces to. A single request is
a one-turn record. The content of a turn is a list of parts, so a turn can hold
literal text (a recorded conversation), a media reference (an image the request
attaches), or a synthetic spec (a trace that recorded only lengths and prefix
block hashes, from which text has to be built). The parts are data, not
behaviour: nothing here knows how to tokenize, schedule or send.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import Field, model_validator

from inference_perf.config.common import StrictBaseModel

# Bumped when a field changes meaning or a required field is added. Additive
# optional fields do not bump it.
SCHEMA_VERSION = "v1alpha1"


class TextPart(StrictBaseModel):
    """Literal text, as recorded."""

    type: Literal["text"] = "text"
    text: str


class MediaPart(StrictBaseModel):
    """A reference to media the request attaches. `ref` is a path, URL or data URI;
    which of those a source may emit is the source's business."""

    type: Literal["media"] = "media"
    kind: Literal["image", "audio", "video"]
    ref: str


class SyntheticPart(StrictBaseModel):
    """Text that has to be built to a spec, because the source recorded lengths
    rather than content.

    `num_tokens` is the length of this part on its own, before any chat template.
    `block_ids` name the leading prefix blocks of `block_size` tokens each: two
    parts that share leading block ids share that prefix, which is what makes a
    KV cache hit reproducible from a trace that carries no text. The blocks may
    cover fewer tokens than `num_tokens` (the tail is unique to this part) but
    never more than the part has room for.

    `scope` says where a block id means the same text: across the whole trace
    (Mooncake, where ids are global) or only within one session (Weka, where
    each session hashes independently).
    """

    type: Literal["synthetic"] = "synthetic"
    num_tokens: int = Field(ge=0)
    block_ids: List[int] = []
    block_size: int = Field(default=512, gt=0)
    scope: Literal["trace", "session"] = "trace"

    @model_validator(mode="after")
    def _blocks_fit(self) -> SyntheticPart:
        max_blocks = math.ceil(self.num_tokens / self.block_size)
        if len(self.block_ids) > max_blocks:
            raise ValueError(
                f"{len(self.block_ids)} block ids of {self.block_size} tokens cannot fit in {self.num_tokens} tokens"
            )
        return self

    @property
    def prefix_tokens(self) -> int:
        """Tokens covered by the named blocks, capped at the part's own length."""
        return min(len(self.block_ids) * self.block_size, self.num_tokens)


ContentPart = Annotated[Union[TextPart, MediaPart, SyntheticPart], Field(discriminator="type")]


class Turn(StrictBaseModel):
    """One message of the conversation. An assistant turn is the reference
    output for the request that elicits it: its `output_tokens` is the length
    to ask the server for, and its parts (if any) are what was recorded.

    A self-contained turn's parts are the whole prompt of the request that
    elicits the next assistant turn; the turns before it are context the
    source already folded in, so they are not sent again. That is how a
    trace that recorded each round's full prompt size (TraceLab, Weka) is
    stated without inventing per-turn splits it never recorded. A turn that
    is not self-contained is one message, and the prompt is it plus the
    turns before it."""

    role: Literal["system", "user", "assistant", "tool"]
    parts: List[ContentPart] = []
    output_tokens: Optional[int] = Field(default=None, ge=0)
    self_contained: bool = False

    @model_validator(mode="after")
    def _fields_match_role(self) -> Turn:
        if self.output_tokens is not None and self.role != "assistant":
            raise ValueError(f"output_tokens is only meaningful on an assistant turn, not {self.role!r}")
        if self.self_contained and self.role == "assistant":
            raise ValueError("an assistant turn is an output, it cannot be a self-contained prompt")
        return self


class Record(StrictBaseModel):
    """A conversation: ordered turns, an id the arrangement can point at, and
    the session it belongs to when the source groups records."""

    id: str
    turns: List[Turn] = Field(min_length=1)
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = {}

    def assistant_turn_indices(self) -> List[int]:
        """Indices of the turns a request can elicit."""
        return [i for i, turn in enumerate(self.turns) if turn.role == "assistant"]
