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
"""The workload APIs: what a benchmark sends, stated as data.

- `record.py`: a record is what one request says, a session the requests of
  one session. When and in what order they went out is the session's
  delivery, kept beside the records and not on them.
- `format.py`: a trace format says what the rows of one session mean, and
  nothing else.
- `reader.py`: where rows come from, which sessions are kept and how recorded
  waits are shaped, once for every format.
- `text.py`: from a record to the messages that go on the wire, with any
  tokenizer that encodes and decodes.

Nothing here schedules, sends or measures, and importing it loads no model
server, tokenizer or dataset library. The schema is versioned
(`SCHEMA_VERSION`) and alpha: it can change between minor releases until it
graduates.
"""

from .format import Row, TraceFormat, format_class, format_names, register
from .reader import (
    FilterExpressionError,
    Selection,
    TraceLocation,
    continues,
    in_seeded_order,
    read_sessions,
    sessions_of,
    starting_at_zero,
    with_waits,
)
from .record import (
    SCHEMA_VERSION,
    AudioPart,
    AudioSpec,
    Delivery,
    ImagePart,
    ImageSpec,
    Message,
    Part,
    Record,
    Schedule,
    ReasoningPart,
    Session,
    TextPart,
    TextSpec,
    Tool,
    ToolCallPart,
    ToolResultPart,
    VideoPart,
    VideoSpec,
)
from .text import BlockTextMaterializer, BuiltMessage, Prompter, Tokenizer

__all__ = [
    "SCHEMA_VERSION",
    "AudioPart",
    "AudioSpec",
    "BlockTextMaterializer",
    "BuiltMessage",
    "Delivery",
    "FilterExpressionError",
    "ImagePart",
    "ImageSpec",
    "Message",
    "Part",
    "Prompter",
    "Record",
    "Row",
    "Schedule",
    "Selection",
    "Session",
    "ReasoningPart",
    "TextPart",
    "TextSpec",
    "Tokenizer",
    "Tool",
    "ToolCallPart",
    "ToolResultPart",
    "TraceFormat",
    "TraceLocation",
    "VideoPart",
    "VideoSpec",
    "continues",
    "format_class",
    "format_names",
    "in_seeded_order",
    "read_sessions",
    "register",
    "sessions_of",
    "starting_at_zero",
    "with_waits",
]
