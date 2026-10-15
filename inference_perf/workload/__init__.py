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

A workload is two things kept apart on purpose:

- records (`record.py`): the pool of content, as ordered turns of role plus
  content parts. A record carries no timing.
- an arrangement (`arrangement.py`): how records become requests over time,
  as nodes that name a record, a turn, a send time and optional dependencies.

A trace format is a `WorkloadSource` (`sources/`) that parses its file into
both. Everything downstream is shared: `materialize.py` turns synthetic content
into text, and the load generators send it.

The schema is versioned (`SCHEMA_VERSION`) and alpha: it can change between
minor releases until it graduates.
"""

from .arrangement import Arrangement, Node
from .record import SCHEMA_VERSION, ContentPart, MediaPart, Record, SyntheticPart, TextPart, Turn

__all__ = [
    "SCHEMA_VERSION",
    "Arrangement",
    "ContentPart",
    "MediaPart",
    "Node",
    "Record",
    "SyntheticPart",
    "TextPart",
    "Turn",
]
