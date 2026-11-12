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
"""The shape a unary chat or completion body must have to be read at all.

Only the fields ``process_response`` reads directly are declared. The envelope
(``id``, ``object``, ``created``, ``model``), ``finish_reason`` values and every
other field differ across TGI, vLLM, SGLang and gateways, so they pass through
unchecked. Tool calls stay with ``tool_output``, which skips a malformed entry
and keeps its siblings. A body that fails here raises pydantic's ``ValidationError`` and the
client records the request as failed.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class _Body(BaseModel):
    model_config = ConfigDict(extra="allow")


class ChatResponseMessage(_Body):
    # null on a reasoning-only or tool-only reply
    content: Optional[str] = None


class ChatChoice(_Body):
    message: ChatResponseMessage


class ChatCompletionResponse(_Body):
    # absent on a body that says nothing, which the empty-response check reports
    choices: List[ChatChoice] = []
    # kept as a plain dict so it is reported exactly as the server sent it
    usage: Optional[Dict[str, object]] = None


class CompletionChoice(_Body):
    text: str


class CompletionResponse(_Body):
    choices: List[CompletionChoice] = []
    usage: Optional[Dict[str, object]] = None
