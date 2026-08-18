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

"""What a 200 can carry that is not a completion.

A non-200 status is a failure by construction. A 200 is only a success if its
body actually is a completion: some servers and proxies answer 200 and put the
failure in the body, and a body with no content and no ``usage`` says nothing at
all. Both used to be recorded as zero-token successes (#713). The exceptions here
are raised by ``process_response`` on such bodies so the client records the
request as failed, with the body preserved, the same way it does for a stream
that breaks partway (``StreamInterruptedError``).
"""

import json
from typing import Any, Optional


class InvalidResponseError(Exception):
    """A 200 response whose body is not a completion.

    Unlike ``StreamInterruptedError`` this is raised after the body was read: the
    transport succeeded, the payload is what failed. ``raw_content`` is the body
    as received, so the client can record what the server actually sent. Streaming
    callers pass the bytes read so far; unary callers leave it empty because the
    client already holds the body it handed to ``process_response``.
    """

    def __init__(self, message: str, raw_content: str = "") -> None:
        super().__init__(message)
        self.raw_content = raw_content


class InBandError(InvalidResponseError):
    """The body carries a top-level ``error`` where a completion was expected.

    ``str(exc)`` is the error payload's JSON text, so it lands in ``error_msg``
    in the same shape as a non-200 body and reportgen's ``parse_error_message``
    pulls the server's message out of it unchanged.
    """


class EmptyResponseError(InvalidResponseError):
    """The body parsed as a completion but yielded no content and no ``usage``.

    A response with neither has nothing that could be measured, and a well-formed
    empty completion always carries ``usage``: it is mandatory in a unary
    completion body, and every streaming request asks for it with
    ``stream_options.include_usage``.
    """

    def __init__(self, raw_content: str = "") -> None:
        super().__init__("200 response with no completion content and no usage", raw_content)


def in_band_error(data: Any) -> Optional[str]:
    """Return the JSON text of ``data`` if it is a body or SSE frame carrying a
    top-level ``error``, else None.

    Only a top-level key counts. That is the shape every server family uses for
    an error it delivers in-band: the OpenAI ``{"error": {...}}`` object (vLLM and
    SGLang emit it mid-stream), TGI's ``{"error": "...", "error_type": ...}``
    string form, and Anthropic's ``{"type": "error", "error": {...}}`` event.
    """
    if isinstance(data, dict) and data.get("error"):
        return json.dumps(data)
    return None
