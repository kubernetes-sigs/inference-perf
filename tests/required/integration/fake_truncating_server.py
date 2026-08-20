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
"""A fake model server that 200s and then breaks the stream, for the
integration tier.

The sim cannot fail on cue, so the truncated-stream condition has to be
induced. This speaks raw HTTP/1.1 rather than going through an server
framework because the failure being reproduced *is* a protocol-level lie: the
response declares a ``Content-Length`` larger than the body it actually sends
and then half-closes. aiohttp raises a real ``ClientPayloadError`` when the
connection ends early, so tests exercise the client's genuine exception path
instead of a patched-in exception (the #606 Integration tier's "fake the
conditions, never the oracle" principle).

With ``missing_bytes=0`` the server keeps its promise and the body arrives
complete, which is the condition #713 needs: a well-formed 200 whose *payload*
is the failure (an in-band error frame, or nothing at all).
"""

import asyncio
from types import TracebackType
from typing import List, Optional, Type


class TruncatingSSEServer:
    """Serves a 200 SSE response that stops short of its declared length.

    ``events`` are the JSON payloads to emit as ``data:`` frames before the
    break; pass an empty list to break before any body byte is sent.
    ``sent_body`` records exactly what went out on the wire so a test can
    assert the client preserved that text and not an approximation of it.

    ``body`` replaces the SSE framing with a verbatim body (a unary JSON
    response, say) and ``content_type`` labels it; ``events`` is ignored then.
    """

    def __init__(
        self,
        events: List[str],
        missing_bytes: int = 64,
        *,
        body: Optional[str] = None,
        content_type: str = "text/event-stream",
    ) -> None:
        self.events = events
        # How far the declared Content-Length overshoots what we actually
        # write. Any positive value triggers the client-side error; 0 delivers
        # the body complete.
        self.missing_bytes = missing_bytes
        self.body = body
        self.content_type = content_type
        self.sent_body = ""
        self._server: Optional[asyncio.AbstractServer] = None
        self.port = 0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def __aenter__(self) -> "TruncatingSSEServer":
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(
        self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]
    ) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    @staticmethod
    async def _drain_request(reader: asyncio.StreamReader) -> None:
        """Read the full request so the client's own write completes cleanly and
        the only failure under test is the response-side truncation."""
        header = await reader.readuntil(b"\r\n\r\n")
        content_length = 0
        for line in header.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                content_length = int(line.split(b":", 1)[1])
        if content_length:
            await reader.readexactly(content_length)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await self._drain_request(reader)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            writer.close()
            return

        body = self.body if self.body is not None else "".join(f"data: {event}\n\n" for event in self.events)
        encoded = body.encode()
        headers = (
            "HTTP/1.1 200 OK\r\n"
            f"Content-Type: {self.content_type}\r\n"
            # The lie (when missing_bytes > 0): promise more than we are going to send.
            f"Content-Length: {len(encoded) + self.missing_bytes}\r\n"
            "\r\n"
        )
        writer.write(headers.encode() + encoded)
        await writer.drain()
        self.sent_body = body
        # Half-close so the partial body is delivered and *then* the stream
        # ends early. An abrupt reset here would surface as a connection error
        # rather than the incomplete-payload case #531 describes.
        if writer.can_write_eof():
            writer.write_eof()
        writer.close()
