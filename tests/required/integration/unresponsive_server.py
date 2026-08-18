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
"""Two ways for a model server endpoint to be dead, as controllable fakes.

``UnresponsiveServer`` is the "hang instead of crash" condition from #620: the
TCP handshake completes, the request bytes are accepted, and not one response
byte is ever written back. Nothing on the server side will ever end such a
request, so only the client's own timeout can, which is what makes it the right
fake for asserting that the timeout exists.

``reserve_unbound_port`` is the complementary condition: a port that is
guaranteed to have been free, with nothing listening on it, so every connect
attempt is refused at the transport layer.

Deliberately minimal: no HTTP parsing, no aiohttp, no response shaping. The
fake supplies the failure condition, the assertions come from the real client
and the real reports.
"""

import asyncio
import socket
from types import TracebackType
from typing import List, Optional, Type


# Returns a port number (e.g. 41234) that was free a moment ago and has nothing listening
# on it, so a connect to 127.0.0.1:<port> is refused.
def reserve_unbound_port(host: str = "127.0.0.1") -> int:
    """Return a TCP port that had nothing listening on it a moment ago.

    Binds an ephemeral port, reads the number the OS assigned, then closes the
    socket so the port is free again. There is an inherent race: between the
    close and the moment the test connects, something else on the machine could
    claim the same port. The window is tiny and the OS does not immediately
    recycle ephemeral ports, so in practice a connect to this port is refused.
    This is the same tradeoff ``e2e/utils/net.py:get_free_port`` documents; it
    is repeated here rather than imported because ``e2e/utils`` is only on the
    import path for the e2e tier.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, 0))
        port: int = sock.getsockname()[1]
    return port


# async-with fake: listens on an ephemeral 127.0.0.1 port, accepts every connection, reads
# and discards whatever arrives, never writes a byte back. .connections counts accepts,
# .base_url is http://127.0.0.1:<port>.
class UnresponsiveServer:
    """Accepts TCP connections, reads whatever arrives, never writes a byte.

    Counting accepted connections is what lets a caller tell a single timed-out
    attempt apart from a retry loop: N requests that each connect once produce
    at most N connections, while a client that retries on timeout produces more.
    """

    def __init__(self, host: str = "127.0.0.1") -> None:
        self.host = host
        self.port = 0
        self.connections = 0
        self._server: Optional[asyncio.AbstractServer] = None
        self._writers: List[asyncio.StreamWriter] = []

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # Per-connection handler: bumps .connections, then drains the socket until the peer closes.
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        self._writers.append(writer)
        try:
            # Keep draining so the client's send never blocks on a full receive
            # buffer, and so this coroutine parks until the peer goes away.
            # read() returns b"" at EOF, which is the only way out.
            while await reader.read(4096):
                pass
        except (ConnectionResetError, asyncio.CancelledError):
            pass

    # Starts listening on port 0 and records the port the OS assigned.
    async def __aenter__(self) -> "UnresponsiveServer":
        self._server = await asyncio.start_server(self._handle, self.host, 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    # Closes every accepted connection, then the listener.
    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        for writer in self._writers:
            try:
                writer.close()
            except Exception:  # noqa: BLE001 - teardown must not mask the test's own failure
                pass
        self._writers.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
