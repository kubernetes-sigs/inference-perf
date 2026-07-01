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
import socket


def get_free_port(host: str = "127.0.0.1") -> int:
    """Allocate an ephemeral TCP port and return it.

    Binds to port 0, reads the OS-assigned port, then releases the socket so a
    subprocess (sim, mock server, Prometheus) can bind it. There is an inherent
    TOCTOU window between release and rebind, but with per-worker allocation
    collisions are vanishingly rare in practice and this is the standard way to
    hand a free port to a child process. Using distinct ports per test is what
    lets the e2e suite run under pytest-xdist without port clashes.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, 0))
        return s.getsockname()[1]
