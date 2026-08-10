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
"""Real vLLM server (CPU mode) for live-oracle e2e tests.

Unlike llm-d-inference-sim, a real vLLM carries the real tokenizer, the real
``usage`` accounting, and the real ``/metrics`` exposition, so tests against
it check our numbers against the server's own numbers (#627, #580, #669).
CPU mode changes only speed, never token counts, which makes those oracles
available without GPU capacity. Keep load tiny: a small model on a CI runner
serves on the order of tens of tokens per second.

Two provisioning modes, resolved in this order:

1. External: ``E2E_VLLM_BASE_URL`` points at an already-running server (CI
   starts the ``vllm/vllm-openai-cpu`` container once and shares it across
   tests; the operator is responsible for serving a model with a chat
   template and with greedy sampling as the generation default, see below).
   The runner never starts or stops anything in this mode, so tests sharing
   it must not assume exclusive access unless run sequentially.
2. Spawned: a ``vllm`` executable on PATH is started per runner with
   ``vllm serve``. Slow (a full model load per test) but hermetic.

Tight token accounting requires greedy sampling
(``--override-generation-config '{"temperature": 0}'``): the client does not
send ``temperature``, so under default sampling every run draws different
outputs and accuracy failures cannot be replayed. Greedy makes a given
prompt's output, and therefore any failure, deterministic. It does NOT make
re-encoding exact: detokenize-then-re-encode is not injective (adjacent
generated tokens like ``'.' + 'I'`` or ``'\\n' + '\\n'`` re-encode as one
token), which is a real property of text-based counting, not a bug in either
side; the accuracy tests bound it instead of assuming it away.

If neither is available, ``is_available()`` is False and tests skip, keeping
the gating e2e job green without any workflow coupling.
"""

import asyncio
import logging
import os
import shutil
import sys
import textwrap
from contextlib import AsyncContextDecorator
from pathlib import Path
from typing import Optional

import aiohttp

from utils.testdata import TEST_E2E_TESTDATA

logger = logging.getLogger(__name__)

ENV_BASE_URL = "E2E_VLLM_BASE_URL"

# vLLM's canonical tiny test model. Its tokenizer prepends a BOS token to
# completion prompts, which keeps the #564-lineage special-token handling in
# play on the completion path.
DEFAULT_MODEL = "facebook/opt-125m"

# Minimal template so models without a built-in chat template (opt-125m)
# still serve /v1/chat/completions.
SIMPLE_CHAT_TEMPLATE = TEST_E2E_TESTDATA / "simple_chat_template.jinja"

# The nix devshell exports python env vars for a torch built against a
# different CPython ABI; a spawned vLLM must not inherit them (same fix as
# the vllm-bench harness). LD_LIBRARY_PATH is deliberately kept.
_HOST_PYTHON_ENV_VARS = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "NIX_PYTHONPATH")


class VLLMServerRunner(AsyncContextDecorator):
    @staticmethod
    def is_available(executable: str = "vllm") -> bool:
        """Whether a real vLLM server can be reached or started."""
        return bool(os.environ.get(ENV_BASE_URL)) or shutil.which(executable) is not None

    _proc: "Optional[asyncio.subprocess.Process]" = None
    _base_url: str
    _external: bool

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *cmd_args: str,
        port: int = 8000,
        max_model_len: int = 2048,
        chat_template: Optional[Path] = SIMPLE_CHAT_TEMPLATE,
        executable: str = "vllm",
        startup_timeout_sec: float = 600,
    ) -> None:
        self.model = model
        self.executable = executable
        self.startup_timeout_sec = startup_timeout_sec
        self._external = bool(os.environ.get(ENV_BASE_URL))
        if self._external:
            self._base_url = os.environ[ENV_BASE_URL].rstrip("/")
        else:
            self._base_url = f"http://127.0.0.1:{port}"
        self.argv = [
            "serve",
            model,
            *("--port", str(port)),
            *("--max-model-len", str(max_model_len)),
            *("--override-generation-config", '{"temperature": 0}'),
            # Token accounting does not depend on compiled kernels, and
            # skipping the torch.compile warmup makes startup faster and
            # more robust on small shared machines.
            "--enforce-eager",
            *(("--chat-template", str(chat_template)) if chat_template else ()),
            *cmd_args,
        ]

    @property
    def base_url(self) -> str:
        return self._base_url

    async def __aenter__(self) -> "VLLMServerRunner":
        if self._external:
            await self.wait_until_ready(timeout_sec=30)
            self.model = await self._served_model()
            logger.debug(f"using external vLLM at {self._base_url} serving {self.model}")
            return self

        if shutil.which(self.executable) is None:
            raise FileNotFoundError(f"executable not found: {self.executable}")

        env = {k: v for k, v in os.environ.items() if k not in _HOST_PYTHON_ENV_VARS}
        env.setdefault("VLLM_CPU_KVCACHE_SPACE", "4")
        logger.debug(f"starting server: {self.argv=}")
        self._proc = await asyncio.create_subprocess_exec(
            self.executable,
            *self.argv,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            # A CPU vLLM boot includes torch import, model download, and
            # graph warmup; there is no pytest-timeout in this suite, so this
            # deadline is what stands between a hung boot and the CI job cap.
            await self.wait_until_ready(timeout_sec=self.startup_timeout_sec)
        except Exception:
            await self.__aexit__(*sys.exc_info())
            raise
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._proc is None:
            return
        terminate_task = asyncio.create_task(self._terminate())
        await self._wait()
        await terminate_task

    async def wait_until_ready(self, polling_sec: float = 1.0, timeout_sec: float = 600) -> None:
        """Waits until /health returns 200 (or the spawned process exits)."""

        async def wait_http() -> bool:
            async with aiohttp.ClientSession() as http:
                while True:
                    try:
                        async with http.get(f"{self._base_url}/health") as resp:
                            if resp.status == 200:
                                return True
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        logger.error(f"vLLM server did not become ready after {timeout_sec}s!")
                        raise
                    except Exception as e:
                        logger.debug(f"http polling error: {e}, retrying...")
                    await asyncio.sleep(polling_sec)

        async def wait_proc() -> None:
            await self._wait()
            raise ConnectionRefusedError("server process exited before becoming ready")

        waiters = [wait_http()] + ([wait_proc()] if self._proc else [])
        done, pending = await asyncio.wait(
            [asyncio.create_task(x) for x in waiters],
            return_when=asyncio.FIRST_COMPLETED,
            timeout=timeout_sec,
        )
        [task.cancel() for task in pending]
        if done:
            [task.result() for task in done]
        else:
            [await task for task in pending]

    async def fetch_metrics(self) -> str:
        """The server's current /metrics exposition text."""
        async with aiohttp.ClientSession() as http:
            async with http.get(f"{self._base_url}/metrics") as resp:
                resp.raise_for_status()
                return await resp.text()

    async def _served_model(self) -> str:
        async with aiohttp.ClientSession() as http:
            async with http.get(f"{self._base_url}/v1/models") as resp:
                resp.raise_for_status()
                models = (await resp.json())["data"]
                assert models, "external vLLM serves no models"
                return str(models[0]["id"])

    async def _wait(self) -> None:
        proc = self._proc
        assert proc

        stdout, _ = await proc.communicate()
        self.stdout = stdout.decode()
        stdout_pretty = textwrap.indent(self.stdout, "  | ")
        logger.debug(f"server exited with status {proc.returncode}, output:\n{stdout_pretty}")

    async def _terminate(self) -> None:
        proc = self._proc
        assert proc

        try:
            proc.terminate()
            await asyncio.sleep(2)
            proc.kill()
        except ProcessLookupError:
            pass  # process already exited
        except Exception as e:
            logger.debug(f"server failed to be terminated: {e}")
            raise
