import aiohttp
import asyncio
import logging
import sys
import textwrap
import shutil
from contextlib import AsyncContextDecorator
from typing import Optional


logger = logging.getLogger(__name__)


class LLMDInferenceSimRunner(AsyncContextDecorator):
    @staticmethod
    def is_available(executable: str = "llm-d-inference-sim") -> bool:
        """
        Returns whether llm-d-inference-sim is present in the local
        environment.
        """
        return shutil.which(executable) is not None

    executable: str
    argv: list[str]

    _host = "127.0.0.1"
    _port: int
    _proc: "Optional[asyncio.subprocess.Process]" = None
    _wait_until_ready: bool

    def __init__(
        self,
        model: str,
        *cmd_args: str,
        port: int = 8000,
        max_waiting_queue_length: int = 10000,
        executable: str = "llm-d-inference-sim",
        wait_until_ready=True,
    ) -> None:
        self.executable = executable
        self.argv = [
            *("--port", str(port)),
            *("--model", model),
            *("--max-waiting-queue-length", str(max_waiting_queue_length)),
            *cmd_args,
        ]
        self._port = port
        self._wait_until_ready = wait_until_ready

    @property
    def host(self):
        return self._host

    @property
    def port(self):
        return self._port

    async def __aenter__(self) -> "LLMDInferenceSimRunner":
        """
        Starts running the llm-d-inference-sim server in the background.
        Once the contextmanager exits, stop the server using a SIGTERM.
        """
        if not LLMDInferenceSimRunner.is_available(self.executable):
            raise FileNotFoundError(f"executable not found: {self.executable}")

        logger.debug(f"starting server: {self.argv=}")
        self._proc = await asyncio.create_subprocess_exec(
            self.executable,
            *self.argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        if self._wait_until_ready:
            try:
                await self.wait_until_ready()
            except Exception:
                await self.__aexit__(*sys.exc_info())
                raise

        return self

    async def __aexit__(self, *exc):
        """
        Sends a SIGTERM to the server and waits a bit for it to stop.
        Returns true if process exited gracefully.
        """
        terminate_task = asyncio.create_task(self._terminate())
        await self._wait()
        await terminate_task

    async def wait_until_ready(
        self,
        polling_sec: float = 0.5,
        timeout_sec: Optional[float] = 10,
    ) -> None:
        """Waits until the server is ready to serve requests."""
        assert self._proc

        async def wait_http():
            async with aiohttp.ClientSession() as http:
                while True:
                    try:
                        async with http.head(f"http://{self._host}:{self._port}") as resp:
                            await resp.read()
                            logger.debug(f"querying server's / endpoint returned {resp.status=}")
                        return True
                    except (asyncio.exceptions.CancelledError, asyncio.exceptions.TimeoutError):
                        logger.error(f"llm-d-inference-sim server did not become ready after {timeout_sec}s!")
                        raise
                    except Exception as e:
                        logger.debug(f"http polling error: {e}, retrying...")
                        await asyncio.sleep(polling_sec)
                        continue

        async def wait_proc():
            await self._wait()
            raise ConnectionRefusedError("server process exited before port was ready")

        done, pending = await asyncio.wait(
            [asyncio.create_task(x) for x in [wait_http(), wait_proc()]],
            return_when=asyncio.FIRST_COMPLETED,
            timeout=timeout_sec,
        )
        [task.cancel() for task in pending]
        if done:
            # either client finished polling or process ended early, so read the
            # result to raise any potential exceptions.
            [task.result() for task in done]
        else:
            # everything timed out, so one of these will have the timeout
            # exception. await it so it's thrown.
            [await task for task in pending]

    async def _wait(self) -> None:
        proc = self._proc
        assert proc

        stdout, _ = await proc.communicate()
        stdout_pretty = textwrap.indent(stdout.decode(), "  | ")
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
