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

from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
import time
import pytest
import logging
import requests

from utils.net import get_free_port


logger = logging.getLogger(__name__)


@dataclass
class PrometheusEnv:
    """Handle to an ephemeral Prometheus and the target it is scraping.

    ``url`` is the Prometheus HTTP base URL. ``sim_port`` is a free port that
    Prometheus is preconfigured to scrape; the test must bind its simulator (or
    mock metrics server) to this port so the exposed metrics get collected.
    """

    url: str
    sim_port: int


@pytest.fixture
def prometheus_server():
    """Starts a lightweight ephemeral Prometheus instance.

    Function-scoped with dynamically allocated ports (both Prometheus' own web
    port and the single scrape target) so multiple copies can run concurrently
    under pytest-xdist without clashing. The test binds its sim/mock server to
    ``PrometheusEnv.sim_port``.
    """
    web_port = get_free_port()
    sim_port = get_free_port()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        config_path = tmp_path / "prometheus.yml"

        # Write minimal config pointing to this test's simulator target.
        config_path.write_text(
            f"""
global:
  scrape_interval: 5s
scrape_configs:
  - job_name: 'llm-d-inference-sim'
    static_configs:
      - targets: ['127.0.0.1:{sim_port}']
""",
            encoding="utf-8",
        )

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        logger.debug(f"Prometheus config and data dirs created at {config_path} and {data_dir}. Starting Prometheus...")

        # Start prometheus
        proc = subprocess.Popen(
            [
                "prometheus",
                f"--config.file={config_path}",
                f"--storage.tsdb.path={data_dir}",
                f"--web.listen-address=127.0.0.1:{web_port}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        logger.debug("Prometheus started. Starting HTTP poll...")

        # Wait for ready
        ready = False
        for _ in range(30):
            try:
                resp = requests.get(f"http://127.0.0.1:{web_port}/-/ready", timeout=1)
                if resp.status_code == 200:
                    logger.debug(f"Prometheus started. Status code: {resp.status_code}")
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(1)

        if not ready:
            proc.terminate()
            stdout, _ = proc.communicate()
            raise Exception(f"Prometheus failed to become ready. Output:\n{stdout.decode()}")

        yield PrometheusEnv(url=f"http://127.0.0.1:{web_port}", sim_port=sim_port)

        # Teardown
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
