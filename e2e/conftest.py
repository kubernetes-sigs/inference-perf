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

from pathlib import Path
import subprocess
import tempfile
import time
import pytest
import requests


@pytest.fixture(scope="module")
def prometheus_server():
    """Starts a lightweight ephemeral Prometheus instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        config_path = tmp_path / "prometheus.yml"

        # Write minimal config pointing to the simulator
        config_path.write_text(
            """
global:
  scrape_interval: 5s
scrape_configs:
  - job_name: 'llm-d-inference-sim'
    static_configs:
      - targets: ['127.0.0.1:18000', '127.0.0.1:18001']
""",
            encoding="utf-8",
        )

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Start prometheus
        proc = subprocess.Popen(
            [
                "prometheus",
                f"--config.file={config_path}",
                f"--storage.tsdb.path={data_dir}",
                "--web.listen-address=127.0.0.1:9090",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Wait for ready
        ready = False
        for _ in range(30):
            try:
                resp = requests.get("http://127.0.0.1:9090/-/ready", timeout=1)
                if resp.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(1)

        if not ready:
            proc.terminate()
            stdout, _ = proc.communicate()
            raise Exception(f"Prometheus failed to become ready. Output:\n{stdout.decode()}")

        yield "http://127.0.0.1:9090"

        # Teardown
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
