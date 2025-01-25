# Copyright 2025
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
from loadgen import LoadGenerator
import subprocess


class InferencePerfRunner:
    # Configuration and benchmarking setup
    def __init__(self) -> None:
        self.load_script = LoadGenerator().generate(url="http://test.k6.io")

    # Local testing with k6 executable
    def run_local(self) -> None:
        filename = "tmp/script.js"
        with open(filename, "w") as scriptfile:
            scriptfile.write(self.load_script)

        subprocess.run(["k6", "run", filename])

    # Distributed testing with Kubernetes
    def run_distributed_k8s(self) -> None:
        pass


if __name__ == "__main__":
    InferencePerfRunner().run_local()
