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
"""Peer tool argument converter (#755): ``inference-perf-convert``."""

from inference_perf.tools.convert.aiperf_profile import convert_aiperf_profile
from inference_perf.tools.convert.cli import main_cli, run
from inference_perf.tools.convert.emit import emit_yaml
from inference_perf.tools.convert.model import Conversion, PeerUsageError, Verdict
from inference_perf.tools.convert.vllm_bench import convert_vllm_bench

__all__ = [
    "Conversion",
    "PeerUsageError",
    "Verdict",
    "convert_aiperf_profile",
    "convert_vllm_bench",
    "emit_yaml",
    "main_cli",
    "run",
]
