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
from typing import Optional

from pydantic import Field

from inference_perf.config.datagen.replay import TraceReplayConfig


class WorkloadReplayConfig(TraceReplayConfig):
    """A recorded trace replayed through the workload record layer (alpha).

    The format names a trace format, built in or `package.module:Class`. A
    format that records sessions needs `load.type: trace_session_replay`, one
    that records independent timestamped requests `trace_replay`; the
    generator refuses the wrong one.
    """

    format: str = Field(description="Trace format: a built-in name, or package.module:Class for a format of your own.")
    block_size: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Tokens per prefix block for formats that record or mint block ids. Defaults to the format's own block size."
        ),
    )
