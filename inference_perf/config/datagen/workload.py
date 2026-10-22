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

from inference_perf.config.common import StrictBaseModel
from inference_perf.config.datagen.replay import SessionReplayConfig


class WorkloadReplayConfig(StrictBaseModel):
    """A recorded workload replayed through the workload record layer (alpha).

    The format names a registered trace parser. Whether the run needs
    `load.type: trace_replay` (timestamped independent requests) or
    `trace_session_replay` (sessions with dependencies) follows from what the
    parser produces; the generator refuses the wrong one.
    """

    format: str = Field(description="Trace format to parse: the name of a registered workload source.")
    file: str = Field(description="Path to the trace file.")
    block_size: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Tokens per prefix block for formats that record or mint block ids. Defaults to the format's own block size."
        ),
    )
    session: Optional[SessionReplayConfig] = Field(
        default=None,
        description=(
            "Session replay settings (wait caps, predecessor timeouts, KV-cache invalidation) for "
            "workloads whose arrangement has dependencies. Ignored for independent requests."
        ),
    )
