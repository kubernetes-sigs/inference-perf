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
"""Unit tests for max_stage_requests, which main.py uses to size the random and synthetic generators.

Those generators build their input and output lengths up front and look a record's lengths up by its
index, so a stage that draws more records than they built would read past the end.
"""

from inference_perf.config import BatchJobConfig, BatchLoadStage, ConcurrentLoadStage, StandardLoadStage
from inference_perf.main import max_stage_requests


# Constant-rate stages of 2 req/s for 3 s and 1 req/s for 10 s draw 6 and 10 records: the most is 10.
def test_standard_stages_draw_rate_times_duration() -> None:
    assert max_stage_requests([StandardLoadStage(rate=2, duration=3), StandardLoadStage(rate=1, duration=10)]) == 10


# Concurrent stages of 7 and 4 requests draw 7 and 4 records: the most is 7.
def test_concurrent_stages_draw_their_request_count() -> None:
    stages = [
        ConcurrentLoadStage(num_requests=7, concurrency_level=1),
        ConcurrentLoadStage(num_requests=4, concurrency_level=1),
    ]
    assert max_stage_requests(stages) == 7


# A batch stage of jobs of 3 and 4 records draws 7, one record per line, and a batch stage of one job of 5
# draws 5: the most is 7.
def test_batch_stages_draw_the_lines_of_all_their_jobs() -> None:
    stages = [
        BatchLoadStage(batches=[BatchJobConfig(batch_size=3), BatchJobConfig(batch_size=4)]),
        BatchLoadStage(batches=[BatchJobConfig(batch_size=5)]),
    ]
    assert max_stage_requests(stages) == 7
