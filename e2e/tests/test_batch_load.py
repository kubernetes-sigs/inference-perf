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
"""`load.type: batch` from the command line to the Batch API and back (#1069).

The unit tests replace the batch job or run it on its own. This runs `inference-perf --config_file` the way a
user does, against an in-process stand-in for /v1/files and /v1/batches, so main.py's wiring, the stage loop
and a run that ends with no request metrics are covered together.
"""

import json
import re

import pytest

from utils.benchmark import run_benchmark_minimal
from utils.fake_batch_gateway import FakeBatchGateway
from utils.testdata import TEST_E2E_TESTDATA, extract_tarball


# Two stages: jobs of 3 lines (30m window) and 2 lines, then one job of 2 lines. The gateway answers every line
# 200 except custom_ids ending in "-1". The run exits 0 and logs each batch: 2 of 3, 1 of 2 and 1 of 2 lines
# succeeded. It writes only the config and validation reports, since a batch run collects no request metrics.
# The gateway got three files, whose lines are completion requests for model "m" with stream false, and stage 1
# started the mock prompts over.
@pytest.mark.asyncio
async def test_batch_load_runs_each_stage_through_the_batch_api() -> None:
    tokenizer = extract_tarball(TEST_E2E_TESTDATA / "models" / "google_gemma-3-270m.tar.gz")
    async with FakeBatchGateway() as gateway:
        result = await run_benchmark_minimal(
            {
                "api": {"type": "completion"},
                "data": {"type": "mock"},
                "load": {
                    "type": "batch",
                    "interval": 0,
                    "stages": [
                        {
                            "batches": [{"batch_size": 3, "completion_window": "30m"}, {"batch_size": 2}],
                            "poll_interval_sec": 0.1,
                        },
                        {"batches": [{"batch_size": 2}], "poll_interval_sec": 0.1},
                    ],
                },
                "server": {"type": "vllm", "base_url": gateway.base_url, "model_name": "m"},
                "tokenizer": {"pretrained_model_name_or_path": str(tokenizer)},
            },
            timeout_sec=300,
        )

    assert result.success, result.stdout
    for stage, job, outcome in [(0, 0, "2 of 3"), (0, 1, "1 of 2"), (1, 0, "1 of 2")]:
        assert re.search(
            rf"Stage {stage} batch {job} \(batch-\d\) is completed after .*: {outcome} lines succeeded", result.stdout
        )
    assert result.reports is not None and sorted(result.reports) == ["config.yaml", "validation.json"]

    lines = [json.loads(line) for upload in gateway.uploads for line in upload.splitlines()]
    assert sorted(line["custom_id"] for line in lines) == [
        "s0-b0-0",
        "s0-b0-1",
        "s0-b0-2",
        "s0-b1-0",
        "s0-b1-1",
        "s1-b0-0",
        "s1-b0-1",
    ]
    assert {(line["url"], line["body"]["model"], line["body"]["stream"]) for line in lines} == {
        ("/v1/completions", "m", False)
    }
    stage_1 = sorted(line["body"]["prompt"] for line in lines if line["custom_id"].startswith("s1-"))
    assert stage_1 == ["1 2 3 1", "1 2 3 2"]
