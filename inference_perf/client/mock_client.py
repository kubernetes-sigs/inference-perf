# Copyright 2025 The Kubernetes Authors.
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
from inference_perf.config import RequestMetric
from inference_perf.datagen import InferenceData
from inference_perf.reportgen import ReportGenerator
from .base import ModelServerClient
import asyncio


class MockModelServerClient(ModelServerClient):
    def __init__(self) -> None:
        pass

    def set_report_generator(self, reportgen: ReportGenerator) -> None:
        self.reportgen = reportgen

    async def process_request(self, payload: InferenceData, stage_id: int) -> None:
        print("Processing request - " + str(payload.data) + " for stage - " + str(stage_id))
        await asyncio.sleep(3)
        self.reportgen.collect_request_metrics(
            RequestMetric(
                stage_id=stage_id,
                prompt_len=0,
                output_len=0,
                start_time=1.23,
                end_time=3.21,
            )
        )
