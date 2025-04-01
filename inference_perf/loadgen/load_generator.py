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
from .load_timer import LoadTimer, ConstantLoadTimer, PoissonLoadTimer
from inference_perf.datagen import DataGenerator
from inference_perf.client import ModelServerClient
from inference_perf.config import LoadType, LoadConfig
from asyncio import TaskGroup, sleep
import time


class LoadGenerator:
    def __init__(self, datagen: DataGenerator, load_config: LoadConfig) -> None:
        self.datagen = datagen
        self.stageInterval = load_config.interval
        self.load_type = load_config.type
        self.stages = load_config.stages

    def get_timer(self, rate: float) -> LoadTimer:
        if self.load_type == LoadType.POISSON:
            return PoissonLoadTimer(rate=rate)
        return ConstantLoadTimer(rate=rate)

    async def run(self, client: ModelServerClient) -> None:
        for stage_id, stage in enumerate(self.stages):
            timer = self.get_timer(stage.rate)
            start_time = time.time()
            end_time = start_time + stage.duration
            print(f"Stage {stage_id} - run started")
            async with TaskGroup() as tg:
                for _, (data, time_index) in enumerate(
                    zip(self.datagen.get_data(), timer.start_timer(start_time), strict=True)
                ):
                    now = time.time()
                    if time_index < end_time and now < end_time:
                        if time_index > now:
                            await sleep(time_index - time.time())
                        tg.create_task(client.process_request(data, stage_id))
                        continue
                    else:
                        break
            print(f"Stage {stage_id} - run completed")
            if self.stageInterval:
                await sleep(self.stageInterval)
