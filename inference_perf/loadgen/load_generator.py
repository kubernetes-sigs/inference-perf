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
from enum import Enum
from .load_timer import LoadTimer, ConstantLoadTimer, PoissonLoadTimer
from inference_perf.datagen import DataGenerator
from inference_perf.client import ModelServerClient
from asyncio import TaskGroup, sleep
import time


class LoadType(Enum):
    CONSTANT = 1
    POISSON = 2


class LoadGenerator:
    def __init__(self, datagen: DataGenerator, load_type: LoadType, rate: float, duration: float) -> None:
        self.datagen = datagen
        self.duration = duration
        self.timer: LoadTimer
        if load_type == LoadType.CONSTANT:
            self.timer = ConstantLoadTimer(rate=rate)
        elif load_type == LoadType.POISSON:
            self.timer = PoissonLoadTimer(rate=rate)
        else:
            raise

    async def run(self, client: ModelServerClient) -> None:
        start_time = time.time()
        end_time = start_time + self.duration
        print("Run started")
        async with TaskGroup() as tg:
            for _, (data, time_index) in enumerate(
                zip(self.datagen.get_data(), self.timer.start_timer(start_time), strict=True)
            ):
                now = time.time()
                if time_index < end_time and now < end_time:
                    if time_index > now:
                        await sleep(time_index - time.time())
                    tg.create_task(client.process_request(data))
                    continue
                else:
                    break
        print("Run completed")
