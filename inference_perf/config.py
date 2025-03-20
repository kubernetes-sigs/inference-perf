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
from pydantic import BaseModel
from typing import Optional
from argparse import ArgumentParser
from enum import Enum
import yaml


class DataConfig(BaseModel):
    type: str


class LoadType(Enum):
    CONSTANT = "constant"
    POISSON = "poisson"


class LoadConfig(BaseModel):
    type: LoadType = LoadType.CONSTANT
    rate: int = 1
    duration: int = 1


class ReportConfig(BaseModel):
    name: str


class MetricsConfig(BaseModel):
    url: str


class VLLMConfig(BaseModel):
    model_name: str
    url: str


class Config(BaseModel):
    data: Optional[DataConfig] = DataConfig(type="")
    load: Optional[LoadConfig] = LoadConfig()
    report: Optional[ReportConfig] = ReportConfig(name="")
    metrics: Optional[MetricsConfig] = MetricsConfig(url="")
    vllm: Optional[VLLMConfig] = None


def read_config() -> Config:
    parser = ArgumentParser()

    parser.add_argument("-c", "--config_file", help="Config File", required=True)

    args = parser.parse_args()

    if args.config_file:
        print("Using configuration from: % s" % args.config_file)
        with open(args.config_file, "r") as stream:
            cfg = yaml.safe_load(stream)

        return Config(**cfg)

    return Config()
