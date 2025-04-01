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
from typing import Optional, List
from argparse import ArgumentParser
from enum import Enum
import yaml


class APIType(Enum):
    Completion = "completion"
    Chat = "chat"


class DataGenType(Enum):
    Mock = "mock"
    ShareGPT = "shareGPT"


class DataConfig(BaseModel):
    type: DataGenType = DataGenType.Mock


class LoadType(Enum):
    CONSTANT = "constant"
    POISSON = "poisson"


class LoadStage(BaseModel):
    rate: int = 1
    duration: int = 1


class LoadConfig(BaseModel):
    type: LoadType = LoadType.CONSTANT
    interval: Optional[int] = 1
    stages: List[LoadStage]


class ReportConfig(BaseModel):
    name: str


class MetricsConfig(BaseModel):
    url: str


class VLLMConfig(BaseModel):
    model_name: str
    api: APIType = APIType.Completion
    url: str


class CustomTokenizerConfig(BaseModel):
    pretrained_model_name_or_path: str
    trust_remote_code: Optional[bool] = None
    token: Optional[str] = None


class Config(BaseModel):
    data: Optional[DataConfig] = DataConfig()
    load: Optional[LoadConfig] = LoadConfig(stages=[LoadStage()])
    report: Optional[ReportConfig] = ReportConfig(name="")
    metrics: Optional[MetricsConfig] = MetricsConfig(url="")
    vllm: Optional[VLLMConfig] = None
    tokenizer: Optional[CustomTokenizerConfig] = None


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
