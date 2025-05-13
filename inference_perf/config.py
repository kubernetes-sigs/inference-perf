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
from abc import ABC
from datetime import datetime

from pydantic import BaseModel, HttpUrl
from typing import Any, Optional, List
from argparse import ArgumentParser
from enum import Enum
import yaml


class APIType(Enum):
    Completion = "completion"
    Chat = "chat"


class DataGenType(Enum):
    Mock = "mock"
    ShareGPT = "shareGPT"
    Synthetic = "synthetic"


# Represents the distribution for input prompts and output generations.
class Distribution(BaseModel):
    min: int = 10
    max: int = 1024
    mean: float = 512
    std_dev: float = 200
    total_count: int = 1000


class DataConfig(BaseModel):
    type: DataGenType = DataGenType.Mock
    # Distributions are only supported for synthetic dataset at this moment
    input_distribution: Optional[Distribution] = Distribution()
    output_distribution: Optional[Distribution] = Distribution()


class LoadType(Enum):
    CONSTANT = "constant"
    POISSON = "poisson"


class LoadStage(BaseModel):
    rate: int
    duration: int


class LoadConfig(BaseModel):
    type: LoadType = LoadType.CONSTANT
    interval: float = 1.0
    stages: List[LoadStage] = []


class StorageConfigBase(BaseModel):
    path: str = f"reports-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    report_file_prefix: Optional[str] = None


class GoogleCloudStorageConfig(StorageConfigBase):
    bucket_name: str


class StorageConfig(BaseModel):
    google_cloud_storage: Optional[GoogleCloudStorageConfig] = None


class PromptMetricsReportConfig(BaseModel):
    summary: Optional[bool] = True
    per_request: Optional[bool] = False


class PrometheusMetricsReportConfig(BaseModel):
    pass


class ReportConfig(BaseModel):
    prompt: PromptMetricsReportConfig = PromptMetricsReportConfig()
    prometheus: Optional[PrometheusMetricsReportConfig] = None



class PrometheusCollectorConfig(ABC, BaseModel):
    scrape_interval: int = 15


class SelfHostedPrometheusCollectorConfig(PrometheusCollectorConfig):
    url: HttpUrl


class GMPCollectorConfig(PrometheusCollectorConfig):
    project_id: str


class MetricsClientConfig(BaseModel):
    self_hosted_prometheus: Optional[SelfHostedPrometheusCollectorConfig] = None
    google_managed_prometheus: Optional[GMPCollectorConfig] = None


class CustomTokenizerConfig(BaseModel):
    pretrained_model_name_or_path: str
    trust_remote_code: Optional[bool] = None
    token: Optional[str] = None


class VLLMConfig(BaseModel):
    model_name: str
    api: APIType
    url: str
    tokenizer: CustomTokenizerConfig


class Config(BaseModel):
    data: DataConfig = DataConfig()
    load: LoadConfig = LoadConfig()
    report: Optional[ReportConfig] = ReportConfig()
    metrics_client: MetricsClientConfig = MetricsClientConfig()
    storage: Optional[StorageConfig] = StorageConfig()
    vllm: Optional[VLLMConfig] = None


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def read_config() -> Config:
    parser = ArgumentParser()

    parser.add_argument("-c", "--config_file", help="Config File", required=True)

    args = parser.parse_args()
    if args.config_file:
        print("Using configuration from: %s" % args.config_file)
        with open(args.config_file, "r") as stream:
            cfg = yaml.safe_load(stream)

        default_cfg = Config().model_dump(mode="json")
        merged_cfg = deep_merge(default_cfg, cfg)

        print(
            f"Benchmarking with the following config:\n\n{yaml.dump(merged_cfg, sort_keys=False, default_flow_style=False)}\n"
        )
        return Config(**merged_cfg)
    return Config()
