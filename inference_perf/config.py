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
from datetime import datetime
import numpy as np
from pydantic import BaseModel
from typing import Any, Optional, List
from argparse import ArgumentParser
from enum import Enum
import yaml



class RequestMetric(BaseModel):
    stage_id: int
    prompt_len: int
    prompt: str
    output_len: int
    output: str
    start_time: float
    end_time: float


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
    rate: int
    duration: int


class LoadConfig(BaseModel):
    type: LoadType = LoadType.CONSTANT
    interval: Optional[float] = 1.0
    stages: List[LoadStage] = []


class StorageConfig(BaseModel):
    path: str = f"reports-{datetime.now().strftime("%Y%m%d-%H%M%S")}"
    report_file_prefix: Optional[str] = None


class GoogleCloudStorageConfig(StorageConfig):
    bucket_name: str


class StorageConfig(BaseModel):
    google_cloud_storage: Optional[GoogleCloudStorageConfig] = None

class ObservedMetricsReportSummaryConfig(BaseModel):
    def get_summarization(self, items: List[float]) -> dict[str, Any]:
        return {
            "mean": float(np.mean(items)),
            "min": float(np.min(items)),
            "p10": float(np.percentile(items, 10)),
            "p50": float(np.percentile(items, 50)),
            "p90": float(np.percentile(items, 90)),
            "max": float(np.max(items)),
        }

    def get_report(self, request_metrics: List[RequestMetric]) -> dict[str, Any]:
        return {
            "total_requests": len(request_metrics),
            "prompt_length": self.get_summarization([x.prompt_len for x in request_metrics]),
            "output_length": self.get_summarization([x.output_len for x in request_metrics]),
            "time_per_request": self.get_summarization([(x.end_time - x.start_time) for x in request_metrics]),
            "per_token_latency": self.get_summarization([(x.end_time - x.start_time) / (x.output_len) if x.output_len != 0 else 0 for x in request_metrics])
        }

class ObservedMetricsReportPerRequestConfig(BaseModel):
    include_inputs: bool = False        # replace input_len with input request body
    include_outputs: bool = False       # replace output_len with output request body

    def get_report(self, request_metrics: List[RequestMetric]) -> Any:
        for metric in request_metrics:
            if not self.include_inputs:
                delattr(metric, 'prompt')
            if not self.include_outputs:
                delattr(metric, 'output')
        return [metric.model_dump() for metric in request_metrics]


class ObservedMetricsReportConfig(BaseModel):
    """What data is presented from what this tool observes about the requests?"""
    summary: ObservedMetricsReportSummaryConfig = ObservedMetricsReportSummaryConfig()
    per_request: Optional[ObservedMetricsReportPerRequestConfig] = None

    def get_report(self, request_metrics: List[RequestMetric]) -> dict[str, Any] | None:
        if len(request_metrics) == 0:
            return None

        report = {}
        summary_report = self.summary.get_report(request_metrics) if self.summary else None
        if summary_report is not None:
            report["summary"] = summary_report
        if self.per_request:
            per_request_report = self.per_request.get_report(request_metrics)
            if per_request_report is not None:
                report["per_request"] = per_request_report
        return report if report else None


class PrometheusMetricsReportConfig(BaseModel):
    """What data should be presented from prometheus metrics?"""
    def get_report(self) -> dict[str, Any] | None:
        return None
    
class ReportConfig(BaseModel):
    observed: ObservedMetricsReportConfig = ObservedMetricsReportConfig()
    prometheus: Optional[PrometheusMetricsReportConfig] = None

    def get_report(self, request_metrics: List[RequestMetric]) -> dict[str, Any] | None:
        report = {}
        if self.observed:
            observed_report = self.observed.get_report(request_metrics)
            if observed_report is not None:
                report["observed"] = observed_report
        if self.prometheus:
            prometheus_report = self.prometheus.get_report()
            if prometheus_report is not None:
                report["prometheus"] = prometheus_report
        return report if report else None

class MetricsConfig(BaseModel):
    pass


class VLLMConfig(BaseModel):
    model_name: str
    api: APIType = APIType.Completion
    url: str


class CustomTokenizerConfig(BaseModel):
    pretrained_model_name_or_path: str
    trust_remote_code: Optional[bool] = None
    token: Optional[str] = None


class Config(BaseModel):
    data: DataConfig = DataConfig()
    load: LoadConfig = LoadConfig()
    report: ReportConfig = ReportConfig()
    metrics: MetricsConfig = MetricsConfig()
    storage: Optional[StorageConfig] = StorageConfig()
    vllm: Optional[VLLMConfig] = None
    tokenizer: Optional[CustomTokenizerConfig] = None


def deep_merge(base: dict, override: dict) -> dict:
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

        default_cfg = Config().model_dump()
        merged_cfg = deep_merge(default_cfg, cfg)
        return Config(**merged_cfg)
    return Config()