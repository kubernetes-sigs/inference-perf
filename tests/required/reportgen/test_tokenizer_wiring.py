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
"""The tokenizer a client resolves has to reach report generation (#809).

Report generation builds its own tokenizer from ``config.tokenizer`` instead of reusing
the client's. A run that lets the client resolve one -- a detected model name, or the
``/v1/models`` entry's ``root`` behind a served alias -- therefore has nothing in
``config.tokenizer`` unless main.py copies the resolved one back, and the report then
silently skips the streamed token-timing correction. These tests drive the copy and then
a real ``generate_reports`` call, so removing either half fails here.
"""

from typing import Any
from unittest.mock import Mock, patch

from inference_perf.client.modelserver import MockModelServerClient
from inference_perf.client.modelserver.metrics import BaseMetrics
from inference_perf.client.server_metrics.base import PerfRuntimeParameters, StageRuntimeInfo, StageStatus
from inference_perf.config import APIConfig, APIType, Config, CustomTokenizerConfig
from inference_perf.config.reportgen.config import ReportConfig
from inference_perf.main import sync_tokenizer_config
from inference_perf.reportgen.base import ReportGenerator

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


class RecordingTokenizer:
    """Stands in for CustomTokenizer and records the path report generation asked for."""

    paths: list[str] = []

    def __init__(self, config: Any = None) -> None:
        RecordingTokenizer.paths.append(config.pretrained_model_name_or_path)


# A client of the mock server type, which loads no tokenizer and touches no network.
def _client() -> MockModelServerClient:
    return MockModelServerClient(
        metrics_collector=Mock(),
        api_config=APIConfig(type=APIType.Completion, streaming=False),
    )


# The runtime parameters of a one-stage run that collected no request metrics.
def _runtime_parameters() -> PerfRuntimeParameters:
    return PerfRuntimeParameters(
        start_time=0.0,
        duration=1.0,
        model_server_metrics=BaseMetrics(),
        stages={0: StageRuntimeInfo(stage_id=0, rate=1.0, start_time=0.0, end_time=1.0, status=StageStatus.COMPLETED)},
    )


# The client resolved the model root while the config had no tokenizer section: the
# report must build its tokenizer from that path, which is what the timing correction
# needs. Reverting either the copy or the report's _build_tokenizer call fails here.
async def test_generate_reports_uses_the_tokenizer_the_client_resolved() -> None:
    RecordingTokenizer.paths.clear()
    config = Config()
    client = _client()
    client.tokenizer_config = CustomTokenizerConfig(pretrained_model_name_or_path=MODEL)

    sync_tokenizer_config(config, client)

    generator = ReportGenerator(metrics_client=None, metrics_collector=Mock(get_metrics=Mock(return_value=[])), config=config)
    with patch("inference_perf.utils.custom_tokenizer.CustomTokenizer", RecordingTokenizer):
        await generator.generate_reports(ReportConfig(), _runtime_parameters())

    assert RecordingTokenizer.paths == [MODEL]


# The mock server type resolves no tokenizer, so its runs must report without one:
# building one there would attempt a download for a server the run never contacted.
async def test_generate_reports_builds_no_tokenizer_when_the_client_resolved_none() -> None:
    RecordingTokenizer.paths.clear()
    config = Config()
    client = _client()

    sync_tokenizer_config(config, client)

    generator = ReportGenerator(metrics_client=None, metrics_collector=Mock(get_metrics=Mock(return_value=[])), config=config)
    with patch("inference_perf.utils.custom_tokenizer.CustomTokenizer", RecordingTokenizer):
        await generator.generate_reports(ReportConfig(), _runtime_parameters())

    assert RecordingTokenizer.paths == []


# A config that pins its own path keeps that object and its settings: the copy must not
# replace what the user wrote (token, trust_remote_code) with the client's default.
def test_sync_keeps_an_explicitly_configured_tokenizer_path() -> None:
    pinned = CustomTokenizerConfig(pretrained_model_name_or_path=MODEL, trust_remote_code=True)
    config = Config(tokenizer=pinned)
    client = _client()
    client.tokenizer_config = CustomTokenizerConfig(pretrained_model_name_or_path="served-alias")

    sync_tokenizer_config(config, client)

    assert config.tokenizer is pinned
    assert config.tokenizer.trust_remote_code is True
