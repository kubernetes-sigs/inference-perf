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
"""Does the run the docs describe need a config that spells nothing out?

``docs/config.md`` says a server that serves one model needs neither
``server.model_name`` nor ``tokenizer.pretrained_model_name_or_path``: the
client reads both off ``/v1/models``. Every other e2e config pins both, and the
unit test for the report tokenizer calls ``sync_tokenizer_config`` directly, so
deleting that call from ``main.py`` passes everything and auto-detected runs go
back to reporting without a tokenizer.

Each case here runs the CLI with both fields unset against the absorber and
reads the resolved tokenizer path back out of the report bundle's
``config.yaml``, which is the config the run ended with rather than the one it
started from.
"""

from typing import Any, Dict

from utils.absorber import AbsorberServer
from utils.benchmark import run_benchmark_minimal
from utils.net import get_free_port
from utils.testdata import extract_tarball

# The tokenizer files the run has to end up pointing at, and the served alias
# that hides them, the way vLLM answers behind --served-model-name.
MODEL_PATH = str(extract_tarball("e2e/testdata/models/google_gemma-3-270m.tar.gz"))
ALIAS = "served-alias"


# The smallest run that reaches report generation: 4 mock prompts at 2 per
# second, one worker, neither a model name nor a tokenizer path. Only base_url
# is filled in, since the port belongs to the test.
def _config(base_url: str) -> Dict[str, Any]:
    return {
        "api": {"type": "completion"},
        "data": {"type": "mock"},
        "load": {"type": "constant", "stages": [{"rate": 2, "duration": 2}], "num_workers": 1},
        "server": {"type": "vllm", "base_url": base_url},
        "report": {"request_lifecycle": {"summary": True, "per_stage": False, "per_request": False}},
    }


# Runs the CLI against base_url and returns the report bundle's config.yaml,
# failing on a run error or a bundle without one.
async def _bundle_config(base_url: str) -> Dict[str, Any]:
    result = await run_benchmark_minimal(_config(base_url), timeout_sec=None)
    assert result.success, f"run failed (rc={result.return_code}):\n{result.stdout[-4000:]}"
    assert result.reports, "no report bundle was written"
    assert "config.yaml" in result.reports, f"bundle has no config.yaml: {sorted(result.reports)}"
    bundle = result.reports["config.yaml"]
    assert isinstance(bundle, dict), f"config.yaml is not a mapping: {type(bundle)}"
    return bundle


# The tokenizer path the bundle records, which is null or absent in a run that
# let the client resolve one but never handed it to report generation. An
# absorber serving id "served-alias" with root "/models/gemma" gives "/models/gemma".
def _tokenizer_path(bundle: Dict[str, Any]) -> str:
    tokenizer = bundle.get("tokenizer") or {}
    path = tokenizer.get("pretrained_model_name_or_path")
    assert isinstance(path, str) and path, f"bundle config.yaml records no tokenizer path: tokenizer={tokenizer!r}"
    return path


# The server serves the extracted model directory as its only id, so the
# client's first candidate already loads: the bundle records that directory as
# the tokenizer path and no root fallback is involved.
async def test_autodetects_a_served_model_path() -> None:
    async with AbsorberServer(port=get_free_port(), model=MODEL_PATH) as absorber:
        bundle = await _bundle_config(absorber.base_url)

    assert _tokenizer_path(bundle) == MODEL_PATH


# The server serves an alias whose root is the extracted model directory, as
# vLLM does behind --served-model-name: the alias cannot load a tokenizer, so
# the bundle records the root path, not the alias.
async def test_autodetects_through_the_root_fallback_behind_an_alias() -> None:
    async with AbsorberServer(port=get_free_port(), model=ALIAS, root=MODEL_PATH) as absorber:
        bundle = await _bundle_config(absorber.base_url)

    assert _tokenizer_path(bundle) == MODEL_PATH
