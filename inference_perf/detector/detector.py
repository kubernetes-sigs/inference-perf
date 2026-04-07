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

import logging
import os
from typing import Any, Optional

import requests
from inference_perf.config import (
    APIType,
    Config,
    MetricsClientConfig,
    MetricsClientType,
    ModelServerType,
    PrometheusClientConfig,
    deep_merge,
)

from enum import Enum, auto


class EnvironmentType(Enum):
    GKE = auto()
    GCP = auto()
    EKS = auto()
    AWS = auto()
    K8S = auto()
    LOCAL = auto()


logger = logging.getLogger(__name__)


def autofill_config(url: str, base_config: Optional[Config] = None) -> Config:
    """Generate or patch a Config object for a given URL by auto-detecting the server type and environment."""
    detected_type = detect_server_type(url)
    if not detected_type:
        logger.warning("Skipping autoconfiguration for %s because server type could not be detected.", url)
        return base_config or Config()

    detected_api_type = detect_api_type(url)
    detected_model_name = detect_model_name(url)
    env = detect_environment()
    metrics_cfg = autofill_metrics_config(env)

    if base_config is None:
        base_config = Config()

    base_dict = base_config.model_dump(mode="json")

    overrides: dict[str, Any] = {
        "server": {
            "type": detected_type.value,
            "base_url": url,
        },
        "api": {
            "type": detected_api_type.value,
        },
    }
    if detected_model_name:
        overrides["server"]["model_name"] = detected_model_name

    if metrics_cfg:
        overrides["metrics"] = metrics_cfg.model_dump(mode="json")

    # Apply default load stages if none are defined
    if "load" not in base_dict or not base_dict["load"].get("stages"):
        overrides["load"] = {"stages": [{"rate": 1.0, "duration": 10}]}

    merged = deep_merge(base_dict, overrides)
    return Config(**merged)


def detect_server_type(url: str) -> Optional[ModelServerType]:
    """Detect the type of model server by probing endpoints.

    Heuristics:
    1. Check /metrics for server-specific prefixes (vllm:, sglang:, tgi_).
    2. Check /v1/models to see if it's OpenAI compatible (defaults to vLLM).
    3. Check /generate (TGI specific).
    """
    # 1. Try /metrics
    try:
        response = requests.get(f"{url}/metrics", timeout=5)
        if response.status_code == 200:
            content = response.text
            if "vllm:" in content:
                logger.info("Auto-detected vLLM server via /metrics")
                return ModelServerType.VLLM
            if "sglang:" in content:
                logger.info("Auto-detected SGLang server via /metrics")
                return ModelServerType.SGLANG
            if "tgi_" in content:
                logger.info("Auto-detected TGI server via /metrics")
                return ModelServerType.TGI
    except Exception as e:
        logger.debug("/metrics probe failed: %s", e)

    # 2. Try /v1/models (OpenAI compatibility)
    try:
        response = requests.get(f"{url}/v1/models", timeout=5)
        if response.status_code == 200:
            logger.info("Auto-detected OpenAI compatible server via /v1/models (defaulting to vLLM)")
            return ModelServerType.VLLM
    except Exception as e:
        logger.debug("/v1/models probe failed: %s", e)

    # 3. Try /generate (TGI specific)
    try:
        response = requests.get(f"{url}/generate", timeout=5)
        # TGI /generate expects POST, so GET might return 405 Method Not Allowed or 400 Bad Request
        if response.status_code in [405, 400]:
            logger.info("Auto-detected TGI server via /generate endpoint")
            return ModelServerType.TGI
    except Exception as e:
        logger.debug("/generate probe failed: %s", e)

    logger.warning("Could not auto-detect model server type for %s", url)
    return None


def detect_api_type(url: str) -> APIType:
    """Detect the supported API type by inspecting /openapi.json.

    Prefers Chat if both are supported.
    """
    try:
        response = requests.get(f"{url}/openapi.json", timeout=5)
        if response.status_code == 200:
            openapi = response.json()
            paths = openapi.get("paths", {})
            if "/v1/chat/completions" in paths:
                logger.info("Auto-detected Chat API support via /openapi.json")
                return APIType.Chat
            if "/v1/completions" in paths:
                logger.info("Auto-detected Completion API support via /openapi.json")
                return APIType.Completion
    except Exception as e:
        logger.debug("/openapi.json probe failed: %s", e)

    logger.warning("Could not auto-detect API type, defaulting to Completion")
    return APIType.Completion


def detect_model_name(url: str) -> Optional[str]:
    """Detect the model name by inspecting /v1/models."""
    try:
        response = requests.get(f"{url}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            if models:
                if len(models) > 1:
                    logger.warning(
                        "Multiple models found at /v1/models. Auto-selecting the first one: %s", models[0].get("id")
                    )
                model_name = models[0].get("id")
                if isinstance(model_name, str):
                    logger.info("Auto-detected model name: %s", model_name)
                    return model_name
    except Exception as e:
        logger.debug("/v1/models probe for model name failed: %s", e)
    return None


def detect_environment() -> EnvironmentType:
    """Detect the environment (GKE, EKS, Local) using metadata servers and env vars."""
    # 1. Check GKE (GCP metadata server + K8s env vars)
    try:
        response = requests.get("http://metadata.google.internal", headers={"Metadata-Flavor": "Google"}, timeout=1)
        if response.status_code == 200:
            if os.environ.get("KUBERNETES_SERVICE_HOST"):
                logger.info("Auto-detected GKE environment")
                return EnvironmentType.GKE
            logger.info("Auto-detected GCP environment")
            return EnvironmentType.GCP
    except Exception:
        pass

    # 2. Check EKS (AWS metadata server + K8s env vars)
    try:
        response = requests.get("http://169.254.169.254/latest/meta-data/", timeout=1)
        if response.status_code == 200:
            if os.environ.get("KUBERNETES_SERVICE_HOST"):
                logger.info("Auto-detected EKS environment")
                return EnvironmentType.EKS
            logger.info("Auto-detected AWS environment")
            return EnvironmentType.AWS
    except Exception:
        pass

    # 3. Fallback
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        logger.info("Auto-detected Generic K8s environment")
        return EnvironmentType.K8S

    logger.info("Defaulting to Local environment")
    return EnvironmentType.LOCAL


def autofill_metrics_config(detected_env: EnvironmentType) -> Optional[MetricsClientConfig]:
    """Generate MetricsClientConfig based on the detected environment."""
    if detected_env in [EnvironmentType.GKE, EnvironmentType.GCP]:
        return MetricsClientConfig(
            type=MetricsClientType.PROMETHEUS,
            prometheus=PrometheusClientConfig(google_managed=True),
        )
    elif detected_env in [EnvironmentType.EKS, EnvironmentType.AWS, EnvironmentType.K8S, EnvironmentType.LOCAL]:
        # Try localhost:9090 as default
        from pydantic import HttpUrl

        return MetricsClientConfig(
            type=MetricsClientType.PROMETHEUS,
            prometheus=PrometheusClientConfig(
                url=HttpUrl("http://localhost:9090"),
            ),
        )
    return None
