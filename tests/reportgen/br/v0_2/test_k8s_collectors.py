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
"""Unit tests for the K8s sub-collectors. The kubernetes client is replaced
with mocks so these tests run without a cluster."""

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from inference_perf.reportgen.br.v0_2.collectors.k8s.component_inspector import (
    ComponentInspector,
)
from inference_perf.reportgen.br.v0_2.collectors.k8s.per_pod_prom_scraper import (
    PerPodPromScraper,
    PodScrapeTarget,
)
from inference_perf.reportgen.br.v0_2.collectors.k8s.replica_status_sampler import (
    ReplicaStatusSampler,
)


def _pod(
    name: str,
    image: str,
    *,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
    gpu_count: Optional[int] = None,
    node: Optional[str] = None,
    ip: str = "10.0.0.1",
    annotations: Optional[Dict[str, str]] = None,
) -> Any:
    env_objs = [SimpleNamespace(name=k, value=v, value_from=None) for k, v in (env or {}).items()]
    requests = {"nvidia.com/gpu": str(gpu_count)} if gpu_count else {}
    container = SimpleNamespace(
        image=image,
        command=[],
        args=args or [],
        env=env_objs,
        resources=SimpleNamespace(requests=requests, limits={}),
    )
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name=name,
            labels=labels or {},
            annotations=annotations or {},
        ),
        spec=SimpleNamespace(containers=[container], node_name=node),
        status=SimpleNamespace(
            pod_ip=ip,
            conditions=[],
        ),
    )


# ---------------------------------------------------------------------------
# ComponentInspector
# ---------------------------------------------------------------------------


def test_component_inspector_aggregates_replicas() -> None:
    pods = [
        _pod(
            "vllm-0",
            "ghcr.io/llm-d/llm-d-cuda:0.3.1",
            args=["--model", "Qwen/Qwen3-0.6B", "--tensor-parallel-size", "8"],
            env={"VLLM_LOGGING_LEVEL": "DEBUG"},
            labels={
                "app.kubernetes.io/instance": "vllm-svc-0",
                "llm-d.ai/role": "decode",
            },
            gpu_count=8,
            node="gpu-node-01",
        ),
        _pod(
            "vllm-1",
            "ghcr.io/llm-d/llm-d-cuda:0.3.1",
            args=["--model", "Qwen/Qwen3-0.6B", "--tensor-parallel-size", "8"],
            env={"VLLM_LOGGING_LEVEL": "DEBUG"},
            labels={
                "app.kubernetes.io/instance": "vllm-svc-0",
                "llm-d.ai/role": "decode",
            },
            gpu_count=8,
            node="gpu-node-02",
        ),
    ]
    node = SimpleNamespace(metadata=SimpleNamespace(labels={"nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3"}))

    class _CoreV1:
        def list_namespaced_pod(self, namespace: str, label_selector: str = "") -> Any:
            return SimpleNamespace(items=pods)

        def read_node(self, name: str) -> Any:
            return node

    with patch(
        "inference_perf.reportgen.br.v0_2.collectors.k8s.component_inspector.load_clients",
        return_value=(_CoreV1(), None),
    ):
        result = ComponentInspector(namespace="ns").inspect()

    assert len(result) == 1
    component = result[0]
    assert component["metadata"]["label"] == "vllm-svc-0"
    assert component["standardized"]["replicas"] == 2
    assert component["standardized"]["role"] == "decode"
    assert component["standardized"]["model"] == {"name": "Qwen/Qwen3-0.6B"}
    assert component["standardized"]["tool"] == "llm-d-cuda"
    assert component["standardized"]["tool_version"] == "ghcr.io/llm-d/llm-d-cuda:0.3.1"
    assert component["standardized"]["accelerator"]["model"] == "NVIDIA-H100-80GB-HBM3"
    assert component["standardized"]["accelerator"]["count"] == 8
    assert component["standardized"]["accelerator"]["parallelism"]["tp"] == 8
    assert component["native"]["args"]["--model"] == "Qwen/Qwen3-0.6B"
    assert component["native"]["envars"]["VLLM_LOGGING_LEVEL"] == "DEBUG"


def test_component_inspector_handles_no_gpu() -> None:
    pod = _pod("epp-0", "ghcr.io/llm-d/scheduler:0.3.2", labels={"app.kubernetes.io/instance": "epp-0"})

    class _CoreV1:
        def list_namespaced_pod(self, namespace: str, label_selector: str = "") -> Any:
            return SimpleNamespace(items=[pod])

    with patch(
        "inference_perf.reportgen.br.v0_2.collectors.k8s.component_inspector.load_clients",
        return_value=(_CoreV1(), None),
    ):
        result = ComponentInspector(namespace="ns").inspect()

    assert len(result) == 1
    assert result[0]["standardized"]["accelerator"] is None
    # No model arg present, so model is None.
    assert result[0]["standardized"]["model"] is None


def test_component_inspector_no_namespace_yields_empty() -> None:
    inspector = ComponentInspector(namespace=None)
    assert inspector.inspect() == []


def test_component_inspector_parses_equals_args() -> None:
    pod = _pod(
        "vllm-0",
        "vllm:latest",
        args=["--model=Qwen/Qwen3-0.6B", "--tensor-parallel-size=4"],
        labels={"app.kubernetes.io/instance": "vllm"},
    )

    class _CoreV1:
        def list_namespaced_pod(self, namespace: str, label_selector: str = "") -> Any:
            return SimpleNamespace(items=[pod])

    with patch(
        "inference_perf.reportgen.br.v0_2.collectors.k8s.component_inspector.load_clients",
        return_value=(_CoreV1(), None),
    ):
        result = ComponentInspector(namespace="ns").inspect()

    assert result[0]["native"]["args"]["--model"] == "Qwen/Qwen3-0.6B"
    assert result[0]["native"]["args"]["--tensor-parallel-size"] == "4"


# ---------------------------------------------------------------------------
# ReplicaStatusSampler
# ---------------------------------------------------------------------------


def _deployment(name: str, *, desired: int, ready: int, available: int, role: str) -> Any:
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name=name,
            labels={"llm-d.ai/role": role, "llm-d.ai/model": "Qwen/Qwen3-0.6B"},
        ),
        spec=SimpleNamespace(replicas=desired),
        status=SimpleNamespace(
            available_replicas=available,
            ready_replicas=ready,
            updated_replicas=ready,
        ),
    )


@pytest.mark.asyncio
async def test_replica_status_sampler_collects_snapshots() -> None:
    deployments = [_deployment("decode", desired=3, ready=3, available=3, role="decode")]

    class _AppsV1:
        def list_namespaced_deployment(self, namespace: str, label_selector: str = "") -> Any:
            return SimpleNamespace(items=deployments)

        def list_namespaced_stateful_set(self, namespace: str, label_selector: str = "") -> Any:
            return SimpleNamespace(items=[])

    sampler = ReplicaStatusSampler(namespace="ns", interval_seconds=1)

    with patch(
        "inference_perf.reportgen.br.v0_2.collectors.k8s.replica_status_sampler.load_clients",
        return_value=(None, _AppsV1()),
    ):
        await sampler.start()
        await asyncio.sleep(0.05)  # let the loop tick once
        await sampler.stop()

    snapshots = sampler.snapshots()
    assert len(snapshots) >= 1
    controllers = snapshots[-1]["controllers"]
    assert controllers[0]["kind"] == "Deployment"
    assert controllers[0]["desired_replicas"] == 3
    assert controllers[0]["ready_replicas"] == 3
    assert controllers[0]["role"] == "decode"


# ---------------------------------------------------------------------------
# PerPodPromScraper
# ---------------------------------------------------------------------------


_VLLM_METRICS_TEXT = """\
# HELP vllm:gpu_cache_usage_perc gpu cache utilization percentage
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="Qwen/Qwen3-0.6B"} 42.5
# HELP vllm:num_requests_running running requests
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="Qwen/Qwen3-0.6B"} 24
# HELP vllm:num_requests_waiting waiting requests
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="Qwen/Qwen3-0.6B"} 3
"""


def test_per_pod_prom_scraper_parses_vllm_metrics() -> None:
    scraper = PerPodPromScraper()
    target = PodScrapeTarget(pod_name="vllm-0", component_label="vllm-svc-0", role="decode", url="http://x")
    scraper.set_targets([target])
    # Feed two ticks of identical metrics; statistics should reflect both.
    scraper._parse_into_samples(target, _VLLM_METRICS_TEXT)
    scraper._parse_into_samples(target, _VLLM_METRICS_TEXT)

    result = scraper.collect()
    kv = result["vllm_kv_cache_usage_perc"]
    assert len(kv["components"]) == 1
    assert kv["components"][0]["pod"] == "vllm-0"
    assert kv["components"][0]["statistics"]["mean"] == pytest.approx(42.5)
    assert kv["components"][0]["statistics"]["units"] == "percent"
    assert kv["aggregated"]["mean"] == pytest.approx(42.5)

    running = result["vllm_num_requests_running"]
    assert running["components"][0]["statistics"]["mean"] == pytest.approx(24.0)
    assert running["components"][0]["statistics"]["units"] == "count"
