#!/usr/bin/env bash
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
#
# Runs the real vLLM CPU server the live-oracle e2e slice tests against.
# CI and local runs share this one invocation, so "passes locally" and
# "passes the merge gate" are claims about the same server.
#
# Usage:
#   e2e/vllm_cpu_server.sh start <release-tag>   # start, wait for /health
#   e2e/vllm_cpu_server.sh logs                  # container logs to stdout
#   e2e/vllm_cpu_server.sh stop                  # remove the container
#
# Release tags come from e2e/vllm_releases.txt. Local repro of the CI slice:
#
#   e2e/vllm_cpu_server.sh start v0.26.0
#   E2E_VLLM_BASE_URL=http://127.0.0.1:8000 E2E_VLLM_VERSION=v0.26.0 \
#     pdm run test:e2e:live
#   e2e/vllm_cpu_server.sh stop
#
# Environment overrides: VLLM_MODEL, VLLM_PORT, VLLM_HF_CACHE (host-side
# model cache), VLLM_CPU_KVCACHE_SPACE, VLLM_CONTAINER_ENGINE,
# VLLM_CONTAINER_NAME (default vllm-cpu; set it to run two servers side by
# side), VLLM_CHAT_TEMPLATE (host path; the default is the text-only template
# opt-125m needs; set it to the empty string to serve with the model's own
# template, which every multimodal model requires), VLLM_MAX_MODEL_LEN
# (default 2048), VLLM_EXTRA_ARGS (extra `vllm serve` flags, word-split),
# VLLM_HEALTH_TIMEOUT_SEC (how long to wait for /health, default 300),
# VLLM_CPU_OMP_THREADS_BIND (passed to vLLM, default auto).
#
# The multimodal slice (e2e/tests/test_vllm_cpu_multimodal.py) starts the
# same image with a vision-language or audio model. Those models profile one
# encoder item of the maximum feature size before they serve, so they need a
# health budget the text-only oracle does not:
#
#   VLLM_MODEL=OpenGVLab/InternVL3-1B-hf VLLM_CHAT_TEMPLATE= \
#     VLLM_MAX_MODEL_LEN=4096 VLLM_HEALTH_TIMEOUT_SEC=600 \
#     e2e/vllm_cpu_server.sh start latest
#
# On an x86 host that profiling pass runs on one core, because vLLM's `auto`
# thread binding takes one logical CPU per physical core and reserves one for
# the framework: measured 118s pinned that way against 44s across four cores.
# CI does not hit this (ARM binding uses every logical CPU), so it is a local
# repro knob: VLLM_CPU_OMP_THREADS_BIND=nobind drops the pinning entirely.

set -euo pipefail

# A docker CLI on PATH is not a working engine (podman hosts often carry a
# daemonless docker shim), so probe for one that actually responds.
detect_engine() {
  local e
  for e in docker podman; do
    if command -v "$e" > /dev/null 2>&1 && "$e" info > /dev/null 2>&1; then
      echo "$e"
      return
    fi
  done
  echo docker # let the run below surface the real connection error
}
ENGINE="${VLLM_CONTAINER_ENGINE:-$(detect_engine)}"
NAME="${VLLM_CONTAINER_NAME:-vllm-cpu}"
PORT="${VLLM_PORT:-8000}"
MODEL="${VLLM_MODEL:-facebook/opt-125m}"
HF_CACHE="${VLLM_HF_CACHE:-$HOME/.cache/huggingface}"
E2E_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
# Seconds to wait for /health. The text-only model serves in about two
# minutes on a CI runner; multimodal startup profiling needs longer, so the
# multimodal workflow raises this rather than every caller paying for the
# slowest model.
HEALTH_TIMEOUT_SEC="${VLLM_HEALTH_TIMEOUT_SEC:-300}"
OMP_THREADS_BIND="${VLLM_CPU_OMP_THREADS_BIND:-auto}"
# ${VAR-default} (no colon) so an explicitly empty VLLM_CHAT_TEMPLATE means
# "no --chat-template flag", not "the default".
CHAT_TEMPLATE="${VLLM_CHAT_TEMPLATE-$E2E_DIR/testdata/simple_chat_template.jinja}"

start() {
  local tag="$1"
  local image="docker.io/vllm/vllm-openai-cpu:$tag"
  mkdir -p "$HF_CACHE"

  local engine_opts=()
  if [ "${ENGINE##*/}" = podman ]; then
    # Rootless podman on SELinux hosts: without this the volume mounts get
    # permission denied, and relabeling the user's real HF cache with :z is
    # not ours to do. Docker paths stay byte-identical to what CI proved.
    engine_opts+=(--security-opt label=disable)
  fi

  # A chat template on the host is mounted into the container under a fixed
  # path; no template means the model's own is used.
  local template_opts=()
  local serve_template_opts=()
  if [ -n "$CHAT_TEMPLATE" ]; then
    template_opts+=(-v "$CHAT_TEMPLATE:/chat_template.jinja")
    serve_template_opts+=(--chat-template /chat_template.jinja)
  fi
  # shellcheck disable=SC2206  # word-splitting VLLM_EXTRA_ARGS is the contract
  local extra_args=(${VLLM_EXTRA_ARGS:-})

  # shm-size/seccomp/SYS_NICE are the invocation vLLM's CPU docs prescribe:
  # the engine's IPC lives in /dev/shm (the 64MB default kills the worker
  # during startup) and thread binding wants SYS_NICE. KV cache defaults
  # sized for a 16GB CI runner. Server-side greedy decoding
  # (temperature 0) is required for zero-tolerance token accounting: under
  # default sampling the model emits special tokens that detokenization
  # drops, so text-based counts undercount the server's own by construction.
  "$ENGINE" run -d --name "$NAME" \
    --shm-size=4g \
    --security-opt seccomp=unconfined \
    --cap-add SYS_NICE \
    "${engine_opts[@]+"${engine_opts[@]}"}" \
    -p "$PORT:8000" \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    "${template_opts[@]+"${template_opts[@]}"}" \
    -e VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE:-2}" \
    -e VLLM_CPU_OMP_THREADS_BIND="$OMP_THREADS_BIND" \
    "$image" \
    "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    "${serve_template_opts[@]+"${serve_template_opts[@]}"}" \
    --override-generation-config '{"temperature": 0}' \
    --enforce-eager \
    "${extra_args[@]+"${extra_args[@]}"}"

  for _ in $(seq 1 $((HEALTH_TIMEOUT_SEC / 5))); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
      echo "vLLM ($tag) ready: E2E_VLLM_BASE_URL=http://127.0.0.1:$PORT E2E_VLLM_VERSION=$tag"
      return 0
    fi
    if [ -z "$("$ENGINE" ps -q -f "name=$NAME")" ]; then
      break
    fi
    sleep 5
  done

  # The container is left in place so the caller can still run logs/stop.
  echo "vLLM CPU server ($tag) failed to become healthy within ${HEALTH_TIMEOUT_SEC}s" >&2
  "$ENGINE" logs "$NAME" 2>&1 | tail -100 >&2 || true
  return 1
}

case "${1:-}" in
  start) start "${2:?usage: $0 start <release-tag>}" ;;
  logs) exec "$ENGINE" logs "$NAME" ;;
  stop) exec "$ENGINE" rm -f "$NAME" ;;
  *)
    echo "usage: $0 {start <release-tag>|logs|stop}" >&2
    exit 2
    ;;
esac
