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

# Run both live /generate cases on a Linux amd64 host with AMX and Docker:
#   bash e2e/sglang_cpu_server.sh
# Override SGLANG_PORT, SGLANG_HF_CACHE or SGLANG_LOG_DIR if needed.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
image=docker.io/lmsysorg/sglang:v0.5.23-xeon
export E2E_SGLANG_MODEL=Qwen/Qwen2.5-0.5B-Instruct
port="${SGLANG_PORT:-30000}"
export E2E_SGLANG_BASE_URL="http://127.0.0.1:$port"
hf_cache="${SGLANG_HF_CACHE:-$HOME/.cache/huggingface}"
log_dir="${SGLANG_LOG_DIR:-$(mktemp -d)}"
mkdir -p "$hf_cache" "$log_dir"
echo "SGLang logs: $log_dir"
docker pull "$image"

cleanup() {
  if [[ -n "$container" ]]; then
    docker logs "$container" > "$log_dir/sglang-${mode}.log" 2>&1 || true
    docker rm -f "$container" >/dev/null 2>&1 || true
    container=""
  fi
}

wait_for_health() {
  local deadline=$((SECONDS + 300))
  while (( SECONDS < deadline )); do
    [[ $(docker inspect --format '{{.State.Running}}' "$container") == true ]] || return 1
    if curl --fail --silent --max-time 2 "$E2E_SGLANG_BASE_URL/health" >/dev/null; then
      return 0
    fi
    sleep 2
  done
  return 1
}

failed=0
container=""
trap cleanup EXIT
for mode in cumulative incremental; do
  args=()
  if [[ "$mode" == incremental ]]; then
    args+=(--incremental-streaming-output)
  fi

  echo "Starting SGLang ($mode)"
  if ! container=$(docker run --detach --shm-size 1g \
    --publish "127.0.0.1:$port:30000" \
    --volume "$hf_cache:/hf-cache" --env HF_HOME=/hf-cache \
    "$image" python3 -m sglang.launch_server --device cpu --disable-overlap-schedule \
    --host 0.0.0.0 --port 30000 --model-path "$E2E_SGLANG_MODEL" \
    --context-length 512 --max-total-tokens 1024 --stream-interval 1 "${args[@]}"); then
    echo "Could not start SGLang ($mode)" >&2
    failed=1
  elif ! wait_for_health; then
    echo "SGLang ($mode) did not become ready; see its server log" >&2
    failed=1
  elif ! E2E_SGLANG_STREAM_MODE="$mode" pdm run pytest -rs \
    e2e/tests/test_sglang_generate_live.py 2>&1 | tee "$log_dir/sglang-${mode}-pytest.log"; then
    echo "SGLang ($mode) live test failed" >&2
    failed=1
  fi
  cleanup
done
exit "$failed"
