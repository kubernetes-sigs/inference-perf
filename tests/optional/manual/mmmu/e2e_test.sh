#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASES_DIR="$SCRIPT_DIR/cases"

NAMESPACE_PREFIX="inference-perf-mmmu-e2e"

IMAGE="quay.io/inference-perf/inference-perf:latest"
CLEANUP=false

usage() {
  cat <<EOF
Usage: $(basename "$0") [-n <namespace_prefix>] [-i <image>] [-c] [-h]

Runs every test case under $CASES_DIR sequentially, each in its own
namespace named "<prefix>-<case>". A failing case is reported but
does not stop the run; the script exits non-zero if any case failed.

  -n <prefix>   Namespace prefix (default: $NAMESPACE_PREFIX)
  -i <image>    Inference-perf docker image (default: $IMAGE)
  -c            Clean up namespaces after the test run completes (default: $CLEANUP)
  -h            Show this help
EOF
}

while getopts ":n:i:ch" opt; do
  case "$opt" in
    n) NAMESPACE_PREFIX="$OPTARG" ;;
    i) IMAGE="$OPTARG" ;;
    c) CLEANUP=true ;;
    h) usage; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 2 ;;
    :)  echo "Option -$OPTARG requires an argument" >&2; usage; exit 2 ;;
  esac
done

if [ ! -d "$CASES_DIR" ]; then
  echo "Error: cases directory not found at $CASES_DIR" >&2
  exit 1
fi

# ConfigMap parameters
CONFIG_MAP_NAME="inference-perf-config"
JOB_NAME="inference-perf"
VLLM_DEPLOYMENT="qwen3-32b-vllm-deployment"

# Subshell function so `set -e` aborts the case on first failure
# without killing the parent loop.
run_case() (
  set -e
  local CASE_DIR="$1"
  local NAMESPACE="$2"

  echo "=== Ensuring namespace $NAMESPACE ==="
  kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

  echo "=== Copying hf-secret from default namespace ==="
  if kubectl get secret hf-secret -n default >/dev/null 2>&1; then
    kubectl get secret hf-secret -n default -o yaml | grep -v -E "namespace:|resourceVersion:|uid:|creationTimestamp:" | kubectl apply -n "$NAMESPACE" -f -
  else
    echo "Warning: hf-secret not found in 'default' namespace! The model server might fail to start if the model is gated."
  fi

  echo "=== Step 1: Ensure Model Server is Running ==="
  if [ -f "$CASE_DIR/vllm.yaml" ]; then
    echo "Applying $CASE_DIR/vllm.yaml"
    kubectl apply -f "$CASE_DIR/vllm.yaml" -n "$NAMESPACE"
  fi
  kubectl rollout status deployment/"$VLLM_DEPLOYMENT" -n "$NAMESPACE"

  echo "=== Step 2: Update ConfigMap ==="
  echo "Updating ConfigMap from $CASE_DIR/config.yml"
  kubectl create configmap "$CONFIG_MAP_NAME" --from-file="$CASE_DIR/config.yml" -n "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

  echo "=== Step 3: Run inference-perf ==="
  echo "Cleaning up old job if exists..."
  kubectl delete job "$JOB_NAME" -n "$NAMESPACE" --ignore-not-found

  if [ -f "$CASE_DIR/manifest.yaml" ]; then
    echo "Deploying inference-perf job from $CASE_DIR/manifest.yaml..."
    kubectl apply -f "$CASE_DIR/manifest.yaml" -n "$NAMESPACE"
  else
    local TEMPLATE_MANIFEST="$SCRIPT_DIR/../../../../deploy/manifests.yaml"
    if [ ! -f "$TEMPLATE_MANIFEST" ]; then
      TEMPLATE_MANIFEST="$SCRIPT_DIR/../../../deploy/manifests.yaml"
    fi
    if [ ! -f "$TEMPLATE_MANIFEST" ]; then
      echo "Error: manifests.yaml template not found at $TEMPLATE_MANIFEST" >&2
      exit 1
    fi
    echo "Deploying inference-perf job from template using image $IMAGE..."
    sed "s|quay.io/inference-perf/inference-perf:latest|$IMAGE|g" "$TEMPLATE_MANIFEST" | kubectl apply -f - -n "$NAMESPACE"
  fi

  echo "=== Step 4: Wait and Extract Results ==="
  echo "Waiting for job to complete..."
  kubectl wait --for=condition=complete "job/$JOB_NAME" -n "$NAMESPACE" --timeout=300s

  local OUTPUT_DIR="$CASE_DIR/output"
  mkdir -p "$OUTPUT_DIR"
  local LOG_FILE="$OUTPUT_DIR/job_logs.txt"

  echo "Extracting reports..."
  kubectl logs "job/$JOB_NAME" -n "$NAMESPACE" > "$LOG_FILE"

  awk '/=== START_SUMMARY ===/{flag=1;next}/=== END_SUMMARY ===/{flag=0}flag' "$LOG_FILE" > "$OUTPUT_DIR/summary_lifecycle_metrics.json"
  awk '/=== START_STAGE_0 ===/{flag=1;next}/=== END_STAGE_0 ===/{flag=0}flag' "$LOG_FILE" > "$OUTPUT_DIR/stage_0_lifecycle_metrics.json"

  rm "$LOG_FILE"
  echo "Reports saved to $OUTPUT_DIR/"

  # Verify that all requests succeeded using jq
  local SUMMARY_FILE="$OUTPUT_DIR/summary_lifecycle_metrics.json"
  if [ -f "$SUMMARY_FILE" ]; then
    local FAIL_COUNT
    FAIL_COUNT=$(jq '.failures.count' "$SUMMARY_FILE")
    local SUCCESS_COUNT
    SUCCESS_COUNT=$(jq '.successes.count' "$SUMMARY_FILE")

    echo "=== Verification ==="
    echo "Successes: $SUCCESS_COUNT, Failures: $FAIL_COUNT"

    if [ -z "$SUCCESS_COUNT" ] || [ "$SUCCESS_COUNT" -eq 0 ] || [ "$SUCCESS_COUNT" = "null" ]; then
      echo "Error: Zero successful requests in case $CASE_NAME!" >&2
      return 1
    fi

    if [ -n "$FAIL_COUNT" ] && [ "$FAIL_COUNT" -gt 0 ] && [ "$FAIL_COUNT" != "null" ]; then
      echo "Error: Case $CASE_NAME had $FAIL_COUNT failed requests!" >&2
      return 1
    fi
    echo "Verification successful for case $CASE_NAME!"
  else
    echo "Error: summary_lifecycle_metrics.json not found!" >&2
    return 1
  fi
)

echo "=== Checking for leftover namespaces from previous runs ==="
LEFTOVER_NAMESPACES=$(kubectl get namespaces -o jsonpath="{.items[*].metadata.name}" | tr ' ' '\n' | grep "^${NAMESPACE_PREFIX}-" || true)

if [ -n "$LEFTOVER_NAMESPACES" ]; then
  echo "Found leftover namespaces:"
  echo "$LEFTOVER_NAMESPACES"
  echo "Deleting leftover namespaces (waiting for termination)..."
  kubectl delete namespace $LEFTOVER_NAMESPACES --ignore-not-found --wait=true
  echo "Cleanup complete!"
else
  echo "No leftover namespaces found."
fi

PASSED=()
FAILED=()
SKIPPED=()

for case_dir in "$CASES_DIR"/*/; do
  CASE_DIR="${case_dir%/}"
  CASE_NAME="$(basename "$CASE_DIR")"
  NAMESPACE_SUFFIX="${CASE_NAME//_/-}"
  NAMESPACE="${NAMESPACE_PREFIX}-${NAMESPACE_SUFFIX}"

  echo
  echo "########################################"
  echo "### Case: $CASE_NAME (namespace: $NAMESPACE)"
  echo "########################################"

  if [ ! -f "$CASE_DIR/config.yml" ]; then
    echo "Skipping $CASE_NAME: config.yml not found"
    SKIPPED+=("$CASE_NAME")
    continue
  fi

  if run_case "$CASE_DIR" "$NAMESPACE"; then
    PASSED+=("$CASE_NAME")
  else
    echo ">>> FAILED: $CASE_NAME"
    FAILED+=("$CASE_NAME")
  fi
done

echo
echo "########################################"
echo "### Summary"
echo "########################################"
echo "Passed:  ${#PASSED[@]} ${PASSED[*]}"
echo "Failed:  ${#FAILED[@]} ${FAILED[*]}"
echo "Skipped: ${#SKIPPED[@]} ${SKIPPED[*]}"

if [ "$CLEANUP" = true ]; then
  echo
  echo "=== Cleaning up namespaces ==="
  for case_dir in "$CASES_DIR"/*/; do
    CASE_DIR="${case_dir%/}"
    CASE_NAME="$(basename "$CASE_DIR")"
    NAMESPACE_SUFFIX="${CASE_NAME//_/-}"
    NAMESPACE="${NAMESPACE_PREFIX}-${NAMESPACE_SUFFIX}"
    if [ -f "$CASE_DIR/config.yml" ]; then
      echo "Deleting namespace $NAMESPACE..."
      kubectl delete namespace "$NAMESPACE" --ignore-not-found --wait=false
    fi
  done
fi

[ "${#FAILED[@]}" -eq 0 ]
