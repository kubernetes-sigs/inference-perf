#!/bin/bash

# Exit on error
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <test_case_directory>"
  echo "Example: $0 tmp/images"
  exit 1
fi

DIR=$1

if [ ! -d "$DIR" ]; then
  echo "Error: Directory $DIR not found"
  exit 1
fi

if [ ! -f "$DIR/config.yml" ]; then
  echo "Error: $DIR/config.yml not found"
  exit 1
fi

NAMESPACE="test-ns"
CONFIG_MAP_NAME="inference-perf-config"
JOB_NAME="inference-perf"
VLLM_DEPLOYMENT="qwen3-32b-vllm-deployment"

# The image you built and pushed
IMAGE="<some-registry>/inference-perf:slabe-multimodal"

echo "=== Step 1: Ensure Model Server is Running ==="
if [ -f "$DIR/vllm.yaml" ]; then
  echo "Applying $DIR/vllm.yaml"
  kubectl apply -f $DIR/vllm.yaml -n $NAMESPACE
fi
kubectl rollout status deployment/$VLLM_DEPLOYMENT -n $NAMESPACE

echo "=== Step 2: Update ConfigMap ==="
echo "Updating ConfigMap from $DIR/config.yml"
kubectl create configmap $CONFIG_MAP_NAME --from-file=$DIR/config.yml -n $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

echo "=== Step 3: Run inference-perf ==="
echo "Cleaning up old job if exists..."
kubectl delete job $JOB_NAME -n $NAMESPACE --ignore-not-found

echo "Deploying inference-perf job from $DIR/manifest.yaml..."
kubectl apply -f $DIR/manifest.yaml

echo "=== Step 4: Wait and Extract Results ==="
echo "Waiting for job to complete..."
kubectl wait --for=condition=complete job/$JOB_NAME -n $NAMESPACE --timeout=300s

echo "Extracting reports..."
LOG_FILE="tmp/job_logs.txt"
kubectl logs job/$JOB_NAME -n $NAMESPACE > $LOG_FILE

# Extract content between markers
mkdir -p $DIR/results
awk '/=== START_SUMMARY ===/{flag=1;next}/=== END_SUMMARY ===/{flag=0}flag' $LOG_FILE > $DIR/results/summary_lifecycle_metrics.json
awk '/=== START_STAGE_0 ===/{flag=1;next}/=== END_STAGE_0 ===/{flag=0}flag' $LOG_FILE > $DIR/results/stage_0_lifecycle_metrics.json

echo "Reports saved to $DIR/results/"
rm $LOG_FILE
