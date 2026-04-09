#!/bin/sh
# Entrypoint wrapper: generates synthetic traces then runs inference-perf.
#
# Generates traces from the embedded default config (or SYNTHETIC_TRACE_CONFIG
# env var if set) before running the benchmark. Traces are generated to
# ./synthetic_traces/ which the bench config references.
#
# The generation is deterministic (seeded) and fast (~5-10s for 500 conversations).

set -e

TRACE_CONFIG="${SYNTHETIC_TRACE_CONFIG:-/workspace/tools/default_trace_config.yaml}"

if [ -f "$TRACE_CONFIG" ] || [ -n "$SYNTHETIC_TRACE_CONFIG" ]; then
    if [ -n "$SYNTHETIC_TRACE_CONFIG" ]; then
        echo "$SYNTHETIC_TRACE_CONFIG" > /tmp/trace_config.yaml
        TRACE_CONFIG="/tmp/trace_config.yaml"
    fi
    echo "Generating synthetic traces from $TRACE_CONFIG..."
    python /workspace/tools/generate_synthetic_traces.py --config "$TRACE_CONFIG"
fi

exec "$@"
