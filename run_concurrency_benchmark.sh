#!/usr/bin/env bash
# ==============================================================================
# SGLang GCS Rapid Bucket Multi-Session Concurrency Benchmark Runner
#
# Usage:
#   ./run_concurrency_benchmark.sh [16k|32k|64k|128k]
# ==============================================================================

set -eo pipefail

cd "$(dirname "$0")"

PREFIX_ARG="${1:-16k}"
case "$PREFIX_ARG" in
    16k|16K)
        PREFIX_LEN=16384
        DRAIN_WAIT=5.0
        ;;
    32k|32K)
        PREFIX_LEN=32768
        DRAIN_WAIT=8.0
        ;;
    64k|64K)
        PREFIX_LEN=65536
        DRAIN_WAIT=12.0
        ;;
    128k|128K)
        PREFIX_LEN=131072
        DRAIN_WAIT=15.0
        ;;
    *)
        PREFIX_LEN="$PREFIX_ARG"
        DRAIN_WAIT=8.0
        ;;
esac

echo "============================================================================"
echo "    Running Concurrency Benchmark: Prefix = ${PREFIX_LEN} tokens"
echo "    Concurrency levels: 1, 4, 8, 16, 32"
echo "============================================================================"

source .venv/bin/activate
export PYTHONPATH=$PWD/python:$PYTHONPATH

python3 benchmark/hicache/benchmark_concurrency.py \
    --base-url http://127.0.0.1:30000 \
    --prefix-len "$PREFIX_LEN" \
    --concurrency-levels 1 4 8 16 32 \
    --output-len 32 \
    --wait-offload-time "$DRAIN_WAIT"

