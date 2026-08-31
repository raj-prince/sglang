#!/usr/bin/env bash
# ==============================================================================
# SGLang HiCache 3-Request Benchmark Runner
#
# Usage:
#   ./run_benchmark.sh          # Runs 16K benchmark by default
#   ./run_benchmark.sh 32k      # Runs 32K prefix benchmark
#   ./run_benchmark.sh 64k      # Runs 64K prefix benchmark
#   ./run_benchmark.sh 128k     # Runs 128K prefix benchmark
#   ./run_benchmark.sh all      # Runs 16K, 32K, 64K, 128K sequentially
#   ./run_benchmark.sh 8192     # Runs custom prefix length
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 1. Environment & Python Setup
if [ -d "${SCRIPT_DIR}/.venv" ]; then
    source "${SCRIPT_DIR}/.venv/bin/activate"
fi

export PYTHONPATH="${SCRIPT_DIR}/python:${PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SERVER_URL="${SERVER_URL:-http://127.0.0.1:30000}"
MODE="${1:-16k}"

echo "============================================================================"
echo "      SGLang HiCache 3-Request Benchmark Runner"
echo "============================================================================"
echo "[*] Target Server : ${SERVER_URL}"

# 2. Check if server is reachable
if ! curl -s "${SERVER_URL}/v1/models" > /dev/null 2>&1; then
    echo "[!] Error: SGLang server is not responding at ${SERVER_URL}"
    echo "    Please start the server first via: ./run.sh"
    exit 1
fi

run_single_benchmark() {
    local prefix_len=$1
    local wait_time=$2
    local output_len=${3:-32}

    echo -e "\n----------------------------------------------------------------------------"
    echo "  Running Benchmark: Prefix = ${prefix_len} tokens (Offload Wait = ${wait_time}s)"
    echo "----------------------------------------------------------------------------"

    python3 benchmark/hicache/benchmark_3_requests.py \
        --base-url "${SERVER_URL}" \
        --prefix-len "${prefix_len}" \
        --output-len "${output_len}" \
        --wait-offload-time "${wait_time}"
}

case "${MODE,,}" in
    16k)
        run_single_benchmark 16384 3.0 32
        ;;
    32k)
        run_single_benchmark 32768 5.0 32
        ;;
    64k)
        run_single_benchmark 65536 10.0 32
        ;;
    128k)
        run_single_benchmark 131072 15.0 32
        ;;
    all)
        echo -e "\n[*] Starting Full Multi-Prefix Benchmark Suite (16K, 32K, 64K, 128K)..."
        run_single_benchmark 16384 3.0 32
        run_single_benchmark 32768 5.0 32
        run_single_benchmark 65536 10.0 32
        run_single_benchmark 131072 15.0 32
        echo -e "\n============================================================================"
        echo "  [✓] All multi-prefix benchmarks completed successfully!"
        echo "============================================================================"
        ;;
    *)
        if [[ "${MODE}" =~ ^[0-9]+$ ]]; then
            wait_time=5.0
            if [ "${MODE}" -gt 65536 ]; then
                wait_time=15.0
            elif [ "${MODE}" -gt 32768 ]; then
                wait_time=10.0
            fi
            run_single_benchmark "${MODE}" "${wait_time}" 32
        else
            echo "[!] Unknown mode: '${MODE}'. Options: 16k, 32k, 64k, 128k, all, or integer tokens."
            exit 1
        fi
        ;;
esac

