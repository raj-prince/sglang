# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""
Multi-Session Concurrency Benchmark for SGLang GCS Rapid Bucket KV-Cache Offloader.

Supports both:
  - --workload distinct : Each session has a 100% unique, non-overlapping document context.
  - --workload shared   : All sessions share a common large prefix with distinct question suffixes.

Benchmarks Time-To-First-Token (TTFT) distribution (P50, P90, P99), total duration,
and aggregate throughput across concurrency levels (1, 4, 8, 16, 32) and token lengths.

Phases evaluated per concurrency level:
  1. Cold Prefill: C concurrent requests hitting un-cached prompts (GPU compute contention).
  2. GCS Rapid Bucket Prefetch: /flush_cache called -> C concurrent requests prefetching distinct/shared cache from GCS.
  3. GPU HBM L1 Cache (Hot): C concurrent requests hitting in-memory RadixCache directly.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
import requests

DEFAULT_MODEL_PATH = "/home/princer_google_com/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Concurrency Benchmark for SGLang GCS Rapid Bucket Offloading"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:30000",
        help="Base URL for SGLang server (default: http://127.0.0.1:30000)",
    )
    parser.add_argument(
        "--workload",
        type=str,
        choices=["distinct", "shared"],
        default="distinct",
        help="Workload pattern: 'distinct' (100%% unique context per session) or 'shared' (common prefix)",
    )
    parser.add_argument(
        "--concurrency-levels",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32],
        help="List of concurrency levels to evaluate (default: 1 4 8 16 32)",
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=16384,
        help="Exact number of prefix tokens per session (default: 16384)",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=32,
        help="Number of tokens to generate per request (default: 32)",
    )
    parser.add_argument(
        "--wait-offload-time",
        type=float,
        default=5.0,
        help="Base seconds to wait for background GCS offload to drain (default: 5.0)",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path or name of HuggingFace tokenizer. Defaults to local model cache.",
    )
    return parser.parse_args()


def flush_server_cache(base_url: str) -> bool:
    """Flush server L1 GPU HBM and L2 Host RAM caches."""
    try:
        resp = requests.post(f"{base_url}/flush_cache", params={"timeout": 30.0}, timeout=45)
        if resp.status_code == 200:
            return True
        print(f"[Warning] /flush_cache returned status {resp.status_code}: {resp.text.strip()}")
        return False
    except Exception as e:
        print(f"[Error] Failed to flush cache: {e}")
        return False


def build_session_requests(
    workload: str,
    target_prefix_tokens: int,
    num_sessions: int,
    tokenizer_path: Optional[str] = None,
) -> List[List[int]]:
    """
    Build token ID sequences for each session:
      - If workload == 'distinct': Every session has a completely unique document context and key hash.
      - If workload == 'shared'  : All sessions share a common document context with unique question suffixes.
    """
    path_to_try = tokenizer_path or DEFAULT_MODEL_PATH
    tokenizer = None
    if os.path.exists(path_to_try) or tokenizer_path is not None:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(path_to_try, trust_remote_code=True)
        except Exception:
            pass

    requests_list = []

    if workload == "distinct":
        for s in range(num_sessions):
            if tokenizer is not None:
                topic_paragraph = (
                    f"Session Document {s + 1}. This unique research paper investigates topic domain {s + 1} "
                    f"regarding high-performance distributed computing, cloud storage architectures, and parallel algorithms. "
                    f"Dataset partition {s + 1} demonstrates unique statistical properties across experimental dimensions {s * 13 + 7}. "
                    f"Google Cloud Storage Rapid Bucket provides disaggregated KV cache offloading for large language model workloads. "
                )
                topic_ids = tokenizer.encode(topic_paragraph, add_special_tokens=False)
                repeats = (target_prefix_tokens // len(topic_ids)) + 1
                session_prefix = (topic_ids * repeats)[:target_prefix_tokens]
                suffix = tokenizer.encode(
                    f"\n\nQuestion: Summarize document {s + 1} in 15 words.\nAnswer:",
                    add_special_tokens=False,
                )
                requests_list.append(session_prefix + suffix)
            else:
                pattern = [(s * 1000 + k) % 30000 + 100 for k in range(500)]
                repeats = (target_prefix_tokens // len(pattern)) + 1
                session_prefix = (pattern * repeats)[:target_prefix_tokens]
                suffix = [50000 + s, 50001 + s, 50002 + s]
                requests_list.append(session_prefix + suffix)
    else:  # shared
        if tokenizer is not None:
            base_text = (
                "Google Cloud Storage Rapid Bucket is an ultra-high performance object storage tier "
                "engineered for AI/ML training and inference workloads. It delivers low-latency "
                "and multi-gigabit per second aggregate throughput for hierarchical KV cache offload "
                "in large language model serving engines such as SGLang. "
            )
            base_ids = tokenizer.encode(base_text, add_special_tokens=False)
            repeats = (target_prefix_tokens // len(base_ids)) + 1
            shared_prefix_ids = (base_ids * repeats)[:target_prefix_tokens]

            for s in range(num_sessions):
                suffix_text = f"\n\nQuestion {s + 1}: Summarize key point {s + 1} in 15 words.\nAnswer:"
                suffix = tokenizer.encode(suffix_text, add_special_tokens=False)
                requests_list.append(shared_prefix_ids + suffix)
        else:
            pattern = list(range(100, 500))
            repeats = (target_prefix_tokens // len(pattern)) + 1
            shared_prefix_ids = (pattern * repeats)[:target_prefix_tokens]
            for s in range(num_sessions):
                requests_list.append(shared_prefix_ids + [1000 + s, 1001 + s, 1002 + s])

    return requests_list


async def async_send_generate_request(
    client: httpx.AsyncClient,
    base_url: str,
    input_ids: List[int],
    max_new_tokens: int,
    session_id: int,
) -> Tuple[float, float, int]:
    """
    Send streaming generation request asynchronously.
    Returns: (ttft_sec, total_latency_sec, session_id)
    """
    url = f"{base_url}/generate"
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
        },
        "stream": True,
    }

    start_time = time.perf_counter()
    first_token_time = None

    try:
        async with client.stream("POST", url, json=payload, timeout=300.0) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
    except Exception as e:
        print(f"[Error Session {session_id}]: {e}")
        end_time = time.perf_counter()
        return (end_time - start_time), (end_time - start_time), session_id

    end_time = time.perf_counter()
    if first_token_time is None:
        first_token_time = end_time

    ttft = first_token_time - start_time
    total = end_time - start_time
    return ttft, total, session_id


async def run_concurrent_batch(
    base_url: str,
    request_id_lists: List[List[int]],
    max_new_tokens: int,
) -> Tuple[List[float], List[float], float]:
    """
    Dispatches a batch of requests simultaneously across async HTTP connections.
    Returns: (ttft_list, total_time_list, wall_clock_batch_duration)
    """
    limits = httpx.Limits(max_connections=128, max_keepalive_connections=64)
    async with httpx.AsyncClient(limits=limits, timeout=300.0) as client:
        batch_start = time.perf_counter()
        tasks = [
            async_send_generate_request(client, base_url, req_ids, max_new_tokens, idx)
            for idx, req_ids in enumerate(request_id_lists)
        ]
        results = await asyncio.gather(*tasks)
        batch_duration = time.perf_counter() - batch_start

    ttfts = [r[0] for r in results]
    totals = [r[1] for r in results]
    return ttfts, totals, batch_duration


def calculate_metrics(ttfts: List[float], batch_duration: float, total_tokens: int) -> Dict[str, float]:
    """Computes percentile metrics and aggregate throughput."""
    return {
        "p50_ttft_ms": float(np.percentile(ttfts, 50) * 1000),
        "p90_ttft_ms": float(np.percentile(ttfts, 90) * 1000),
        "p99_ttft_ms": float(np.percentile(ttfts, 99) * 1000),
        "mean_ttft_ms": float(np.mean(ttfts) * 1000),
        "batch_duration_s": batch_duration,
        "throughput_tok_s": total_tokens / batch_duration if batch_duration > 0 else 0.0,
    }


def main():
    args = parse_args()
    base_url = args.base_url.rstrip("/")

    print("=" * 90)
    print("       SGLang GCS Rapid Bucket Multi-Session Concurrency Benchmark")
    print("=" * 90)
    print(f"[*] Target Server      : {base_url}")
    print(f"[*] Workload Pattern   : {args.workload.upper()} ({'100% unique context per session' if args.workload == 'distinct' else 'shared common prefix'})")
    print(f"[*] Prefix Length      : {args.prefix_len:,} tokens per session")
    print(f"[*] Output Length      : {args.output_len:,} tokens")
    print(f"[*] Concurrency Levels : {args.concurrency_levels}")
    print(f"[*] Base Offload Wait  : {args.wait_offload_time}s")

    max_c = max(args.concurrency_levels)
    all_session_requests = build_session_requests(
        args.workload, args.prefix_len, max_c, args.tokenizer_path
    )

    results_table = []

    for c in args.concurrency_levels:
        current_requests = all_session_requests[:c]
        total_prompt_tokens = sum(len(r) for r in current_requests)
        offload_wait = args.wait_offload_time + (c * 0.5)

        print("\n" + "#" * 90)
        print(f"  EVALUATING CONCURRENCY C = {c:2d} ({args.workload.upper()} WORKLOAD: {args.prefix_len:,} tokens x {c} sessions = {total_prompt_tokens:,} tokens)")
        print("#" * 90)

        # ----------------------------------------------------------------------
        # Stage 1: Cold Prefill Concurrency
        # ----------------------------------------------------------------------
        print(f"\n[*] [C={c}] Step 1/3: Cold Prefill (Flushing cache -> Dispatching {c} concurrent requests)...")
        flush_server_cache(base_url)
        time.sleep(1.0)

        cold_ttfts, cold_totals, cold_duration = asyncio.run(
            run_concurrent_batch(base_url, current_requests, args.output_len)
        )
        cold_m = calculate_metrics(cold_ttfts, cold_duration, total_prompt_tokens)
        print(f"    [+] Cold Prefill P50 TTFT: {cold_m['p50_ttft_ms']:7.1f} ms | P90: {cold_m['p90_ttft_ms']:7.1f} ms | P99: {cold_m['p99_ttft_ms']:7.1f} ms")
        print(f"    [+] Batch Duration       : {cold_m['batch_duration_s']:7.3f} s  | Throughput: {cold_m['throughput_tok_s']:,.0f} tokens/s")

        print(f"\n[*] Waiting {offload_wait:.1f}s for GCS Rapid Bucket background writes to complete ({c} distinct objects)...")
        time.sleep(offload_wait)

        # ----------------------------------------------------------------------
        # Stage 2: Warm GCS Rapid Bucket Prefetch Concurrency
        # ----------------------------------------------------------------------
        print(f"\n[*] [C={c}] Step 2/3: Warm GCS Rapid Bucket Hit (Flushing GPU HBM -> Dispatching {c} concurrent requests)...")
        flush_server_cache(base_url)
        time.sleep(1.0)

        gcs_ttfts, gcs_totals, gcs_duration = asyncio.run(
            run_concurrent_batch(base_url, current_requests, args.output_len)
        )
        gcs_m = calculate_metrics(gcs_ttfts, gcs_duration, total_prompt_tokens)
        speedup_p50 = cold_m['p50_ttft_ms'] / gcs_m['p50_ttft_ms'] if gcs_m['p50_ttft_ms'] > 0 else 0.0
        print(f"    [+] GCS Prefetch P50 TTFT: {gcs_m['p50_ttft_ms']:7.1f} ms | P90: {gcs_m['p90_ttft_ms']:7.1f} ms | P99: {gcs_m['p99_ttft_ms']:7.1f} ms")
        print(f"    [+] Batch Duration       : {gcs_m['batch_duration_s']:7.3f} s  | Throughput: {gcs_m['throughput_tok_s']:,.0f} tokens/s")
        print(f"    [+] GCS P50 TTFT Speedup : {speedup_p50:.2f}x faster than cold prefill")

        # ----------------------------------------------------------------------
        # Stage 3: Hot GPU HBM L1 Cache Concurrency
        # ----------------------------------------------------------------------
        print(f"\n[*] [C={c}] Step 3/3: Hot GPU HBM Cache Hit (Immediate dispatch without flush)...")
        hbm_ttfts, hbm_totals, hbm_duration = asyncio.run(
            run_concurrent_batch(base_url, current_requests, args.output_len)
        )
        hbm_m = calculate_metrics(hbm_ttfts, hbm_duration, total_prompt_tokens)
        speedup_hbm = cold_m['p50_ttft_ms'] / hbm_m['p50_ttft_ms'] if hbm_m['p50_ttft_ms'] > 0 else 0.0
        print(f"    [+] GPU HBM Hot P50 TTFT : {hbm_m['p50_ttft_ms']:7.1f} ms | P90: {hbm_m['p90_ttft_ms']:7.1f} ms | P99: {hbm_m['p99_ttft_ms']:7.1f} ms")
        print(f"    [+] GPU HBM TTFT Speedup : {speedup_hbm:.2f}x faster than cold prefill")

        results_table.append({
            "c": c,
            "tokens": total_prompt_tokens,
            "cold": cold_m,
            "gcs": gcs_m,
            "hbm": hbm_m,
            "gcs_speedup": speedup_p50,
            "hbm_speedup": speedup_hbm,
        })

    # ==========================================================================
    # Final Summary Matrix
    # ==========================================================================
    print("\n" + "=" * 100)
    print(f"       CONCURRENCY BENCHMARK RESULTS ({args.workload.upper()} WORKLOAD: {args.prefix_len:,} tokens/session)")
    print("=" * 100)
    print(f"{'Concurrency':<12} | {'Batch Tokens':<14} | {'Cold P50 (ms)':<14} | {'GCS P50 (ms)':<14} | {'GCS P99 (ms)':<14} | {'HBM P50 (ms)':<14} | {'GCS Speedup':<12}")
    print("-" * 100)
    for r in results_table:
        print(
            f"C = {r['c']:<8} | "
            f"{r['tokens']:<14,} | "
            f"{r['cold']['p50_ttft_ms']:<14.1f} | "
            f"{r['gcs']['p50_ttft_ms']:<14.1f} | "
            f"{r['gcs']['p99_ttft_ms']:<14.1f} | "
            f"{r['hbm']['p50_ttft_ms']:<14.1f} | "
            f"{r['gcs_speedup']:<10.2f}x"
        )
    print("=" * 100)
    print("\n[✓] Multi-session distinct concurrency benchmark completed successfully!")


if __name__ == "__main__":
    main()
