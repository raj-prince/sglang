# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""
3-Request Benchmark for GCS Rapid Bucket KV-Cache Offloading in SGLang.

This script benchmarks hierarchical KV cache offloading across 3 distinct caching tiers:
  1. Request 1 (Cold Prefill): Full compute on GPU, offloading KV cache to GCS Rapid Bucket in background.
  2. Cache Flush: Calls /flush_cache to evict GPU HBM and Host RAM, leaving cache solely in GCS.
  3. Request 2 (GCS Rapid Bucket Hit): Prefetches KV cache from GCS Rapid Bucket into GPU, skipping prefill.
  4. Request 3 (GPU HBM Cache Hit): Instantaneous in-memory RadixCache hit without flush.

Guarantees 100% exact input token counts by dispatching structured `input_ids`.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests

DEFAULT_MODEL_PATH = "/home/princer_google_com/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"


def parse_args():
    parser = argparse.ArgumentParser(
        description="3-Request Benchmark for SGLang GCS Rapid Bucket Offloading"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:30000",
        help="Base URL for SGLang server (default: http://127.0.0.1:30000)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name. If None, auto-detected from /v1/models",
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=16384,
        help="Exact number of tokens in the shared prefix prompt (default: 16384)",
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
        help="Seconds to wait after Request 1 for background GCS backup to complete (default: 5.0)",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path or name of HuggingFace tokenizer. Defaults to local model cache.",
    )
    return parser.parse_args()


def get_model_info(base_url: str) -> str:
    """Fetch the active model name from SGLang server."""
    try:
        resp = requests.get(f"{base_url}/get_model_info", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("model_path", "Qwen2.5-0.5B-Instruct")
    except Exception:
        pass
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return data["data"][0]["id"]
    except Exception:
        pass
    return "Qwen2.5-0.5B-Instruct"


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


def build_exact_input_ids(
    target_prefix_tokens: int,
    tokenizer_path: Optional[str] = None,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Builds exact token ID sequences for the benchmark:
      - shared_prefix_ids: exactly `target_prefix_tokens` long
      - suffix1_ids, suffix2_ids, suffix3_ids: unique suffixes to distinguish questions
    """
    path_to_try = tokenizer_path or DEFAULT_MODEL_PATH
    tokenizer = None
    if os.path.exists(path_to_try) or tokenizer_path is not None:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(path_to_try, trust_remote_code=True)
        except Exception as e:
            print(f"[Notice] AutoTokenizer load failed ({e}). Using synthetic deterministic IDs.")

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

        suffix1_ids = tokenizer.encode("\n\nQuestion: Summarize Section 1 in 15 words.\nAnswer:", add_special_tokens=False)
        suffix2_ids = tokenizer.encode("\n\nQuestion: Summarize Section 2 in 15 words.\nAnswer:", add_special_tokens=False)
        suffix3_ids = tokenizer.encode("\n\nQuestion: Summarize Section 3 in 15 words.\nAnswer:", add_special_tokens=False)
    else:
        # Synthetic deterministic token IDs [100..1000]
        pattern = list(range(100, 500))
        repeats = (target_prefix_tokens // len(pattern)) + 1
        shared_prefix_ids = (pattern * repeats)[:target_prefix_tokens]
        suffix1_ids = [901, 902, 903, 904, 905]
        suffix2_ids = [906, 907, 908, 909, 910]
        suffix3_ids = [911, 912, 913, 914, 915]

    assert len(shared_prefix_ids) == target_prefix_tokens, (
        f"Mismatch: expected {target_prefix_tokens}, got {len(shared_prefix_ids)}"
    )

    return shared_prefix_ids, suffix1_ids, suffix2_ids, suffix3_ids


def send_streaming_generate_request(
    base_url: str,
    input_ids: List[int],
    max_new_tokens: int,
) -> Tuple[float, float, str, int]:
    """
    Send generation request using SSE streaming with exact `input_ids`.
    Returns: (ttft_sec, total_latency_sec, generated_text, output_tokens)
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
    generated_chunks = []

    with requests.post(url, json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data_str = line_str[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    generated_chunks.append(chunk.get("text", ""))
                except json.JSONDecodeError:
                    continue

    end_time = time.perf_counter()

    if first_token_time is None:
        first_token_time = end_time

    ttft = first_token_time - start_time
    total_time = end_time - start_time
    full_text = "".join(generated_chunks)

    return ttft, total_time, full_text, max_new_tokens


def main():
    args = parse_args()
    base_url = args.base_url.rstrip("/")

    print("=" * 80)
    print("      SGLang GCS Rapid Bucket Hierarchical Cache Benchmark")
    print("=" * 80)

    # 1. Server Info
    model_name = args.model or get_model_info(base_url)
    print(f"[*] Target Server : {base_url}")
    print(f"[*] Serving Model : {model_name}")
    print(f"[*] Exact Prefix  : {args.prefix_len:,} tokens")
    print(f"[*] Output Length : {args.output_len:,} tokens")
    print(f"[*] Offload Drain : {args.wait_offload_time}s")

    # 2. Build Exact Token Arrays
    print(f"\n[*] Generating exact {args.prefix_len:,}-token prompt array...")
    shared_prefix_ids, suffix1, suffix2, suffix3 = build_exact_input_ids(
        args.prefix_len, args.tokenizer_path
    )

    req1_ids = shared_prefix_ids + suffix1
    req2_ids = shared_prefix_ids + suffix2
    req3_ids = shared_prefix_ids + suffix3

    print(f"    - Shared Prefix Tokens : {len(shared_prefix_ids):,}")
    print(f"    - Request 1 Total Tokens: {len(req1_ids):,} (Prefix: {len(shared_prefix_ids):,} + Suffix: {len(suffix1)})")
    print(f"    - Request 2 Total Tokens: {len(req2_ids):,} (Prefix: {len(shared_prefix_ids):,} + Suffix: {len(suffix2)})")
    print(f"    - Request 3 Total Tokens: {len(req3_ids):,} (Prefix: {len(shared_prefix_ids):,} + Suffix: {len(suffix3)})")

    # 3. Step 0: Flush initial cache
    print("\n[*] Step 0: Flushing server memory cache to ensure clean baseline...")
    flush_server_cache(base_url)
    time.sleep(1.0)

    # 4. Request 1: Cold Prefill + GCS Offload
    print("\n" + "-" * 80)
    print(f"  [Request 1] Cold Prefill ({len(req1_ids):,} input tokens -> Full GPU Compute -> Offload to GCS)")
    print("-" * 80)
    print(">> Sending Request 1...")
    ttft_1, total_1, text_1, out_tokens_1 = send_streaming_generate_request(
        base_url, req1_ids, args.output_len
    )
    print(f"   [+] TTFT (Cold Prefill Latency): {ttft_1 * 1000:.2f} ms ({ttft_1:.3f} s)")
    print(f"   [+] Total Request Latency      : {total_1:.3f} s")
    print(f"   [+] Output Text Preview        : {text_1.strip()[:80]}...")

    print(f"\n[*] Waiting {args.wait_offload_time}s for background GCS Rapid Bucket write to complete...")
    time.sleep(args.wait_offload_time)

    # 5. Flush Cache (Simulates GPU memory eviction / separate server instance)
    print("\n[*] Step 1.5: Calling /flush_cache (Evicting GPU HBM & Host RAM -> Cache exists only in GCS)...")
    if not flush_server_cache(base_url):
        print("[Warning] /flush_cache did not return success. Continuing...")
    time.sleep(1.0)

    # 6. Request 2: Warm Prefill from GCS Rapid Bucket
    print("\n" + "-" * 80)
    print(f"  [Request 2] Warm Prefill ({len(shared_prefix_ids):,} tokens Hit from GCS Rapid Bucket)")
    print("-" * 80)
    print(">> Sending Request 2 with identical prefix...")
    ttft_2, total_2, text_2, out_tokens_2 = send_streaming_generate_request(
        base_url, req2_ids, args.output_len
    )
    speedup_gcs = ttft_1 / ttft_2 if ttft_2 > 0 else float("inf")
    print(f"   [+] TTFT (GCS Prefetch Hit)    : {ttft_2 * 1000:.2f} ms ({ttft_2:.3f} s)")
    print(f"   [+] Total Request Latency      : {total_2:.3f} s")
    print(f"   [+] GCS TTFT Speedup           : {speedup_gcs:.2f}x faster than Cold Prefill")
    print(f"   [+] Output Text Preview        : {text_2.strip()[:80]}...")

    # 7. Request 3: GPU HBM L1 Cache Hit (Immediate, no flush)
    print("\n" + "-" * 80)
    print(f"  [Request 3] Hot Cache ({len(shared_prefix_ids):,} tokens Hit in GPU HBM RadixCache L1)")
    print("-" * 80)
    print(">> Sending Request 3 immediately without flush...")
    ttft_3, total_3, text_3, out_tokens_3 = send_streaming_generate_request(
        base_url, req3_ids, args.output_len
    )
    speedup_hbm = ttft_1 / ttft_3 if ttft_3 > 0 else float("inf")
    print(f"   [+] TTFT (GPU HBM Hit)         : {ttft_3 * 1000:.2f} ms ({ttft_3:.3f} s)")
    print(f"   [+] Total Request Latency      : {total_3:.3f} s")
    print(f"   [+] GPU HBM TTFT Speedup       : {speedup_hbm:.2f}x faster than Cold Prefill")
    print(f"   [+] Output Text Preview        : {text_3.strip()[:80]}...")

    # 8. Summary Table
    print("\n" + "=" * 80)
    print(f"         BENCHMARK RESULTS SUMMARY (Prefix: {args.prefix_len:,} tokens)")
    print("=" * 80)
    print(f"{'Tier / Request Stage':<35} | {'TTFT (s)':<10} | {'TTFT (ms)':<10} | {'Total (s)':<10} | {'Speedup':<8}")
    print("-" * 80)
    print(f"{'1. Cold Prefill (GPU Compute)':<35} | {ttft_1:<10.3f} | {ttft_1*1000:<10.1f} | {total_1:<10.3f} | {'1.00x':<8}")
    print(f"{'2. GCS Rapid Bucket Offload':<35} | {ttft_2:<10.3f} | {ttft_2*1000:<10.1f} | {total_2:<10.3f} | {f'{speedup_gcs:.2f}x':<8}")
    print(f"{'3. GPU HBM L1 Cache (Hot)':<35} | {ttft_3:<10.3f} | {ttft_3*1000:<10.1f} | {total_3:<10.3f} | {f'{speedup_hbm:.2f}x':<8}")
    print("=" * 80)
    print("\n[✓] Benchmark completed successfully!")


if __name__ == "__main__":
    main()
