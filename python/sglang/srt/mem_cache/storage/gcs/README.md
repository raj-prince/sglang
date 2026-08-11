# Google Cloud Storage (GCS) Backend for HiCache

This directory contains the native **Google Cloud Storage (GCS)** integration for SGLang **HiCache**, enabling high-throughput KV cache tiering directly to GCS buckets.

## Overview

The GCS backend (`HiCacheGCS`) implements the page-oriented storage interface defined in [`HiCacheStorage`](../../hicache_storage.py), allowing SGLang worker ranks to offload and prefetch Key-Value cache pages across GPU memory, Host memory, and Google Cloud Storage buckets.

### Key Features
- **Parallel Multi-Threaded I/O**: Leverages Python's `ThreadPoolExecutor` to perform concurrent blob transfers across page batches, maximizing TCP connection throughput to GCS.
- **Fast In-Memory Metadata Cache**: Maintains an in-memory `MetadataCache` pre-populated during server startup and updated on writes to eliminate HTTP `HEAD` latency during local match checks.
- **MLA Rank Optimization**: For DeepSeek-style MLA models where KV activations are identical across TP ranks, non-zero TP ranks skip write-back calls locally to prevent redundant network I/O.
- **Hybrid Memory Pool Support**: Supports auxiliary sidecar state pools (e.g., Mamba SSM states, SWA, DSA) in addition to core KV pages.

## Installation

Ensure the Google Cloud Storage SDK is installed in your python environment:

```bash
pip install google-cloud-storage
```

Authenticating to Google Cloud can be configured using standard GCP credentials:
- Set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json`, or
- Use GKE Workload Identity / Application Default Credentials (ADC).

## Usage

### Launching SGLang Server with GCS HiCache Backend

To enable HiCache with the GCS storage backend, launch the server using `--hicache-storage-backend gcs` and specify the bucket configuration using `--hicache-storage-backend-extra-config`:

```bash
python3 -m sglang.launch_server \
  --model-path /path/to/your/model \
  --host 0.0.0.0 \
  --port 30000 \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-io-backend kernel \
  --hicache-write-policy write_through \
  --hicache-storage-prefetch-policy timeout \
  --hicache-storage-backend gcs \
  --hicache-storage-backend-extra-config '{"bucket_name": "my-sglang-kv-bucket", "prefix": "deepseek-r1-cache", "max_workers": 64}'
```

### Configuration Options

The JSON string or TOML file passed to `--hicache-storage-backend-extra-config` supports the following knobs:

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `bucket_name` | string | *Required* | GCS bucket name (can also be supplied via `SGLANG_HICACHE_GCS_BUCKET` environment variable). |
| `prefix` | string | `"sglang_hicache"` | Root folder prefix inside the bucket for stored KV blobs. |
| `max_workers` | integer | `32` | Number of concurrent worker threads for page uploads/downloads. |
| `enable_metadata_cache` | boolean | `true` | Maintain an in-memory TTL metadata cache of stored blobs to accelerate page existence checks. |
| `metadata_ttl` | float | `300.0` | In-memory metadata TTL in seconds. |
