#!/usr/bin/env bash
cd ~/sglang

export PYTHONPATH=$PWD/python:$PYTHONPATH
export GOOGLE_APPLICATION_CREDENTIALS=/home/princer_google_com/.config/gcloud/application_default_credentials.json
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source .venv/bin/activate

# Free port 30000 if previously occupied
fuser -k 30000/tcp 2>/dev/null || true

python3 -m sglang.launch_server \
  --model-path /home/princer_google_com/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
  --served-model-name Qwen2.5-0.5B-Instruct \
  --tp 1 \
  --host 0.0.0.0 \
  --port 30000 \
  --context-length 135000 \
  --mem-fraction-static 0.80 \
  --enable-hierarchical-cache \
  --page-size 4096 \
  --hicache-size 15 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-backend gcs \
  --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-backend-extra-config '{"protocol": "gcs", "bucket": "princer-rapid-uscentral1a", "prefix": "sglang_kv_cache", "num_workers": 64, "metadata_ttl": 300}'
