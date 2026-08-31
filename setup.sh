#!/usr/bin/env bash
# ==============================================================================
# SGLang VM Fast Setup Script
#
# - Installs pre-compiled SGLang binary & PyTorch CUDA dependencies
# - Configures Google Cloud Storage (gcsfs / fsspec) libraries
# - Links local SGLang repository tree in editable development mode
#
# Usage:
#   bash setup.sh
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo "============================================================================"
echo "    SGLang Fast Setup (Core Binary + Editable GCS Rapid Bucket Offloader)"
echo "============================================================================"

# --- Step 1: System Packages ---
echo -e "\n[*] Step 1/4: Installing system prerequisites..."
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv python3-dev git htop tmux aria2 libssl-dev

# --- Step 2: Virtual Environment ---
echo -e "\n[*] Step 2/4: Configuring Python virtual environment (.venv)..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel

# --- Step 3: Install Core SGLang & GCS Dependencies ---
echo -e "\n[*] Step 3/4: Installing PyTorch, SGLang, and Google Cloud Storage packages..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install "sglang[all]"
pip install --upgrade "fsspec>=2026.7.0" "gcsfs>=2026.7.0" "google-auth" "google-cloud-storage" "requests" "transformers"

# --- Step 4: Link Local SGLang Development Tree ---
echo -e "\n[*] Step 4/4: Linking local SGLang package into .venv..."
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
rm -rf "${SITE_PACKAGES}/sglang"
ln -sfn "${SCRIPT_DIR}/python/sglang" "${SITE_PACKAGES}/sglang"

# Ensure execution permissions for companion scripts
chmod +x "${SCRIPT_DIR}/run_server.sh" "${SCRIPT_DIR}/run_benchmark.sh" 2>/dev/null || true

# --- Verification ---
echo -e "\n============================================================================"
echo "                           Verification"
echo "============================================================================"

python3 -c "
import torch
import sglang
import fsspec
import gcsfs
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory

print('[✓] SGLang Core Version   :', sglang.__version__)
print('[✓] PyTorch CUDA Support  :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('[✓] GPU Device Detected   :', torch.cuda.get_device_name(0))
print('[✓] fsspec / gcsfs        : fsspec=' + fsspec.__version__ + ', gcsfs=' + gcsfs.__version__)
assert 'gcs' in StorageBackendFactory._registry, 'GCS storage backend missing from factory registry!'
print('[✓] GCS Storage Backend   : Registered in StorageBackendFactory (OK)')
"

echo "============================================================================"
echo "  [✓] Setup complete! All prerequisites & GCS offloader are ready."
echo "  To start the server: ./run_server.sh"
echo "  To run benchmarks  : ./run_benchmark.sh [16k|32k|64k|128k|all]"
echo "============================================================================"
