#!/usr/bin/env bash
set -euo pipefail

cd /workspace/capacitylab

python -m pip install --upgrade pip setuptools wheel

# Install CUDA-enabled PyTorch wheels that match the CUDA 12.4 base image.
python -m pip install \
  torch \
  torchvision \
  torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

# Install PyG after torch so pip resolves the matching binary wheels.
python -m pip install \
  torch-geometric

python -m pip install -r requirements.txt
python -m pip install -r src/traffic_flow_gui/requirements.txt

python - <<'PY'
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not visible inside container. Check Docker + NVIDIA runtime on the host.")
PY
