#!/usr/bin/env bash
set -euo pipefail

echo "=== Hardware Check ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "WARNING: nvidia-smi not found (running without GPU?)"
fi
echo "CPU cores: $(nproc)"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "=== Installing uv ==="
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "=== Installing git-lfs ==="
  if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" -eq 0 ]; then
    apt-get update -qq && apt-get install -y -qq git-lfs
  else
    echo "WARNING: git-lfs missing and cannot apt-get without root"
  fi
fi

if command -v git-lfs >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git lfs install --local || true
  git lfs pull || true
fi

echo "=== Syncing Python 3.13 + ViZDoom + PyTorch (.venv) ==="
uv sync

echo "=== Verifying Colab Environment ==="
.venv/bin/python - <<'PY'
import sys
import torch
import vizdoom
import doom_env

print(f"Python:   {sys.version.split()[0]}")
print(f"PyTorch:  {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
if torch.cuda.is_available():
    print(f"GPU:      {torch.cuda.get_device_name(0)}")
print(f"ViZDoom:  {vizdoom.__version__}")
print(f"Contract: {doom_env.CONTRACT_VERSION}")

env = doom_env.make_env("maps/two_rooms.wad", render_buffers=False)
obs, _ = env.reset(seed=0)
obs, reward, term, trunc, info = env.step(env.action_space.sample())
env.close()
print("doom_env smoke step: OK (rays:", obs["rays"]["distance"].shape, "vitals:", obs["vitals"].shape, ")")
PY
