#!/usr/bin/env bash
set -euo pipefail
export PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL=
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=8 FORCE_CUDA=1
export TMPDIR=/linxi/tmp/vipe
PY=/linxi/tmp/vipe/venv/bin/python
SRC=/linxi/tmp/vipe/pytorch3d-33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba
if ! "$PY" -c 'import torch; from pytorch3d import _C; from pytorch3d.renderer.fisheyecameras import FishEyeCameras' >/dev/null 2>&1; then
 curl -fL --retry 5 https://codeload.github.com/facebookresearch/pytorch3d/tar.gz/33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba -o /linxi/tmp/vipe/pytorch3d.tar.gz
 tar -xzf /linxi/tmp/vipe/pytorch3d.tar.gz -C /linxi/tmp/vipe
 "$PY" -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple iopath
 "$PY" -m pip install --no-build-isolation --no-deps "$SRC"
fi
"$PY" -c 'import torch; from pytorch3d import _C; from pytorch3d.renderer.fisheyecameras import FishEyeCameras; print("PyTorch3D extension import OK")'
exec /linxi/tmp/vipe/venv/bin/python /linxi/Badminton-Game/scripts/run_ego_prior_short_windows.py
