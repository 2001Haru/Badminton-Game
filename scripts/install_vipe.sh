#!/usr/bin/env bash
set -euo pipefail
export PIP_CONFIG_FILE=/dev/null
export PIP_EXTRA_INDEX_URL=
export TMPDIR=/linxi/tmp/vipe
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=8
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/linxi/models/worldwander/vipe/hf
export TORCH_HOME=/linxi/models/worldwander/vipe/torch
export TORCH_EXTENSIONS_DIR=/linxi/tmp/vipe/extensions
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR"
ENV=/linxi/tmp/vipe/venv
if [ ! -e "$ENV" ]; then
    cp -a /linxi/tmp/egox/venv "$ENV"
    /opt/conda/bin/python -m venv --upgrade "$ENV"
fi
if [ ! -f /usr/include/eigen3/Eigen/Core ]; then
    apt-get update -qq
    apt-get install -y --no-install-recommends libeigen3-dev
fi
PY="$ENV/bin/python"
"$PY" -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple wheel setuptools ninja
"$PY" -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r /linxi/Badminton-Game/requirements-vipe-cu124.txt
"$PY" -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple 'git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a'
cd /linxi/Badminton-Game/third_party/EgoPriorRenderer
"$PY" -m pip install --no-build-isolation --no-deps -e .
"$PY" -c 'import torch,vipe_ext; from moge.model.v2 import MoGeModel; import vipe.cli.main; print("ViPE CUDA extension and pipeline imports OK", torch.__version__)'
touch /linxi/tmp/vipe/environment.ready
