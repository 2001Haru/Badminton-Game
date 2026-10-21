#!/usr/bin/env bash
set -euo pipefail
export TMPDIR=/linxi/tmp/worldwander
export PIP_CONFIG_FILE=/dev/null
export PIP_EXTRA_INDEX_URL=
ENV=/linxi/tmp/worldwander/venv-conda
"$ENV/bin/pip" install -r /linxi/Badminton-Game/requirements-inference.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
"$ENV/bin/python" -c 'import torch, torchvision, diffusers, pytorch_lightning; print(torch.__version__, diffusers.__version__); print(torch.cuda.get_device_name(0)); print(torch.ones(8, device="cuda").sum().item())'
