#!/usr/bin/env bash
set -euo pipefail
export PIP_CONFIG_FILE=/dev/null
export PIP_EXTRA_INDEX_URL=
export TMPDIR=/linxi/tmp/egox
mkdir -p "$TMPDIR"
ENV=/linxi/tmp/egox/venv
if [ ! -e "$ENV" ]; then
    cp -a /linxi/tmp/worldwander/venv-conda "$ENV"
    /opt/conda/bin/python -m venv --upgrade "$ENV"
fi
# Inference does not import DeepSpeed; keep training-only dependencies separate.
"$ENV/bin/python" -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    diffusers==0.34.0 transformers==4.49.0 accelerate==1.5.2 peft==0.17.1 \
    sentencepiece tyro opencv-python-headless
cd /linxi/Badminton-Game/third_party/EgoX
"$ENV/bin/python" -c 'import torch; import infer; from core.finetune.models.wan_i2v.sft_trainer import WanWidthConcatImageToVideoPipeline; print("EgoX imports OK", torch.__version__)'
touch /linxi/tmp/egox/environment.ready
