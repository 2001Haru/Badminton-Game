"""Verify complete weights, input frames, and CUDA before an expensive run."""
import hashlib
import json
from pathlib import Path
import sys

root = Path('/linxi/models/worldwander')
manifest = json.loads((root/'model-manifest.json').read_text())
for item in manifest['files']:
    path = root/'Wan2.2-TI2V-5B-Diffusers'/item['path']
    if not path.is_file() or path.stat().st_size != item['size']:
        raise RuntimeError(f'Missing/incomplete model: {path}')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(16*1024*1024):
            h.update(block)
    if h.hexdigest() != item['sha256']:
        raise RuntimeError(f'Checksum mismatch: {path}')
    print('SHA256 OK', item['path'], flush=True)

from PIL import Image
data = Path('/linxi/dataset/badminton/prepared')
clips = json.loads((data/'manifest.json').read_text())
for clip in clips['clips']:
    for window in clip['windows']:
        frames = sorted((data/'frames'/window['name']).glob('*.png'))
        assert len(frames) == 49, window
        for frame in frames:
            with Image.open(frame) as im:
                assert im.size == (1280,704), frame
print('All five frame windows OK', flush=True)

import torch
import torch.nn.functional as F
assert torch.cuda.is_available()
q = torch.randn(1, 4, 128, 64, device='cuda', dtype=torch.bfloat16)
assert torch.isfinite(F.scaled_dot_product_attention(q,q,q)).all()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'third_party/WorldWander'))
from src.wan2_inference import WorldWanderInferenceSystem
print('CUDA BF16 SDPA and WorldWander imports OK:', torch.cuda.get_device_name(0), flush=True)
