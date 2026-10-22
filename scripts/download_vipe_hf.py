"""Prefetch the HF weights needed by the official lyra pipeline into persistent caches."""
import os
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor
import json

root=Path('/linxi/models/worldwander/vipe')
os.environ['HF_HOME']=str(root/'hf')
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
from huggingface_hub import hf_hub_download,snapshot_download

jobs=[
    ('Ruicheng/moge-2-vitl-normal','model.pt',None),
    ('Rain729/Prior-Depth-Anything','depth_anything_v2_vitl.pth',None),
    ('Rain729/Prior-Depth-Anything','depth_anything_v2_vitb.pth',None),
    ('Rain729/Prior-Depth-Anything','prior_depth_anything_vitb_1_1.pth',None),
    ('ShilongLiu/GroundingDINO','groundingdino_swint_ogc.pth','groundingdino_swint_ogc.pth'),
    ('depth-anything/Video-Depth-Anything-Large','video_depth_anything_vitl.pth','video_depth_anything_vitl.pth'),
]
def download(job):
    repo,name,torch_name=job
    print('Downloading',repo,name,flush=True)
    path=Path(hf_hub_download(repo,name))
    if torch_name:
        dest=root/'torch/hub/checkpoints'/torch_name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
    print('Ready',repo,name,flush=True)
    return {'repo':repo,'file':name,'cache_path':str(path)}
with ThreadPoolExecutor(max_workers=4) as pool:
    manifest=list(pool.map(download,jobs))
bert=snapshot_download('bert-base-uncased',allow_patterns=['*.json','vocab.txt','model.safetensors'])
manifest.append({'repo':'bert-base-uncased','cache_path':bert})
(root/'hf-download-manifest.json').write_text(json.dumps(manifest,indent=2))
Path('/linxi/tmp/vipe/hf-weights.ready').touch()
print('HF weights ready',flush=True)
