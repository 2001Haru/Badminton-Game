"""Download pinned EgoX checkpoints; default to the requested HF mirror."""
import argparse
import os
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor

p=argparse.ArgumentParser()
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--endpoint',default='https://hf-mirror.com')
a=p.parse_args()
os.environ['HF_ENDPOINT']=a.endpoint
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
from huggingface_hub import snapshot_download

def download(item):
    print('Downloading',item['repo'],item['revision'],flush=True)
    snapshot_download(item['repo'],revision=item['revision'],
        local_dir=str(a.output/item['repo'].split('/')[-1]),
        allow_patterns=[x['path'] for x in item['files']],max_workers=4)
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(download,json.loads(a.manifest.read_text())))
print('All downloads complete',flush=True)
