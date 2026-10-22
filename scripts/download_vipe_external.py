"""Prefetch the two directly hosted official checkpoints on the cloud node."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess
import hashlib
import json

root=Path('/linxi/models/worldwander/vipe/torch/hub')
jobs=[
    ('https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar','geocalib/pinhole.tar'),
    ('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth','sam/sam_vit_b_01ec64.pth'),
]
def download(job):
    url,name=job;dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        partial=Path(str(dest)+'.partial')
        subprocess.run(['curl','-fL','--retry','8','--retry-all-errors','--connect-timeout','20',
            '--speed-limit','1024','--speed-time','60','-C','-','-o',str(partial),url],check=True)
        if partial.stat().st_size<1_000_000:raise ValueError('Checkpoint unexpectedly small')
        partial.replace(dest)
    h=hashlib.sha256()
    with dest.open('rb') as stream:
        while block:=stream.read(8*1024*1024):h.update(block)
    print('Ready',name,flush=True)
    return {'url':url,'path':str(dest),'sha256':h.hexdigest()}
with ThreadPoolExecutor(max_workers=2) as pool:
    records=list(pool.map(download,jobs))
(root/'external-manifest.json').write_text(json.dumps(records,indent=2))
