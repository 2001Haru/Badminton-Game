"""Resumable curl fallback for Windows environments with unreliable HF/Xet."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
from huggingface_hub import HfApi

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
repo = 'Wan-AI/Wan2.2-TI2V-5B-Diffusers'
info = HfApi().model_info(repo, files_metadata=True)
def download(s):
    target = a.output / s.rfilename
    if target.exists() and target.stat().st_size == s.size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.part')
    print('Downloading', s.rfilename, s.size, flush=True)
    subprocess.run(['curl.exe', '-f', '-L', '--retry', '8', '--retry-all-errors', '--connect-timeout', '30',
                    '-C', '-', '-o', str(partial), f'https://huggingface.co/{repo}/resolve/{info.sha}/{s.rfilename}'], check=True)
    if partial.stat().st_size != s.size:
        raise ValueError(f'Incomplete download: {partial}')
    partial.replace(target)
    print('Completed', s.rfilename, flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(download, [s for s in info.siblings if not s.rfilename.lower().endswith(('.md','.png','.jpg'))]))
print('All model files downloaded at revision', info.sha, flush=True)
