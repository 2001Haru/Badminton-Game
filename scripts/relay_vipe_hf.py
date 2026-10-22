"""Download remaining official ViPE weights and checksum-verify cloud uploads."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
import subprocess
import hashlib
import time

root = Path(__file__).resolve().parents[1] / 'artifacts/models/vipe_hf_relay'
root.mkdir(parents=True, exist_ok=True)
ssh = ['ssh', '-i', 'C:/Users/HAL/.ssh/aistation_task', '-p', '42086', 'root@172.16.78.10']
scp = ['scp', '-i', 'C:/Users/HAL/.ssh/aistation_task', '-P', '42086']
jobs = []
for repo, wanted in [('Ruicheng/moge-2-vitl-normal', {'model.pt'}), ('depth-anything/Video-Depth-Anything-Large', {'video_depth_anything_vitl.pth'}), ('bert-base-uncased', None)]:
    response = requests.get('https://huggingface.co/api/models/' + repo, params={'blobs': 'true'}, timeout=60)
    response.raise_for_status()
    meta = response.json()
    for entry in meta['siblings']:
        name = entry['rfilename']
        if (wanted and name in wanted) or (wanted is None and (name.endswith('.json') or name in ['vocab.txt', 'model.safetensors'])):
            jobs.append((repo, meta['sha'], entry))

def transfer(job):
    repo, revision, entry = job
    name = entry['rfilename']
    dest = root / repo.replace('/', '--') / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    part = Path(str(dest) + '.partial')
    if not dest.exists():
        with requests.get(f'https://huggingface.co/{repo}/resolve/{revision}/{name}', stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with part.open('wb') as handle:
                for chunk in response.iter_content(4 * 1024 * 1024):
                    handle.write(chunk)
        if entry.get('size') and part.stat().st_size != entry['size']:
            raise ValueError('Size mismatch: ' + name)
        part.replace(dest)
    with dest.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    expected = entry.get('lfs', {}).get('sha256')
    if expected and digest != expected:
        raise ValueError('HF SHA mismatch: ' + name)
    print('Downloaded', repo, name, dest.stat().st_size, 'seconds', round(time.monotonic() - start, 1), flush=True)
    remote = '/linxi/models/worldwander/vipe/hf/hub/models--' + repo.replace('/', '--') + '/snapshots/' + revision + '/' + name
    subprocess.run(ssh + ['mkdir -p ' + remote.rsplit('/', 1)[0]], check=True)
    subprocess.run(scp + [str(dest), 'root@172.16.78.10:' + remote + '.relay'], check=True)
    actual = subprocess.check_output(ssh + ['sha256sum ' + remote + '.relay'], text=True).split()[0]
    if actual != digest:
        raise ValueError('Upload checksum mismatch')
    subprocess.run(ssh + ['mv ' + remote + '.relay ' + remote], check=True)
    print('Verified uploaded', repo, name, flush=True)

with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(transfer, jobs))
print('All relay uploads verified', flush=True)
