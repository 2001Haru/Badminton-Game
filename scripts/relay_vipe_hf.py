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
        size = entry['size']
        url = f'https://hf-mirror.com/{repo}/resolve/{revision}/{name}'
        def fetch_range(index):
            begin = index * 16 * 1024 * 1024
            end = min(size, begin + 16 * 1024 * 1024) - 1
            piece = Path(str(part) + f'.{index}')
            if piece.exists() and piece.stat().st_size == end - begin + 1:
                return piece
            for attempt in range(5):
                try:
                    with requests.get(url, headers={'Range': f'bytes={begin}-{end}'}, stream=True, timeout=(30, 60)) as response:
                        response.raise_for_status()
                        if response.status_code != 206 and size > end - begin + 1:
                            raise ValueError('Server ignored byte range')
                        with piece.open('wb') as handle:
                            for chunk in response.iter_content(1024 * 1024):
                                handle.write(chunk)
                    if piece.stat().st_size != end - begin + 1:
                        raise ValueError('Range size mismatch')
                    return piece
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2)
        with ThreadPoolExecutor(max_workers=4) as ranges:
            pieces = list(ranges.map(fetch_range, range((size + 16 * 1024 * 1024 - 1) // (16 * 1024 * 1024))))
        with part.open('wb') as handle:
            for piece in pieces:
                handle.write(piece.read_bytes())
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
subprocess.run(ssh + ['nohup python /linxi/Badminton-Game/scripts/download_vipe_hf.py > /linxi/tmp/vipe/download-hf-relay.log 2>&1 < /dev/null &'], check=True)
