"""Download pinned HF weight files in verified, resumable HTTP ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests

p = argparse.ArgumentParser()
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--endpoint', default='https://hf-mirror.com')
p.add_argument('--workers', type=int, default=12)
a = p.parse_args()
m = json.loads(a.manifest.read_text())
chunk_size = 64 * 1024 * 1024
def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

for item in m['files']:
    target = a.output / item['path']
    if target.exists() and target.stat().st_size == item['size']:
        if file_sha256(target) == item['sha256']:
            print('Verified existing', item['path'], flush=True)
            continue
    target.parent.mkdir(parents=True, exist_ok=True)
    parts = target.parent / (target.name + '.parts')
    parts.mkdir(exist_ok=True)
    url = f"{a.endpoint}/{m['repo']}/resolve/{m['revision']}/{item['path']}"
    # Resolve once; reuse the signed CDN URL across workers.
    head = requests.head(url, allow_redirects=True, timeout=60)
    head.raise_for_status()
    resolved = head.url
    def get_part(start):
        end = min(start + chunk_size, item['size']) - 1
        dest = parts / str(start)
        if dest.exists() and dest.stat().st_size == end-start+1:
            return dest
        for attempt in range(5):
            try:
                with requests.get(resolved, headers={'Range': f'bytes={start}-{end}'}, stream=True, timeout=(30,120)) as r:
                    r.raise_for_status()
                    if r.status_code != 206 or r.headers.get('Content-Range') != f"bytes {start}-{end}/{item['size']}":
                        raise ValueError('Server returned an unexpected byte range')
                    with dest.open('wb') as out:
                        for block in r.iter_content(1024 * 1024):
                            out.write(block)
                if dest.stat().st_size != end-start+1:
                    raise ValueError('Incomplete range')
                print(item['path'], start, 'complete', flush=True)
                return dest
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        files = list(pool.map(get_part, range(0, item['size'], chunk_size)))
    temporary = target.with_name(target.name + '.assembling')
    digest = hashlib.sha256()
    with temporary.open('wb') as out:
        for part in files:
            with part.open('rb') as source:
                while block := source.read(8 * 1024 * 1024):
                    out.write(block)
                    digest.update(block)
    if digest.hexdigest() != item['sha256']:
        raise ValueError('SHA-256 mismatch: ' + str(target))
    temporary.replace(target)
    print('Verified complete', item['path'], flush=True)
