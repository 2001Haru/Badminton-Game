"""Upload completed local shards while other shards continue downloading."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--model-dir', type=Path, required=True)
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--host', required=True)
p.add_argument('--port', required=True)
p.add_argument('--identity', required=True)
a = p.parse_args()
m = json.loads(a.manifest.read_text())
remote = '/linxi/models/worldwander/Wan2.2-TI2V-5B-Diffusers/'
ssh = ['ssh', '-i', a.identity, '-p', a.port, a.host]
scp = ['scp', '-i', a.identity, '-P', a.port]
probe = "import json,pathlib; m=json.load(open('/linxi/models/worldwander/model-manifest.json')); r=pathlib.Path('/linxi/models/worldwander/Wan2.2-TI2V-5B-Diffusers'); print(json.dumps({x['path']:(r/x['path']).stat().st_size if (r/x['path']).exists() else 0 for x in m['files']}))"
while True:
    sizes = json.loads(subprocess.check_output(ssh + ['python -c ' + shlex.quote(probe)], text=True))
    if all(sizes.get(x['path']) == x['size'] for x in m['files']):
        print('All cloud weight files have expected sizes; run preflight for hashes.', flush=True)
        break
    for item in m['files']:
        if sizes.get(item['path']) == item['size']:
            continue
        src = a.model_dir/item['path']
        if not src.exists() or src.stat().st_size != item['size']:
            continue
        dest = remote + item['path']
        print('Uploading', item['path'], flush=True)
        subprocess.run(ssh + ['mkdir -p ' + shlex.quote(str(Path(dest).parent).replace('\\','/'))], check=True)
        subprocess.run(scp + [str(src), a.host + ':' + dest + '.uploading'], check=True)
        subprocess.run(ssh + ['mv ' + shlex.quote(dest + '.uploading') + ' ' + shlex.quote(dest)], check=True)
        print('Uploaded', item['path'], flush=True)
    time.sleep(15)
