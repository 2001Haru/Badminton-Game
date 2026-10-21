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
connection_options = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4']
ssh = ['ssh', '-i', a.identity, '-p', a.port] + connection_options + [a.host]
sftp = ['sftp', '-i', a.identity, '-P', a.port] + connection_options + ['-b', '-', a.host]
def retry(command, **kwargs):
    for attempt in range(6):
        try:
            return subprocess.run(command, check=True, **kwargs)
        except subprocess.CalledProcessError:
            if attempt == 5:
                raise
            print('Connection failed; resuming in 10 seconds', flush=True)
            time.sleep(10)
probe = "import json,pathlib; m=json.load(open('/linxi/models/worldwander/model-manifest.json')); r=pathlib.Path('/linxi/models/worldwander/Wan2.2-TI2V-5B-Diffusers'); print(json.dumps({x['path']:(r/x['path']).stat().st_size if (r/x['path']).exists() else 0 for x in m['files']}))"
while True:
    sizes = json.loads(retry(ssh + ['python -c ' + shlex.quote(probe)], text=True, stdout=subprocess.PIPE).stdout)
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
        retry(ssh + ['mkdir -p ' + shlex.quote(str(Path(dest).parent).replace('\\','/')) + ' && touch ' + shlex.quote(dest + '.uploading')])
        retry(sftp, input=f'reput "{src.resolve().as_posix()}" "{dest}.uploading"\n', text=True)
        retry(ssh + ['mv ' + shlex.quote(dest + '.uploading') + ' ' + shlex.quote(dest)])
        print('Uploaded', item['path'], flush=True)
    time.sleep(15)
