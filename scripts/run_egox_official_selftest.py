"""Wait for pinned weights and the isolated environment, then run official ITW #0."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
from experiment_record import record_condition

p=argparse.ArgumentParser()
p.add_argument('--round-number',type=int,required=True)
p.add_argument('--wait-hours',type=float,default=24)
a=p.parse_args()
repo=Path(__file__).resolve().parents[1]
models=Path('/linxi/models/worldwander/egox')
items=json.loads((models/'manifests.json').read_text())
ready=Path('/linxi/tmp/egox/environment.ready')
deadline=time.monotonic()+a.wait_hours*3600
previous=None
while True:
    missing=[]
    for item in items:
        for f in item['files']:
            path=models/item['repo'].split('/')[-1]/f['path']
            if not path.is_file() or path.stat().st_size!=f['size']:
                missing.append(str(path))
    state=(len(missing),ready.exists())
    if state!=previous:
        print('Waiting: missing files, environment ready =',state,flush=True)
        previous=state
    if not missing and ready.exists(): break
    if time.monotonic()>deadline: raise TimeoutError('Weights/environment not ready within deadline')
    time.sleep(20)
for item in items:
    for f in item['files']:
        if not f['sha256']: continue
        path=models/item['repo'].split('/')[-1]/f['path']
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            while block:=stream.read(16*1024*1024): digest.update(block)
        if digest.hexdigest()!=f['sha256']: raise ValueError('Checksum mismatch: '+str(path))
        print('SHA256 OK',f['path'],flush=True)
while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
    if time.monotonic()>deadline: raise TimeoutError('GPU remains occupied')
    time.sleep(20)
stamp=datetime.datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y%m%d_%H%M%S')
out=Path('/linxi/dataset/badminton/results')/('egox_official_'+stamp)
out.mkdir()
exe='/linxi/tmp/egox/venv/bin/python'
command=[exe,'infer.py','--meta_data_file','./example/in_the_wild/meta.json',
    '--model_path',str(models/'Wan2.1-I2V-14B-480P-Diffusers'),
    '--lora_path',str(models/'EgoX/pytorch_lora_weights.safetensors'),
    '--lora_rank','256','--out',str(out),'--seed','846514','--use_GGA',
    '--cos_sim_scaling_factor','3.0','--in_the_wild','--idx','0']
meta=json.loads((repo/'third_party/EgoX/example/in_the_wild/meta.json').read_text())['test_datasets'][0]
metadata={'round':a.round_number,'command':command,'official_sample':meta,'output':str(out)}
(out/'run.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
record_condition(repo/'data.md','EgoX 官方 ITW 样例 joker', '官方 meta.json 第 0 项原文，完整提示词见输出目录 run.json。',out)
with (repo/'data.md').open('a',encoding='utf-8') as log:
    log.write('- 模型：Wan2.1-I2V-14B-480P + EgoX LoRA rank=256；seed=846514；GGA=True；cos_sim_scaling_factor=3.0；50 steps；CFG=5；49 帧，30 fps；exo 784×448，ego 448×448。\n')
print('OUTPUT='+str(out),flush=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1')
start=time.monotonic()
with (out/'inference.log').open('w') as log:
    proc=subprocess.run(command,cwd=repo/'third_party/EgoX',env=env,stdout=log,stderr=subprocess.STDOUT)
metadata.update(exit_code=proc.returncode,wall_time_seconds=time.monotonic()-start)
(out/'run.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
if proc.returncode==0:
    with (repo/'data.md').open('a',encoding='utf-8') as log:
        log.write(f'- 推理总耗时（含加载）：{metadata["wall_time_seconds"]:.1f} 秒。\n')
        for line in (out/'inference.log').read_text(errors='replace').splitlines():
            if line.startswith('EgoX peak'): log.write('- '+line+'\n')
print('EXIT_CODE='+str(proc.returncode),flush=True)
sys.exit(proc.returncode)
