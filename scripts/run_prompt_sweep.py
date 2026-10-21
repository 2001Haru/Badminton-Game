"""Run the three requested prompt variants sequentially on the same two clips."""
import datetime
import json
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo
import fcntl

repo = Path(__file__).resolve().parents[1]
root = Path('/linxi/dataset/badminton/results')
root.mkdir(parents=True, exist_ok=True)
lock = (root/'.prompt_sweep.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
stamp = datetime.datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y%m%d_%H%M%S')
batch = root / ('prompt_sweep_' + stamp)
batch.mkdir()
prompts = [
    ('p1_official', 'Transform it into the first-person perspective.'),
    ('p2_closest_player', 'Transform it into the first-person perspective of the player closest to the camera.'),
    ('p3_closest_badminton_player', 'Transform it into the first-person perspective of the badminton player closest to the camera.'),
]
status = {'batch': str(batch), 'commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
          'seed_per_variant': 1234, 'windows_per_variant': 5, 'steps': 50,
          'resolution': [1280,704], 'fps': 8,
          'variants': [{'label': label, 'prompt': prompt, 'status': 'queued', 'output': str(batch/label)} for label,prompt in prompts]}
def save():
    temporary = batch/'status.json.tmp'
    temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
    temporary.replace(batch/'status.json')
save()
with (repo/'data.md').open('a', encoding='utf-8') as log:
    log.write(f'\n## Prompt sweep {stamp}\n- 用户要求：启动后不持续跟踪；单 GPU 顺序执行三组，每组包含两个完整片段。\n')
    log.write('- batch: ' + str(batch) + '\n')
    for label,prompt in prompts:
        log.write(f'- {label}: {prompt}\n')
print('BATCH_DIR=' + str(batch), flush=True)
for variant in status['variants']:
    variant['status'] = 'running'
    save()
    print('START ' + variant['label'], flush=True)
    with (batch/(variant['label']+'.launcher.log')).open('w') as stream:
        result = subprocess.run([sys.executable,str(repo/'scripts/run_round.py'),
            '--prompt',variant['prompt'],'--label',variant['label'],'--output-dir',variant['output']],
            cwd=repo, stdout=stream, stderr=subprocess.STDOUT)
    variant['returncode'] = result.returncode
    variant['status'] = 'completed' if result.returncode == 0 else 'failed'
    save()
    with (repo/'data.md').open('a',encoding='utf-8') as log:
        log.write(f'- sweep {variant["label"]}: {variant["status"]}, exit={result.returncode}\n')
    print(variant['status'].upper() + ' ' + variant['label'], flush=True)
sys.exit(0 if all(v['status']=='completed' for v in status['variants']) else 1)
