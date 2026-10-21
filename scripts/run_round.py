"""Each invocation appends its command, status and outputs to ignored data.md."""
import datetime
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

repo = Path(__file__).resolve().parents[1]
stamp = datetime.datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y%m%d_%H%M%S')
default_prompt = ('Transform it into the first-person perspective of Shi Yu Qi, the badminton player wearing a red shirt. '
          'The camera is at his eye level, looking toward the net and his opponent. '
          'Preserve the rally timing, court, racket and shuttlecock motion.')
parser = argparse.ArgumentParser()
parser.add_argument('--prompt', default=default_prompt)
parser.add_argument('--output-dir', type=Path)
parser.add_argument('--label', default='red_player')
args = parser.parse_args()
out = args.output_dir or Path('/linxi/dataset/badminton/results') / stamp
out.mkdir(parents=True)
prompt = args.prompt
cmd = [sys.executable, 'src/wan2_inference.py', '--config=configs/wan2-2_lora_three2one_realworld.yaml',
       '--ckpt_path=/linxi/models/worldwander/wan2-2_lora_three2one_realworld.ckpt',
       '--original_video_root=/linxi/dataset/badminton/prepared/frames', '--pred_path=' + str(out), '--seed=1234',
       'model_id=/linxi/models/worldwander/Wan2.2-TI2V-5B-Diffusers', 'dataset.num_workers=0',
       'prompt=' + prompt]
with (repo / 'data.md').open('a', encoding='utf-8') as log:
    log.write(f'\n## 启动 {stamp}\n- seed: 1234; 49 frames/window; 8 fps; 704x1280; 50 steps\n')
    log.write(f'- variant: {args.label}\n')
    log.write('- command: ' + json.dumps(cmd, ensure_ascii=False) + '\n')
    log.write('- output: ' + str(out) + '\n')
    log.write('- commit: ' + subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() + '\n')
with (out/'environment.txt').open('w') as log:
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=log, check=True)
    subprocess.run(['nvidia-smi'], stdout=log, check=True)
env = dict(os.environ, PYTHONPATH=str(repo/'third_party/WorldWander'), CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1')
start = time.monotonic()
with (out/'inference.log').open('w') as log:
    result = subprocess.run(cmd, cwd=repo/'third_party/WorldWander', env=env, stdout=log, stderr=subprocess.STDOUT)
with (repo/'data.md').open('a', encoding='utf-8') as log:
    log.write(f'- exit code: {result.returncode}\n')
    log.write(f'- wall time seconds: {time.monotonic()-start:.1f}\n')
if result.returncode == 0:
    subprocess.run([sys.executable, str(repo/'scripts/assemble_outputs.py'),
                    '--manifest', '/linxi/dataset/badminton/prepared/manifest.json',
                    '--results', str(out)], check=True)
sys.exit(result.returncode)
