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
from experiment_record import begin_round, record_condition, record_measurements

repo = Path(__file__).resolve().parents[1]
stamp = datetime.datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y%m%d_%H%M%S')
default_prompt = ('Transform it into the first-person perspective of Shi Yu Qi, the badminton player wearing a red shirt. '
          'The camera is at his eye level, looking toward the net and his opponent. '
          'Preserve the rally timing, court, racket and shuttlecock motion.')
parser = argparse.ArgumentParser()
parser.add_argument('--prompt', default=default_prompt)
parser.add_argument('--output-dir', type=Path)
parser.add_argument('--label', default='red_player')
parser.add_argument('--round-number', type=int)
args = parser.parse_args()
out = args.output_dir or Path('/linxi/dataset/badminton/results') / stamp
out.mkdir(parents=True)
prompt = args.prompt
cmd = [sys.executable, 'src/wan2_inference.py', '--config=configs/wan2-2_lora_three2one_realworld.yaml',
       '--ckpt_path=/linxi/models/worldwander/wan2-2_lora_three2one_realworld.ckpt',
       '--original_video_root=/linxi/dataset/badminton/prepared/frames', '--pred_path=' + str(out), '--seed=1234',
       'model_id=/linxi/models/worldwander/Wan2.2-TI2V-5B-Diffusers', 'dataset.num_workers=0',
       'prompt=' + prompt]
round_number = args.round_number or begin_round(repo/'data.md', '第一视角生成实验')
record_condition(repo/'data.md', args.label, prompt, out)
metadata = {'round': round_number, 'command': cmd, 'commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()}
(out/'run.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
with (out/'environment.txt').open('w') as log:
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=log, check=True)
    subprocess.run(['nvidia-smi'], stdout=log, check=True)
env = dict(os.environ, PYTHONPATH=str(repo/'third_party/WorldWander'), CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1')
start = time.monotonic()
with (out/'inference.log').open('w') as log:
    result = subprocess.run(cmd, cwd=repo/'third_party/WorldWander', env=env, stdout=log, stderr=subprocess.STDOUT)
elapsed = time.monotonic()-start
metadata.update(inference_exit_code=result.returncode, wall_time_seconds=elapsed)
(out/'run.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
record_measurements(repo/'data.md', out/'inference.log', elapsed)
if result.returncode == 0:
    subprocess.run([sys.executable, str(repo/'scripts/assemble_outputs.py'),
                    '--manifest', '/linxi/dataset/badminton/prepared/manifest.json',
                    '--results', str(out)], check=True)
sys.exit(result.returncode)
