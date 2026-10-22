"""Run the official real-world exo-to-ego example with its official prompt."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
from experiment_record import begin_round, record_condition, record_measurements

repo = Path(__file__).resolve().parents[1]
vendor = repo/'third_party/WorldWander'
stamp = datetime.datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y%m%d_%H%M%S')
output = Path('/linxi/dataset/badminton/results')/('official_selftests_'+stamp)/'worldwander'
output.mkdir(parents=True)
number = begin_round(repo/'data.md', '官方样例自检', fields=[
    'WorldWander：官方 realworld_scenarios/Third_Video/case1，配套 First_Video/case1 作为参考。',
    'WorldWander 参数：three2one_realworld；seed=1234；50 steps；CFG=5；1280×704；49 帧；输出 8 fps。',
])
record_condition(repo/'data.md', 'WorldWander 官方真实场景样例', 'Transform it into the first-person perspective.', output)
command = [sys.executable,'src/wan2_inference.py',
    '--config=configs/wan2-2_lora_three2one_realworld.yaml',
    '--ckpt_path=/linxi/models/worldwander/wan2-2_lora_three2one_realworld.ckpt',
    '--original_video_root=examples/realworld_scenarios/Third_Video',
    '--pred_path='+str(output),'--seed=1234',
    'model_id=/linxi/models/worldwander/Wan2.2-TI2V-5B-Diffusers','dataset.num_workers=0']
metadata = {'round':number,'command':command,'output':str(output)}
(output/'run.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(json.dumps(metadata),flush=True)
env = dict(os.environ,PYTHONPATH=str(vendor),CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1')
started=time.monotonic()
with (output/'inference.log').open('w') as log:
    process=subprocess.run(command,cwd=vendor,env=env,stdout=log,stderr=subprocess.STDOUT)
elapsed=time.monotonic()-started
metadata.update(exit_code=process.returncode,wall_time_seconds=elapsed)
(output/'run.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
record_measurements(repo/'data.md',output/'inference.log',elapsed)
if process.returncode==0:
    import imageio.v2 as iio
    import numpy as np
    from PIL import Image
    for source,name in [('First_Video','official_ego_reference'),('Third_Video','official_exo_input')]:
        frames=sorted((vendor/'examples/realworld_scenarios'/source/'case1').glob('*.png'))
        with iio.get_writer(str(output/(name+'.mp4')),fps=8,codec='libx264',macro_block_size=None) as writer:
            for frame in frames:
                writer.append_data(np.asarray(Image.open(frame).convert('RGB')))
print('EXIT_CODE='+str(process.returncode),flush=True)
sys.exit(process.returncode)
