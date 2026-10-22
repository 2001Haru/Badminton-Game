"""Run official lyra on both approved fixed-camera windows, without ego pose annotation."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
from experiment_record import begin_round

repo=Path(__file__).resolve().parents[1]
root=Path('/linxi/dataset/badminton/short_windows')
deadline=time.monotonic()+24*3600
assets=Path('/linxi/models/worldwander/vipe/torch/hub')
required_assets=['geocalib/pinhole.tar','sam/sam_vit_b_01ec64.pth','aot/R50_DeAOTL_PRE_YTB_DAV.pth','droid_slam/droid.pth']
while not (all(Path('/linxi/tmp/vipe',f).exists() for f in ['environment.ready','hf-weights.ready'])
           and all((assets/f).is_file() and (assets/f).stat().st_size>1_000_000 for f in required_assets)):
    if time.monotonic()>deadline:raise TimeoutError('ViPE environment/weights not ready')
    time.sleep(20)
names=['clip1_t02_49f','clip2_t04_49f']
for name in names:
    audit=json.loads((root/name/'camera_audit.json').read_text())
    if not audit['fixed_camera_candidate']:raise ValueError('Camera motion requires review: '+name)
while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
    if time.monotonic()>deadline:raise TimeoutError('GPU remains occupied')
    time.sleep(20)
stamp=datetime.datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y%m%d_%H%M%S')
resume=os.environ.get('VIPE_RESUME_OUTPUT')
out=Path(resume) if resume else Path('/linxi/dataset/badminton/vipe_runs')/stamp
out.mkdir(parents=True,exist_ok=bool(resume))
if resume:
    previous=(out/'run.json').read_text()
    number=json.loads(previous)['round']
    (out/('run-before-'+stamp+'.json')).write_text(previous)
    for old_log in out.glob('clip*.log'):
        old_log.rename(old_log.with_name(old_log.stem+'-before-'+stamp+'.log'))
else:
    number=begin_round(repo/'data.md','两个短窗口的 ViPE 场景重建',fields=[
    '方法：EgoX-EgoPriorRenderer，lyra（MoGe2 + Video Depth Anything），固定外部机位假设。',
    '输入：短片段 1 的 00:02 起、短片段 2 的 00:04 起；各 49 帧，30 fps，784×448，无补帧。',
    '目标人物：红衣石宇奇；本轮仅重建场景，不标注 ego 轨迹、不运行视频生成。',
    '运行版本：PyTorch 2.5.1 / CUDA 12.4 兼容环境；不同于官方 2.7.0 / CUDA 12.8 锁定版本。',
    '输出：'+str(out),
])
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_ENDPOINT='https://hf-mirror.com',
    HF_HOME='/linxi/models/worldwander/vipe/hf',TORCH_HOME='/linxi/models/worldwander/vipe/torch',
    TORCH_EXTENSIONS_DIR='/linxi/tmp/vipe/extensions',PYTHONUNBUFFERED='1',
    TOKENIZERS_PARALLELISM='false')
state={'round':number,'output':str(out),'clips':[]}
print('OUTPUT='+str(out),flush=True)
for name in names:
    command=['/linxi/tmp/vipe/venv/bin/vipe','infer',str(root/name/'exo.mp4'),
             '--start_frame','0','--end_frame','48','--assume_fixed_camera_pose',
             '--pipeline','lyra','--output',str(out/'reconstruction')]
    entry={'name':name,'command':command,'status':'running'}
    state['clips'].append(entry);(out/'run.json').write_text(json.dumps(state,indent=2))
    start=time.monotonic()
    with (out/(name+'.log')).open('w') as log:
        result=subprocess.run(command,cwd=repo/'third_party/EgoPriorRenderer',env=env,stdout=log,stderr=subprocess.STDOUT)
    entry.update(status='completed' if result.returncode==0 else 'failed',exit_code=result.returncode,seconds=time.monotonic()-start)
    (out/'run.json').write_text(json.dumps(state,indent=2))
    if result.returncode:
        print('FAILED '+name,flush=True)
        sys.exit(result.returncode)
    audit=json.loads((root/name/'camera_audit.json').read_text())
    with (repo/'data.md').open('a',encoding='utf-8') as log:
        log.write(f'\n### {name}\n\n- 耗时：{entry["seconds"]:.1f} 秒。\n')
        log.write(f'- 背景抽查帧最大拟合角点位移：{max(x["max_corner_drift_px"] for x in audit["measurements"]):.4f} 像素（784×448）。\n')
print('Both reconstructions completed',flush=True)
