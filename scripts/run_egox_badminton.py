"""Run two badminton samples through the previously reproduced official EgoX pipeline."""
from pathlib import Path
import subprocess,datetime,os,json,time
from zoneinfo import ZoneInfo
from experiment_record import begin_round
repo=Path('/linxi/Badminton-Game');data=Path('/linxi/dataset/badminton');models=Path('/linxi/models/worldwander/egox')
subprocess.run(['/linxi/tmp/vipe/venv/bin/python',str(repo/'scripts/prepare_egox_badminton.py')],check=True)
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied; refusing overlapping inference')
out=data/'results'/('egox_badminton_'+datetime.datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True)
meta=json.loads((data/'egox_badminton_v1/meta.json').read_text());(out/'input_meta.json').write_text(json.dumps(meta,indent=2))
number=begin_round(repo/'data.md','两段羽毛球当前先验直接进入官方 EgoX',fields=['输入：两段 49 帧/30fps 窗口；红衣石宇奇；采用 shiyuqi_v1 原轨迹与第 5 轮转正官方粗渲染；不修朝向、不清除记分牌。','模型：Wan2.1-I2V-14B-480P + EgoX rank256；50 steps、CFG5、GGA=True、cos_sim_scaling_factor=3.0、seed=846514；--in_the_wild。','输出：exo+ego 并排 1232×448，ego 为 448×448；49 帧/30fps。','条件限制：官方 GGA 保留硬编码 Aria 畸变参数，与当前 90° 透视粗渲染不完全一致；本轮按用户要求不修改官方几何实现。','完整输入、提示词及参数：'+str(out/'input_meta.json')])
with (repo/'data.md').open('a',encoding='utf-8') as f:
 for sample in meta['test_datasets']:f.write('\n### '+Path(sample['exo_path']).parent.name+'\n\n```text\n'+sample['prompt'].strip()+'\n```\n')
cmd=['/linxi/tmp/egox/venv/bin/python','infer.py','--meta_data_file','./example/badminton_v1/meta.json','--model_path',str(models/'Wan2.1-I2V-14B-480P-Diffusers'),'--lora_path',str(models/'EgoX/pytorch_lora_weights.safetensors'),'--lora_rank','256','--out',str(out),'--seed','846514','--use_GGA','--cos_sim_scaling_factor','3.0','--in_the_wild','--idx','-1']
state={'round':number,'command':cmd,'output':str(out),'status':'running'};(out/'run.json').write_text(json.dumps(state,indent=2));print('OUTPUT='+str(out),flush=True)
start=time.monotonic()
with (out/'inference.log').open('w') as log:result=subprocess.run(cmd,cwd=repo/'third_party/EgoX',env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT)
state.update(status='completed' if result.returncode==0 else 'failed',exit_code=result.returncode,seconds=time.monotonic()-start);(out/'run.json').write_text(json.dumps(state,indent=2))
if result.returncode==0:
 with (repo/'data.md').open('a',encoding='utf-8') as f:f.write(f'\n- 两段合计耗时（含加载）：{state["seconds"]:.1f} 秒。\n')
print('EXIT_CODE='+str(result.returncode),flush=True)
raise SystemExit(result.returncode)
