"""Run unmodified official perspective Ego Prior rendering for both annotated windows."""
from pathlib import Path
import os,json,subprocess,time,datetime
from zoneinfo import ZoneInfo
from experiment_record import begin_round
repo=Path('/linxi/Badminton-Game');root=Path('/linxi/dataset/badminton')
out=root/'ego_prior_runs'/datetime.datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True)
round_id=begin_round(repo/'data.md','两段石宇奇官方 Ego Prior 粗渲染',fields=['输入：ViPE 第 4 轮重建及 shiyuqi_v1 近似头部轨迹；每段 49 帧、30fps。','渲染器：EgoX-EgoPriorRenderer 官方 scripts/render_vipe_pointcloud.py，未修改。','参数：透视投影，448×448，90° 视场，point_size=5.0，use_mean_bg=True，包含动态点云，不启用鱼眼；无生成 prompt。','PyTorch3D：v0.7.9（33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba），PyTorch 2.5.1/CUDA 12.4。','输出：'+str(out)])
state={'round':round_id,'output':str(out),'clips':[]};print('OUTPUT='+str(out),flush=True)
for name in ['clip1_t02_49f','clip2_t04_49f']:
 cmd=['/linxi/tmp/vipe/venv/bin/python','scripts/render_vipe_pointcloud.py','--input_dir',str(root/'vipe_runs/20261023_000015/reconstruction'/name),'--out_dir',str(out/'official'),'--meta_json_path',str(root/'annotations/shiyuqi_v1/meta.json'),'--point_size','5.0','--start_frame','0','--end_frame','48','--use_mean_bg']
 item={'name':name,'command':cmd,'status':'running'};state['clips'].append(item);(out/'run.json').write_text(json.dumps(state,indent=2));start=time.monotonic()
 with (out/(name+'.log')).open('w') as f:r=subprocess.run(cmd,cwd=repo/'third_party/EgoPriorRenderer',stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0'))
 item.update(status='completed' if r.returncode==0 else 'failed',exit_code=r.returncode,seconds=time.monotonic()-start);(out/'run.json').write_text(json.dumps(state,indent=2))
 if r.returncode:raise SystemExit(r.returncode)
 with (repo/'data.md').open('a',encoding='utf-8') as f:f.write(f'\n- {name}：渲染耗时 {item["seconds"]:.1f} 秒，退出码 0。\n')
print('Both official renders completed',flush=True)
