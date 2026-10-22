"""Package current unchanged annotations/priors for official EgoX depth-path convention."""
from pathlib import Path
import json,shutil,zipfile,sys
import numpy as np
repo=Path('/linxi/Badminton-Game');data=Path('/linxi/dataset/badminton');out=data/'egox_badminton_v1';out.mkdir(exist_ok=True)
sys.path.insert(0,str(repo/'third_party/EgoPriorRenderer/scripts'))
from convert_depth_zip_to_npy import read_exr_depth
meta=json.loads((data/'annotations/shiyuqi_v1/meta.json').read_text())
for sample in meta['test_datasets']:
 name=Path(sample['exo_path']).parent.name;video=out/'videos'/name;video.mkdir(parents=True,exist_ok=True)
 shutil.copy2(sample['exo_path'],video/'exo.mp4')
 shutil.copy2(data/'ego_prior_runs/20261023_003437/official'/name/'ego_Prior_upright.mp4',video/'ego_Prior.mp4')
 sample['exo_path']=f'./example/badminton_v1/videos/{name}/exo.mp4';sample['ego_prior_path']=f'./example/badminton_v1/videos/{name}/ego_Prior.mp4';sample['prompt']=(data/'short_windows'/name/'prompt_draft.txt').read_text()
 depth=out/'depth_maps'/name;depth.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(data/'vipe_runs/20261023_000015/reconstruction'/name/'depth/exo.zip') as archive:
  for file in sorted(archive.namelist()):np.save(depth/(Path(file).stem+'.npy'),read_exr_depth(archive.read(file)).astype(np.float32))
 assert len(list(depth.glob('*.npy')))==49 and len(sample['ego_extrinsics'])==49
 for file in depth.glob('*.npy'):
  d=np.load(file);assert d.shape==(448,784) and np.isfinite(d).all() and (d>0).all()
 print('Prepared',name,flush=True)
(out/'meta.json').write_text(json.dumps(meta,indent=2))
link=repo/'third_party/EgoX/example/badminton_v1'
if not link.exists():link.symlink_to(out,target_is_directory=True)
assert link.resolve()==out
print('INPUTS_READY='+str(out),flush=True)
