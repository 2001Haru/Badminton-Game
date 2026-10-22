"""Persist only Badminton project runtime prerequisites before node replacement."""
from pathlib import Path
import subprocess,tarfile,json,shutil,re,hashlib,datetime
root=Path('/linxi/dataset/badminton/migration/20261023');root.mkdir(parents=True,exist_ok=True)
base=Path('/opt/conda');stdlib=base/'lib/python3.10'
files={base/'bin/python3.10'}
for path in stdlib.rglob('*'):
 if path.is_file() and 'site-packages' not in path.parts and '__pycache__' not in path.parts:files.add(path)
# Preserve only base-Python libraries actually required by this interpreter/stdlib.
for path in list(files):
 if path.suffix=='.so' or path.name=='python3.10':
  result=subprocess.run(['ldd',str(path)],capture_output=True,text=True)
  for dep in re.findall(r'(/opt/conda/[^\s()]+)',result.stdout):
   p=Path(dep)
   if p.is_file():files.add(p)
archive=root/'python310-runtime.tar.gz'
with tarfile.open(archive,'w:gz',dereference=True) as tar:
 for path in sorted(files):tar.add(path,arcname=str(path.relative_to(base)),recursive=False)
for name,env in [('egox','/linxi/tmp/egox/venv'),('vipe','/linxi/tmp/vipe/venv'),('worldwander','/linxi/tmp/worldwander/venv-conda')]:
 result=subprocess.run([env+'/bin/python','-m','pip','freeze','--all'],capture_output=True,text=True)
 (root/(name+'-pip-freeze.txt')).write_text(result.stdout)
 if result.returncode:raise RuntimeError(result.stderr)
 shutil.copy2(env+'/pyvenv.cfg',root/(name+'-pyvenv.cfg'))
for wheel in Path('/root/.cache/pip/wheels').rglob('pytorch3d-*.whl'):shutil.copy2(wheel,root/wheel.name)
for name,cmd in [('system-packages.txt',['dpkg-query','-W']),('gpu-driver.txt',['nvidia-smi']),('cuda-version.txt',['/usr/local/cuda/bin/nvcc','--version']),('mounts.txt',['findmnt','-T','/linxi'])]:
 r=subprocess.run(cmd,capture_output=True,text=True);(root/name).write_text(r.stdout+r.stderr)
(root/'README.txt').write_text('''Badminton project migration snapshot, 2026-10-23 Asia/Singapore.
Persistent GPFS: /linxi including /linxi/tmp; preserve and mount the same volume on the new node.
Project: /linxi/Badminton-Game (GitHub); ignored experiment journal data.md is here and on local Windows.
Data/results/annotations/input packages: /linxi/dataset/badminton.
All model roots: /linxi/models/worldwander, including egox and vipe.
Existing envs: /linxi/tmp/worldwander/venv-conda, /linxi/tmp/egox/venv, /linxi/tmp/vipe/venv.
These envs link to /opt/conda/bin/python (Python 3.10.15), not persistent by default.
python310-runtime.tar.gz backs up base interpreter, stdlib (excluding site-packages), and referenced Conda shared libraries only. The existing persistent venvs retain their installed packages.
To restore, prefer the same base image/Python 3.10 path; archive is recovery material, NOT a tested portable environment. Do not overwrite a different new-node Conda installation blindly.
Need compatible NVIDIA driver and CUDA runtime; rebuilding CUDA extensions may be needed for a different GPU/CUDA. ViPE build used CUDA 12.4, TORCH_CUDA_ARCH_LIST=8.0, Eigen; OpenCV needs libgl1/libglib2.0-0. Record packages provided.
PyTorch3D compiled wheel copied from ephemeral pip cache. ViPE extension and source/build tree already live under /linxi.
No other projects were backed up or modified. SSH backup remains /linxi/.ssh_backup; private keys are not included in this archive.
''')
manifest={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in root.iterdir() if p.is_file() and p.name!='manifest.json'}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'directory':str(root),'files':len(manifest),'runtime_archive_bytes':archive.stat().st_size},indent=2))
