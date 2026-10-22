"""Inspect original official renderer output; create explicitly rotation-corrected previews."""
import sys,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path('artifacts/vipe_inspect_deps').resolve()))
import cv2,numpy as np
root=Path(sys.argv[1]);report={}
for name in ['clip1_t02_49f','clip2_t04_49f']:
 folder=root/'official'/name;source=folder/'ego_Prior.mp4';upright=folder/'ego_Prior_upright.mp4'
 subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(source),'-vf','transpose=2','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(upright)],check=True)
 cap=cv2.VideoCapture(str(upright));fps=cap.get(cv2.CAP_PROP_FPS);frames=[]
 while True:
  ok,f=cap.read()
  if not ok:break
  frames.append(f)
 cap.release();assert len(frames)==49
 report[name]={'frames':len(frames),'fps':fps,'width':frames[0].shape[1],'height':frames[0].shape[0],'near_black_fraction_min_max':[float(fn([np.mean(np.max(f,axis=2)<10) for f in frames])) for fn in [min,max]],'upright_transform':'90 degrees counterclockwise, undo official hardcoded clockwise rotation'}
 rows=[]
 exo=cv2.VideoCapture(f'artifacts/results/vipe_20261023_000015/reconstruction/{name}/rgb/exo.mp4')
 for i in [0,12,24,36,48]:
  exo.set(cv2.CAP_PROP_POS_FRAMES,i);ok,src=exo.read();src=cv2.resize(src,(784,448));pair=np.concatenate([src,frames[i]],axis=1);cv2.putText(pair,f'{name} f{i}: exo / official ego prior, upright',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1);rows.append(cv2.resize(pair,(924,336)))
 exo.release();cv2.imwrite(str(folder/'contact.jpg'),np.concatenate(rows,axis=0))
(root/'inspection.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
