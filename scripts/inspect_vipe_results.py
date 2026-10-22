from pathlib import Path
import sys,zipfile,json,subprocess
sys.path.insert(0,str(Path('artifacts/vipe_inspect_deps').resolve()))
import numpy as np,cv2,OpenEXR
root=Path('artifacts/results/vipe_20261023_000015');report={}
for folder in (root/'reconstruction').iterdir():
 out=root/folder.name;out.mkdir(exist_ok=True)
 with zipfile.ZipFile(folder/'depth/exo.zip') as archive:archive.extractall(out/'depth_exr')
 depths=np.stack([OpenEXR.File(str(f)).channels()['Z'].pixels.astype(np.float32) for f in sorted((out/'depth_exr').glob('*.exr'))])
 cap=cv2.VideoCapture(str(folder/'rgb/exo.mp4'));frames=[]
 while True:
  ok,frame=cap.read()
  if not ok:break
  frames.append(frame)
 cap.release()
 valid=np.isfinite(depths)&(depths>0);lo,hi=np.percentile(depths[valid],[2,98])
 poses=np.load(folder/'pose/exo.npz')['data'];intr=np.load(folder/'intrinsics/exo.npz')['data']
 report[folder.name]={'depth_frames':len(depths),'rgb_frames':len(frames),'depth_shape':list(depths.shape),'valid_positive_fraction':float(valid.mean()),'depth_p2_p98':[float(lo),float(hi)],'pose_max_difference_from_first':float(np.max(np.abs(poses-poses[0]))),'intrinsics_first':intr[0].tolist(),'intrinsics_max_difference':float(np.max(np.abs(intr-intr[0])))}
 panels=[]
 for i,(rgb,d) in enumerate(zip(frames,depths)):
  color=cv2.applyColorMap(np.uint8(np.clip((d-lo)/(hi-lo),0,1)*255),cv2.COLORMAP_TURBO)
  color=cv2.resize(color,(rgb.shape[1],rgb.shape[0]))
  panel=np.concatenate([rgb,color],axis=1)
  cv2.putText(panel,f'{folder.name} frame {i} | RGB / Depth (fixed scale: near blue, far red)',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
  panels.append(panel)
 writer=cv2.VideoWriter(str(out/'preview.avi'),cv2.VideoWriter_fourcc(*'MJPG'),30,(panels[0].shape[1],panels[0].shape[0]))
 for panel in panels:writer.write(panel)
 writer.release()
 subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(out/'preview.avi'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(out/'rgb_depth.mp4')],check=True)
 cv2.imwrite(str(out/'contact.jpg'),np.concatenate([cv2.resize(panels[i],(1176,336)) for i in [0,12,24,36,48]],axis=0))
(root/'inspection_metrics.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
