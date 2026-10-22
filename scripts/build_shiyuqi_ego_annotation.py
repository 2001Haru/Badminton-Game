"""Build explicitly approximate Shi Yu Qi ego trajectories from reviewed keyframes."""
from pathlib import Path
import sys,json,zipfile,subprocess
sys.path.insert(0,str(Path('artifacts/vipe_inspect_deps').resolve()))
import cv2,numpy as np,OpenEXR
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation,Slerp
BASE=Path('artifacts/results/vipe_20261023_000015')
OUT=Path('artifacts/annotations/shiyuqi_v1')
ann=json.loads((OUT/'keyframes.json').read_text());meta={'test_datasets':[]};report={}
def unit(x):return x/np.linalg.norm(x)
def backproject(uv,z,k):return np.array([(uv[0]-k[2])*z/k[0],(uv[1]-k[3])*z/k[1],z])
def write_video(path,frames):
 h,w=frames[0].shape[:2];tmp=path.with_suffix('.avi');v=cv2.VideoWriter(str(tmp),cv2.VideoWriter_fourcc(*'MJPG'),30,(w,h))
 for f in frames:v.write(f)
 v.release();subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(tmp),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(path)],check=True)
for name,clip in ann['clips'].items():
 out=OUT/name;out.mkdir(exist_ok=True);folder=BASE/'reconstruction'/name
 ds=np.stack([OpenEXR.File(str(f)).channels()['Z'].pixels.astype(float) for f in sorted((BASE/name/'depth_exr').glob('*.exr'))])
 cap=cv2.VideoCapture(str(folder/'rgb/exo.mp4'));rgb=[]
 while True:
  ok,f=cap.read()
  if not ok:break
  rgb.append(f)
 cap.release();k=np.load(folder/'intrinsics/exo.npz')['data'][0];yy,xx=np.mgrid[:448,:784]
 # Approximate court up from reconstructed green floor; iterative trimmed plane fit.
 median=np.median(ds,axis=0);hsv=cv2.cvtColor(rgb[0],cv2.COLOR_BGR2HSV)
 select=(hsv[:,:,0]>35)&(hsv[:,:,0]<90)&(hsv[:,:,1]>65)&(yy>210)&(yy<420)&(xx>190)&(xx<590)
 xyz=np.stack([(xx-k[2])*median/k[0],(yy-k[3])*median/k[1],median],axis=-1)
 floor=xyz[select][::8];fit=floor.copy()
 for _ in range(5):
  center=fit.mean(0);normal=np.linalg.svd(fit-center,full_matrices=False)[2][-1];err=abs((floor-center)@normal);fit=floor[err<=np.percentile(err,75)]
 up=normal if normal[1]<0 else -normal
 forward=unit(np.array([0.,0.,1.])-up*up[2]);right=unit(np.cross(forward,up))
 times=np.array([x['frame'] for x in clip['keyframes']]);heads=[];rot=[];radii=[]
 masks_for_depth=zipfile.ZipFile(folder/'mask/exo.zip');raw_head=[];torso=[]
 for key in clip['keyframes']:
  u,v=key['head_uv'];i=key['frame'];raw_head.append(float(np.median(ds[i,v-2:v+3,u-2:u+3])))
  mask=cv2.imdecode(np.frombuffer(masks_for_depth.read(f'{i:05d}.png'),np.uint8),-1)
  patch=ds[i,v+6:v+35,u-8:u+9];support=mask[v+6:v+35,u-8:u+9]==1
  torso.append(float(np.median(patch[support])))
 offset=float(np.median(np.array(raw_head)-np.array(torso)))
 for j,key in enumerate(clip['keyframes']):
  u,v=key['head_uv'];z=torso[j]+offset;heads.append(backproject((u,v),z,k));radii.append(key['head_radius_px']*z/k[0]);key.update(depth_raw_head_median=raw_head[j],depth_upper_torso=torso[j],depth_regularized=z,head_torso_offset_model_units=offset)
  yaw,pitch=np.radians([key['yaw_deg'],key['pitch_deg']]);f=unit(np.cos(pitch)*(np.cos(yaw)*forward+np.sin(yaw)*right)+np.sin(pitch)*up)
  r=unit(np.cross(f,up));down=np.cross(f,r);rot.append(np.stack([r,down,f],axis=1))
 head_xyz=PchipInterpolator(times,np.array(heads),axis=0)(np.arange(49));rotations=Slerp(times,Rotation.from_matrix(rot))(np.arange(49)).as_matrix()
 radius=np.interp(np.arange(49),times,radii);positions=head_xyz+2*radius[:,None]*rotations[:,:,2]
 c2w=np.tile(np.eye(4),(49,1,1));c2w[:,:3,:3]=rotations;c2w[:,:3,3]=positions;w2c=np.linalg.inv(c2w)
 uv=np.stack([head_xyz[:,0]/head_xyz[:,2]*k[0]+k[2],head_xyz[:,1]/head_xyz[:,2]*k[1]+k[3]],axis=1)
 np.savez(out/'trajectory.npz',c2w=c2w,w2c=w2c,head_xyz=head_xyz,head_uv=uv,frames=np.arange(49),fps=30)
 (out/'trajectory.json').write_text(json.dumps({'camera_convention':'OpenCV +x right,+y down,+z forward; world=ViPE exo camera; matrices world-to-camera','assumptions':ann['orientation_definition'],'frames':[{'frame':i,'head_uv':uv[i].tolist(),'camera_position':positions[i].tolist(),'w2c':w2c[i,:3].tolist()} for i in range(49)]},indent=2))
 egoK=np.array([[224.,0,224],[0,224.,224],[0,0,1.]])
 meta['test_datasets'].append({'exo_path':f'/linxi/dataset/badminton/short_windows/{name}/exo.mp4','ego_prior_path':f'/linxi/dataset/badminton/annotations/shiyuqi_v1/{name}/ego_prior.mp4','camera_intrinsics':[[float(k[0]),0,float(k[2])],[0,float(k[1]),float(k[3])],[0,0,1]],'camera_extrinsics':np.eye(4)[:3].tolist(),'ego_intrinsics':egoK.tolist(),'ego_extrinsics':w2c[:,:3,:].tolist()})
 overlays=[];previews=[];headsheets=[];coverage=[];selfids=[]
 masks=zipfile.ZipFile(folder/'mask/exo.zip')
 for i,im in enumerate(rgb):
  u,v=np.rint(uv[i]).astype(int);overlay=im.copy();cv2.circle(overlay,(u,v),7,(0,255,255),1)
  tip=head_xyz[i]+1.5*rotations[i,:,2];tuv=(int(tip[0]/tip[2]*k[0]+k[2]),int(tip[1]/tip[2]*k[1]+k[3]));cv2.arrowedLine(overlay,(u,v),tuv,(255,0,255),1,tipLength=.2)
  cv2.putText(overlay,f'{name} f{i}: head / estimated forward (NOT measured gaze)',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
  overlays.append(overlay);patch=overlay[max(0,v-25):v+25,max(0,u-30):u+30];patch=cv2.resize(patch,(120,100));cv2.putText(patch,str(i),(2,14),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1);headsheets.append(patch)
  # Diagnostic CPU point projection, not the official EgoPriorRenderer.
  z=ds[i];pts=np.stack([(xx-k[2])*z/k[0],(yy-k[3])*z/k[1],z],axis=-1)
  mask=cv2.imdecode(np.frombuffer(masks.read(f'{i:05d}.png'),np.uint8),-1)
  nearby=mask[v:v+32,u-5:u+6];labels,counts=np.unique(nearby[nearby>0],return_counts=True);selfid=int(labels[np.argmax(counts)]) if len(labels) else -1;selfids.append(selfid)
  use=(mask!=selfid)&~((xx<190)&(yy<65))
  p=(pts[use]-positions[i])@rotations[i];colors=im[use];positive=p[:,2]>.1;p=p[positive];colors=colors[positive]
  xy=np.rint(p[:,:2]/p[:,2,None]*224+224).astype(int);inside=(xy>=0).all(1)&(xy<448).all(1);xy=xy[inside];p=p[inside];colors=colors[inside]
  canvas=np.zeros((448,448,3),np.uint8);order=np.argsort(p[:,2])[::-1];canvas[xy[order,1],xy[order,0]]=colors[order];coverage.append(float(np.any(canvas>0,axis=2).mean()))
  cv2.putText(canvas,'Diagnostic sparse projection / not ego generation',(5,15),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1)
  previews.append(canvas)
 write_video(out/'annotation_overlay.mp4',overlays);write_video(out/'diagnostic_projection.mp4',previews)
 cv2.imwrite(str(out/'all_frames_head_check.jpg'),np.concatenate([np.concatenate(headsheets[j:j+7],axis=1) for j in range(0,49,7)],axis=0))
 cv2.imwrite(str(out/'projection_contact.jpg'),np.concatenate([cv2.resize(previews[i],(224,224)) for i in [0,12,24,36,48]],axis=1))
 err=np.linalg.norm(uv[times]-np.array([x['head_uv'] for x in clip['keyframes']]),axis=1)
 report[name]={'keyframes':len(times),'output_frames':49,'rotation_det_min':float(np.linalg.det(rotations).min()),'orthogonality_max_abs':float(np.max(abs(rotations.transpose(0,2,1)@rotations-np.eye(3)))),'w2c_c2w_identity_error':float(np.max(abs(w2c@c2w-np.eye(4)))),'keyframe_reprojection_error_max_px':float(err.max()),'head_height_above_fitted_plane_range':[(float(x)) for x in [np.min((head_xyz-center)@up),np.max((head_xyz-center)@up)]],'plane_fit_residual_median_model_units':float(np.median(abs((fit-center)@up))),'diagnostic_coverage_range':[min(coverage),max(coverage)],'self_mask_ids':sorted(set(selfids)),'orientation_confidence':'low, manual hypothesis, roll zero; 90 degree perspective FOV assumption'}
for name in report:
 t=np.load(OUT/name/'trajectory.npz');c=t['c2w'];r=Rotation.from_matrix(c[:,:3,:3]);report[name]['max_rotation_step_deg']=float(np.max((r[:-1].inv()*r[1:]).magnitude())*180/np.pi);report[name]['max_camera_translation_step_model_units']=float(np.max(np.linalg.norm(np.diff(c[:,:3,3],axis=0),axis=1)))
(OUT/'meta.json').write_text(json.dumps(meta,indent=2));(OUT/'keyframes_with_depth.json').write_text(json.dumps(ann,indent=2));(OUT/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
