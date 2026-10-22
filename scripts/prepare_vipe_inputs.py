"""Derive official-size inputs and measure static-background image motion."""
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np
import imageio_ffmpeg

root=Path('/linxi/dataset/badminton/short_windows')
for name in ['clip1_t02_49f','clip2_t04_49f']:
    folder=root/name
    output=folder/'exo.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-i',str(folder/'source_30fps.mp4'),
                    '-vf','scale=784:448','-an','-c:v','libx264','-crf','0','-pix_fmt','yuv420p',str(output)],check=True)
    cap=cv2.VideoCapture(str(output))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==49
    assert abs(cap.get(cv2.CAP_PROP_FPS)-30)<1e-6
    frames=[]
    while True:
        ok,frame=cap.read()
        if not ok:break
        frames.append(frame)
    cap.release()
    h,w=frames[0].shape[:2]
    mask=np.zeros((h,w),np.uint8);mask[int(.12*h):int(.43*h),:]=255
    mask[:int(.22*h),:int(.3*w)]=0
    detector=cv2.SIFT_create(nfeatures=2500)
    key0,desc0=detector.detectAndCompute(cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY),mask)
    rows=[]
    for index in [12,24,36,48]:
        key,desc=detector.detectAndCompute(cv2.cvtColor(frames[index],cv2.COLOR_BGR2GRAY),mask)
        pairs=cv2.BFMatcher().knnMatch(desc0,desc,k=2)
        matches=[x[0] for x in pairs if len(x)==2 and x[0].distance<.7*x[1].distance]
        p0=np.float32([key0[x.queryIdx].pt for x in matches]);p1=np.float32([key[x.trainIdx].pt for x in matches])
        transform,inliers=cv2.estimateAffinePartial2D(p0,p1,method=cv2.RANSAC,ransacReprojThreshold=2)
        if transform is None:raise RuntimeError('Background registration failed')
        corners=np.float32([[0,0],[w,0],[0,h],[w,h]]).reshape(1,-1,2)
        drift=np.linalg.norm(cv2.transform(corners,transform)-corners,axis=-1).max()
        rows.append({'frame':index,'matches':len(matches),'inliers':int(inliers.sum()),'max_corner_drift_px':float(drift),'affine':transform.tolist()})
    audit={'size':[w,h],'frames':49,'fps':30,'background_roi':'y=12%-43%, excluding upper-left scoreboard',
           'measurements':rows,'fixed_camera_candidate':all(x['inliers']>=30 and x['max_corner_drift_px']<=3 for x in rows),
           'interpretation':'Image-based screening only, not proof of zero camera motion or calibrated intrinsics.'}
    (folder/'camera_audit.json').write_text(json.dumps(audit,indent=2))
    print(name,json.dumps(audit),flush=True)
