"""Extract the two agreed 49-frame windows without interpolation or padding."""
import argparse
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--ffmpeg',default='ffmpeg')
p.add_argument('--ffprobe',default='ffprobe')
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
manifest=[]
for source,start,broadcast,name in [
    ('_lcVkJifJnA_1655_1662.mp4',60,1655,'clip1_t02_49f'),
    ('_lcVkJifJnA_1773_1789.mp4',120,1773,'clip2_t04_49f')]:
    src=a.source/source
    meta=json.loads(subprocess.check_output([a.ffprobe,'-v','error','-select_streams','v:0','-show_streams','-of','json',str(src)]))['streams'][0]
    assert meta['r_frame_rate']=='30/1',meta['r_frame_rate']
    folder=a.output/name
    folder.mkdir(exist_ok=True)
    target=folder/'source_30fps.mp4'
    subprocess.run([a.ffmpeg,'-v','error','-y','-i',str(src),'-vf',f'trim=start_frame={start}:end_frame={start+49},setpts=PTS-STARTPTS',
        '-an','-c:v','libx264','-crf','0','-preset','fast','-fps_mode','passthrough',str(target)],check=True)
    info=json.loads(subprocess.check_output([a.ffprobe,'-v','error','-select_streams','v:0','-count_frames','-show_streams','-of','json',str(target)]))['streams'][0]
    assert int(info['nb_read_frames'])==49 and info['r_frame_rate']=='30/1',info
    manifest.append({'name':name,'source':source,'source_first_frame':start,'source_last_frame':start+48,
        'fps':30,'frames':49,'first_to_last_seconds':1.6,'encoded_duration_seconds':49/30,
        'broadcast_start_seconds':broadcast+start/30,'broadcast_last_frame_seconds':broadcast+(start+48)/30,
        'target':'Shi Yu Qi, red shirt','padding':False,'interpolation':False})
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
