"""Decode at 8 fps, form 49-frame windows with one shared boundary frame."""
import argparse
import json
import subprocess
from pathlib import Path
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--ffmpeg', default='ffmpeg')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
manifest = {'fps': 8, 'height': 704, 'width': 1280, 'window_frames': 49, 'stride': 48, 'clips': []}
for name, duration in [('_lcVkJifJnA_1655_1662', 7), ('_lcVkJifJnA_1773_1789', 16)]:
    raw = subprocess.run([a.ffmpeg, '-v', 'error', '-i', str(a.source / (name + '.mp4')),
        '-vf', 'fps=8,scale=1280:720,crop=1280:704', '-frames:v', str(duration * 8),
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], check=True, stdout=subprocess.PIPE).stdout
    size = 1280 * 704 * 3
    assert len(raw) == duration * 8 * size, (name, len(raw))
    frames = [raw[i:i+size] for i in range(0, len(raw), size)]
    clip = {'name': name, 'duration': duration, 'frames': len(frames), 'windows': []}
    for start in range(0, len(frames) - 1, 48):
        window = f'{name}_f{start:04d}'
        dest = a.output / 'frames' / window
        dest.mkdir(parents=True, exist_ok=True)
        for j in range(49):
            Image.frombytes('RGB', (1280, 704), frames[min(start+j, len(frames)-1)]).save(dest / f'{j:05d}.png')
        clip['windows'].append({'name': window, 'start': start, 'valid': min(49, len(frames)-start)})
    manifest['clips'].append(clip)
(a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest, indent=2))
