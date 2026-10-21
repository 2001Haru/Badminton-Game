"""Trim padding/overlap; concatenate independently generated windows without smoothing."""
import argparse
import json
from pathlib import Path
import imageio.v2 as iio

p = argparse.ArgumentParser()
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--results', type=Path, required=True)
a = p.parse_args()
m = json.loads(a.manifest.read_text())
for clip in m['clips']:
    for suffix in ['', '_ego']:
        dest = a.results / (clip['name'] + suffix + '_full.mp4')
        count = 0
        with iio.get_writer(str(dest), fps=m['fps'], codec='libx264', macro_block_size=None) as writer:
            for n, w in enumerate(clip['windows']):
                with iio.get_reader(str(a.results / (w['name'] + suffix + '.mp4'))) as reader:
                    for i, frame in enumerate(reader):
                        if i >= w['valid']:
                            break
                        if n and i == 0:
                            continue
                        writer.append_data(frame)
                        count += 1
        if count != clip['frames']:
            raise ValueError(f'{dest}: expected {clip["frames"]} frames, got {count}')
        print(dest, count, 'frames', flush=True)
