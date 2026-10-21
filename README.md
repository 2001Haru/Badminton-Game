# Badminton exo-to-ego experiment

WorldWander upstream: https://github.com/showlab/WorldWander
Vendored commit: c9f12e85730d3976f1e89df5ee8248f7f3575a7a.
Inner Git metadata is archived outside the source tree in ignored artifacts/.

Source: https://www.youtube.com/watch?v=_lcVkJifJnA
Segments: 27:35–27:42 and 29:33–29:49.
Target: Shi Yu Qi, the player wearing red.

Code changes originate locally, then GitHub push, then cloud pull.
Launch records belong in ignored data.md. Media and weights are not committed.
Cloud dataset: /linxi/dataset/badminton
Cloud weights: /linxi/models/worldwander

Assess viewpoint/identity, body, racket, shuttlecock, temporal consistency separately.
No paired ground-truth ego video exists; plausibility is not reconstruction accuracy.

## First round

Install PyTorch 2.5.1 + torchvision 0.20.1 (CUDA 12.4) in a separate environment,
then `pip install -r requirements-inference.txt`.
Run `scripts/download_models.py --output /linxi/models/worldwander`.
Run `scripts/prepare_clips.py --source /linxi/dataset/badminton --output /linxi/dataset/badminton/prepared`.
Run `python scripts/run_round.py` from the cloud clone.

The 7s and 16s clips are decoded at 8 fps and center-cropped from 1280x720 to
1280x704, matching the official real-world resolution. There are 2 and 3
49-frame windows respectively, with one shared frame at each boundary. The
last window repeats its final frame as padding, trimmed on assembly. Windows
are generated independently; seams at 6s (and 12s) are a separate limitation.
This baseline cannot evaluate shuttle trajectories above the 8 fps sampling rate.
Each window produces an ego-only video and an input/output comparison.
The full outputs preserve 7s and 16s durations; generated outputs contain no audio.
