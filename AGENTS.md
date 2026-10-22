# Development rules

- Modify code in the local Windows repository, push to GitHub, then pull on cloud.
- Never commit data.md. Use consecutive numeric headings (1, 2, 3, ...) for experimental rounds; multiple comparison groups in one round share a heading. Record experimental inputs, prompts, parameters, measurements, outputs, and qualitative observations of the generated videos. Clearly label qualitative observations and their inspection scope, separate them from measured data, and avoid unsupported speculation. Keep code errors, upload progress, launch commands and tracking notes out of data.md; retain operational details in run logs instead.
- Keep datasets and generated outputs out of Git. Cloud data lives in /linxi/dataset/badminton.
- Model files live in /linxi/models/worldwander and are not committed.
- First-round target is Shi Yu Qi (red shirt), not whichever player is easiest to generate.
- Keep failure assessment separate for viewpoint, anatomy, racket, shuttle and time.
