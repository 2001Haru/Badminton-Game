"""Number experimental rounds; keep operational details in per-run logs."""
import re
from pathlib import Path


def begin_round(path: Path, title: str) -> int:
    existing = path.read_text(encoding='utf-8') if path.exists() else '# 实验记录\n'
    numbers = [int(n) for n in re.findall(r'^## (\d+)(?:\s|$)', existing, re.MULTILINE)]
    number = max(numbers, default=0) + 1
    with path.open('w', encoding='utf-8') as log:
        log.write(existing.rstrip() + f'\n\n## {number} {title}\n\n')
        log.write('- 模型：WorldWander three2one_realworld；基座 Wan2.2-TI2V-5B。\n')
        log.write('- 输入：_lcVkJifJnA，27:35–27:42（7 秒）、29:33–29:49（16 秒）。\n')
        log.write('- 参数：seed=1234；50 steps；CFG=5；1280×704；8 fps；49 帧/窗口，共 5 个窗口。\n')
    return number


def record_condition(path: Path, label: str, prompt: str, output: Path):
    with path.open('a', encoding='utf-8') as log:
        log.write(f'\n### {label}\n\n- Prompt：{prompt}\n- 输出：{output}\n')


def record_measurements(path: Path, inference_log: Path, elapsed: float):
    text = inference_log.read_text(encoding='utf-8', errors='replace')
    measurements = re.findall(r'Completed (\S+) in ([\d.]+)s; peak CUDA allocation ([\d.]+) GiB', text)
    if measurements:
        with path.open('a', encoding='utf-8') as log:
            log.write(f'- 推理总耗时（含加载）：{elapsed:.1f} 秒。\n')
            log.write('- 窗口耗时（按输入顺序，秒）：' + ' / '.join(x[1] for x in measurements) + '。\n')
            log.write('- 峰值 CUDA 分配：' + str(max(float(x[2]) for x in measurements)) + ' GiB。\n')
