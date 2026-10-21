"""Run locally or on cloud. Copy completed files if cloud HF access is blocked."""
import argparse
from huggingface_hub import hf_hub_download, snapshot_download
p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a = p.parse_args()
hf_hub_download('QuintonSung/WorldWander', 'wan2-2_lora_three2one_realworld.ckpt', local_dir=a.output)
snapshot_download('Wan-AI/Wan2.2-TI2V-5B-Diffusers', local_dir=a.output + '/Wan2.2-TI2V-5B-Diffusers',
                  ignore_patterns=['*.md', '*.png', '*.jpg'], max_workers=4)
