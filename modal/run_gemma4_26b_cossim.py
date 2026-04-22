"""E2E logits cos-sim: SDPA vs triton_gqa on Gemma-4-26B-A4B (B200).

One forward each (SDPA + triton), N=2048 with 2 image spans, prints cos sim.

Usage:
    export HF_TOKEN=hf_...
    modal run modal/run_gemma4_26b_cossim.py

First run downloads ~52GB into a Modal Volume (~5 min). Subsequent runs reuse.
"""
from __future__ import annotations

import pathlib

import modal


CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu22.04"
KERNEL_DIR = pathlib.Path(__file__).parent.parent  # /mnt/.../kernel

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    .pip_install(
        "torch==2.9.1",
        "triton==3.5.1",
        "transformers==5.5.4",
        "accelerate>=1.0",
        "hf_transfer",
        "numpy",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/huggingface"})
    .workdir("/root")
    # Mount flash_attn under BOTH names: bench imports `gemma_triton_flash_attn`
    # (pyproject remaps that to flash_attn dir), but the kernel internally uses
    # relative imports rooted at flash_attn. Easiest to expose both.
    .add_local_dir(local_path=str(KERNEL_DIR / "flash_attn"),
                   remote_path="/root/flash_attn")
    .add_local_dir(local_path=str(KERNEL_DIR / "flash_attn"),
                   remote_path="/root/gemma_triton_flash_attn")
    .add_local_dir(local_path=str(KERNEL_DIR / "benchmarks"),
                   remote_path="/root/benchmarks")
)

hf_cache_vol = modal.Volume.from_name("hf-cache-gemma4", create_if_missing=True)

app = modal.App("kernel-b200-gemma4-cossim", image=image)


@app.function(
    gpu="B200",
    timeout=60 * 60,                 # generous: first download is the long pole
    volumes={"/cache/huggingface": hf_cache_vol},
    secrets=[modal.Secret.from_local_environ(["HF_TOKEN"])],
)
def run_cossim():
    import os
    import subprocess
    import sys

    import torch
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"[smoke] device: {name}  cap: {cap}  hbm: "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
    if cap != (10, 0):
        print(f"[smoke] WARNING: expected sm_100, got {cap}")

    # Auth: huggingface_hub picks up HF_TOKEN env var automatically.
    assert os.environ.get("HF_TOKEN", "").startswith("hf_"), \
        "HF_TOKEN missing — pass via local env, modal Secret transfers it."

    # Run only the correctness section (single forward each), with per-layer
    # cos sim printed for every decoder layer.
    cmd = [
        "python", "-u", "/root/benchmarks/bench_multimodal_moe.py",
        "--skip-perf", "--all-layers",
    ]
    print(f"[bench] cmd: {' '.join(cmd)}")
    sys.path.insert(0, "/root")
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root"
    result = subprocess.run(cmd, env=env, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"bench exited with code {result.returncode}")
    return "ok"


@app.local_entrypoint()
def main():
    print("[local] dispatching B200 cos-sim job ...")
    run_cossim.remote()
    print("[local] done")
