"""B200 dQ + dKV config sweep at the Gemma-4-26B-A4B MoE shape.

Mirrors run_b200_moe_tune.py but for the bwd kernels. Each individual config
runs in its own subprocess inside the container so wgmma faults are contained.

Usage:
    modal run modal/run_b200_moe_bwd_tune.py
"""
from __future__ import annotations

import pathlib

import modal


CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu22.04"

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    .pip_install(
        "torch==2.9.1",
        "triton==3.5.1",
        "transformers==5.5.4",
        "numpy",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .workdir("/root")
    .add_local_dir(
        local_path=str(pathlib.Path(__file__).parent.parent / "flash_attn"),
        remote_path="/root/flash_attn",
    )
    .add_local_dir(
        local_path=str(pathlib.Path(__file__).parent.parent / "benchmarks"),
        remote_path="/root/benchmarks",
    )
)


app = modal.App("kernel-b200-moe-bwd-tune", image=image)


@app.function(gpu="B200", timeout=60 * 50)
def run_sweep():
    import subprocess
    import sys

    import torch
    cap = torch.cuda.get_device_capability(0)
    print(f"[smoke] device: {torch.cuda.get_device_name(0)}  cap: {cap}")
    if cap != (10, 0):
        print(f"[smoke] WARNING: expected sm_100, got {cap}")

    sys.path.insert(0, "/root")
    cmd = ["python", "-u", "/root/benchmarks/b200_moe_bwd_tune.py"]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sweep exited with {result.returncode}")
    return "ok"


@app.local_entrypoint()
def main():
    print("[local] dispatching B200 MoE bwd-tune sweep ...")
    run_sweep.remote()
    print("[local] done")
