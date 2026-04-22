"""B200 module-level correctness sweep.

Runs the three Triton-kernel correctness tests on a Modal B200:

  * tests/test_packed_dkv.py          (packed dKV vs reference, fwd+bwd)
  * tests/test_image_group_mask.py    (image OR-mask, fwd+bwd)
  * tests/test_noncausal_vision_shape.py (non-causal vision shape)

Same image as run_attn_b200.py — just adds pytest. Streams pytest output
live so a failure surfaces immediately.

Usage:
    modal run modal/run_correctness_b200.py
"""
from __future__ import annotations

import pathlib

import modal


CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu22.04"
KERNEL_DIR = pathlib.Path(__file__).parent.parent

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    .pip_install(
        "torch==2.9.1",
        "triton==3.5.1",
        "transformers==5.5.4",
        "pytest",
        "numpy",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .workdir("/root")
    # Mount under both names so tests using either import path resolve.
    .add_local_dir(local_path=str(KERNEL_DIR / "flash_attn"),
                   remote_path="/root/flash_attn")
    .add_local_dir(local_path=str(KERNEL_DIR / "flash_attn"),
                   remote_path="/root/gemma_triton_flash_attn")
    .add_local_dir(local_path=str(KERNEL_DIR / "tests"),
                   remote_path="/root/tests")
)

app = modal.App("kernel-b200-correctness", image=image)


@app.function(gpu="B200", timeout=60 * 30)
def run_tests():
    import subprocess
    import sys

    import torch
    cap = torch.cuda.get_device_capability(0)
    print(f"[smoke] device: {torch.cuda.get_device_name(0)}  cap: {cap}")
    if cap[0] < 10:
        print(f"[smoke] WARNING: expected sm_100, got {cap}")

    sys.path.insert(0, "/root")

    tests = [
        "/root/tests/test_packed_dkv.py",
        "/root/tests/test_image_group_mask.py",
        "/root/tests/test_noncausal_vision_shape.py",
    ]
    failed = []
    for t in tests:
        print(f"\n{'=' * 70}\n[run] {t}\n{'=' * 70}")
        rc = subprocess.call(
            ["python", "-u", "-m", "pytest", t, "-v", "--tb=short"],
            cwd="/root",
        )
        print(f"[run] {t} → exit {rc}")
        if rc != 0:
            failed.append(t)

    print(f"\n{'=' * 70}")
    if failed:
        print(f"[summary] {len(failed)} test file(s) FAILED:")
        for t in failed:
            print(f"  - {t}")
        raise RuntimeError(f"{len(failed)} test files failed")
    print(f"[summary] all {len(tests)} test files PASSED on B200")
    return "ok"


@app.local_entrypoint()
def main():
    print("[local] dispatching B200 correctness job ...")
    run_tests.remote()
    print("[local] done")
