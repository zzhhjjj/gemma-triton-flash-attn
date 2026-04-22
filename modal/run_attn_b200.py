"""Run the attention-only kernel benchmark on a Modal B200.

The point of this script is "first get it running" — it pins every version
to what already works locally on H100 (`torch 2.9.1+cu128`, `triton 3.5.1`),
adds the local kernel + bench source, and runs `benchmarks/attn_only_all_n.py`
on a B200. No retuning, no model downloads.

Usage:
    pip install modal
    modal setup                                 # one-time: link your account
    modal run modal/run_attn_b200.py            # default: --quick (~2 min)
    modal run modal/run_attn_b200.py --mode full   # full sweep (~15 min)

What you'll see:
    1. Image build: ~3-5 min the first time (downloads ~3 GB of torch+triton).
       Cached after — subsequent runs reuse it.
    2. Cold start on B200: ~10-30 s.
    3. Smoke check: prints GPU name + compute capability (must be (10, 0) on
       Blackwell sm_100).
    4. Bench output: same table as local — Tri vs SDPA fwd/bwd at every N.

Cost (assuming Modal's published B200 rate ~$6/h):
    --quick : ~$0.20-0.30 per run
    --full  : ~$1.50-2.00 per run
"""
from __future__ import annotations

import pathlib

import modal


# =====================================================================
# Image
# =====================================================================
# CUDA 12.8 is the first toolkit with Blackwell sm_100 support.
# Pin the same versions that already work on H100 (verified
# locally: torch 2.9.1+cu128 + triton 3.5.1 boots on (9,0) and ships
# Blackwell codegen). transformers is needed only because
# `flash_attn/__init__.py` transitively imports `hf_integration`.

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
    # add_local_* MUST be last — Modal injects local files at container start, not build time.
    .add_local_dir(
        local_path=str(pathlib.Path(__file__).parent.parent / "flash_attn"),
        remote_path="/root/flash_attn",
    )
    .add_local_dir(
        local_path=str(pathlib.Path(__file__).parent.parent / "benchmarks"),
        remote_path="/root/benchmarks",
    )
)


app = modal.App("kernel-b200-attn-bench", image=image)


# =====================================================================
# Smoke check + bench
# =====================================================================

@app.function(gpu="B200", timeout=60 * 30)
def run_bench(mode: str = "quick"):
    """Run attention-only fwd / fwd+bwd benchmark on the attached B200.

    `mode` ∈ {"quick", "full"} — passed through as `--quick` flag (or absent).
    """
    import subprocess
    import sys

    # 1) Smoke check: must actually be on Blackwell. If we got an H100 by
    #    mistake (Modal scheduling fallback), abort loud.
    import torch
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"[smoke] device: {name}  compute_capability: {cap}")
    print(f"[smoke] torch: {torch.__version__}  cuda: {torch.version.cuda}")
    import triton
    print(f"[smoke] triton: {triton.__version__}")
    if cap != (10, 0):
        # Don't waste budget benchmarking on the wrong GPU.
        print(f"[smoke] WARNING: expected sm_100 (Blackwell B200), got sm_{cap[0]}{cap[1]}")
        print(f"[smoke] continuing anyway so you see *something* — but cost is on the wrong arch")

    # 2) Quick functional check before the bench: a tiny fwd should compile
    #    and run end-to-end. If Triton can't codegen for sm_100, this is
    #    where it'll die with a cleaner error than the bench.
    print("\n[probe] tiny fwd to confirm Triton codegen works on this arch...")
    sys.path.insert(0, "/root")
    from flash_attn.attention import flash_attn_gqa_train
    q = torch.randn(1, 8, 256, 128, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 4, 256, 128, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 4, 256, 128, dtype=torch.float16, device="cuda")
    out = flash_attn_gqa_train(q, k, v, causal=True, slide_size=0)
    torch.cuda.synchronize()
    print(f"[probe] OK — out shape: {tuple(out.shape)}, dtype: {out.dtype}")

    # 2b) Diagnose probe at one of the bench shapes — bench swallows all errors
    #     into 'OOM', so we pre-flight a real config here and let it raise.
    #     Config A from attn_only_all_n.py at N=1024: B=1, H_Q=32, H_KV=16,
    #     D=256, slide=1024, fp16. Both fwd and bwd.
    print("\n[diag] running real bench shape (config A, N=1024) so any error surfaces...")
    B, HQ, HKV, N, D = 1, 32, 16, 1024, 256
    q = torch.randn(B, HQ,  N, D, dtype=torch.float16, device="cuda", requires_grad=True)
    k = torch.randn(B, HKV, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
    v = torch.randn(B, HKV, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
    try:
        out = flash_attn_gqa_train(q, k, v, causal=True, slide_size=1024)
        torch.cuda.synchronize()
        print(f"[diag] fwd OK — out shape: {tuple(out.shape)}")
        out.sum().backward()
        torch.cuda.synchronize()
        print(f"[diag] bwd OK — grads computed")
    except Exception as e:
        import traceback
        print(f"[diag] FAILED with: {type(e).__name__}: {e}")
        traceback.print_exc()
        # Don't re-raise — let the bench still attempt so we see if other
        # shapes also fail and we get the autotune log too.

    # 3) Run the actual bench. Stream output line-by-line so the table shows
    #    up live in the local terminal instead of after the whole run.
    print("\n[bench] launching benchmarks/attn_only_all_n.py ...")
    cmd = ["python", "-u", "/root/benchmarks/attn_only_all_n.py"]
    if mode == "quick":
        cmd.append("--quick")
    elif mode != "full":
        raise ValueError(f"--mode must be 'quick' or 'full', got {mode!r}")
    print(f"[bench] cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"bench exited with code {result.returncode}")
    return "ok"


@app.local_entrypoint()
def main(mode: str = "quick"):
    """CLI: `modal run modal/run_attn_b200.py [--mode quick|full]`."""
    print(f"[local] dispatching B200 job (mode={mode}) ...")
    run_bench.remote(mode=mode)
    print("[local] done")
