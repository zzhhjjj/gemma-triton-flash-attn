"""Re-run everything against SDPA and produce figures for the README.

Collects three datasets across sequence lengths:
  1. Kernel-only SWA forward (D=256, H_Q=32, H_KV=16, slide=1024) vs SDPA causal
  2. Kernel-only SWA fwd+bwd vs SDPA causal
  3. Gemma-4-E2B E2E forward (real model, 5.1B, 35 layers) vs SDPA

Also: peak memory (forward-only) on Gemma-4-E2B across N, SDPA vs Triton.

Outputs:
  benchmarks/results.json            — raw data
  benchmarks/speedup_vs_sdpa.png     — figure 1
  benchmarks/memory_vs_sdpa.png      — figure 2

Run:
    source /opt/tiger/flash_gemma/bin/activate
    export HF_TOKEN="hf_..."
    python benchmarks/run_final_benchmark.py
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch

import transformers  # noqa: F401
from gemma_triton_flash_attn import (
    patch_transformers_5_5_4_flash_attn_key,
    register_triton_attention,
    attention_flash_gqa,
    attention_gqa_ref,
    flash_attn_gqa_train,
)
from gemma_triton_flash_attn.utils import benchmark_fn

patch_transformers_5_5_4_flash_attn_key()

from transformers import AutoModelForCausalLM  # noqa: E402


MB = 1024 ** 2
OUT_DIR = Path(__file__).parent
MODEL_ID = os.environ.get("GEMMA4_MODEL", "google/gemma-4-E2B")


# Gemma4 full-attention config (the SDPA slow path: D=512 falls off cuDNN fast-path)
_FULL_B, _FULL_H_Q, _FULL_H_KV, _FULL_D = 1, 32, 4, 512


def bench_kernel_fwd(seq_lens):
    """Kernel-only full-causal forward: SDPA vs Triton on Gemma4 full config."""
    B, H_Q, H_KV, D = _FULL_B, _FULL_H_Q, _FULL_H_KV, _FULL_D
    from functools import partial
    ref = partial(attention_gqa_ref, causal=True)
    tri = partial(attention_flash_gqa, causal=True, slide_size=0)

    results = []
    for N in seq_lens:
        q = torch.randn(B, H_Q, N, D, dtype=torch.float16, device="cuda")
        k = torch.randn(B, H_KV, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(B, H_KV, N, D, dtype=torch.float16, device="cuda")
        rep = 15 if N <= 4096 else (8 if N <= 16384 else 4)
        try:
            t_ref = benchmark_fn(ref, q, k, v, warmup=5, rep=rep)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect(); t_ref = None
        try:
            t_tri = benchmark_fn(tri, q, k, v, warmup=5, rep=rep)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect(); t_tri = None
        sp = (t_ref / t_tri) if t_ref and t_tri else None
        results.append({"N": N, "sdpa_ms": t_ref, "triton_ms": t_tri, "speedup": sp})
        print(f"  [kernel-fwd] N={N:>6}: sdpa={t_ref} triton={t_tri}  {sp}")
        del q, k, v; torch.cuda.empty_cache()
    return results


def bench_kernel_fwdbwd(seq_lens):
    """Kernel-only full-causal fwd+bwd: SDPA vs Triton on Gemma4 full config."""
    B, H_Q, H_KV, D = _FULL_B, _FULL_H_Q, _FULL_H_KV, _FULL_D

    def run_ref(N):
        q = torch.randn(B, H_Q, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H_KV, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H_KV, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        attention_gqa_ref(q, k, v, causal=True).backward(
            torch.ones(B, H_Q, N, D, dtype=torch.float16, device="cuda"))

    def run_tri(N):
        q = torch.randn(B, H_Q, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn(B, H_KV, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        v = torch.randn(B, H_KV, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
        flash_attn_gqa_train(q, k, v, causal=True, slide_size=0).backward(
            torch.ones(B, H_Q, N, D, dtype=torch.float16, device="cuda"))

    results = []
    for N in seq_lens:
        rep = 10 if N <= 4096 else (5 if N <= 16384 else 3)
        try:
            t_ref = benchmark_fn(lambda: run_ref(N), warmup=3, rep=rep)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            t_ref = None
        try:
            t_tri = benchmark_fn(lambda: run_tri(N), warmup=3, rep=rep)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            t_tri = None
        sp = (t_ref / t_tri) if t_ref and t_tri else None
        results.append({"N": N, "sdpa_ms": t_ref, "triton_ms": t_tri, "speedup": sp})
        print(f"  [kernel-f+b] N={N:>6}: sdpa={t_ref}  triton={t_tri}  {sp}")
        torch.cuda.empty_cache()
    return results


def bench_e2e_gemma4(seq_lens):
    """Real Gemma-4-E2B forward: SDPA vs Triton."""
    register_triton_attention()
    print("[load] Gemma-4-E2B (BF16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    def run(impl, N):
        model.config._attn_implementation = impl
        if hasattr(model.config, "text_config"):
            model.config.text_config._attn_implementation = impl
        ids = torch.randint(0, 1000, (1, N), device="cuda")
        with torch.no_grad():
            model(ids)

    results = []
    for N in seq_lens:
        try:
            # warmup
            for impl in ("sdpa", "triton_gqa"):
                run(impl, N)
            torch.cuda.synchronize()
            rep = 5 if N <= 4096 else (3 if N <= 16384 else 2)
            t_sdpa = benchmark_fn(lambda: run("sdpa", N), warmup=2, rep=rep)
            t_tri = benchmark_fn(lambda: run("triton_gqa", N), warmup=2, rep=rep)
            results.append({"N": N, "sdpa_ms": t_sdpa, "triton_ms": t_tri,
                            "speedup": t_sdpa / t_tri})
            print(f"  [e2e-fwd] N={N:>6}: sdpa={t_sdpa:.2f}ms triton={t_tri:.2f}ms  {t_sdpa/t_tri:.2f}x")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            print(f"  [e2e-fwd] N={N}: OOM (likely SDPA)")
            # try triton only
            try:
                t_tri = benchmark_fn(lambda: run("triton_gqa", N), warmup=1, rep=2)
                results.append({"N": N, "sdpa_ms": None, "triton_ms": t_tri, "speedup": None})
                print(f"             Triton still runs: {t_tri:.2f}ms")
            except Exception:
                results.append({"N": N, "sdpa_ms": None, "triton_ms": None, "speedup": None})
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return results


def bench_memory(seq_lens):
    """Peak memory during Gemma-4-E2B forward pass, SDPA vs Triton."""
    register_triton_attention()
    print("[load] Gemma-4-E2B for memory benchmark...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    def measure(impl, N):
        model.config._attn_implementation = impl
        if hasattr(model.config, "text_config"):
            model.config.text_config._attn_implementation = impl
        # warmup
        warm_ids = torch.randint(0, 1000, (1, 128), device="cuda")
        with torch.no_grad():
            model(warm_ids)
        del warm_ids
        torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()

        ids = torch.randint(0, 1000, (1, N), device="cuda")
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(ids)
        torch.cuda.synchronize()
        peak = (torch.cuda.max_memory_allocated() - base) / MB
        del ids; torch.cuda.empty_cache()
        return peak

    results = []
    for N in seq_lens:
        try:
            sdpa_peak = measure("sdpa", N)
        except torch.cuda.OutOfMemoryError:
            sdpa_peak = None
            torch.cuda.empty_cache(); gc.collect()
        try:
            tri_peak = measure("triton_gqa", N)
        except torch.cuda.OutOfMemoryError:
            tri_peak = None
            torch.cuda.empty_cache(); gc.collect()
        results.append({"N": N, "sdpa_mb": sdpa_peak, "triton_mb": tri_peak})
        print(f"  [memory] N={N:>6}: sdpa={sdpa_peak} MB, triton={tri_peak} MB")
    del model
    torch.cuda.empty_cache(); gc.collect()
    return results


H100_PEAK_FP16_TFLOPS = 989.5


def _fwd_flops_causal(N, B=_FULL_B, H=_FULL_H_Q, D=_FULL_D):
    """2 matmuls (QK^T, PV), each 2·B·H·N²·D, halved for causal → 2·B·H·N²·D total."""
    return 2 * B * H * N * N * D


def _fwdbwd_flops_causal(N, B=_FULL_B, H=_FULL_H_Q, D=_FULL_D):
    """fwd (2·B·H·N²·D) + bwd (~5·B·H·N²·D) = 7·B·H·N²·D for dense causal."""
    return 7 * B * H * N * N * D


def _ms_to_tflops(ms, flops):
    return flops / (ms * 1e-3) / 1e12


def _format_seq_len(n):
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}K"
    return str(n)


def _apply_linear_seqlen_xaxis(ax, xs):
    import matplotlib.ticker as mticker
    ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: _format_seq_len(int(v))))
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)


def plot_speedup(kernel_fwd, kernel_fwdbwd, e2e, out_path):
    """Full-causal Triton vs SDPA. Linear axes. Both impls do the same work,
    so speedup in ms equals speedup in TFLOPS/s directly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, ax = plt.subplots(figsize=(11, 6.5))

    def add_curve(data, flops_fn, impl, label, color, marker, linestyle="-"):
        xs, ys = [], []
        for d in data:
            ms = d.get(f"{impl}_ms")
            if ms is None or d.get("N") is None:
                continue
            xs.append(d["N"])
            ys.append(_ms_to_tflops(ms, flops_fn(d["N"])))
        if xs:
            ax.plot(xs, ys, marker=marker, label=label, color=color,
                    linewidth=2, markersize=8, linestyle=linestyle)
        return xs, ys

    add_curve(kernel_fwd, _fwd_flops_causal, "sdpa",
              "SDPA fwd", "#d62728", "o", "--")
    add_curve(kernel_fwd, _fwd_flops_causal, "triton",
              "Triton fwd", "#1f77b4", "o", "-")
    add_curve(kernel_fwdbwd, _fwdbwd_flops_causal, "sdpa",
              "SDPA fwd+bwd", "#ff7f0e", "s", "--")
    add_curve(kernel_fwdbwd, _fwdbwd_flops_causal, "triton",
              "Triton fwd+bwd", "#2ca02c", "s", "-")

    ax.axhline(H100_PEAK_FP16_TFLOPS, color="gray", linestyle=":", alpha=0.8,
               label=f"H100 peak FP16 ({H100_PEAK_FP16_TFLOPS:.0f} TFLOPS)")

    all_xs = sorted({d["N"] for d in kernel_fwd + kernel_fwdbwd if d.get("N")})
    _apply_linear_seqlen_xaxis(ax, all_xs)
    ax.set_xlabel("Sequence length (N)", fontsize=12)
    ax.set_ylabel("Throughput (TFLOPS/s)", fontsize=12)
    ax.set_title(
        f"Full-causal attention throughput — Gemma4 full-attn config, "
        f"H100 FP16\nB={_FULL_B}, H_Q={_FULL_H_Q}, H_KV={_FULL_H_KV} "
        f"(GQA 8:1), D={_FULL_D}",
        fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    ax.set_ylim(bottom=0)

    # Annotate peak speedups
    def annotate_speedup(data, flops_fn, color, dy):
        best = max((d for d in data if d.get("speedup") is not None),
                   key=lambda d: d["speedup"], default=None)
        if best is None:
            return
        y = _ms_to_tflops(best["triton_ms"], flops_fn(best["N"]))
        ax.annotate(
            f"{best['speedup']:.2f}× vs SDPA\n@ N={_format_seq_len(best['N'])}",
            xy=(best["N"], y),
            xytext=(-75, dy), textcoords="offset points",
            fontsize=10, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

    annotate_speedup(kernel_fwd, _fwd_flops_causal, "#1f77b4", 20)
    annotate_speedup(kernel_fwdbwd, _fwdbwd_flops_causal, "#2ca02c", 40)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def plot_e2e(e2e, out_path):
    """E2E Gemma-4-E2B forward time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    xs_sdpa = [d["N"] for d in e2e if d.get("sdpa_ms")]
    ys_sdpa = [d["sdpa_ms"] for d in e2e if d.get("sdpa_ms")]
    xs_tri = [d["N"] for d in e2e if d.get("triton_ms")]
    ys_tri = [d["triton_ms"] for d in e2e if d.get("triton_ms")]
    if xs_sdpa:
        ax.plot(xs_sdpa, ys_sdpa, marker="o", linestyle="--",
                color="#d62728", label="SDPA", linewidth=2, markersize=7)
    if xs_tri:
        ax.plot(xs_tri, ys_tri, marker="s", color="#2ca02c",
                label="Triton", linewidth=2, markersize=7)
    _apply_linear_seqlen_xaxis(ax, sorted(set(xs_sdpa + xs_tri)))
    ax.set_xlabel("Sequence length (N)", fontsize=12)
    ax.set_ylabel("Forward latency (ms)", fontsize=12)
    ax.set_title("Gemma-4-E2B E2E forward — SDPA vs Triton (H100, BF16)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def plot_memory(memory, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6.5))

    xs = [d["N"] for d in memory]
    sdpa_ys = [d["sdpa_mb"] / 1024 if d["sdpa_mb"] else None for d in memory]
    tri_ys = [d["triton_mb"] / 1024 if d["triton_mb"] else None for d in memory]

    sdpa_valid = [(x, y) for x, y in zip(xs, sdpa_ys) if y is not None]
    tri_valid = [(x, y) for x, y in zip(xs, tri_ys) if y is not None]
    if sdpa_valid:
        ax.plot([p[0] for p in sdpa_valid], [p[1] for p in sdpa_valid],
                marker="o", label="SDPA", color="#d62728",
                linestyle="--", linewidth=2, markersize=8)
    if tri_valid:
        ax.plot([p[0] for p in tri_valid], [p[1] for p in tri_valid],
                marker="s", label="Triton", color="#2ca02c",
                linewidth=2, markersize=8)

    # Mark where SDPA OOMs
    for x, sy, ty in zip(xs, sdpa_ys, tri_ys):
        if sy is None and ty is not None:
            ax.scatter([x], [ty], marker="x", color="#d62728", s=120, zorder=5)
            ax.annotate("SDPA OOM", xy=(x, ty), xytext=(8, -16),
                        textcoords="offset points", fontsize=10,
                        color="#d62728", fontweight="bold")

    ax.axhline(80, color="black", linestyle=":", alpha=0.5, label="H100 80GB limit")
    _apply_linear_seqlen_xaxis(ax, xs)
    ax.set_xlabel("Sequence length (N)", fontsize=12)
    ax.set_ylabel("Peak GPU memory above weights (GB)", fontsize=12)
    ax.set_title("Gemma-4-E2B forward peak memory — SDPA vs Triton (H100, BF16)",
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-e2e", action="store_true")
    p.add_argument("--skip-memory", action="store_true")
    args = p.parse_args()

    results = {}

    print("=== 1. Kernel-only full-causal forward (Gemma4 full config) ===")
    kernel_lens = [1024, 2048, 4096, 8192, 16384, 32768]
    results["kernel_fwd"] = bench_kernel_fwd(kernel_lens)

    print("\n=== 2. Kernel-only full-causal fwd+bwd (Gemma4 full config) ===")
    kernel_fb_lens = [1024, 2048, 4096, 8192, 16384]
    results["kernel_fwdbwd"] = bench_kernel_fwdbwd(kernel_fb_lens)

    if not args.skip_e2e:
        print("\n=== 3. Gemma-4-E2B E2E forward ===")
        e2e_lens = [512, 1024, 2048, 4096, 8192, 16384, 32768]
        results["e2e_fwd"] = bench_e2e_gemma4(e2e_lens)
    else:
        results["e2e_fwd"] = []

    if not args.skip_memory:
        print("\n=== 4. Gemma-4-E2B peak memory ===")
        mem_lens = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
        results["memory"] = bench_memory(mem_lens)
    else:
        results["memory"] = []

    # Save JSON
    json_path = OUT_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {json_path}")

    # Generate plots
    print("\n=== Generating plots ===")
    if results["kernel_fwd"] or results["kernel_fwdbwd"]:
        plot_speedup(
            results["kernel_fwd"], results["kernel_fwdbwd"], results["e2e_fwd"],
            OUT_DIR / "flops_vs_sdpa.png",
        )
    if results["e2e_fwd"]:
        plot_e2e(results["e2e_fwd"], OUT_DIR / "e2e_latency_vs_sdpa.png")
    if results["memory"]:
        plot_memory(results["memory"], OUT_DIR / "memory_vs_sdpa.png")


if __name__ == "__main__":
    sys.exit(main())
