#!/usr/bin/env python3
"""H100 阶段从 flash_attn/attention.py 迁出的历史 benchmark。

该脚本保留原有 mha/gemma4/causal/long/bf16/sweep/swa/swa_bwd 模式，
仅用于复现实验，不是当前跨架构 canonical benchmark。请从仓库根目录运行。
"""

import sys
from functools import partial

import torch

from flash_attn.attention import (
    attention,
    attention_flash_gqa,
    attention_gqa_ref,
    attention_swa_ref,
    attention_triton,
    attention_triton_opt,
    flash_attn_gqa_train,
)
from flash_attn.utils import benchmark, benchmark_fn


def main():
    def make_qkv(shape, dtype, device):
        B, H, N, D = shape
        q = torch.randn(B, H, N, D, dtype=dtype, device=device)
        k = torch.randn(B, H, N, D, dtype=dtype, device=device)
        v = torch.randn(B, H, N, D, dtype=dtype, device=device)
        return (q, k, v)

    def make_qkv_gqa(shape, dtype, device):
        """shape = (B, H_Q, N, D, H_KV)"""
        B, H_Q, N, D, H_KV = shape
        q = torch.randn(B, H_Q, N, D, dtype=dtype, device=device)
        k = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)
        v = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)
        return (q, k, v)

    mode = sys.argv[1] if len(sys.argv) > 1 else "mha"

    if mode == "gemma4":
        # Gemma4 config: H_Q=32, H_KV=4, D=512
        print("=== Gemma4 GQA Benchmark (H_Q=32, H_KV=4, D=512) ===")
        benchmark(
            implementations={
                "pytorch_sdpa": attention_gqa_ref,
                "flash_gqa": attention_flash_gqa,
            },
            input_shapes=[
                # (B, H_Q, N, D, H_KV)
                (1, 32, 128, 512, 4),
                (1, 32, 256, 512, 4),
                (1, 32, 512, 512, 4),
                (1, 32, 1024, 512, 4),
                (1, 32, 2048, 512, 4),
                (1, 32, 4096, 512, 4),
                (2, 32, 1024, 512, 4),
                (2, 32, 2048, 512, 4),
            ],
            input_fn=make_qkv_gqa,
            dtype=torch.float16,
            device="cuda",
            warmup=10,
            rep=100,
            verify=True,
            atol=5e-2,
            rtol=5e-2,
        )

    elif mode == "causal":
        # Gemma4 causal attention benchmark
        print("=== Gemma4 GQA Causal Benchmark (H_Q=32, H_KV=4, D=512) ===")
        ref_causal = partial(attention_gqa_ref, causal=True)
        triton_causal = partial(attention_flash_gqa, causal=True)
        benchmark(
            implementations={
                "pytorch_sdpa_causal": ref_causal,
                "flash_gqa_causal": triton_causal,
            },
            input_shapes=[
                (1, 32, 128, 512, 4),
                (1, 32, 256, 512, 4),
                (1, 32, 512, 512, 4),
                (1, 32, 1024, 512, 4),
                (1, 32, 2048, 512, 4),
                (1, 32, 4096, 512, 4),
                (2, 32, 2048, 512, 4),
            ],
            input_fn=make_qkv_gqa,
            dtype=torch.float16,
            device="cuda",
            warmup=10,
            rep=100,
            verify=True,
            atol=5e-2,
            rtol=5e-2,
        )

    elif mode == "long":
        # Long sequence benchmark (Gemma4 non-causal + causal)
        print("=== Gemma4 Long Sequence Benchmark (H_Q=32, H_KV=4, D=512) ===")
        ref_causal = partial(attention_gqa_ref, causal=True)
        triton_causal = partial(attention_flash_gqa, causal=True)
        benchmark(
            implementations={
                "sdpa_causal": ref_causal,
                "triton_causal": triton_causal,
            },
            input_shapes=[
                (1, 32, 4096, 512, 4),
                (1, 32, 8192, 512, 4),
                (1, 32, 16384, 512, 4),
            ],
            input_fn=make_qkv_gqa,
            dtype=torch.float16,
            device="cuda",
            warmup=5,
            rep=20,
            verify=True,
            atol=5e-2,
            rtol=5e-2,
        )

    elif mode == "bf16":
        # BF16 benchmark
        print("=== Gemma4 GQA BF16 Benchmark (H_Q=32, H_KV=4, D=512) ===")
        benchmark(
            implementations={
                "pytorch_sdpa": attention_gqa_ref,
                "flash_gqa": attention_flash_gqa,
            },
            input_shapes=[
                (1, 32, 512, 512, 4),
                (1, 32, 1024, 512, 4),
                (1, 32, 2048, 512, 4),
                (1, 32, 4096, 512, 4),
            ],
            input_fn=make_qkv_gqa,
            dtype=torch.bfloat16,
            device="cuda",
            warmup=10,
            rep=100,
            verify=True,
            atol=5e-2,
            rtol=5e-2,
        )

    elif mode == "sweep":
        # Sweep block sizes for Gemma4 to find best config
        print("=== Block Size Sweep for Gemma4 (B=1, H_Q=32, H_KV=4, N=1024, D=512) ===")
        shape = (1, 32, 1024, 512, 4)
        args = make_qkv_gqa(shape, torch.float16, "cuda")
        ref_out = attention_gqa_ref(*args)
        ref_time = benchmark_fn(attention_gqa_ref, *args, warmup=10, rep=50)
        print(f"PyTorch SDPA: {ref_time:.4f} ms\n")

        configs = [
            # (BLOCK_Q, BLOCK_KV, BLOCK_D, num_warps, num_stages)
            (16, 32, 128, 4, 2),
            (16, 32, 128, 4, 3),
            (16, 32, 128, 8, 2),
            (16, 64, 128, 4, 2),
            (16, 64, 128, 4, 3),
            (16, 64, 128, 8, 2),
            (16, 64, 64, 4, 3),
            (16, 64, 256, 4, 3),
            (32, 32, 128, 4, 2),
            (32, 32, 128, 8, 2),
            (32, 64, 128, 4, 2),
            (32, 64, 128, 8, 2),
            (64, 32, 128, 4, 2),
            (64, 64, 128, 4, 2),
            (16, 128, 128, 4, 2),
            (16, 128, 128, 8, 2),
        ]
        print(f"{'Config (BQ,BKV,BD,W,S)':<30} {'Time (ms)':>10} {'vs SDPA':>8} {'Correct':>8}")
        print("-" * 60)
        for bq, bkv, bd, nw, ns in configs:
            try:
                fn = partial(attention_flash_gqa,
                             BLOCK_Q=bq, BLOCK_KV=bkv, BLOCK_D=bd,
                             num_warps=nw, num_stages=ns)
                out = fn(*args)
                correct = torch.allclose(ref_out, out, atol=5e-2, rtol=5e-2)
                t = benchmark_fn(fn, *args, warmup=10, rep=50)
                ratio = ref_time / t
                print(f"({bq:>2},{bkv:>3},{bd:>3},{nw},{ns})"
                      f"{'':>16} {t:>10.4f} {ratio:>7.2f}x {'OK' if correct else 'FAIL':>8}")
            except Exception as e:
                print(f"({bq:>2},{bkv:>3},{bd:>3},{nw},{ns})"
                      f"{'':>16} {'ERROR':>10} {str(e)[:40]}")

    elif mode == "swa":
        # Sliding Window Attention benchmark and correctness test.
        # Gemma4 Sliding layer config: H_Q=32, H_KV=16, D=256, slide_size=1024
        # (distinct from the full-attention layer config H_KV=4, D=512).
        SLIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
        print(f"=== Gemma4 Sliding Attention (slide_size={SLIDE}, H_Q=32, H_KV=16, D=256) ===")
        print()

        # --- Correctness check ---
        print("Correctness check:")
        for N in [128, 256, 512, 1024, 2048, 4096]:
            shape = (1, 32, N, 256, 16)
            args = make_qkv_gqa(shape, torch.float16, "cuda")
            ref_out = attention_swa_ref(*args, slide_size=SLIDE)
            triton_out = attention_flash_gqa(*args, causal=True, slide_size=SLIDE)
            ok = torch.allclose(ref_out, triton_out, atol=5e-2, rtol=5e-2)
            max_err = (ref_out - triton_out).abs().max().item()
            print(f"  N={N:>5}: {'OK' if ok else 'FAIL'} (max_err={max_err:.4f})")

        print()

        # --- Forward benchmark: SWA vs full causal vs SDPA ---
        print("Forward benchmark:")
        triton_causal = partial(attention_flash_gqa, causal=True)
        triton_swa = partial(attention_flash_gqa, causal=True, slide_size=SLIDE)
        sdpa_causal = partial(attention_gqa_ref, causal=True)
        benchmark(
            implementations={
                "sdpa_causal":   sdpa_causal,
                "triton_causal": triton_causal,
                f"triton_swa_{SLIDE}": triton_swa,
            },
            input_shapes=[
                (1, 32, 512,  256, 16),
                (1, 32, 1024, 256, 16),
                (1, 32, 2048, 256, 16),
                (1, 32, 4096, 256, 16),
                (1, 32, 8192, 256, 16),
                (1, 32, 16384, 256, 16),
            ],
            input_fn=make_qkv_gqa,
            dtype=torch.float16,
            device="cuda",
            warmup=10,
            rep=30,
            verify=False,
        )

        print()

        # --- Backward correctness check ---
        print("Backward correctness check (train mode):")
        for N in [128, 256, 512, 1024, 2048]:
            shape = (1, 32, N, 256, 16)
            args_ref  = make_qkv_gqa(shape, torch.float16, "cuda")
            args_tri  = [t.clone().requires_grad_(True) for t in args_ref]
            args_ref  = [t.clone().requires_grad_(True) for t in args_ref]

            ref_out = attention_swa_ref(*args_ref, slide_size=SLIDE)
            ref_out.sum().backward()

            tri_out = flash_attn_gqa_train(*args_tri, causal=True, slide_size=SLIDE)
            tri_out.sum().backward()

            dq_ok = torch.allclose(args_ref[0].grad, args_tri[0].grad, atol=1e-1, rtol=1e-1)
            dk_ok = torch.allclose(args_ref[1].grad, args_tri[1].grad, atol=1e-1, rtol=1e-1)
            dv_ok = torch.allclose(args_ref[2].grad, args_tri[2].grad, atol=1e-1, rtol=1e-1)
            print(f"  N={N:>5}: dQ={'OK' if dq_ok else 'FAIL'} dK={'OK' if dk_ok else 'FAIL'} dV={'OK' if dv_ok else 'FAIL'}")

    elif mode == "swa_bwd":
        SLIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
        print("=== SWA Fwd+Bwd Benchmark (slide=%d, H_Q=32, H_KV=16, D=256) ===" % SLIDE)
        print()

        def sdpa_causal_fwd_bwd(q, k, v):
            ratio = q.shape[1] // k.shape[1]
            k_exp = k.repeat_interleave(ratio, dim=1)
            v_exp = v.repeat_interleave(ratio, dim=1)
            out = torch.nn.functional.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
            out.sum().backward()
            q.grad = k.grad = v.grad = None

        def triton_swa_fwd_bwd(q, k, v):
            out = flash_attn_gqa_train(q, k, v, causal=True, slide_size=SLIDE)
            out.sum().backward()
            q.grad = k.grad = v.grad = None

        print("%6s | %16s | %15s | %8s" % ("N", "SDPA-causal (ms)", "Triton-SWA (ms)", "Speedup"))
        print("-" * 56)
        for N in [512, 1024, 2048, 4096, 8192]:
            q = torch.randn(1, 32, N, 256, dtype=torch.float16, device="cuda").requires_grad_(True)
            k = torch.randn(1, 16, N, 256, dtype=torch.float16, device="cuda").requires_grad_(True)
            v = torch.randn(1, 16, N, 256, dtype=torch.float16, device="cuda").requires_grad_(True)
            t_sdpa   = benchmark_fn(sdpa_causal_fwd_bwd, q, k, v, warmup=5, rep=20)
            t_triton = benchmark_fn(triton_swa_fwd_bwd,  q, k, v, warmup=5, rep=20)
            print("%6d | %16.3f | %15.3f | %7.2fx" % (N, t_sdpa, t_triton, t_sdpa/t_triton))

    else:
        # Original MHA benchmark
        benchmark(
            implementations={
                "pytorch": attention,
                "triton": attention_triton,
                "triton_opt": attention_triton_opt,
            },
            input_shapes=[
                # (B, H, N, D)
                (1, 8, 128, 64),
                (1, 8, 256, 64),
                (1, 8, 512, 64),
                (1, 8, 1024, 64),
                (2, 8, 1024, 64),
                (1, 8, 2048, 64),
                (1, 8, 4096, 64),
            ],
            input_fn=make_qkv,
            dtype=torch.float16,
            device="cuda",
            warmup=10,
            rep=100,
            verify=True,
            atol=5e-2,
            rtol=5e-2,
        )


if __name__ == "__main__":
    main()
