#!/bin/bash
# B300 (sm103) sweep for D=512 backward kernels via GTFA_* env overrides.
# Fresh process per config to avoid CUDA-context poisoning from bad configs.
set -u
PY=${PY:-python}
REPO=${REPO:-/tmp/gemma-triton-flash-attn}

probe() {
  local tag="$1"; shift
  env "$@" PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES=0 $PY - <<'EOF' 2>/dev/null | tail -1 | sed "s|^|$tag |"
import torch, time
from flash_attn.attention import flash_attn_gqa_train, attention_flash_gqa

S, Hq, Hkv, D = 8192, 32, 4, 512
q = torch.randn(1, Hq, S, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
dout = torch.randn_like(q)

def run():
    q_, k_, v_ = (t.detach().requires_grad_(True) for t in (q, k, v))
    out = flash_attn_gqa_train(q_, k_, v_, causal=True, slide_size=0)
    torch.autograd.grad(out, (q_, k_, v_), dout)

def fwd():
    with torch.no_grad():
        attention_flash_gqa(q, k, v, causal=True, slide_size=0)

def bench(fn, n=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3

try:
    f = bench(fwd)
    fb = bench(run)
    fl = 3.5 * 4.0 * Hq * (S * (S + 1) // 2) * D
    print(f"fwd {f:8.2f} ms | fwd+bwd {fb:8.2f} ms | bwd {fb-f:8.2f} ms | {fl/(fb/1e3)/1e12:6.1f} TFLOPS")
except Exception as e:
    print(f"FAILED {type(e).__name__}: {str(e)[:90]}")
EOF
}

echo "--- baseline (branch defaults) ---"
probe "default              "

echo "--- dQ sweep (dKV default) ---"
probe "dQ 32/32/8/2         " GTFA_DQ_BQ=32 GTFA_DQ_BKV=32 GTFA_DQ_W=8
probe "dQ 64/32/8/2         " GTFA_DQ_BQ=64 GTFA_DQ_BKV=32 GTFA_DQ_W=8
probe "dQ 32/64/4/2         " GTFA_DQ_W=4
probe "dQ 32/64/8/1         " GTFA_DQ_ST=1
probe "dQ 16/64/8/2         " GTFA_DQ_BQ=16
probe "dQ 64/64/8/2         " GTFA_DQ_BQ=64

echo "--- dKV sweep (dQ default) ---"
probe "dKV 16/32/4/2        " GTFA_DKV_BQ=32
probe "dKV 32/32/4/2        " GTFA_DKV_BKV=32 GTFA_DKV_BQ=32
probe "dKV 32/32/8/2        " GTFA_DKV_BKV=32 GTFA_DKV_BQ=32 GTFA_DKV_W=8
probe "dKV 16/64/8/2        " GTFA_DKV_W=8
probe "dKV 8/64/4/2         " GTFA_DKV_BKV=8
probe "dKV 16/128/4/2       " GTFA_DKV_BQ=128
probe "dKV 16/64/4/1        " GTFA_DKV_ST=1
