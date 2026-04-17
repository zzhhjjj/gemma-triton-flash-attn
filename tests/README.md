# Tests

## Layout

```
tests/
├── test_packed_dkv.py           ← ✅ CORE correctness for hot-path dKV kernel (run every commit)
├── gemma4_integration/          ← integration tests (real HF model, needs clean env)
│   ├── test_adapter.py          ← HF attention registry + adapter unit tests (24 cases)
│   ├── test_gemma4.py           ← real Gemma-4-E2B E2E forward + throughput
│   └── test_memory.py           ← Peak memory SDPA vs Triton on real model
└── legacy/                      ← experiments that didn't pan out, kept for reference
    ├── test_fused_backward.py   ← A.2 atomic-fused dQ+dKV (net-negative, unused)
    └── test_grouped_forward.py  ← A.1 multi-head fusion (register spills, unused)
```

## When to run

### Tier 0 — Pre-commit smoke test (< 1 min)
Required before any commit touching `flash_attn/attention.py`:

```bash
python tests/test_packed_dkv.py
```

This exercises pack-GQA dKV correctness across 7 shape configs (D=256/512, H_KV∈{1,4,16},
N∈{512,1K,2K}, causal + SWA slide). Must all be OK.

If your change is backward-path only, this is sufficient for correctness. If you added or
renamed kernel parameters, also run Tier-1 bench to catch missing call-site updates.

### Tier 3 — Integration (release-time, needs clean env)

```bash
source /opt/tiger/flash_gemma/bin/activate
python tests/gemma4_integration/test_adapter.py    # 24/24 cases must PASS
python tests/gemma4_integration/test_gemma4.py     # cos sim > 0.999, top-1 100%
python tests/gemma4_integration/test_memory.py     # no regression in peak memory
```

Environment: requires `transformers==5.5.4`, `torch==2.9.1`, `triton==3.5.1` installed in
the `/opt/tiger/flash_gemma` venv, plus Gemma-4-E2B weights cached under `~/.cache/huggingface`.

The `patch_transformers_5_5_4_flash_attn_key()` helper (in `flash_attn/hf_integration.py`)
works around a known 5.5.4 `KeyError: 'flash_attn'` bug.

## What NOT to run routinely

- `tests/legacy/*` — these are **kept-for-reference** experiments that failed. They still pass
  correctness (they test the atomic-fused / grouped kernels which exist in `attention.py`) but
  are not on any hot path. Running them on every commit is a waste of GPU time.

## Correctness conventions (if adding a new test)

1. **Always use fp32 cosine similarity** for Triton vs SDPA comparison:
   ```python
   def cs(a, b):
       return F.cosine_similarity(a.detach().float().flatten(),
                                  b.detach().float().flatten(), dim=0).item()
   ```
   fp16 cos sim produces false failures due to accumulation noise.
   See `skills/2026-04-17_fp16-cossim-pitfall.md`.

2. **PASS threshold: cos sim > 0.9999** (fp16/bf16 input). Lower than 0.9999 → real bug.

3. **For SWA**: SDPA has no native sliding-window, so compare Triton fwd+bwd against
   itself (run twice, verify determinism) OR against `attention_swa_ref` for ref.
   No-NaN check is the minimum correctness bar when ref doesn't exist.
