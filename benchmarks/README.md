# Benchmarks

Organized by **when to run** rather than what they measure.

## Layout

```
benchmarks/
├── attn_only_all_n.py          ← ⭐ Canonical kernel-only (all configs × N=1K-32K)
├── real_gemma4_fwdbwd.py       ← ⭐ Real Gemma-4-E2B training throughput
├── run_final_benchmark.py      ← ⭐ Release bench + generates plots
├── replot.py                   ← Regenerate plots from saved results.json
│
├── profile_short_n.py          ← Per-kernel breakdown N=1K/2K/4K
├── profile_swa.py              ← Per-kernel breakdown N≥8K SWA
├── profile_n8k.py              ← Per-kernel breakdown D=512 N=8K
├── bwd_breakdown.py            ← delta/dQ/dKV isolated timing
├── diag_short_n.py             ← SM util / SDPA backend / launch floor
├── dump_kernel_regs.py         ← Compiled kernel regs / spills / shmem
│
├── alloc_overhead.py           ← torch.autograd + alloc overhead decomposition
├── triton_launch_overhead.py   ← Triton launch wrapper cost isolation
│
└── archive/                    ← One-off tuning sweeps (landed; rarely re-run)
    ├── dq_sweep_D512.py
    ├── dkv_sweep_D512.py
    ├── dkv_swa_sweep_D256.py
    ├── dkv_qsplits_sweep.py
    ├── dkv_bkv_qsplits_joint.py
    ├── dkv_config_a_sweep.py
    ├── dkv_config_b_bkv64.py
    ├── dq_config_a_sweep.py
    ├── dq_config_b_sweep.py
    ├── dkv_stages_sweep.py     ← Failed experiment (kept as reference)
    ├── dkv_split_bench.py      ← Failed experiment (kept as reference)
    └── swa_e2e_bench.py        ← N-gate development bench (landed)
```

## Commit workflow (4 tiers)

### Tier 0 — Smoke test (< 1 min)
Every commit touching kernels. **Required**.
```bash
python tests/test_packed_dkv.py
```

### Tier 1 — Perf regression check (< 3 min)
Commits that may affect perf. Compare against previously saved JSON.
```bash
python benchmarks/attn_only_all_n.py --quick    # N=1K/4K/16K only
```
Compare output manually (or diff `attn_only_all_n_quick.json` against checkpoint). If any
speedup regresses by > 5% at any N, investigate before merging.

### Tier 2 — Full bench (< 20 min)
Major refactors / before release. Produces canonical numbers + plots.
```bash
python benchmarks/attn_only_all_n.py            # full N sweep, all 3 configs
python benchmarks/run_final_benchmark.py        # kernel + E2E + memory + plots
```

### Tier 3 — Real model integration (needs clean env)
Release-time. See `tests/README.md` for environment setup.
```bash
source /opt/tiger/flash_gemma/bin/activate
python tests/gemma4_integration/test_adapter.py
python tests/gemma4_integration/test_gemma4.py
python tests/gemma4_integration/test_memory.py
python benchmarks/real_gemma4_fwdbwd.py         # real model fwd+bwd throughput
```

## When to use profile / diagnostic scripts

### `profile_short_n.py` / `profile_swa.py` / `profile_n8k.py`
**Use when**: a perf regression appeared and you need per-kernel breakdown (fwd / delta /
dQ / dKV time + speedup vs SDPA). These identify *which* kernel is slower, not *why*.

### `bwd_breakdown.py`
**Use when**: specifically investigating backward path on Gemma4 full-attn (D=512).

### `diag_short_n.py`
**Use when**: suspecting SM occupancy issues or launch-overhead bottlenecks.
Reports raw grid × SM count, measures noop-kernel launch floor, detects which
SDPA backend PyTorch picks (cuDNN vs FA2 vs math).

### `dump_kernel_regs.py`
**Use when**: suspecting register spills (see 2026-04-17 dKV attack). Pulls
`n_regs / n_spills / shared_memory` from CompiledKernel metadata without needing ncu.

### `alloc_overhead.py` / `triton_launch_overhead.py`
**Use when**: measuring Python-side overhead (autograd.Function, allocator, launch
wrappers). Short-N Config B specifically.

## When to use archive/ scripts

**Don't run by default.** These are tuning sweeps that established the current config in
`attention.py`. Re-run **only** if:
1. Kernel structure changed (new param/branch) → re-sweep to confirm optimal config
2. Moved to a different GPU (not H100) → rerun everything
3. Triton major version update → may shift optimal block configs

Reference list:
| Script | Established | Finding |
|--------|-------------|---------|
| `dkv_sweep_D512.py` | 2026-04-16 | BKV=16, BQ=64, w=4 for D=512 pack-GQA |
| `dkv_swa_sweep_D256.py` | 2026-04-16 / 2026-04-17 | N-gated BKV config for D=256 |
| `dkv_qsplits_sweep.py` | 2026-04-17 | raw_grid-based QS heuristic (target 128/256) |
| `dkv_bkv_qsplits_joint.py` | 2026-04-17 | BKV=32 vs 64 interaction with QS |
| `dkv_config_a_sweep.py` | 2026-04-17 | BKV=64 is optimal Config A all-N (翻案) |
| `dkv_config_b_bkv64.py` | 2026-04-17 | BKV=64 rescue for extreme SM-starve (grid≤16) |
| `dq_sweep_D512.py` | 2026-04-17 | dQ default (BQ=32, BKV=64, w=8, s=2) for D=512 |
| `dq_config_a/b_sweep.py` | 2026-04-17 | dQ default (BQ=64, BKV=64, w=4, s=2) for D=256 |
| `dkv_stages_sweep.py` | 2026-04-17 | s=2 is only viable for D=512 (s=1/3 OOM shmem) — FAILED |
| `dkv_split_bench.py` | 2026-04-17 | Split into dV + dK kernels net-negative (+35% time) — FAILED |
| `swa_e2e_bench.py` | 2026-04-17 | N-gate dKV config for short vs long N in E2E training |

## Output conventions

- **JSON results**: each bench saves `{script_name}.json` next to itself. `results.json` is
  `run_final_benchmark.py`'s composite output (used by `replot.py`).
- **PNG figures**: only `run_final_benchmark.py` produces plots (`flops_vs_sdpa.png`,
  `e2e_latency_vs_sdpa.png`, `memory_vs_sdpa.png`). Embedded in README.md.

## Measurement gotchas

1. **GPU contamination**: if another process uses the same GPU (check `nvidia-smi`), absolute
   timings are unreliable. Ratios (speedup) still roughly hold. Use a dedicated GPU for
   release numbers.
2. **Triton JIT cold start**: first call compiles. Always warmup ≥ 10 reps before timing.
3. **CUDA event timing** (`torch.cuda.Event`) is more reliable than `time.perf_counter`
   for GPU kernels. `time.perf_counter` is fine for full E2E (fwd+bwd+optimizer) steps.
4. **Median > mean** for noisy environments. All scripts sort `ts` and take `ts[len//2]`.

## Adding a new benchmark

1. Name it by purpose: `what_scenario_kind.py` (e.g. `dq_config_c_sweep.py`).
2. If it's a one-off sweep, move to `archive/` after the finding is landed in `attention.py`.
3. Always save a JSON with same stem as the script for reproducibility.
4. Use `flash_attn.utils.benchmark_fn` or the local `time_cuda` helper; don't reinvent.
