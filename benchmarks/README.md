# Benchmark 指南

## 当前正式入口

| 用途 | 入口 | 说明 |
| --- | --- | --- |
| batched canonical | `benchmark_registry.py` | 走 public API 与 registry，先做同语义正确性，再记录 latency/MFU/selection |
| varlen canonical | `benchmark_varlen_registry.py` | packed/cu_seqlens 正式性能入口 |
| 结果回归 | `compare_registry_results.py` | 严格匹配 cell，检查缺失、正确性和性能退化 |
| NCU/NSYS target | `profile_varlen_target.py` | 只 profile registry 实际选择的 production path |
| B200 D512 候选复现 | `probe_varlen_d512_candidate.py` | candidate-only；不是 production benchmark |

示例：

```bash
python benchmarks/benchmark_registry.py \
  --profile gemma4_e2b_text_full \
  --seq-len 2048 \
  --phase forward_backward \
  --dtype bfloat16

python benchmarks/benchmark_varlen_registry.py \
  --profile gemma4_e2b_text_full \
  --lengths 2048,2048,2048,2048 \
  --phase forward_backward \
  --dtype bfloat16
```

正式结论必须同时保存 GPU/软件栈、git dirty 状态、registry selection、正确性、原始样本和分位数；不同硬件或提交不得覆盖旧结果。

## 三代硬件资产

- H100：保留历史 tuning 脚本、JSON/PNG 和 product override。旧的内嵌 benchmark 已迁到 `history/h100/attention_embedded_benchmark.py`。
- H200：保留 varlen 早期实现、测试和结果；当前 production 使用 `sm90` compile-safe base，尚未形成独立 tuned override。
- B200：当前 canonical 证据位于 `exp/b200_speedup/`，D512 production 六个 workload 为 1.927–3.144× SDPA。
- 失败实验：grouped forward、fused backward、split dKV 等必须连同复现脚本保留，但不再作为生产门禁。

当前根目录仍有一批 H100/H200 历史脚本，后续会按硬件移动；移动前不删除，不把旧数字当作跨硬件结论。

## 历史脚本与结论索引

以下结论来自早期 H100/H200 调优，继续保留：

| 脚本 | 日期 | 保存的结论 |
| --- | --- | --- |
| `archive/dkv_sweep_D512.py` | 2026-04-16 | H100 D512 pack-GQA：BKV16、BQ64、warps4 |
| `archive/dkv_swa_sweep_D256.py` | 2026-04-16/17 | H100 D256 SWA 需要按 N 选择 dKV 配置 |
| `archive/dkv_qsplits_sweep.py` | 2026-04-17 | raw-grid Q_SPLITS target 为 128/256 |
| `archive/dkv_bkv_qsplits_joint.py` | 2026-04-17 | BKV32/64 与 Q_SPLITS 存在耦合 |
| `archive/dkv_config_a_sweep.py` | 2026-04-17 | H100 Config A 中 BKV64 优于旧结论 |
| `archive/dkv_config_b_bkv64.py` | 2026-04-17 | 极端 SM-starve（grid≤16）可由 BKV64 改善 |
| `archive/dq_sweep_D512.py` | 2026-04-17 | H100 D512 dQ：BQ32/BKV64/warps8/stages2 |
| `archive/dq_config_a_sweep.py`、`dq_config_b_sweep.py` | 2026-04-17 | H100 D256 dQ：BQ64/BKV64/warps4/stages2 |
| `archive/dkv_stages_sweep.py` | 2026-04-17 | D512 stages1/3 shared-memory OOM，stages2 可用 |
| `archive/dkv_split_bench.py` | 2026-04-17 | dV+dK 拆分净退化约 35% |
| `archive/swa_e2e_bench.py` | 2026-04-17 | SWA dKV 的 short/long N-gate |
| `bench_varlen.py` + `varlen_bench.json` | H200/Triton 3.2 | Zipf packed workload 为 3.14–24.76× padded-batched；详见 `docs/varlen.md` |

历史诊断入口也保留：`profile_short_n.py`、`profile_swa.py`、
`profile_n8k.py`、`bwd_breakdown.py`、`diag_short_n.py`、
`dump_kernel_regs.py`、`alloc_overhead.py` 和
`triton_launch_overhead.py`。它们用于定位旧配置，不是当前 canonical
release benchmark。

## 当前门禁

```bash
# CPU 合同与逻辑测试
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q

# 目标 GPU 数值门禁
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q --run-gpu
```

截至 e051 防回退门禁：CPU 100 passed、50 skipped；B200 GPU 50/50；
D512 六个 production cell 为 1.946–3.095× SDPA，且相对 e042 的 Triton
median 最大绝对变化 0.71%。H100/H200 需要在对应实机重新认证，不能复用
B200 结果。

## 历史脚本规则

- `history/<gpu>/`：可复现但不再是当前入口；
- `archive/`：旧 tuning sweep，暂待按硬件细分；
- 旧脚本若使用硬编码 H100 峰值、旧 Torch/Triton 或非同语义 SDPA，必须在文件头标注；
- 失败实验保留负面结论与环境，不因失败删除；
- H100/H200/B200 相关脚本、配置和结果只允许迁移与加标签，不允许删除；
- 新实验使用唯一 ID，结果写入新的时间戳目录，禁止覆盖。

历史测量约束同样保留：先确认 GPU 无污染；排除 Triton JIT cold start；
GPU kernel 使用 CUDA event；保存原始样本和 median，不能只保留单个均值。
