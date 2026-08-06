# Benchmark 指南

## 当前正式入口

| 用途 | 入口 | 说明 |
| --- | --- | --- |
| batched canonical | `benchmark_registry.py` | 走 public API 与 registry，先做同语义正确性，再记录 latency/MFU/selection |
| varlen canonical | `benchmark_varlen_registry.py` | packed/cu_seqlens 正式入口；记录正确性、吞吐、显存峰值与 selection |
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

Backward 大梯度范围可用 `--grad-output-scale` 压测；绝对误差门槛会随尺度线性调整。

正式结论必须同时保存 GPU/软件栈、git dirty 状态、registry selection、正确性、
原始样本、分位数和增量峰值 allocated/reserved；不同硬件或提交不得覆盖旧结果。

## 三代硬件资产

- H100：保留历史 tuning 脚本、JSON/PNG 和 product override。旧的内嵌 benchmark 已迁到 `history/h100/attention_embedded_benchmark.py`。
- H200：保留 varlen 早期实现、测试和结果；当前 production 使用 `sm90` compile-safe base，尚未形成独立 tuned override。
- B200：当前 canonical 证据位于 `exp/b200_speedup/`。stage3+qsplit production 的
  D512 单序列 2K–256K 均超过 1.5× SDPA；E2B 2K 已到 4.33×。固定总长 256K 的不同 packed
  分布为 2.39–8.38×。E2B/MoE 256K 峰值显存约为 SDPA 的 22.5%；B200
  qsplit 独立 scratch 与提前释放 delta 使 BF16 8K 峰值下降 4.67%。FP16
  qsplit 进一步使用同 dtype scratch，峰值约再降14%，full F+B 快1.3%–5.7%；
  BF16 E2B full batch1 当前使用 head-grid：raw32–67为q3+FP32 scratch，
  raw68–105为q2/w4+BF16x2，raw106–536为q1+BF16x2；
  MoE BF16 2K 使用 q14/w8，其他 raw64–95 使用 q8/w8、96–127 使用 q4/w8。
  FP16 保留 E2B raw32 q13 与 MoE raw64–71 q9。forward BKV64 仅覆盖已验证的 E2B
  raw32–240 与 MoE raw64–96；packed、sliding 和更长区间保持原配置。
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

截至 e331：CPU 150 passed、50 skipped；B200 GPU 完整门禁 200/200，GPU
1–7 各另复测 varlen 8/8。单序列 head-grid/qsplit、BF16x2 scratch、packed
防回退与2K–256K矩阵均正确。
H100/H200 需要在对应实机重新认证，
不能复用 B200 结果。

## 历史脚本规则

- `history/<gpu>/`：可复现但不再是当前入口；
- `archive/`：旧 tuning sweep，暂待按硬件细分；
- 旧脚本若使用硬编码 H100 峰值、旧 Torch/Triton 或非同语义 SDPA，必须在文件头标注；
- 失败实验保留负面结论与环境，不因失败删除；
- H100/H200/B200 相关脚本、配置和结果只允许迁移与加标签，不允许删除；
- 新实验使用唯一 ID，结果写入新的时间戳目录，禁止覆盖。

历史测量约束同样保留：先确认 GPU 无污染；排除 Triton JIT cold start；
GPU kernel 使用 CUDA event；保存原始样本和 median，不能只保留单个均值。
