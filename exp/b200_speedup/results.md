# Experiment Results

所有时间固定为 PST（UTC-8）。实验必须先登记再启动；retry 使用新 ID。

## 当前结论

- 目标已达到：最终单序列/packed矩阵16/16正确，全部超过1.5×同语义SDPA。
- 首轮 production：1.956–3.146×，几何平均 2.389×。
- 50-repeat 跨卡复测：1.927–3.144×，几何平均 2.387×。
- output/dQ/dK/dV：e067 的 24/24 production benchmark cells 全通过。
- 当前production：B200 BF16 E2B batch1 dKV按raw-grid选择q3 `[32,68)`、q2 `[68,106)`、q1 `[106,537)`；长端与packed使用已验证BKV64或安全路径。
- total-128K 已覆盖 32K×4、64K×2、128K×1 和不均匀组成；production 为 2.323–2.475× SDPA，几何平均 2.417×。
- 原始 baseline 为 0.36–0.64×，几何平均 0.483×。
- 256K NSYS：dKV/dQ/forward分别占43.0%/32.3%/24.6%；e072/e076/e077三核NCU均已归档。
- e070 起统一报告吞吐与增量峰值显存；显存测量独立于 latency，正确性先于性能。
- e070阶段total-256K packed 8/8为2.283–2.439×；历史结果保留。
- e075阶段2K–256K单序列10/10为1.563–2.516×；历史结果保留。
- qsplit 晋级后单序列 2K–256K 为 2.22–2.88× SDPA；packed、H100/H200 和未知 sm100 路径不变。
- qsplit 后 8K NSYS 为 dKV/dQ/forward 41.4%/31.8%/26.8%；e129/e132 未找到第二个可靠 winner。
- B200 qsplit 独立 scratch + 提前释放 delta 使 8K 峰值 185,073,664→176,422,912 bytes（-4.67%）；50-repeat 吞吐无回退。
- MoE full D512 单序列 2K–<4K 使用 w8，full F+B 再快 2.44%–5.34%、显存不变；4K+、E2B、packed、sliding 保持原路径。
- e179阶段E2B raw32–105使用q8/w8，2K为1.367ms、3.50×SDPA；后续已由head-grid取代，历史结果保留。
- q8 后 E2B2K 的 dKV/dQ/forward 为49.3%/26.5%/23.5%；dKV相对q2约快55%。
- forward BKV64 仅在B200 D512 full batch1安全区间晋级：E2B raw32–240、MoE raw64–96。E2B2K为1.325ms、3.61×SDPA、0.041GiB；MoE2K为2.242ms、3.00×。
- relaxed 后 BF16 q14 曾晋级 E2B raw32–34/45–76 与 MoE 精确2K；当时E2B/MoE 2K为4.20×/3.38×，历史结论保留。
- 当前BF16矩阵：E2B单序列geo2.899×、packed geo2.768×，16/16正确；MoE为2.46×–3.38×。
- B200 BF16 E2B head-grid 已晋级：raw32–67使用q3，raw68–105使用q2/w4+BF16x2，raw106–536使用q1+BF16x2。e314将raw281–316吞吐提高52.6%–53.5%。
- raw106–223 相对真实 q4 production 快 1.1%–4.7%，并把增量峰值显存降低约14%；raw224–280、317–536在已测收益岛快约2%–10.6%，显存不增。
- raw68–105 使用 q2/w4+BF16x2：raw68–71双seed full F+B 吞吐提升0.69%–0.77%，allocator增量峰值降低14.12%–14.88%；raw64出现dV超门槛，未纳入。
- e331最终门禁：CPU150；B200 GPU200+7×8/8；raw537精确回落，H100/H200/未知sm100未接管。

| Profile | Workload | Baseline | 当前 Production | 最终 Speedup |
| --- | --- | ---: | ---: | ---: |
| E2B | balanced | 30.360 ms | 4.997 ms | 3.847× SDPA |
| E2B | skewed | 36.322 ms | 8.148 ms | 2.529× SDPA |
| E2B | dominant | 130.158 ms | 21.028 ms | 2.791× SDPA |
| MoE | balanced | 56.812 ms | 8.320 ms | 3.257× SDPA |
| MoE | skewed | 70.350 ms | 10.959 ms | 2.895× SDPA |
| MoE | dominant | 278.036 ms | 35.084 ms | 2.840× SDPA |

e067 门禁已完成：CPU 107 passed；B200 GPU 50/50 passed；24/24 production 性能单元数值与 selection 通过。H100/H200 和未知 sm100 路径未修改。

代码整理状态：首批仅把 `attention.py` 内嵌的 H100 历史 benchmark 移到
`benchmarks/history/h100/`，并更新当前测试/benchmark 文档；未修改 kernel、
registry 或配置。新增 12 个 H200 selection 防回退 case 后，CPU 为
100 passed、50 GPU skipped。

`e051` 已完成整理后防回退复测：CPU 100 passed、B200 GPU 50/50；
D512 六个 cell 为 1.946–3.095× SDPA，几何平均 2.378×，selection 与数值
全部通过。相对 e042 的 Triton median 最大绝对变化 0.71%，无回退。

| ID | Run type | Parent | 唯一变量 | 预测与判断标准 | Status | Verdict | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `e000_baseline_b200` | superseded plan | — | 错误地以 batched API 为目标 | 用户澄清只优化 varlen；未启动、无运行 artifact | stopped | diagnostic | scope corrected before execution |
| `e001_baseline_b200_repeat` | superseded plan | `e000_baseline_b200` | 错误 batched baseline 的 repeat | 未启动、无运行 artifact | stopped | diagnostic | superseded with e000 |
| `e005_varlen_tooling_smoke` | probe | — | 新增 registry-aware canonical varlen benchmark；kernel/config 不变 | CPU 合约、B200 varlen unit gate、一个 full 与一个 true-sliding smoke 全通过；失败则先修工具，不启动 baseline | completed | improved | `runs/e005_varlen_tooling_smoke/summary.json` |
| `e010_varlen_baseline_b200` | baseline | — | 无；冻结当前 sm100 varlen registry | 用户补充要求 SDPA speedup 后工具口径改变；已完成 8 cells 保留为无 SDPA 的 diagnostic，不参与新 baseline | stopped | diagnostic | `runs/e010_varlen_baseline_b200/` |
| `e011_varlen_baseline_repeat` | independent repeat | `e010_varlen_baseline_b200` | 无；独立重复旧口径 varlen baseline | 未启动；随无 SDPA 的 e010 一起过期 | stopped | diagnostic | — |
| `e012_varlen_baseline_parallel_continuation` | baseline continuation | `e010_varlen_baseline_b200` | 余下 16 cells 固定映射到 GPU 0–7 并行；代码/config/workload 不变 | 未启动；SDPA 指标要求使旧 baseline schema 过期 | stopped | diagnostic | `runs/e012_varlen_baseline_parallel_continuation/` |
| `e013_sdpa_metric_tooling_smoke` | probe | `e005_varlen_tooling_smoke` | canonical cell 新增 exact-semantics per-sample PyTorch SDPA timing、MFU 与 speedup | B200 full/sliding smoke 通过；Triton/SDPA 都保存 raw latency、semantic TFLOPS/MFU，speedup 有限正值 | completed | improved | `runs/e013_sdpa_metric_tooling_smoke/summary.json` |
| `e014_varlen_sdpa_baseline_b200` | baseline | `e013_sdpa_metric_tooling_smoke` | 无；8 卡固定分片采集新口径 24-cell baseline | 24/24 通过；GPU 0–3 在测量前已有外部 Ray actor，故其 D512 cells 需 clean-GPU retry；GPU 4–7 D256 cells 可用 | completed | uncertain | `runs/e014_varlen_sdpa_baseline_b200/summary.md` |
| `e015_varlen_sdpa_baseline_repeat` | independent repeat | `e014_varlen_sdpa_baseline_b200` | 原计划同 workload→GPU 映射独立重复 | 因 GPU 污染改由 e016 clean-GPU retry 取代；未启动 | stopped | diagnostic | — |
| `e016_d512_baseline_clean_gpu_retry` | retry | `e014_varlen_sdpa_baseline_b200` | D512 的 12 cells 从受外部 Ray 干扰的 GPU 0–3 移到空闲 GPU 4–7；代码/config/workload 不变 | 与 e014 selection/correctness 一致；若 latency 差异超 dispersion，以 e016 作为 D512 parent | completed | diagnostic | `runs/e016_d512_baseline_clean_gpu_retry/summary.md` |
| `e020_d512_backward_nsys` | profile | `e016_d512_baseline_clean_gpu_retry` | 对 clean GPU 上 E2B D512 balanced F+B 仅采一个 warm target 的 CUDA/NVTX timeline | dKV 26.156 ms，占三个 Triton 内核时间 92.5%；dQ 1.159 ms，forward 0.958 ms | completed | diagnostic | `runs/e020_d512_backward_nsys/summary.md` |
| `e021_d512_dkv_ncu` | profile | `e020_d512_backward_nsys` | 仅对当前 registry 选择的 D512 varlen dKV 采 NCU full 指标；代码/config/workload 不变 | 166.91 KB shared/block 将 occupancy 限制到 6.25%；eligible 3.45%，52.1M local spill requests；首个单变量轴选 BQ | completed | diagnostic | `runs/e021_d512_dkv_ncu/summary.md` |
| `e030_d512_dkv_bq_probe` | probe | `e016_d512_baseline_clean_gpu_retry` | 只改变 D512 varlen dKV 的 BQ：16/32/64/128；BKV=16、w=4、s=2、QS=1 固定 | BQ16 全数值门禁通过，F+B 6.222 ms、3.144× SDPA；BQ64 28.263 ms、0.686×，BQ32 44.369 ms、0.444×；BQ128 shared-memory OOR | completed | improved | `runs/e030_d512_dkv_bq_probe/summary.md` |
| `e031_d512_dkv_bq16_confirm` | confirmation | `e030_d512_dkv_bq_probe` | 无；BQ16 候选、workload 与口径不变，仅将 warmup/repetitions 提高到 10/50 | 全数值门禁通过；F+B 6.221 ms、3.096× SDPA，p20–p80 为 6.208–6.231 ms | completed | improved | `runs/e031_d512_dkv_bq16_confirm/summary.md` |
| `e032_d512_dkv_bq16_repeat` | independent repeat | `e030_d512_dkv_bq_probe` | 无；在另一张空闲 B200 独立复测 BQ16，10 warmup/50 repetitions | 全数值门禁通过；F+B 6.214 ms、3.083× SDPA，与 e031 中位数仅差 0.12% | completed | improved | `runs/e032_d512_dkv_bq16_repeat/summary.md` |
| `e033_d512_bq16_family_sweep` | benchmark | `e032_d512_dkv_bq16_repeat` | 配置固定为 BQ16；workload 扩展到 E2B/MoE × balanced/skewed/dominant 六个 D512 full F+B cells | 6/6 数值门禁通过；各 cell 1.956–3.144× SDPA，几何平均 2.387×，均超过 1.5× | completed | improved | `runs/e033_d512_bq16_family_sweep/summary.md` |
| `e040_promote_bq16_registry` | code change | `e033_d512_bq16_family_sweep` | 将 BQ16 作为 B200+Torch2.11+Triton3.6 专用 varlen D512 dKV tuned override；保留 sm90 与未知 sm100 安全基线 | CPU 合同 80/80、B200 varlen FP16/BF16 数值 8/8 通过；B200 选择 BQ16/QS1，未知 sm100 保持 BQ64 基线 | completed | improved | `runs/e040_promote_bq16_registry/summary.md` |
| `e041_production_d512_family` | benchmark | `e040_promote_bq16_registry` | 不再注入候选；用 production registry/public API 重跑六个 D512 full F+B cells | 6/6 选择 B200 tuned override且数值通过；1.956–3.146× SDPA，几何平均 2.389× | completed | improved | `runs/e041_production_d512_family/summary.md` |
| `e042_production_family_repeat` | independent repeat | `e041_production_d512_family` | production 配置与 workload 不变；换卡并提高到 10 warmup/50 repetitions | 6/6 数值与 selection 一致；1.927–3.144× SDPA，几何平均 2.387×；raw p20–p80 稳定 | completed | improved | `runs/e042_production_family_repeat/summary.md` |
| `e050_b200_release_gate` | validation | `e042_production_family_repeat` | 性能配置不变；并行执行 batched/varlen/image-group/vision/语义不变量 50-case GPU gate 与完整 CPU suite | CPU 88 passed；B200 GPU 50/50 passed，无失败或 required skip | completed | improved | `runs/e050_b200_release_gate/summary.md` |
| `e051_refactor_no_regression` | validation | `e042_production_family_repeat` | production kernel/config 不变；迁移内嵌 H100 benchmark、补 H200 selection 合同并更新文档 | CPU 100 passed、B200 GPU 50/50；D512 6/6 正确且 1.946–3.095×；最大 Triton latency delta 0.71% | completed | no-impact | `runs/e051_refactor_no_regression/summary.md` |
| `e060_d512_long_matrix_baseline` | baseline | `e051_refactor_no_regression` | production config 不变；新增 4K×4、8K×4、16K×2、32K×1、ragged-16K，覆盖 E2B/MoE 共 10 cells | 10/10 正确且 selection 不变；1.59–2.50× SDPA，几何平均约 1.82×；32K×1 最弱 | completed | diagnostic | `runs/e060_d512_long_matrix_baseline/summary.md` |
| `e061_d512_128k_probe` | probe | `e060_d512_long_matrix_baseline` | total packed tokens 固定 131072，组成改为 32K×4、64K×2、128K×1、64K 主导 ragged；E2B/MoE 共 8 cells | 8/8 正确、无 OOM/fallback；1.504–1.592× SDPA，几何平均 1.563×；E2B 128K×1 最弱 | completed | diagnostic | `runs/e061_d512_128k_probe/summary.md` |
| `e062_d512_128k_nsys` | profile | `e061_d512_128k_probe` | 只对 E2B 128K×1 production F+B 采一次 CUDA/NVTX timeline | dKV 3076.992 ms（62.8%）、dQ 1043.720 ms（21.3%）、forward 782.546 ms（16.0%）；首轴选择 dKV BKV | completed | diagnostic | `runs/e062_d512_128k_nsys/summary.md` |
| `e063_d512_128k_dkv_bkv_probe` | probe | `e061_d512_128k_probe` | E2B 128K×1；固定 BQ16/warps4/stages2/QS1，只改变 dKV BKV=8/16/32/64 | 全部正确；BKV64 3203.026 ms、2.328×，比 BKV16 快 34.9%；BKV32 退化到 21186.588 ms、0.351× | completed | improved | `runs/e063_d512_128k_dkv_bkv_probe/summary.md` |
| `e064_d512_128k_bkv64_family` | benchmark | `e063_d512_128k_dkv_bkv_probe` | 固定 BKV64 winner，扩展到 E2B/MoE × 4 个 total-128K workload | 8/8 正确；2.323–2.470× SDPA，几何平均 2.418×；每个 cell 比 BKV16 快 34.8–35.3% | completed | improved | `runs/e064_d512_128k_bkv64_family/summary.md` |
| `e065_d512_bkv64_crossover` | benchmark | `e064_d512_128k_bkv64_family` | BKV64 固定，回测原 6-cell 短矩阵与 e060 10-cell 长矩阵 | 16/16 正确；2.391–3.848× SDPA；15 cells 明显更快，E2B 短不均匀跨实验慢 1.2%，需同卡 A/B 消歧 | completed | improved | `runs/e065_d512_bkv64_crossover/summary.md` |
| `e066_d512_bkv64_paired_confirm` | confirmation | `e065_d512_bkv64_crossover` | 同卡顺序运行 BKV16/BKV64；覆盖 E2B/MoE 短不均匀与 128K×1 | 8/8 正确；E2B 短不均匀确认回退 0.86%，MoE 快 22.1%；128K 两 profile 均快约 35.2% | completed | improved | `runs/e066_d512_bkv64_paired_confirm/summary.md` |
| `e067_promote_bkv64_grid_gate` | code change | `e066_d512_bkv64_paired_confirm` | 新增 B200-only BKV64 高优先级 override，raw-grid 仅覆盖 `[128,257)∪[512,+∞)`；保留 BKV16 fallback | CPU 107 passed、GPU 50/50；24/24 production 正确；短/长/128K 分别为 2.529–3.847×、2.389–3.234×、2.323–2.475× | completed | improved | `runs/e067_promote_bkv64_grid_gate/summary.md` |
| `e068_d512_160k_256k_scaling` | baseline | `e067_promote_bkv64_grid_gate` | production 配置不变；新增 E2B/MoE × 160K/192K/224K/256K 单序列 | 7 cells 正确且 2.274–2.354×；MoE 256K 在 cosine 的单次 `torch.dot` 32 位长度上限失败，非 kernel 失败 | failed | diagnostic | `runs/e068_d512_160k_256k_scaling/summary.md` |
| `e069_chunked_metrics_moe256k_retry` | tooling fix + retry | `e068_d512_160k_256k_scaling` | 只把超大张量 dot 改为分块 FP32 dot；production kernel/config/workload 不变 | CPU 108 passed；MoE 256K output/dQ/dK/dV 与 selection 通过，25996.464 ms vs SDPA 59865.969 ms（2.30×） | completed | no-impact | `runs/e069_chunked_metrics_moe256k_retry/summary.md` |
| `e070_d512_256k_packed_family` | baseline | `e069_chunked_metrics_moe256k_retry` | production 配置不变；E2B/MoE × 64K×4、128K×2、256K×1、128K 主导不均匀；新增独立显存峰值观测 | 8/8 正确；2.283–2.439×；Triton 显存比 SDPA 少 65.4%–77.5% | completed | diagnostic | `runs/e070_d512_256k_packed_family/summary.md` |
| `e071_d512_256k_nsys` | profile | `e068_d512_160k_256k_scaling` | 只对已正确通过的 E2B 256K×1 production F+B 采一次 CUDA/NVTX timeline | dKV 5606.015 ms（43.0%）、dQ 4208.199 ms（32.3%）、forward 3209.655 ms（24.6%） | completed | diagnostic | `runs/e071_d512_256k_nsys/summary.md` |
| `e072_d512_256k_dkv_ncu` | profile | `e071_d512_256k_nsys` | 只对占比最高的 production dKV kernel 采 NCU full 指标；代码/config/workload 不变 | 255 regs、166.03 KiB shared、6.25% occupancy、9.55M spill、87.66% no-eligible | completed | diagnostic | `runs/e072_d512_256k_dkv_ncu/summary.md` |
| `e073_d512_256k_ragged_nsys` | profile | `e070_d512_256k_packed_family` | 对已正确通过的 E2B total-256K ragged production F+B 采一次 NSYS | dKV/dQ/forward 为 43.0%/32.5%/24.6%，与 256K×1 占比基本一致 | completed | diagnostic | `runs/e073_d512_256k_ragged_nsys/summary.md` |
| `e074_d512_256k_moe_nsys` | profile | `e070_d512_256k_packed_family` | 对 MoE 256K×1 与 total-256K ragged 各采一次 NSYS；production config 不变 | single 为 43.4%/32.3%/24.3%，ragged 为 43.0%/32.4%/24.5%；与 E2B 一致 | completed | diagnostic | `runs/e074_d512_256k_moe_nsys/summary.md` |
| `e075_d512_memory_scaling_baseline` | baseline | `e070_d512_256k_packed_family` | production config 不变；E2B/MoE 单序列 2K、8K、32K、128K，256K 复用 e070 | 10/10 正确；1.563–2.516×；Triton 显存线性，比 SDPA 少 77.5%–82.8% | completed | diagnostic | `runs/e075_d512_memory_scaling_baseline/summary.md` |
| `e076_d512_256k_dq_ncu` | profile | `e071_d512_256k_nsys` | 只对占 32.3% 的 production dQ kernel 采 NCU full 指标 | 181 regs、200.70 KiB shared、12.5% occupancy、0 spill；short-scoreboard 为主 | completed | diagnostic | `runs/e076_d512_256k_dq_ncu/summary.md` |
| `e077_d512_256k_forward_ncu` | profile | `e071_d512_256k_nsys` | 只对稳定占约 24.5% 的 production forward kernel 采 NCU full 指标 | 255 regs、100.35 KiB shared、12.5% occupancy、9.67B spill；先扫 warps | completed | diagnostic | `runs/e077_d512_256k_forward_ncu/summary.md` |
| `e080_d512_dkv_warps_probe` | probe | `e021_d512_dkv_ncu` | E2B 128K×1；固定 BQ16/BKV64/stages2/QS1，只改变 dKV warps=2/4/8 | w2 严重退化，w8 慢 0.87%；3/3 正确、显存相同，保持 w4 | completed | slower | `runs/e080_d512_dkv_warps_probe/summary.md` |
| `e081_d512_dkv_stages_probe` | probe | `e021_d512_dkv_ncu` | E2B 128K×1；固定 BQ16/BKV64/w4/QS1，只改变 stages=1/2/3/4 | s3 比 s2 快 5.6%，显存不变；s1 慢 7.1%，s4 快 5.1%；全部正确 | completed | improved | `runs/e081_d512_dkv_stages_probe/summary.md` |
| `e082_d512_dkv_s3_confirm` | confirmation + repeat | `e081_d512_dkv_stages_probe` | s3 winner、E2B 128K×1 不变；两张物理卡提高 warmup/repetitions | 两卡差 0.10%，相对 s2 快 5.40%/5.31%；正确且显存不变 | completed | improved | `runs/e082_d512_dkv_s3_confirm/summary.md` |
| `e083_d512_dkv_s3_family_probe` | benchmark | `e081_d512_dkv_stages_probe` | s3 winner 固定；workload 扩到 E2B/MoE 8K、E2B 32K、MoE 128K | 4/4 正确；full F+B 快 2.44%–5.45%，显存不变 | completed | improved | `runs/e083_d512_dkv_s3_family_probe/summary.md` |
| `e084_d512_dkv_s3_256k_family` | benchmark | `e083_d512_dkv_s3_family_probe` | s3 winner 固定；与 e070 同口径覆盖 E2B/MoE × 64K×4、128K×2、256K×1、ragged | 8/8 正确；full F+B 快 3.91%–6.35%，显存不变 | completed | improved | `runs/e084_d512_dkv_s3_256k_family/summary.md` |
| `e085_d512_128k_dkv_s3_ncu` | profile | `e082_d512_dkv_s3_confirm` | 对已确认的 E2B 128K s3 dKV 采 NCU full 指标 | occupancy 6.25%、spill 7.27M、no-eligible 85.5%；等待 s2 control 差分 | completed | diagnostic | `runs/e085_d512_128k_dkv_s3_ncu/summary.md` |
| `e086_d512_dkv_s3_8k_repeat` | independent repeat | `e083_d512_dkv_s3_family_probe` | E2B 8K s3 不变；同物理卡提高到 10 warmup/50 repetitions | 18.810 ms，比 production 快 2.69%；正确且显存不变 | completed | improved | `runs/e086_d512_dkv_s3_8k_repeat/summary.md` |
| `e087_d512_dkv_s3_moe8k_repeat` | independent repeat | `e083_d512_dkv_s3_family_probe` | MoE 8K s3 不变；提高到 10 warmup/50 repetitions | 25.922 ms，比 production 快 3.72%；正确且显存不变 | completed | improved | `runs/e087_d512_dkv_s3_moe8k_repeat/summary.md` |
| `e088_d512_dkv_s3_32k_repeat` | independent repeat | `e083_d512_dkv_s3_family_probe` | E2B 32K s3 不变；提高到 10 warmup/20 repetitions | 194.261 ms，比 production 快 2.39%；正确且显存不变 | completed | improved | `runs/e088_d512_dkv_s3_32k_repeat/summary.md` |
| `e089_d512_128k_dkv_s2_ncu` | profile control | `e085_d512_128k_dkv_s3_ncu` | 同一 E2B 128K workload 改回 production s2，采 NCU full 指标 | s3 duration -10.8%，issue 提升；shared +19.8%、spill +52.1%，occupancy 不变 | completed | diagnostic | `runs/e089_d512_128k_dkv_s2_ncu/summary.md` |
| `e090_d512_dkv_s3_160k_224k_scaling` | benchmark | `e084_d512_dkv_s3_256k_family` | s3 winner 固定；E2B/MoE × 160K/192K/224K 单序列 | 6/6 正确；相对 e068 快 5.62%–6.01%，显存线性 | completed | improved | `runs/e090_d512_dkv_s3_160k_224k_scaling/summary.md` |
| `e091_promote_dkv_s3_grid_gate` | code change | `e084_d512_dkv_s3_256k_family` | 只将 B200 BKV64 grid-gated override 的 stages 2→3；其他 config 不变 | registry/CPU/GPU gate 通过；production 2K–256K 无正确性、吞吐或显存回退 | completed | improved | `runs/e091_promote_dkv_s3_grid_gate/` |
| `e093/e094_rawgrid448` | probe + paired confirmation | `e066_d512_bkv64_paired_confirm` | E2B raw-grid 448，同卡 BKV16/s2 对 BKV64/s3 | 8.2219→8.0293 ms（+2.34%），正确且显存相同；暂不扩大 gate | completed | improved | `runs/e094_d512_rawgrid448_pair/summary.md` |
| `e100_d512_dq_stage_sweep` | probe | `e076_d512_256k_dq_ncu` | E2B 128K；固定 dQ BQ32/BKV64/w8，只改 stages | s2 比 s1 快 11.5%；s3/s4 shared OOR；保持 s2 | completed | no-impact | `runs/e100_d512_dq_stage_sweep/summary.md` |
| `e101_d512_fwd_warp_sweep` | probe | `e077_d512_256k_forward_ncu` | E2B 128K；固定 forward BQ32/BKV32/s2，只改 warps | w4 最快；w2/w8 分别慢 21.6%/16.7% | completed | no-impact | `runs/e101_d512_fwd_warp_sweep/summary.md` |
| `e102_d512_dq_tile_sweep` | probe | `e100_d512_dq_stage_sweep` | dQ 固定 s2，缩小 BQ 或 BKV | 慢 15.1%–25.1%；无显存收益 | completed | slower | `runs/e102_d512_dq_tile_sweep/summary.md` |
| `e103/e104_forward_tile` | probe | `e101_d512_fwd_warp_sweep` | forward 扫 stages、BQ、BKV 与资源组合 | 正确候选慢 1.2%–26.2%；另有 OOR/失败签名；无晋级 | completed | slower | `runs/e103_d512_fwd_tile_stage/summary.md` |
| `e105_d512_dkv_qsplits` | probe | `e085_d512_128k_dkv_s3_ncu` | 固定 BQ16/BKV64/w4/s3，只改 q_splits | 无吞吐收益；峰值显存增加约 0.50 GiB | completed | slower | `runs/e105_d512_dkv_qsplits/summary.md` |
| `e106_d512_production_stage3_grid` | production benchmark | `e091_promote_dkv_s3_grid_gate` | 真实 registry；E2B/MoE 单序列 2K–256K | 全部 >1.5×；E2B/MoE 256K 为 2.39×/2.48×，显存约为 SDPA 22.5% | completed | improved | `runs/e106_d512_production_stage3_grid/summary.md` |
| `e107_d512_production_long_grid` | production benchmark | `e106_d512_production_stage3_grid` | E2B 160/192/224K、E2B/MoE ragged 256K | 全部正确；2.39–2.54×，显存线性 | completed | improved | `runs/e107_d512_production_long_grid/summary.md` |
| `e108_d512_production_packed_grid` | production benchmark | `e106_d512_production_stage3_grid` | 固定总长 256K，扫描 2K×128 到 256K×1 | E2B 2.39–8.38×，MoE 2.48–6.52×；显存由总 token 决定 | completed | diagnostic | `runs/e108_d512_production_packed_grid/summary.md` |
| `e109_b200_stage3_release_gate` | validation | `e106_d512_production_stage3_grid` | 完整 CPU/GPU、跨卡 varlen、compile/diff/selection 门禁 | CPU 108；B200 50/50；另 3 卡 varlen 各 8/8；全部通过 | completed | no-impact | `runs/e109_b200_stage3_release_gate/summary.md` |
| `e110–e113_dkv_reloadv` | source probe + revert | `e085_d512_128k_dkv_s3_ncu` | dKV 只改变 V tile live range；以额外 L2 load 换 shared/spill | s3 OOR；s2 正确但慢 10.6%；源码完整撤回 | completed | slower | `runs/e110_d512_dkv_reloadv/summary.md` |
| `e114_dkv_reloadq` | source probe + NCU + revert | `e085_d512_128k_dkv_s3_ncu` | dK 前重载 Q，尝试缩短 live range | latency 与 regs/shared/occupancy 均不变；源码撤回 | completed | no-impact | `runs/e114_d512_dkv_reloadq/summary.md` |
| `e115_dkv_bkv32_short` | probe | `e114_dkv_reloadq` | raw128 改 BKV32 以增加 blocks，扫 stages | 最佳仍慢 25%–29%，其余严重退化 | completed | slower | `runs/e115_d512_dkv_bkv32_short_boundary/summary.md` |
| `e116–e119_dkv_qsplit_single` | probe + boundary | `e115_dkv_bkv32_short` | BKV64/s3 固定，扫描 qs 与 raw16–256 | raw16–192 快 7%–48%；raw256 profile 不一致 | completed | improved | `runs/e116_d512_dkv_qsplits_short_boundary/summary.md` |
| `e120–e124_qsplit_packed_guard` | packed probe | `e116–e119_dkv_qsplit_single` | 固定/动态 split，覆盖 homogeneous/ragged packed | 发现 1K×8/512×8 回退；收紧为 batch=1/query≥2K | completed | diagnostic | `runs/e122_d512_dkv_qsplit_target256_packed/summary.md` |
| `e125_qs4_upper_boundary` | paired confirmation | `e116–e119_dkv_qsplit_single` | raw208/224/240/248 双 profile、50-repeat | raw208 稳定 >4.7%；raw240/248 MoE <2%；上界取 224 exclusive | completed | diagnostic | `runs/e125_d512_dkv_qs4_upper_boundary/summary.md` |
| `e126_promote_dkv_qsplit_single` | code change + production | `e125_qs4_upper_boundary` | B200 batch1/query≥2K；raw32–63 qs2、64–223 qs4 | E2B 2K–8K 快 26%–42%；MoE 2K–6K 快 7%–41%；显存仍为 SDPA 21%–26% | completed | improved | `runs/e126_promote_dkv_qsplit_single/summary.md` |
| `e127_qsplit_packed_no_regression` | validation | `e126_promote_dkv_qsplit_single` | packed selection/latency、完整 CPU/GPU 门禁 | packed 保持旧 q1；CPU118、B200 50/50，全通过 | completed | no-impact | `runs/e127_qsplit_packed_no_regression/summary.md` |
| `e128_d512_dkv_qs4_ncu` | profile | `e126_promote_dkv_qsplit_single` | production E2B 8K q4 dKV NCU | grid 128→512，NCU duration 17.97→9.22 ms；单 block 资源不变 | completed | diagnostic | `runs/e128_d512_dkv_qs4_ncu/summary.md` |
| `e129_d512_dkv_qs4_resource_sweep` | 8-GPU probe | `e128_d512_dkv_qs4_ncu` | q4 下扫 warps/stages/BQ/BKV | BQ8 仅快 0.15%；其余慢 0.85%–严重退化；全部正确且显存相同 | completed | no-impact | `runs/e129_d512_dkv_qs4_resource_sweep/summary.md` |
| `e130_d512_qsplit_8k_nsys` | profile | `e128_d512_dkv_qs4_ncu` | production E2B 8K full F+B NSYS | dKV/dQ/forward 为 41.4%/31.8%/26.8% | completed | diagnostic | `runs/e130_d512_qsplit_8k_nsys/summary.md` |
| `e131_d512_qsplit_8k_other_ncu` | profile | `e130_d512_qsplit_8k_nsys` | 8K dQ 与 forward NCU | dQ 无 spill、受 shared scoreboard 限制；forward 255 regs、9.52M spill | completed | diagnostic | `runs/e131_d512_qsplit_8k_other_ncu/summary.md` |
| `e132_d512_forward_split_d512` | source probe + revert | `e131_d512_qsplit_8k_other_ncu` | D512 forward 启用 causal 两段循环 | E2B8K/MoE4K 慢 0.16%/0.18%；源码完整撤回 | completed | no-impact | `runs/e132_d512_forward_split_d512/summary.md` |
| `e133_d512_dkv_nonpower_qsplit` | 8-GPU probe | `e128_d512_dkv_qs4_ncu` | E2B8K/MoE4K 扫 q3/q5/q6/q7 | 全部正确且显存相同；无候选超过 q4 | completed | slower | `runs/e133_d512_dkv_nonpower_qsplit/summary.md` |
| `e134_d512_qsplit_separate_scratch` | 8-GPU memory probe | `e133_d512_dkv_nonpower_qsplit` | 拆分 FP32 dK/dV scratch 并顺序转 dtype | 8/8 正确；8K 峰值 -4.53%，吞吐在噪声内 | completed | improved | `runs/e134_d512_qsplit_separate_scratch/summary.md` |
| `e135_promote_qsplit_separate_scratch` | code change + production | `e134_d512_qsplit_separate_scratch` | 只对 B200 qs2/qs4 开启独立 scratch | production 8/8 正确；H100/H200/packed/q1 保持旧路径 | completed | improved | `runs/e135_promote_qsplit_separate_scratch/summary.md` |
| `e136_qsplit_memory_release_gate` | validation | `e135_promote_qsplit_separate_scratch` | 完整 CPU/GPU 与 packed/q1 防回退 | CPU118、GPU50/50；packed 三点 selection/latency 不变 | completed | no-impact | `runs/e136_qsplit_memory_release_gate/summary.md` |
| `e137_qsplit_memory_repeat` | independent repeat | `e135_promote_qsplit_separate_scratch` | E2B8K/MoE4K 10 warmup/50-repeat | 13.449/7.698 ms，均 0.165 GiB；峰值 -4.53% | completed | improved | `runs/e137_qsplit_memory_repeat/summary.md` |
| `e138_varlen_release_delta_early` | memory probe | `e137_qsplit_memory_repeat` | dKV 后提前释放 delta | 8K 再省 0.25 MiB；正确且 latency 不变 | completed | improved | `runs/e138_varlen_release_delta_early/summary.md` |
| `e139/e140_forward_hoist_q` | source probe + NCU + revert | `e131_d512_qsplit_8k_other_ncu` | forward Q tile 移到 KV 循环外 | latency 与 NCU 资源不变；源码撤回 | completed | no-impact | `runs/e139_d512_forward_hoist_q/summary.md` |
| `e141_delta_early_final_confirm` | confirmation | `e138_varlen_release_delta_early` | 最终 memory 版本 50-repeat + packed | 8K 峰值总计 -4.67%；13.442/7.692 ms；packed q1 不变 | completed | improved | `runs/e141_delta_early_final_confirm/summary.md` |
| `e142/e144_varlen_split_delta` | source probe + NSYS + revert | `e131_d512_qsplit_8k_other_ncu` | delta 从 dQ prologue 拆成独立 kernel | dQ 只快 11.8 µs，delta 花 41.1 µs；净慢约 29 µs | completed | slower | `runs/e144_d512_varlen_split_delta_nsys_retry/summary.md` |
| `e143_split_delta_nsys_failed` | failed profile | `e142_d512_varlen_split_delta` | 首次 NSYS 采集 | trace 无 CUDA kernel data；由 e144 重试取代 | failed | diagnostic | `runs/e143_d512_varlen_split_delta_nsys/summary.md` |
| `e145_d512_dq_warps` | probe | `e131_d512_qsplit_8k_other_ncu` | dQ 只改 warps=4/8/16 | w8 最优；w4 慢 0.63%，w16 慢约 18%–19% | completed | slower | `runs/e145_d512_dq_warps/summary.md` |
| `e146_final_release_audit` | validation | `e141_delta_early_final_confirm` | 最终 diff、CPU、8 卡 GPU 与三代资产审计 | CPU119；GPU50/50 + 3×varlen8/8；无删除/重命名 | completed | no-impact | `runs/e146_final_release_audit/summary.md` |
| `e147_gemma4_e2e_environment_audit` | environment audit | `e146_final_release_audit` | 检查真实模型验收前置条件 | 无本地 gated 权重/专用环境；系统 HF 依赖冲突，未伪造 E2E 结果 | completed | diagnostic | `runs/e147_gemma4_e2e_environment_audit/summary.md` |
| `e148_qsplit_memory_fp16` | dtype regression | `e141_delta_early_final_confirm` | FP16 E2B8K/MoE4K production 50-repeat | 2.37×/2.17× SDPA；峰值与 BF16 同为 176,422,912 bytes | completed | improved | `runs/e148_qsplit_memory_fp16/summary.md` |
| `e149_qs2_fp16` | dtype regression | `e148_qsplit_memory_fp16` | FP16 E2B2K qs2 50-repeat | 2.201 ms、0.041 GiB、约 1.50× FP16 SDPA；正确 | completed | diagnostic | `runs/e149_qs2_fp16/summary.md` |
| `e150_dkv_warp_specialize` | 8-GPU source probe + revert | `e131_d512_qsplit_8k_other_ncu` | dKV query loop 只启用 warp-specialize；覆盖 2K–256K/packed/GPU gate | D512 shared 266,640 > 232,448 bytes；GPU 2/8；源码撤回 | failed | failed | `runs/e150_dkv_warp_specialize/summary.md` |
| `e151_warp_specialize_revert_gate` | 8-GPU validation | `e150_dkv_warp_specialize` | 回退后复测 2K–256K/packed/GPU gate | GPU8/8；2.19–2.84×；显存/selection 恢复，无残留回退 | completed | no-impact | `runs/e151_warp_specialize_revert_gate/summary.md` |
| `e152_dkv_runtime_gqa_loop` | 8-GPU source probe + revert | `e131_d512_qsplit_8k_other_ncu` | dKV 静态 GQA loop 改普通 constexpr loop | 正确/显存不变，但各点慢 0.8%–8.5%；源码撤回 | completed | slower | `runs/e152_dkv_runtime_gqa_loop/summary.md` |
| `e153_moe4k_nsys` | profile + 8-GPU validation | `e130_d512_qsplit_8k_nsys` | MoE4K production NSYS；同步稳定版回归 | dKV/dQ/forward=47.0%/28.8%/24.2%；GPU8/8，性能/显存稳定 | completed | diagnostic | `runs/e153_moe4k_nsys/summary.md` |
| `e154_moe4k_dkv_resource_sweep` | 8-GPU probe | `e153_moe4k_nsys` | MoE4K q4 扫 BQ/BKV/warps/stages | w8 快 1.71%、显存不变；其余慢或 OOR | completed | improved | `runs/e154_moe4k_dkv_resource_sweep/summary.md` |
| `e155_moe4k_dkv_w8_paired` | 8-GPU paired confirmation | `e154_moe4k_dkv_resource_sweep` | 同卡 w4→w8，10 warmup/50-repeat | 8/8 快 1.49%–1.62%；显存/正确性一致，待长度族 | completed | improved | `runs/e155_moe4k_dkv_w8_paired/summary.md` |
| `e156_moe_dkv_w8_length_family` | 8-GPU paired family | `e155_moe4k_dkv_w8_paired` | MoE2K–7K 同卡 w4→w8 | 2K–3.5K 快 2.44%–5.34%；7K 回退0.12%；gate raw64–127 | completed | improved | `runs/e156_moe_dkv_w8_length_family/summary.md` |
| `e157_moe_dkv_w8_boundary_confirm` | 8-GPU paired confirm | `e156_moe_dkv_w8_length_family` | 2K/3.5K 各4卡 w4→w8 | 2K 快4.88%–5.16%；3.5K 快2.53%–2.70%；显存一致 | completed | improved | `runs/e157_moe_dkv_w8_boundary_confirm/summary.md` |
| `e158_promote_moe_dkv_w8` | code change + production | `e157_moe_dkv_w8_boundary_confirm` | 仅 B200 full Q16/KV2 raw64–127 选 w8 | 2K–3.5K 2.67–2.81×；GPU8/8；4K/E2B/packed 不变 | completed | improved | `runs/e158_promote_moe_dkv_w8/summary.md` |
| `e159_moe_w8_release_gate` | validation | `e158_promote_moe_dkv_w8` | 完整 CPU、8卡GPU、diff 与三代配置门禁 | CPU123；GPU50/50 + 7×varlen8/8；11文件仅修改 | completed | no-impact | `runs/e159_moe_w8_release_gate/summary.md` |
| `e160_moe2k_w8_nsys` | profile + dtype regression | `e158_promote_moe_dkv_w8` | MoE2K w8 NSYS；FP16/回退并行门禁 | dKV/dQ/forward=51.0%/26.6%/22.4%；FP16与回退正确 | completed | diagnostic | `runs/e160_moe2k_w8_nsys/summary.md` |
| `e161_moe2k_w8_dkv_ncu` | profile | `e160_moe2k_w8_nsys` | production w8 dKV NCU full | 255regs、198.93KiB、12.5% occupancy、84.17% no-eligible | completed | diagnostic | `runs/e161_moe2k_w8_dkv_ncu/summary.md` |
| `e162_moe2k_dkv_resource_sweep` | 7-GPU probe | `e161_moe2k_w8_dkv_ncu` | MoE2K q4 扫 BQ/warps/stages | BQ8仅+0.22%；其他慢；无晋级 | completed | no-impact | `runs/e162_moe2k_dkv_resource_sweep/summary.md` |
| `e163_moe_w8_qsplit_sweep` | 8-GPU probe | `e161_moe2k_w8_dkv_ncu` | MoE2K/3.5K w8 扫 q1/q2/q4/q8 | 2K q8快4.34%、显存同q4；3.5K仅+1.23% | completed | improved | `runs/e163_moe_w8_qsplit_sweep/summary.md` |
| `e164_moe_w8_qs8_crossover` | 8-GPU paired family | `e163_moe_w8_qsplit_sweep` | 2K–3.5K 各两卡 q4→q8 | 2K/2.5K +3.27%–4.31%；3K起<2% | completed | improved | `runs/e164_moe_w8_qs8_crossover/summary.md` |
| `e165_moe_w8_qs8_upper_boundary` | 8-GPU paired boundary | `e164_moe_w8_qs8_crossover` | raw88/94 各4卡 q4→q8 | 全部+2.32%–2.64%；显存一致；gate=[64,96) | completed | improved | `runs/e165_moe_w8_qs8_upper_boundary/summary.md` |
| `e166_promote_moe_qs8_w8` | code change + production | `e165_moe_w8_qs8_upper_boundary` | B200 full Q16/KV2 raw64–95选q8/w8 | 2K–3008为2.77×–2.94×；raw96/128边界与GPU8/8通过 | completed | improved | `runs/e166_promote_moe_qs8_w8/summary.md` |
| `e167_moe_w8_large_qsplit` | 8-GPU probe | `e166_promote_moe_qs8_w8` | MoE2K/2.5K 比较q4/q8/q16/q32 | q16/q32均慢；q8最优，停止dKV split轴 | completed | slower | `runs/e167_moe_w8_large_qsplit/summary.md` |
| `e168_moe2k_qs8_post_nsys` | profile + dtype regression | `e166_promote_moe_qs8_w8` | q8后NSYS；FP16/边界并行门禁 | dKV/dQ/fwd=48.4%/27.5%/23.4%；FP16/回退正确 | completed | diagnostic | `runs/e168_moe2k_qs8_post_nsys/summary.md` |
| `e169_moe2k_dq_ncu` | profile | `e168_moe2k_qs8_post_nsys` | MoE2K production dQ NCU full | 181regs、200.70KiB、12.5% occupancy、80.59% no-eligible | completed | diagnostic | `runs/e169_moe2k_dq_ncu/summary.md` |
| `e170_moe2k_dq_resource_sweep` | 7-GPU probe | `e169_moe2k_dq_ncu` | dQ扫warps/BQ/BKV/stages | 全部候选慢；显存/正确性一致 | completed | slower | `runs/e170_moe2k_dq_resource_sweep/summary.md` |
| `e171_moe2k_forward_ncu` | profile | `e168_moe2k_qs8_post_nsys` | MoE2K production forward NCU full | 255regs、100.35KiB、11.85% occupancy、72.11% no-eligible | completed | diagnostic | `runs/e171_moe2k_forward_ncu/summary.md` |
| `e172_moe2k_forward_resource_sweep` | 7-GPU probe | `e171_moe2k_forward_ncu` | forward扫warps/BQ/BKV/stages | 全部候选慢；显存/正确性一致 | completed | slower | `runs/e172_moe2k_forward_resource_sweep/summary.md` |

| `e173–e180_e2b_low_grid` | profile + probe + production | `e172_moe2k_forward_resource_sweep` | E2B 2K热点与qsplit长度族 | q8/w8在raw32–105晋级；2K从q2的2.191ms降至1.367ms，显存/正确性通过 | completed | improved | `runs/e180_e2b2k_qs8_post_profile/summary.md` |
| `e181–e190_forward_bkv64` | profile + probe + production | `e180_e2b2k_qs8_post_profile` | 大split、dQ与forward资源/长度轴 | forward BKV64按E2B/MoE短区间晋级；E2B2K 1.325ms；长序列安全回退 | completed | improved | `runs/e190_e2b2k_q8_dkv_ncu/summary.md` |
| `e191–e195_e2b_q11` | tail-wave + production + profile | `e190_e2b2k_q8_dkv_ncu` | q8 partial-wave下扫描q9–q14 | raw32–40使用q11；2K 1.264–1.268ms、3.75×–3.79×；显存不变 | completed | improved | `runs/e195_e2b2k_q11_post_profile/summary.md` |
| `e196/e197_forward_resource` | NCU + 8-GPU probe | `e195_e2b2k_q11_post_profile` | BKV64 forward资源轴 | w8仅+0.37%，其余更慢；保持原配置 | completed | no-impact | `runs/e197_e2b2k_bkv64_forward_resource_sweep/summary.md` |
| `e198_moe_qsplit` | 8-GPU probe | `e168_moe2k_qs8_post_nsys` | MoE2K q8–q15 | q9仅+1.56%，不增加碎片化配置 | completed | no-impact | `runs/e198_moe2k_qsplit_8_15/summary.md` |
| `e199_forward_split` | source probe + revert | `e196_e2b2k_bkv64_forward_ncu` | D512 BKV64 forward两段循环 | 2K–32K无收益；源码完整撤回 | completed | no-impact | `runs/e199_forward_split_bkv64_source_probe/summary.md` |
| `e200–e202_e2b_q9` | tail-wave + production | `e195_e2b2k_q11_post_profile` | E2B q8的raw41+细分 | raw41–44使用q9，快2.49%–3.78%；8卡生产门禁正确且显存不回退 | completed | improved | `runs/e202_promote_e2b_q9/summary.md` |
| `e203_e2b_q9_post_profile` | profile + 8-GPU regression | `e202_promote_e2b_q9` | raw44 NSYS与FP16/sliding/MoE/packed/8K–256K | dKV仍占46.4%；全数值正确；256K 2.42×、4.516GiB | completed | diagnostic | `runs/e203_e2b_q9_post_profile/summary.md` |
| `e204_e2b_q10_crossover` | paired probe | `e203_e2b_q9_post_profile` | raw45–48只改变q8→q10 | raw45略慢，raw46–48仅+0.3%–0.9%；显存相同 | completed | no-impact | `runs/e204_e2b_q10_crossover/summary.md` |
| `e205_q9_release_gate` | validation | `e202_promote_e2b_q9` | 完整CPU/GPU、三代selection、compile/diff | CPU142；B200 50/50 + 3×varlen8/8；无删除/重命名 | completed | no-impact | `runs/e205_q9_release_gate/summary.md` |
| `e206–e208_fp16_scratch_all_dtype` | source probe + paired/multiseed | `e205_q9_release_gate` | 所有qsplit dtype改用FP16 scratch | 常规吞吐快1%–5%、显存-9.5%，但BF16范围未证明 | completed | uncertain | `runs/e207_fp16_scratch_paired/summary.md` |
| `e209/e210_bf16_grad_scale` | correctness stress | `e206_qsplit_fp16_scratch_probe` | BF16 dOutput scale1–16384 | scale8192起候选非有限、reference为0；拒绝全dtype方案 | failed | failed | `runs/e210_fp16_scratch_grad_scale_retry/summary.md` |
| `e211/e212_dtype_gated_scratch` | code change + paired confirmation | `e210_fp16_scratch_grad_scale_retry` | 仅FP16输入使用FP16 scratch；BF16保持FP32 | FP16快1.3%–5.7%、显存约-14%；BF16大梯度恢复 | completed | improved | `runs/e212_fp16_dtype_scratch_paired/summary.md` |
| `e213_dtype_scratch_release_gate` | validation | `e211/e212_dtype_gated_scratch` | CPU/GPU、BF16大梯度、compile/diff与三代selection | CPU142；B200 50/50+8/8；无删除/重命名 | completed | no-impact | `runs/e213_dtype_scratch_release_gate/summary.md` |
| `e214/e215_sequential_split_scratch` | source probe + revert | `e190_e2b2k_q8_dkv_ncu` | BF16 dV/dK顺序执行并复用一份FP32 scratch | 显存-4%–5%，但full F+B慢7%–18%；撤回后恢复 | completed | slower | `runs/e215_sequential_split_revert_gate/summary.md` |
| `e216_bf16_atomic_micro` | 8-GPU micro-probe | `e210_fp16_scratch_grad_scale_retry` | PTX原生BF16 atomic编译/数值/时延 | 类型修正后成功；BF16约0.050ms、FP32约0.048ms | completed | diagnostic | `runs/e216_bf16_atomic_micro/summary.md` |
| `e217/e218_bf16_atomic_dkv` | source probe + revert | `e216_bf16_atomic_micro` | 真实B200 BF16 qsplit改用标量BF16 atomic | 显存-14%，但full F+B慢20%–75%；撤回后恢复 | completed | slower | `runs/e218_bf16_atomic_revert_gate/summary.md` |
| `e219/e220_bf16x2_atomic_dkv` | source probe + revert | `e216_bf16_atomic_micro` | 真实B200 BF16 qsplit改用成对BF16 atomic | 显存-14%，但full F+B慢0.8%–7.4%；撤回后恢复 | completed | slower | `runs/e220_bf16x2_atomic_revert_gate/summary.md` |
| `e221_fp16_dtype_gate_length_matrix` | 8-GPU regression | `e211/e212_dtype_gated_scratch` | FP16 E2B/MoE 2K–256K与packed/ragged | E2B 2.23×–2.75×、MoE 2.26×–2.43×；全正确、显存线性 | completed | improved | `runs/e221_fp16_dtype_gate_length_matrix/summary.md` |
| `e222/e223_fp16_e2b_q13` | 8-GPU probe + paired boundary | `e221_fp16_dtype_gate_length_matrix` | 原生FP16 scratch下扫描E2B q8–q16，并确认raw32–44 | q13在raw32稳定快约2.4%；raw33起低于2%，显存相同 | completed | improved | `runs/e223_fp16_e2b_q13_confirm/summary.md` |
| `e224–e226_fp16_moe_q9` | 8-GPU probe + paired boundary | `e221_fp16_dtype_gate_length_matrix` | 原生FP16 scratch下扫描MoE q4–q12，并确认raw64–72 | q9在raw64–70快1.8%–2.7%；raw72持平，显存相同 | completed | improved | `runs/e226_fp16_narrow_gate_confirm/summary.md` |
| `e227_fp16_e2b_q13_boundary` | 8-GPU paired boundary | `e223_fp16_e2b_q13_confirm` | E2B q11/q13复测raw33–35 | 收益降至1.0%–1.8%；只保留raw32窄门控 | completed | diagnostic | `runs/e227_fp16_e2b_q13_boundary/summary.md` |
| `e228_fp16_production_gate` | code change + 8-GPU production | `e226/e227` | FP16-only E2B raw32 q13、MoE raw64–71 q9；BF16/packed回归 | E2B/MoE 2K为2.82×/2.48×；8/8正确、显存不增 | completed | improved | `runs/e228_fp16_production_gate/summary.md` |
| `e229_fp16_tailwave_post_profile` | profile + regression | `e228_fp16_production_gate` | FP16 E2B/MoE 2K NSYS；128K/256K、边界与packed回归 | dKV仍约45%；长端2.30×/2.37×，全部正确 | completed | diagnostic | `runs/e229_fp16_tailwave_post_profile/summary.md` |
| `e230–e232_fp16_three_kernel_ncu` | NCU + 8-GPU resource probe | `e229_fp16_tailwave_post_profile` | dKV、dQ、forward资源与单变量配置 | 高shared/低occupancy；所有warps/stages/tile候选无winner | completed | no-impact | `runs/e232_fp16_forward_profile_resource/summary.md` |
| `e233/e234_empty_qsplit_early_exit` | source probe + 8-GPU paired | `e230_fp16_dkv_ncu` | 空尾q-split在K/V加载与atomic前直接退出 | E2B四卡+0.96%–1.03%，MoE四卡+1.08%–1.11%；显存不变 | completed | improved | `runs/e234_empty_qsplit_paired/summary.md` |
| `e235/e236_empty_qsplit_release` | validation + NSYS retry | `e234_empty_qsplit_paired` | CPU/GPU、边界/packed与post-profile | CPU147、GPU50/50+4×8/8；dKV快2.10%–2.23%；无回退 | completed | improved | `runs/e236_empty_qsplit_nsys_retry/summary.md` |
| `e237–e240_empty_exit_qsplit_rescan` | 4×8-GPU probe | `e236_empty_qsplit_nsys_retry` | FP16/BF16、E2B/MoE重新扫描q-split | FP16原q13/q9仍最优；BF16候选收益≤1.9%，显存/正确性相同，不晋级 | completed | no-impact | `runs/e240_empty_exit_bf16_moe_qsplit_rescan/summary.md` |
| `e241/e242_dkv_ds_mask` | source probe + 8-GPU paired control | `e230_fp16_dkv_ncu` | 删除dKV第二次ds mask | 仅快0.10%–0.29%；显存/正确性相同，源码恢复 | completed | no-impact | `runs/e241_dkv_redundant_ds_mask_probe/summary.md` |
| `e243_dkv_dense_causal_mask` | source probe + 8-GPU paired | `e242_dkv_ds_mask_control` | full-causal稠密Q块跳过对角比较 | FP16略慢、BF16收益≤0.15%；显存/正确性相同，源码撤回 | completed | no-impact | `runs/e243_dkv_dense_causal_mask_probe/summary.md` |
| `e244/e245_dkv_loop_interchange` | source probe + resource sweep | `e230_fp16_dkv_ncu` | 交换动态Q循环与静态GQA头循环 | 原配置shared OOR；唯一可运行点慢约46%，源码撤回 | completed | slower | `runs/e245_dkv_q_gqa_interchange_resource/summary.md` |
| `e246_dkv_no_acc_multibuffer` | compiler attribute probe + 8-GPU paired | `e242_dkv_ds_mask_control` | 禁止Q循环dot accumulator多缓冲 | 与control差异≤0.06%；显存/正确性相同，属性撤回 | completed | no-impact | `runs/e246_dkv_no_acc_multibuffer_probe/summary.md` |
| `e247_dkv_dense_causal_long` | 8-GPU long regression + revert | `e243_dkv_dense_causal_mask_probe` | 32K/128K复测causal稠密分支 | FP16/BF16约慢0.8%–1.0%；显存/正确性相同，源码撤回 | completed | slower | `runs/e247_dkv_dense_causal_long_probe/summary.md` |
| `e248_dkv_constexpr_scale` | source probe + 8-GPU paired | `e242_dkv_ds_mask_control` | dKV scale编译期常量化 | 与control基本相同；显存/正确性不变，签名恢复 | completed | no-impact | `runs/e248_dkv_constexpr_scale_probe/summary.md` |
| `e249_fp16_tailwave_dkv_bq_resource` | 8-GPU resource probe | `e237/e238` | 当前FP16 q13/q9下复扫BQ/warps/stages | BQ8略慢，s2/w4更慢；显存/正确性相同 | completed | slower | `runs/e249_fp16_tailwave_dkv_bq_resource/summary.md` |
| `e250_fp16_moe_dq_resource` | 8-GPU resource probe | `e231_fp16_dq_profile_resource` | MoE FP16 dQ扫描BQ/BKV/warps/stages | 全部候选慢，s3 OOR；显存/正确性相同 | completed | slower | `runs/e250_fp16_moe_dq_resource/summary.md` |
| `e251_fp16_moe_forward_resource` | 8-GPU resource probe | `e232_fp16_forward_profile_resource` | MoE FP16 forward扫描BQ/BKV/warps/stages | w8仅快约0.5%；其余慢或OOR，不晋级 | completed | no-impact | `runs/e251_fp16_moe_forward_resource/summary.md` |
| `e252_dkv_atomic_ptx_audit` | PTX audit | `e230_fp16_dkv_ncu` | 检查FP16/FP32 atomic向量宽度与内存序 | 已为v8.f16/v4.f32；默认acq_rel，转测relaxed | completed | diagnostic | `runs/e252_dkv_atomic_ptx_audit/summary.md` |
| `e253/e254_dkv_relaxed_atomic` | source probe + 8-GPU paired control | `e252_dkv_atomic_ptx_audit` | qsplit atomic acq_rel→relaxed | FP16 full快约4%，BF16快约7%–8%；显存/正确性相同 | completed | improved | `runs/e253_dkv_relaxed_atomic_probe/summary.md` |
| `e255_relaxed_atomic_production_gate` | code change + 8-GPU production | `e253/e254` | B200-only registry策略；边界/8K/packed | 8/8正确；2K达2.61×–4.08×SDPA，q1/H100/H200不变 | completed | improved | `runs/e255_relaxed_atomic_production_gate/summary.md` |
| `e256_relaxed_atomic_stress` | 8-GPU correctness stress | `e255_relaxed_atomic_production_gate` | BF16 scale8192/16384、多seed/边界/8K | 8/8正确且全有限；吞吐/显存稳定 | completed | improved | `runs/e256_relaxed_atomic_stress/summary.md` |
| `e257_relaxed_atomic_post_profile` | NSYS + NCU + 8-GPU validation | `e255/e256` | relaxed后E2B/MoE热点与PTX确认 | dKV快约8.7%–8.9%，占比降至约43%；四卡数值8/8 | completed | diagnostic | `runs/e257_relaxed_atomic_post_profile/summary.md` |
| `e258_relaxed_atomic_release_gate` | CPU + 8-GPU release gate | `e255–e257` | 完整数值、三代配置、compile与diff门禁 | CPU147；GPU197 + 7×8/8；无删除/重命名 | completed | no-impact | `runs/e258_relaxed_atomic_release_gate/summary.md` |
| `e259/e260_relaxed_fp16_qsplit` | 2×8-GPU probe | `e257` | relaxed后FP16两profile重扫split | 最多仅快0.8%/1.3%；显存/正确性相同，不晋级 | completed | no-impact | `runs/e260_relaxed_fp16_moe_qsplit_rescan/summary.md` |
| `e261/e262_relaxed_bf16_qsplit` | 2×8-GPU probe | `e257` | relaxed后BF16两profile重扫split | q14初测快2.8%/3.8%；显存/正确性相同 | completed | improved | `runs/e262_relaxed_bf16_moe_qsplit_rescan/summary.md` |
| `e263_relaxed_bf16_q14_paired` | 8-GPU paired confirm | `e261/e262` | BF16当前split→q14，100-repeat | E2B四卡+2.49%–2.62%；MoE四卡+3.77%–3.84% | completed | improved | `runs/e263_relaxed_bf16_q14_paired/summary.md` |
| `e264/e265_bf16_e2b_q14_boundary` | 2×8-GPU paired family | `e263` | raw32–52 精确 crossover | raw32–34与raw45–52稳定获益；中间区不晋级 | completed | improved | `runs/e265_relaxed_bf16_e2b_q14_boundary2/summary.md` |
| `e266_bf16_e2b_q14_family` | 8-GPU paired family | `e265` | raw54–105 q8→q14 | raw54–72快2.4%–3.5%；raw80+低于门槛 | completed | improved | `runs/e266_relaxed_bf16_e2b_q14_family/summary.md` |
| `e267_bf16_moe_q14_family` | 8-GPU paired family | `e263` | raw64–95 q8→q14 | raw64–68快3.4%–3.8%；后续非单调并逐步衰减 | completed | improved | `runs/e267_relaxed_bf16_moe_q14_family/summary.md` |
| `e268/e269_bf16_e2b_q14_upper` | 2×8-GPU paired family | `e266` | raw49–79补点与上边界 | 已测raw45–76均>2%；raw77+低于门槛 | completed | improved | `runs/e269_relaxed_bf16_e2b_q14_gapfill/summary.md` |
| `e270_bf16_moe_q14_gapfill` | 8-GPU paired family | `e267` | 同一raw-grid内tile尾长度消歧 | 收益不可由raw-grid安全区分；只留精确2K/raw64候选 | completed | diagnostic | `runs/e270_relaxed_bf16_moe_q14_gapfill/summary.md` |
| `e271_bf16_e2b_q14_gapfill` | 8-GPU paired family | `e269` | raw45–76最后空点 | 8/8快2.2%–3.8%；连续gate证据闭合 | completed | improved | `runs/e271_relaxed_bf16_e2b_q14_final_gapfill/summary.md` |
| `e272_bf16_q14_production_gate` | code change + 8-GPU production | `e263–e271` | BF16 E2B两段与MoE精确2K晋级 | 8/8正确；E2B 3.23×–4.20×、MoE2K 3.38×SDPA，显存不变 | completed | improved | `runs/e272_bf16_q14_production_gate/summary.md` |
| `e273_bf16_q14_release_gate` | CPU + 8-GPU release/stress | `e272` | 全套GPU与BF16 scale16384 | CPU152；GPU202+3×8/8；四个压力点正确且峰值不变 | completed | no-impact | `runs/e273_bf16_q14_release_gate/summary.md` |
| `e274_bf16_q14_release_length_matrix` | 8-GPU length regression | `e273` | 两profile × 2K/8K/128K/256K | 8/8正确；2.42×–4.20×，256K峰值4.516/9.031GiB | completed | improved | `runs/e274_bf16_q14_release_length_matrix/summary.md` |
| `e275_bf16_q14_post_profile` | NSYS + NCU + 8-GPU validation | `e274` | q14后三核占比与dKV资源 | dKV仍约43%；E2B 796.7us、3.03waves，资源瓶颈未变 | completed | diagnostic | `runs/e275_bf16_q14_post_profile/summary.md` |
| `e276_bf16_q14_dkv_resource` | 8-GPU paired resource probe | `e275` | q14下BQ/BKV/warps/stages单变量 | BQ8持平，其余慢1%–48%；数值/显存不变，不晋级 | completed | slower | `runs/e276_bf16_q14_dkv_resource_paired/summary.md` |
| `e277_bf16x2_relaxed_micro` | 8-GPU micro-probe | `e252/e216` | 显式BF16x2 relaxed atomic编译/数值/时延 | 8卡可运行，但不快于FP32 relaxed | completed | diagnostic | `runs/e277_bf16x2_relaxed_micro/summary.md` |
| `e278_bf16x2_relaxed_dkv` | source probe + 8-GPU paired | `e277` | BF16 scratch与BF16x2 relaxed真实dKV | 显存-14.1%，但慢1.5%–1.9%且1/8数值失败；已撤回 | completed | slower | `runs/e278_bf16x2_relaxed_dkv_paired/summary.md` |
| `e279–e281_bf16_v4_micro` | PTX + 3×8-GPU micro | `e278` | 四元素BF16 atomic对齐、执行映射与时延 | e279/e280失败；e281数值恢复且与FP32时延持平 | completed | diagnostic | `runs/e281_bf16_v4_aligned_retry/summary.md` |
| `e282–e284_bf16_v4_dkv` | source probe + 3×8-GPU retry | `e281` | 四元素BF16 atomic真实dKV | 显存-14.1%，但慢9%–10%且1/8数值失败；已撤回 | completed | slower | `runs/e284_bf16_v4_dkv_compile_retry2/summary.md` |
| `e285_bf16_atomic_revert` | 8-GPU validation | `e284` | 候选撤回后的varlen数值门禁 | 8×8/8通过，production恢复 | completed | no-impact | `runs/e285_bf16_atomic_revert_gate/summary.md` |
| `e286/e287_split_gqa_head` | scan + 8-GPU paired | `e275` | 将8个GQA head映射到dKV grid | E2B快2.5%–2.7%，MoE快1.8%；显存/正确性不变 | completed | improved | `runs/e287_split_gqa_head_paired/summary.md` |
| `e288/e289_split_gqa_boundary` | 2×8-GPU paired | `e287` | E2B raw32–77与production分段比较 | e288格式修正；e289八点快2.4%–4.7% | completed | improved | `runs/e289_split_gqa_e2b_boundary_corrected/summary.md` |
| `e290_split_gqa_upper` | 8-GPU paired | `e289` | raw80–256 throughput/memory边界 | raw80–105有效；raw106–223基线错误，由e302取代 | completed | diagnostic | `runs/e290_split_gqa_e2b_upper/summary.md` |
| `e291–e294_split_gqa_bf16x2` | scan + paired + stress | `e290` | head-grid/q1用BF16x2消除FP32 scratch | raw224+对q1有效；raw106–223由e302纠正；scale16384正确 | completed | improved | `runs/e294_split_gqa_bf16x2_upper_stress/summary.md` |
| `e295–e297_split_gqa_long` | long boundary + independent retry | `e294` | 32K–256K吞吐/显存/正确性 | 可运行到256K且显存相同；96K+需同卡消歧 | completed | diagnostic | `runs/e297_split_gqa_bf16x2_long_independent/summary.md` |
| `e298_split_gqa_long_paired` | 8-GPU paired long | `e297` | 32K–192K同卡顺序A/B | 只快0.6%–1.3%；正确/显存相同，长端不晋级 | completed | no-impact | `runs/e298_split_gqa_bf16x2_long_paired/summary.md` |
| `e299–e301_split_gqa_islands` | 3×8-GPU paired boundary | `e294` | raw257–639补点与收益岛边界 | 保守gate为`[257,281)`、`[317,537)` | completed | improved | `runs/e301_split_gqa_bf16x2_island_gapfill/summary.md` |
| `e302_production_corrected` | 8-GPU paired correction | `e301` | raw106–223改用真实q4 control | 对q4快1.1%–4.7%且显存约-14%；8/8正确 | completed | improved | `runs/e302_split_gqa_bf16x2_production_corrected/summary.md` |
| `e303_headgrid_production_gate` | code change + 8-GPU production | `e302` | B200 BF16 E2B三个head-grid gate | 8/8正确；2.47×–4.33×SDPA，selection/显存符合预期 | completed | improved | `runs/e303_production_headgrid_gate/summary.md` |
| `e304_headgrid_scale_boundary` | 8-GPU correctness stress | `e303` | scale8192/16384与281/316/537边界 | 8/8正确；新配置与回退配置均精确命中 | completed | no-impact | `runs/e304_production_boundary_scale_stress/summary.md` |
| `e305_headgrid_packed_long` | 8-GPU production matrix | `e303` | packed、ragged、32K–256K | 8/8正确；2.42×–4.01×，256K 4.516GiB | completed | no-impact | `runs/e305_production_packed_long_matrix/summary.md` |
| `e306_headgrid_post_profile` | 2×NCU + 2×NSYS + regression | `e303` | q3/q1生产dKV profile | dKV仍占约42%；12.5% occupancy、约82% no-eligible | completed | diagnostic | `runs/e306_headgrid_post_profile/summary.md` |
| `e307_headgrid_gqa_group` | source probe + 8-GPU scan + revert | `e306` | 每program处理1/2/4/8个GQA head | group1最优；group2起变慢，候选源码撤回 | completed | slower | `runs/e307_headgrid_gqa_group_scan/summary.md` |
| `e308_bf16x2_full_block` | source probe + 8-GPU paired + revert | `e306` | 完整KV块绕过safe pointer | 8/8慢0.19%–0.63%；显存/正确性相同，源码撤回 | completed | slower | `runs/e308_bf16x2_full_block_atomic/summary.md` |
| `e309_headgrid_q3_bf16x2` | 8-GPU paired candidate | `e306` | q3 FP32 scratch→BF16x2 | 显存约-14%，但8/8慢约4%–5%；不晋级 | completed | slower | `runs/e309_headgrid_q3_bf16x2/summary.md` |
| `e310_headgrid_q1_resource` | 8-GPU resource scan | `e306` | BQ8/BKV32/stages2 | BQ8仅约+0.2%；其余慢2%–47%，无winner | completed | slower | `runs/e310_headgrid_q1_resource_scan/summary.md` |
| `e311_raw106_other_ncu` | 2×NCU + 6-GPU regression | `e306` | raw106 dQ/forward资源证据 | dQ 12.5% occupancy；forward 6.25%且9.87M spill | completed | diagnostic | `runs/e311_headgrid_raw106_other_ncu/summary.md` |
| `e312_raw106_other_resource` | 8-GPU resource scan | `e311` | dQ/forward BQ/BKV/warps/stages | w8仅+0.19%；其余慢2.8%–22%，无winner | completed | slower | `runs/e312_raw106_other_resource/summary.md` |
| `e313_headgrid_release_gate` | CPU + 8-GPU release gate | `e303–e312` | 完整数值、三代selection、compile/diff | CPU147；GPU197 + 7×8/8；无删除/重命名 | completed | no-impact | `runs/e313_headgrid_release_gate/summary.md` |
| `e314_headgrid_gap_corrected` | 8-GPU same-shape A/B | `e313` | raw281–316旧生产配置→head-grid q1+BF16x2 | 延迟-34.5%–34.8%，吞吐+52.6%–53.5%；显存逐字节一致，8/8正确 | completed | improved | `runs/e314_headgrid_gap_production_corrected/summary.md` |
| `e315_gap_production_gate` | code change + 8-GPU production | `e314` | 合并q1 gate为`[106,537)` | 边界/内部8/8正确，2.46×–2.59×SDPA，raw537正确回落 | completed | improved | `runs/e315_gap_promotion_production_gate/summary.md` |
| `e316_gap_scale_isolation` | 8-GPU correctness stress | `e315` | scale8192/16384及FP16/packed隔离 | 8/8正确；新区间2.53×–2.59×，非目标路径未接管 | completed | no-impact | `runs/e316_gap_scale_isolation/summary.md` |
| `e317_headgrid_upper_crossover` | 8-GPU same-card A/B | `e315` | raw537–544改用head-grid q1+BF16x2 | 收益-0.03%–+0.16%，显存相同；不扩gate | completed | no-impact | `runs/e317_headgrid_upper_crossover/summary.md` |
| `e318_q3_bf16x2_split_tradeoff` | 8-GPU throughput/memory probe | `e309` | 短端q1/q2/q4/q5 + BF16x2 | 显存-14.12%，但吞吐慢1.62%–8.47%；两点dV超门槛 | completed | slower | `runs/e318_q3_bf16x2_split_tradeoff/summary.md` |
| `e319_q2_bf16x2_resource` | 8-GPU resource probe | `e318` | q2下BQ/BKV/warps/stages | 仅w4双赢：raw76/105吞吐+0.90%/+1.64%，显存-14.12% | completed | improved | `runs/e319_q2_bf16x2_resource_scan/summary.md` |
| `e320_q2_w4_bf16x2_family` | 8-GPU 100-repeat family | `e319` | q2+w4+BF16x2扫raw32–105 | raw80–105吞吐+1.13%–1.66%，全点显存-14.12%；8/8正确 | completed | improved | `runs/e320_q2_w4_bf16x2_family/summary.md` |
| `e321_q2_w4_bf16x2_boundary` | 8-GPU dual-seed confirm | `e320` | raw72/76/80/96各双seed | 吞吐+0.84%–1.46%、显存-14.12%，8/8正确 | completed | improved | `runs/e321_q2_w4_bf16x2_boundary_confirm/summary.md` |
| `e322_q2_w4_bf16x2_production` | code change + 8-GPU production | `e321` | B200 BF16 E2B raw72–105晋级 | 边界/scale16384/FP16/packed 8/8正确 | completed | improved | `runs/e322_q2_w4_bf16x2_production_gate/summary.md` |
| `e323_q2_w4_bf16x2_release` | CPU + 8-GPU release gate | `e322` | 完整数值、三代selection、compile/diff | CPU149；GPU199 + 7×8/8；无删除/重命名 | completed | no-impact | `runs/e323_q2_w4_bf16x2_release_gate/summary.md` |
| `e324_final_single_matrix` | 8-GPU production matrix | `e323` | 2K–256K单序列 | 8/8正确；2.42×–4.33×SDPA，峰值0.041–4.516GiB | completed | no-impact | `runs/e324_final_single_length_matrix/summary.md` |
| `e325_final_packed_matrix` | 8-GPU production matrix | `e323` | packed/ragged到total-256K | 8/8正确；2.46×–4.02×，packed未被batch1 gate接管 | completed | no-impact | `runs/e325_final_packed_distribution_matrix/summary.md` |
| `e326_q2_w4_post_profile` | NSYS + NCU + 6-GPU regression | `e323` | 新q2/w4路径资源与热点 | dKV 40.4%；255regs/198.93KiB/6.25% occupancy/85.5% no-eligible | completed | diagnostic | `runs/e326_q2_w4_post_profile/summary.md` |
| `e327_q2_w4_resource_followup` | 8-GPU resource probe | `e326` | BQ/BKV/stages/warps单变量 | BQ8持平；stages2慢1.7%，BKV32/w2严重退化；显存不变 | completed | slower | `runs/e327_q2_w4_resource_followup/summary.md` |
| `e328_q2_w4_bin_tail_gate` | 8-GPU same-card tail gate | `e323` | raw72/105桶首桶尾+scale16384 | 吞吐+0.66%–1.64%、显存-14.12%–14.34%；8/8正确 | completed | improved | `runs/e328_q2_w4_bin_tail_gate/summary.md` |
| `e329_q2_w4_lower_boundary` | 8-GPU dual-seed boundary | `e328` | raw64/68/70/71下边界 | raw68–71吞吐+0.69%–0.77%、显存约-14%；raw64一例dV失败 | completed | improved | `runs/e329_q2_w4_lower_boundary/summary.md` |
| `e330_q2_w4_lower_production` | code change + 8-GPU production | `e329` | 门控扩至`[68,106)`并做边界/隔离 | 8/8正确；raw67/68、scale16384、FP16/packed选择正确 | completed | improved | `runs/e330_q2_w4_lower_production/summary.md` |
| `e331_q2_w4_lower_release` | CPU + 8-GPU release gate | `e330` | 完整数值、三代selection、compile/diff | CPU150；GPU200 + 7×8/8；无删除/重命名 | completed | no-impact | `runs/e331_q2_w4_lower_release_gate/summary.md` |
| `e332_q2_w4_fp32_short` | 8-GPU throughput/memory scan | `e329` | raw32–67用q2/w4但保留FP32 scratch | 全慢3.10%–7.38%，allocator峰值相同；8/8正确 | completed | slower | `runs/e332_q2_w4_fp32_short_scan/summary.md` |
| `e333_packed_headgrid_baseline` | 8-GPU packed baseline | `e325` | 8种分布复测并量化空网格 | 8/8正确；ragged active仅46.9%/28.6%，指向紧凑block table | completed | diagnostic | `runs/e333_packed_headgrid_scan/summary.md` |
| `e334_registry_reachability` | CPU selection sweep | `e331` | B200 raw1–4096与H100/H200代表点 | 新分段精确；FP16/MoE旧配置仍可达，三代无回退 | completed | no-impact | `runs/e334_registry_reachability_audit/summary.md` |
| `e335_memory_floor_audit` | canonical JSON audit | `e324/e325` | 2K–256K峰值按token/分布归一化 | 长端稳定18.0625KiB/token；2K q3多约14% | completed | diagnostic | `runs/e335_memory_floor_audit/summary.md` |
| `e336_packed_block_table_design` | source/design audit | `e333` | 用prefix blocks消除ragged矩形空网格 | 方案与风险已定义；待GPU candidate A/B，不改production | pending | diagnostic | `runs/e336_packed_block_table_design/summary.md` |
| `e337_profile_direction_audit` | NCU + history audit | `e326/e327` | 去重D-split、dK/dV拆分与资源扫描 | 局部资源轴已穷尽；packed紧凑网格优先 | completed | diagnostic | `runs/e337_profile_direction_audit/summary.md` |
| `e338_registry_policy_invariants` | CPU invariant audit | `e334` | 80配置的策略字段隔离 | 14个策略配置、0违规；仅sm100 dKV | completed | no-impact | `runs/e338_registry_policy_invariant_audit/summary.md` |
| `e339_registry_cache_overhead` | CPU microbenchmark | `e338` | 80配置下缓存resolve开销 | 每角色约1.40µs；非吞吐瓶颈 | completed | no-impact | `runs/e339_registry_cache_overhead/summary.md` |
| `e340_public_docs_audit` | docs audit | `e331` | 当前B200路径、门禁与总计划同步 | 5份文档更新；历史H100/H200/失败结论保留 | completed | no-impact | `runs/e340_public_docs_audit/summary.md` |
| `e341_bf16x2_margin_audit` | 178-cell JSON audit | `e300–e330` | q1/q2 BF16x2数值裕量 | dV max-abs最紧93.75%；raw64失败保留，不放宽门槛 | completed | diagnostic | `runs/e341_bf16x2_correctness_margin_audit/summary.md` |
| `e342_q2_w4_aggregate` | 26-pair JSON audit | `e320/e321/e328/e329` | raw68–105同卡A/B聚合 | 吞吐+0.664%–+1.662%，26/26为正；显存约-14% | completed | improved | `runs/e342_q2_w4_aggregate_confidence/summary.md` |
| `e343_cross_hardware_selection_fuzz` | 14,592-case CPU sweep | `e334/e338` | 四profile/两dtype/四batch/四runtime/三role | 0 ambiguity、0空洞、0 B200策略泄漏 | completed | no-impact | `runs/e343_cross_hardware_selection_fuzz/summary.md` |
| `e344_final_matrix_aggregate` | 16-cell JSON audit | `e324/e325` | 单序列与packed最终指标聚合 | 单序列geo2.899×、packed geo2.768×；16/16正确 | completed | improved | `runs/e344_final_matrix_aggregate/summary.md` |
| `e345_dkv_amdahl_budget` | NSYS derived analysis | `e326` | dKV=40.4%的full收益上限 | dKV快10%→full约+3.81%；过滤噪声级候选 | completed | diagnostic | `runs/e345_dkv_amdahl_budget/summary.md` |
| `e346_production_pareto_audit` | evidence synthesis | `e134–e345` | production吞吐/显存/正确性联合审查 | 所有晋级均吞吐不退、显存不增；负权衡候选均拒绝 | completed | no-impact | `runs/e346_production_pareto_audit/summary.md` |
| `e347_evidence_tracking` | gitignore audit | `e340` | 顶层中文状态文档可提交 | 放行5份文档；raw/profiler/runs继续忽略 | completed | no-impact | `runs/e347_evidence_tracking_audit/summary.md` |
| `e348_packed_empty_grid` | historical NSYS disambiguation | `e073/e333` | active25%的ragged对单序列 | 实测32.4%≈理论33.34%；空CTA成本不可测，table降级 | completed | diagnostic | `runs/e348_packed_empty_grid_disambiguation/summary.md` |
| `e349_dkv_shared_bound` | NCU resource analysis | `e326/e150` | 双CTA驻留所需shared | 需从198.93KiB降至≤113.5KiB（-42.9%）；仅s1未闭合 | completed | diagnostic | `runs/e349_dkv_shared_residency_bound/summary.md` |
| `e350_q2_s1_residency_plan` | profile-driven pending | `e327/e349` | q2/w4仅改stages3→1 | 双驻留阈值的最后局部假设；待GPU配额恢复 | pending | diagnostic | `runs/e350_q2_s1_residency_plan/summary.md` |
| `e351_final_acceptance` | CPU/GPU/evidence audit | `e331–e350` | 最终性能、显存、正确性、三代与提交边界 | 当前production验收通过；pending只属于下一轮 | completed | no-impact | `runs/e351_final_acceptance_audit/summary.md` |

Verdict 使用：`improved / no-impact / slower / failed / diagnostic / uncertain / pending`。
