# Evidence Ledger

Baseline commit：`c6a0cf860612792597382e7745d1514a52d6ca58`，branch `refactor`，起始 working tree 为 dirty。

## Passing evidence

| Timestamp | Evidence | Result | Raw artifact |
| --- | --- | --- | --- |
| 08-04 17:34 PST | `e040` production registry 与数值门禁 | CPU 80/80、B200 varlen FP16/BF16 8/8 通过；B200 选 BQ16，未知 sm100 保留安全基线 | `runs/e040_promote_bq16_registry/summary.md` |
| 08-04 17:43 PST | `e041` production D512 family | 6/6 selection/correctness 通过；1.956–3.146× SDPA，几何平均 2.389× | `runs/e041_production_d512_family/summary.md` |
| 08-04 17:49 PST | `e042` production 50-repeat 独立复测 | 6/6 selection/correctness 通过；1.927–3.144× SDPA，几何平均 2.387× | `runs/e042_production_family_repeat/summary.md` |
| 08-04 17:56 PST | `e050` B200 kernel release gate | CPU 88 passed；B200 GPU 50/50 passed，无失败或 required skip | `runs/e050_b200_release_gate/summary.md` |
| 08-04 17:38 PST | `e051` 三代硬件整理防回退 | CPU 100 passed；B200 GPU 50/50；D512 6/6 正确且 1.946–3.095×，最大 Triton latency delta 0.71% | `runs/e051_refactor_no_regression/summary.md` |
| 08-04 19:07 PST | `e060` D512 长序列 baseline | E2B/MoE 共 10/10 正确，1.59–2.50× SDPA；32K×1 最弱 | `runs/e060_d512_long_matrix_baseline/summary.md` |
| 08-04 19:10 PST | `e061` D512 total-128K probe | 8/8 正确，无 OOM/fallback；1.504–1.592× SDPA，E2B 128K×1 最弱 | `runs/e061_d512_128k_probe/summary.md` |
| 08-04 19:11 PST | `e062` E2B 128K×1 NSYS | dKV 62.8%、dQ 21.3%、forward 16.0%；首个优化轴仍为 dKV | `runs/e062_d512_128k_nsys/summary.md` |
| 08-04 19:15 PST | `e063` 128K dKV BKV probe | BKV64 比 BKV16 快 34.9%，full F+B 达 2.328× SDPA；BKV32 严重退化 | `runs/e063_d512_128k_dkv_bkv_probe/summary.md` |
| 08-04 19:18 PST | `e064` 128K BKV64 family | 8/8 正确；2.323–2.470× SDPA；每个 cell 比 BKV16 快 34.8–35.3% | `runs/e064_d512_128k_bkv64_family/summary.md` |
| 08-04 19:22 PST | `e065` BKV64 长短序列交叉验证 | 16/16 正确；2.391–3.848× SDPA；15 cells 明显更快，1 个短不均匀边界待同卡消歧 | `runs/e065_d512_bkv64_crossover/summary.md` |
| 08-04 19:30 PST | `e066` BKV64 同卡配对确认 | 8/8 正确；E2B 短不均匀确认回退 0.86%；128K 两 profile 均快约 35.2%，采用保守 grid gate | `runs/e066_d512_bkv64_paired_confirm/summary.md` |
| 08-04 19:35 PST | `e067` BKV64 grid gate production 晋级 | CPU 107 passed、GPU 50/50；24/24 production 正确；total-128K 2.323–2.475× | `runs/e067_promote_bkv64_grid_gate/summary.md` |
| 08-04 21:58 PST | `e071` E2B 256K×1 NSYS | dKV/dQ/forward 分别占 43.0%/32.3%/24.6%；下一步先采 dKV NCU | `runs/e071_d512_256k_nsys/summary.md` |
| 08-04 22:00 PST | `e069` MoE 256K retry | output/dQ/dK/dV 与 selection 通过；25996.464 ms vs SDPA 59865.969 ms（2.303×） | `runs/e069_chunked_metrics_moe256k_retry/summary.md` |
| 08-04 22:02 PST | `e073` E2B total-256K ragged NSYS | dKV/dQ/forward 为 43.0%/32.5%/24.6%，与 256K×1 占比基本一致 | `runs/e073_d512_256k_ragged_nsys/summary.md` |
| 08-04 22:08 PST | `e074` MoE total-256K NSYS | single/ragged 的 dKV/dQ/forward 都约为 43%/32%/24%，与 E2B 一致 | `runs/e074_d512_256k_moe_nsys/summary.md` |
| 08-04 22:11 PST | `e070` total-256K packed family | 8/8 正确；2.283–2.439×；Triton 显存比 SDPA 少 65.4%–77.5% | `runs/e070_d512_256k_packed_family/summary.md` |
| 08-04 22:11 PST | `e075` 2K–256K 吞吐/显存尺度 | 10/10 正确；1.563–2.516×；Triton 显存线性，比 SDPA 少 77.5%–82.8% | `runs/e075_d512_memory_scaling_baseline/summary.md` |
| 08-04 22:13 PST | `e081` dKV stages probe | s3 比 s2 快 5.6%，增量显存不变；所有候选正确 | `runs/e081_d512_dkv_stages_probe/summary.md` |
| 08-04 22:17 PST | `e083` dKV s3 family probe | E2B/MoE、8K–128K 的 4/4 正确；full F+B 快 2.44%–5.45%，显存不变 | `runs/e083_d512_dkv_s3_family_probe/summary.md` |
| 08-04 22:19 PST | `e080` dKV warps probe | w2 严重退化，w8 慢 0.87%；保持 w4 | `runs/e080_d512_dkv_warps_probe/summary.md` |
| 08-04 22:21 PST | `e082` dKV s3 双卡确认 | 两卡差 0.10%，相对 s2 快 5.40%/5.31%；正确且显存不变 | `runs/e082_d512_dkv_s3_confirm/summary.md` |
| 08-04 22:24 PST | `e086/e087/e088` s3 crossover repeats | E2B 8K、MoE 8K、E2B 32K 分别快 2.69%/3.72%/2.39%；均正确且显存不变 | `runs/e086_d512_dkv_s3_8k_repeat/summary.md`、`runs/e087_d512_dkv_s3_moe8k_repeat/summary.md`、`runs/e088_d512_dkv_s3_32k_repeat/summary.md` |
| 08-04 22:31 PST | `e084` dKV s3 total-256K family | 8/8 正确；full F+B 快 3.91%–6.35%，显存不变 | `runs/e084_d512_dkv_s3_256k_family/summary.md` |
| 08-04 22:36 PST | `e085` E2B 128K s3 dKV NCU | occupancy 6.25%、spill 7.27M、no-eligible 85.5%；需与 e089 s2 control 差分 | `runs/e085_d512_128k_dkv_s3_ncu/summary.md` |
| 08-05 08:46 PST | `e072/e076/e077` 256K 三 kernel NCU | dKV 为低 occupancy/spill；dQ 为 shared short-scoreboard；forward 为 255 regs 与 9.67B spill | `runs/e072_d512_256k_dkv_ncu/summary.md`、`runs/e076_d512_256k_dq_ncu/summary.md`、`runs/e077_d512_256k_forward_ncu/summary.md` |
| 08-05 08:47 PST | `e089` 128K dKV s2/s3 NCU 差分 | s3 duration -10.8%，但 shared +19.8%、spill +52.1%；记录吞吐/局部访存权衡 | `runs/e089_d512_128k_dkv_s2_ncu/summary.md` |
| 08-05 08:47 PST | `e090` s3 160K–224K scaling | 6/6 正确；比 s2 快 5.62%–6.01%，增量显存线性 | `runs/e090_d512_dkv_s3_160k_224k_scaling/summary.md` |
| 08-05 09:53 PST | `e094` raw-grid 448 同卡配对 | BKV64/s3 比 BKV16/s2 快 2.34%，正确且显存相同；保守不扩 gate | `runs/e094_d512_rawgrid448_pair/summary.md` |
| 08-05 09:56 PST | `e100–e104` dQ/forward NCU 跟进 | stages/warps/tile 均无新 winner；OOR 与失败签名保留 | `runs/e100_d512_dq_stage_sweep/summary.md`、`runs/e101_d512_fwd_warp_sweep/summary.md`、`runs/e103_d512_fwd_tile_stage/summary.md` |
| 08-05 09:58 PST | `e105` dKV q_splits | 无吞吐收益，峰值显存多约 0.50 GiB；拒绝以显存换噪声 | `runs/e105_d512_dkv_qsplits/summary.md` |
| 08-05 10:04 PST | `e106/e107` stage3 production 2K–256K | GPU 数值 8/8；单序列与 ragged 全部 >1.5×，E2B/MoE 256K 为 2.39×/2.48× | `runs/e106_d512_production_stage3_grid/summary.md`、`runs/e107_d512_production_long_grid/summary.md` |
| 08-05 10:04 PST | `e108` total-256K 分布矩阵 | E2B 2.39–8.38×，MoE 2.48–6.52×；Triton 显存只随总 token 变化 | `runs/e108_d512_production_packed_grid/summary.md` |
| 08-05 10:09 PST | `e109` B200 stage3 release gate | CPU 108；B200 GPU 50/50；GPU5/6/7 各 varlen 8/8；compile/diff/selection 全通过 | `runs/e109_b200_stage3_release_gate/summary.md` |
| 08-05 10:15 PST | `e110–e113` dKV reload-V | s3 shared OOR；s2 比 control 慢 10.6%；源码撤回且 smoke 恢复 | `runs/e110_d512_dkv_reloadv/summary.md` |
| 08-05 10:18 PST | `e114` dKV reload-Q | 18.8048 vs 18.8049 ms；NCU 资源指标完全相同；源码撤回 | `runs/e114_d512_dkv_reloadq/summary.md` |
| 08-05 10:20 PST | `e115` BKV32 短边界 | blocks 增加但最佳仍慢 25%–29%；淘汰 | `runs/e115_d512_dkv_bkv32_short_boundary/summary.md` |
| 08-05 10:23 PST | `e116–e119` qsplit 单序列边界 | raw16–192 快 7%–48%；raw256 无一致收益 | `runs/e116_d512_dkv_qsplits_short_boundary/summary.md`、`runs/e119_d512_dkv_qs4_lower_boundary/summary.md` |
| 08-05 10:27 PST | `e120–e124` packed 防回退消歧 | packed 短分布可回退；候选限制为 batch1/query≥2K | `runs/e122_d512_dkv_qsplit_target256_packed/summary.md`、`runs/e124_d512_dkv_bkv64_qs1_lower/summary.md` |
| 08-05 10:30 PST | `e125` qsplit 上边界 | raw208 稳定收益；raw240/248 MoE <2%；上界取 224 exclusive | `runs/e125_d512_dkv_qs4_upper_boundary/summary.md` |
| 08-05 10:33 PST | `e126` qsplit production 晋级 | E2B 2K–8K 快 26%–42%，MoE 2K–6K 快 7%–41%；正确且仍远低于 SDPA 显存 | `runs/e126_promote_dkv_qsplit_single/summary.md` |
| 08-05 10:35 PST | `e127` qsplit 最终门禁 | packed 保持旧 q1；CPU118、B200 50/50、compile/diff 全通过 | `runs/e127_qsplit_packed_no_regression/summary.md` |
| 08-05 10:39 PST | `e128` q4 dKV NCU | grid 扩大 4×，NCU duration 17.97→9.22 ms；资源/occupancy 不变 | `runs/e128_d512_dkv_qs4_ncu/summary.md` |
| 08-05 10:41 PST | `e129` q4 资源扫描 | 8 卡候选全正确；无超过噪声门槛的 winner，显存不变 | `runs/e129_d512_dkv_qs4_resource_sweep/summary.md` |
| 08-05 10:42 PST | `e130` qsplit 后 8K NSYS | dKV/dQ/forward 为 41.4%/31.8%/26.8% | `runs/e130_d512_qsplit_8k_nsys/summary.md` |
| 08-05 10:44 PST | `e131` 8K dQ/forward NCU | dQ shared-scoreboard；forward 255 regs 与 9.52M spill | `runs/e131_d512_qsplit_8k_other_ncu/summary.md` |
| 08-05 10:45 PST | `e132` forward 两段循环 | 正确但慢 0.16%–0.18%；源码撤回 | `runs/e132_d512_forward_split_d512/summary.md` |
| 08-05 10:47 PST | `e133` 非 2 次幂 qsplit | 8 卡全正确；q3/q5/q6/q7 均未超过 q4 | `runs/e133_d512_dkv_nonpower_qsplit/summary.md` |
| 08-05 10:49 PST | `e134` qsplit memory probe | 8K 峰值 -4.53%，吞吐在噪声内 | `runs/e134_d512_qsplit_separate_scratch/summary.md` |
| 08-05 10:51 PST | `e135` qsplit memory 晋级 | 仅 B200 qs2/qs4 开启；production 8/8 正确 | `runs/e135_promote_qsplit_separate_scratch/summary.md` |
| 08-05 10:52 PST | `e136` memory release gate | CPU118、GPU50/50；packed/H100/H200 无回退 | `runs/e136_qsplit_memory_release_gate/summary.md` |
| 08-05 10:54 PST | `e137` memory 50-repeat | E2B8K/MoE4K 峰值 -4.53%，延迟变化 ≤0.22% | `runs/e137_qsplit_memory_repeat/summary.md` |
| 08-05 10:56 PST | `e138` 提前释放 delta | 8K 再省 0.25 MiB，正确且 latency 不变 | `runs/e138_varlen_release_delta_early/summary.md` |
| 08-05 10:57 PST | `e139/e140` forward hoist-Q | NCU 编译资源不变；源码撤回 | `runs/e139_d512_forward_hoist_q/summary.md` |
| 08-05 10:59 PST | `e141` 最终 memory 确认 | 8K 峰值总计 -4.67%；50-repeat/packed/门禁通过 | `runs/e141_delta_early_final_confirm/summary.md` |
| 08-05 11:02 PST | `e142/e143` split-delta | full 无收益；首份 NSYS 无 CUDA kernel data，保留失败签名 | `runs/e143_d512_varlen_split_delta_nsys/summary.md` |
| 08-05 11:03 PST | `e144` split-delta NSYS retry | dQ -11.8 µs、delta +41.1 µs；保持融合 | `runs/e144_d512_varlen_split_delta_nsys_retry/summary.md` |
| 08-05 11:05 PST | `e145` dQ warps | w8 最优；w16 慢约 18%–19% | `runs/e145_d512_dq_warps/summary.md` |
| 08-05 11:08 PST | `e146` 最终 release audit | CPU119；GPU50/50 + 3×varlen8/8；无三代资产删除 | `runs/e146_final_release_audit/summary.md` |
| 08-05 11:10 PST | `e147` Gemma-4 E2E 环境审计 | 无本地 gated 权重/兼容环境；未启动模型 run | `runs/e147_gemma4_e2e_environment_audit/summary.md` |
| 08-05 11:12 PST | `e148` FP16 qsplit memory | 2.37×/2.17× SDPA；峰值与 BF16 相同 | `runs/e148_qsplit_memory_fp16/summary.md` |
| 08-05 11:13 PST | `e149` qs2 FP16 | 2.201 ms、0.041 GiB、约 1.50×；正确 | `runs/e149_qs2_fp16/summary.md` |

## Diagnostic / superseded evidence

| Timestamp | Evidence | Result | Interpretation | Raw artifact |
| --- | --- | --- | --- | --- |
| 07-31 22:38 PST | 初版计划把 batched API 当成性能目标 | 用户在任何 run 启动前纠正为 varlen-only | `e000/e001` stopped；没有 benchmark 或代码优化 artifact | none |

## Partial setup evidence

| Timestamp | Evidence | Result | Raw artifact |
| --- | --- | --- | --- |
| 07-31 22:38 PST | B200 host inventory | 8× NVIDIA B200 空闲；driver 580.105.08 | 计划制定时的会话记录；随 `e010` 重新捕获到 run artifact |

## 当前性能证据

| Timestamp | Evidence | Result | Raw artifact |
| --- | --- | --- | --- |
| 08-04 16:50 PST | `e020` D512 balanced-8K 公共 varlen API 的 NSYS 时间线 | dKV 26.156 ms，占三个 Triton 内核时间 92.5%；后续只优化 dKV | `runs/e020_d512_backward_nsys/summary.md` |
| 08-04 17:02 PST | `e030` D512 dKV 单变量 BQ probe | BQ16 全数值门禁通过；公共 varlen F+B 从 BQ64 的 28.263 ms 降到 6.222 ms，达到 3.144× SDPA | `runs/e030_d512_dkv_bq_probe/summary.md` |
| 08-04 17:14 PST | `e033` BQ16 D512 family sweep | E2B/MoE × balanced/skewed/dominant 6/6 正确；1.956–3.144× SDPA，几何平均 2.387× | `runs/e033_d512_bq16_family_sweep/summary.md` |
| 08-04 17:43 PST | `e041` production registry family | 不使用临时候选，6/6 选择 B200 tuned override；1.956–3.146× SDPA | `runs/e041_production_d512_family/summary.md` |
| 08-04 17:49 PST | `e042` production 50-repeat | 换卡复测仍为 1.927–3.144×；性能目标完成 | `runs/e042_production_family_repeat/summary.md` |
| 08-04 21:58 PST | `e071` production E2B 256K×1 NSYS | 三个主 kernel 合计 13023.870 ms；dKV 43.0% 为最大单项，dQ+forward 已占 56.9% | `runs/e071_d512_256k_nsys/summary.md` |
| 08-05 11:15 PST | `e150` dKV warp-specialize | D512 shared OOR、GPU 2/8；源码完整撤回 | `runs/e150_dkv_warp_specialize/summary.md` |
| 08-05 11:18 PST | `e151` 回退门禁 | GPU8/8；2K–256K 2.19–2.84×；显存/selection 恢复 | `runs/e151_warp_specialize_revert_gate/summary.md` |
| 08-05 11:22 PST | `e152` dKV GQA loop | 正确/显存不变，但慢 0.8%–8.5%；恢复静态展开 | `runs/e152_dkv_runtime_gqa_loop/summary.md` |
| 08-05 11:27 PST | `e153` MoE4K NSYS | dKV/dQ/forward=47.0%/28.8%/24.2%；8 卡门禁稳定 | `runs/e153_moe4k_nsys/summary.md` |
| 08-05 11:28 PST | `e154` MoE4K dKV 资源扫描 | w8 快 1.71%、显存不变；其余慢或 OOR | `runs/e154_moe4k_dkv_resource_sweep/summary.md` |
| 08-05 11:29 PST | `e155` MoE4K w8 八卡配对 | 8/8 快 1.49%–1.62%；待长度族 | `runs/e155_moe4k_dkv_w8_paired/summary.md` |
| 08-05 11:31 PST | `e156` MoE w8 长度族 | 2K–3.5K 快 2.44%–5.34%；4K 后衰减，7K 回退 | `runs/e156_moe_dkv_w8_length_family/summary.md` |
| 08-05 11:33 PST | `e157` MoE w8 边界复核 | 2K/3.5K 多卡均 >2%；显存一致 | `runs/e157_moe_dkv_w8_boundary_confirm/summary.md` |
| 08-05 11:36 PST | `e158` MoE w8 production | 仅 full Q16/KV2 raw64–127 晋级；GPU8/8，回退点不变 | `runs/e158_promote_moe_dkv_w8/summary.md` |
| 08-05 11:40 PST | `e159` MoE w8 release gate | CPU123；GPU50/50 + 7×varlen8/8；无删除/重命名 | `runs/e159_moe_w8_release_gate/summary.md` |
| 08-05 11:42 PST | `e160` MoE2K w8 NSYS | dKV/dQ/forward=51.0%/26.6%/22.4%；FP16/回退通过 | `runs/e160_moe2k_w8_nsys/summary.md` |
| 08-05 11:44 PST | `e161` MoE2K w8 dKV NCU | 255regs、198.93KiB、12.5% occupancy、84.17% no-eligible | `runs/e161_moe2k_w8_dkv_ncu/summary.md` |
| 08-05 11:45 PST | `e162` MoE2K dKV 资源扫参 | BQ8仅+0.22%；s2/w4/w16均慢 | `runs/e162_moe2k_dkv_resource_sweep/summary.md` |
| 08-05 11:46 PST | `e163` MoE w8 q-split | 2K q8比q4快4.34%、显存相同；3.5K仅+1.23% | `runs/e163_moe_w8_qsplit_sweep/summary.md` |
| 08-05 11:48 PST | `e164` MoE w8 q8 crossover | 2K/2.5K稳定+3.27%–4.31%；3K起<2% | `runs/e164_moe_w8_qs8_crossover/summary.md` |
| 08-05 11:50 PST | `e165` MoE w8 q8 上边界 | raw88/94均+2.32%–2.64%；gate=[64,96) | `runs/e165_moe_w8_qs8_upper_boundary/summary.md` |
| 08-05 11:52 PST | `e166` MoE q8/w8 production | 2K–3008为2.77×–2.94×；raw96/128边界通过 | `runs/e166_promote_moe_qs8_w8/summary.md` |
| 08-05 11:53 PST | `e167` MoE 大q-split | q16/q32均慢；q8最优，停止dKV split轴 | `runs/e167_moe_w8_large_qsplit/summary.md` |
| 08-05 11:55 PST | `e168` MoE2K q8 post-profile | dKV/dQ/fwd=48.4%/27.5%/23.4%；FP16/边界通过 | `runs/e168_moe2k_qs8_post_nsys/summary.md` |
| 08-05 11:57 PST | `e169/e170` MoE2K dQ | 181regs/200.70KiB/12.5%；所有候选慢 | `runs/e169_moe2k_dq_ncu/summary.md`、`runs/e170_moe2k_dq_resource_sweep/summary.md` |
| 08-05 11:59 PST | `e171/e172` MoE2K forward | 255regs/100.35KiB/11.85%；所有候选慢 | `runs/e171_moe2k_forward_ncu/summary.md`、`runs/e172_moe2k_forward_resource_sweep/summary.md` |
| 08-05 12:04 PST | `e173/e174` E2B2K低grid | dKV占68.2%；q4/w8使full F+B快32.98% | `runs/e173_e2b2k_qs2_nsys/summary.md`、`runs/e174_e2b2k_dkv_low_grid_sweep/summary.md` |
| 08-05 12:09 PST | `e175–e178` E2B q8边界 | raw32–105稳定获益，raw107+保留旧配置；显存/正确性一致 | `runs/e178_e2b_q8_crossover/summary.md` |
| 08-05 12:12 PST | `e179` E2B q8/w8 production | 2K–6720为2.81×–3.50×；CPU120、GPU8/8，边界/packed通过 | `runs/e179_promote_e2b_qs8_w8/summary.md` |
| 08-05 12:15 PST | `e180` E2B2K post-profile | dKV/dQ/fwd=49.3%/26.5%/23.5%；FP16与回退门禁通过 | `runs/e180_e2b2k_qs8_post_profile/summary.md` |
| 08-05 12:18 PST | `e181/e182` split与dQ | q16/q32无可靠收益；dQ候选全慢或OOR | `runs/e181_e2b_large_qsplit/summary.md`、`runs/e182_e2b2k_dq_resource_sweep/summary.md` |
| 08-05 12:21 PST | `e183/e184` forward BKV64 | E2B 2K–12K快2.3%–4.0%，显存/正确性一致；待泛化 | `runs/e184_e2b_forward_bkv64_family/summary.md` |
| 08-05 12:28 PST | `e185/e186` forward BKV64 泛化 | 128K回退；安全gate为E2B raw32–240、MoE raw64–96 | `runs/e185_forward_bkv64_generalize/summary.md`、`runs/e186_forward_bkv64_crossover/summary.md` |
| 08-05 12:32 PST | `e187` forward BKV64 production | E2B2K 3.61×、MoE2K 3.00×；CPU129、GPU8/8，边界/packed通过 | `runs/e187_promote_forward_bkv64/summary.md` |
| 08-05 12:38 PST | `e188–e190` post-profile/NCU | dKV/dQ/fwd=50.9%/27.4%/21.0%；q8为1.73waves、85.08% no-eligible | `runs/e188_forward_bkv64_post_profile/summary.md`、`runs/e190_e2b2k_q8_dkv_ncu/summary.md` |
| 08-05 12:43 PST | `e191–e193` q11 tail-wave | raw32–40快2.96%–4.25%；raw41后低于门槛 | `runs/e193_e2b_q11_crossover/summary.md` |
| 08-05 12:46 PST | `e194` q11 production | 2K四卡3.75×–3.79×、0.041GiB；CPU131、GPU8/8 | `runs/e194_promote_e2b_q11/summary.md` |
| 08-05 12:50 PST | `e195` q11 post-profile | dKV/dQ/fwd=48.8%/28.6%/21.9%；dtype/边界/packed正确 | `runs/e195_e2b2k_q11_post_profile/summary.md` |
| 08-05 12:54 PST | `e196/e197` BKV64 forward NCU/资源 | 低occupancy/高stall；资源候选无可靠winner | `runs/e196_e2b2k_bkv64_forward_ncu/summary.md`、`runs/e197_e2b2k_bkv64_forward_resource_sweep/summary.md` |
| 08-05 12:57 PST | `e198/e199` 负实验 | MoE q9低于门槛；forward两段循环无收益且撤回 | `runs/e198_moe2k_qsplit_8_15/summary.md`、`runs/e199_forward_split_bkv64_source_probe/summary.md` |
| 08-05 13:04 PST | `e200/e201` E2B q9 crossover | raw41–44快2.49%–3.78%；raw45+保持q8 | `runs/e200_e2b_q9_grid_family/summary.md`、`runs/e201_e2b_q9_crossover/summary.md` |
| 08-05 13:09 PST | `e202` q9 production | raw41/44双卡3.49×–3.68×；CPU133、GPU8/8、回退正确 | `runs/e202_promote_e2b_q9/summary.md` |
| 08-05 13:17 PST | `e203` q9 post-profile/全长度 | dKV/dQ/fwd=46.4%/30.0%/23.2%；2K–256K、dtype、packed正确 | `runs/e203_e2b_q9_post_profile/summary.md` |
| 08-05 13:23 PST | `e204` q10尾波 | raw45略慢，raw46–48仅+0.3%–0.9%；显存相同，不晋级 | `runs/e204_e2b_q10_crossover/summary.md` |
| 08-05 13:27 PST | `e205` q9 release gate | CPU142；B200 50/50 + 3×varlen8/8；无删除/重命名 | `runs/e205_q9_release_gate/summary.md` |
| 08-05 13:37 PST | `e206–e208` 全dtype FP16 scratch | 常规吞吐/显存改善，但未覆盖BF16动态范围；不晋级 | `runs/e207_fp16_scratch_paired/summary.md` |
| 08-05 13:44 PST | `e209/e210` BF16大梯度 | scale8192起候选出现非有限、reference为0；拒绝全dtype方案 | `runs/e210_fp16_scratch_grad_scale_retry/summary.md` |
| 08-05 13:47 PST | `e211` dtype-gated scratch | BF16大梯度恢复；仅FP16使用FP16 scratch | `runs/e211_dtype_gated_fp16_scratch/summary.md` |
| 08-05 13:51 PST | `e212` FP16同卡确认 | 8卡100-repeat快1.3%–5.7%，峰值约-14%，数值通过 | `runs/e212_fp16_dtype_scratch_paired/summary.md` |
| 08-05 14:01 PST | `e213` dtype gate release | CPU142；B200 50/50 + varlen8/8；BF16大梯度/compile/diff通过 | `runs/e213_dtype_scratch_release_gate/summary.md` |
| 08-05 14:05 PST | `e214` 顺序split单scratch | 显存-4%–5%，但full F+B慢7%–18%；源码撤回 | `runs/e214_sequential_split_scratch_probe/summary.md` |
| 08-05 14:07 PST | `e215` split撤回门禁 | 8卡双点延迟/显存/数值恢复，无源码残留 | `runs/e215_sequential_split_revert_gate/summary.md` |
| 08-05 14:11 PST | `e216` BF16 atomic micro | pointer类型失败后重试成功；BF16约0.050ms、FP32约0.048ms | `runs/e216_bf16_atomic_micro/summary.md` |
| 08-05 14:15 PST | `e217` BF16 atomic dKV | 数值/范围/显存通过，但full F+B慢20%–75%；撤回 | `runs/e217_bf16_atomic_dkv_probe/summary.md` |
| 08-05 14:17 PST | `e218` BF16 atomic撤回 | 8卡延迟/显存/数值恢复，无源码残留 | `runs/e218_bf16_atomic_revert_gate/summary.md` |
| 08-05 14:19 PST | `e216` BF16x2 micro | 8卡正确；0.044–0.046ms，比FP32 atomic快4%–8% | `runs/e216_bf16_atomic_micro/summary.md` |
| 08-05 14:22 PST | `e219` BF16x2真实dKV | 显存-14%，但full F+B慢0.8%–7.4%；撤回 | `runs/e219_bf16x2_atomic_dkv_probe/summary.md` |
| 08-05 14:24 PST | `e220` BF16x2撤回 | 8卡延迟/显存/数值恢复，无源码残留 | `runs/e220_bf16x2_atomic_revert_gate/summary.md` |
| 08-05 14:37 PST | `e221` FP16 2K–256K/packed | E2B 2.23×–2.75×、MoE 2.26×–2.43×；全正确 | `runs/e221_fp16_dtype_gate_length_matrix/summary.md` |
| 08-05 14:43 PST | `e222/e223` FP16 E2B q13 | raw32稳定快约2.4%；raw33+低于门槛 | `runs/e223_fp16_e2b_q13_confirm/summary.md` |
| 08-05 14:49 PST | `e224–e226` FP16 MoE q9 | raw64–70快1.8%–2.7%；raw72保持q8 | `runs/e226_fp16_narrow_gate_confirm/summary.md` |
| 08-05 14:52 PST | `e227` E2B q13边界 | raw33–35仅快1.0%–1.8%；gate只取raw32 | `runs/e227_fp16_e2b_q13_boundary/summary.md` |
| 08-05 14:55 PST | `e228` FP16生产门控 | E2B/MoE 2K为2.82×/2.48×；8卡边界/BF16/packed正确 | `runs/e228_fp16_production_gate/summary.md` |
| 08-05 15:02 PST | `e229` FP16 post-profile | dKV仍约占45%；128K/256K和回退矩阵正确 | `runs/e229_fp16_tailwave_post_profile/summary.md` |
| 08-05 15:07 PST | `e230–e232` 三核NCU/资源 | 三核高shared/低occupancy；配置扫描无winner | `runs/e232_fp16_forward_profile_resource/summary.md` |
| 08-05 15:16 PST | `e233/e234` 空qsplit提前退出 | 8卡full F+B稳定快约1.0%–1.1%，显存不变 | `runs/e234_empty_qsplit_paired/summary.md` |
| 08-05 15:20 PST | `e235` release与失败trace | CPU147、GPU50/50+4×8/8；E2B NSYS无CUDA数据，由e236取代 | `runs/e235_empty_qsplit_release_gate/summary.md` |
| 08-05 15:24 PST | `e236` NSYS重试/边界 | E2B/MoE dKV快2.10%/2.23%；边界、8K、ragged正确 | `runs/e236_empty_qsplit_nsys_retry/summary.md` |
| 08-05 15:28 PST | `e237/e238` FP16 split复扫 | E2B q13、MoE q9仍最优；显存/正确性相同 | `runs/e238_empty_exit_moe_qsplit_rescan/summary.md` |
| 08-05 15:32 PST | `e239/e240` BF16 split复扫 | 新候选收益≤1.9%，不增加配置碎片 | `runs/e240_empty_exit_bf16_moe_qsplit_rescan/summary.md` |
| 08-05 15:38 PST | `e241/e242` dKV ds mask | 8卡同卡A/B仅快0.10%–0.29%；显式mask已恢复 | `runs/e241_dkv_redundant_ds_mask_probe/summary.md` |
| 08-05 15:43 PST | `e243` dKV causal稠密分支 | FP16略慢、BF16收益≤0.15%；源码已撤回 | `runs/e243_dkv_dense_causal_mask_probe/summary.md` |
| 08-05 15:49 PST | `e244/e245` dKV循环交换 | 原配置659,472B shared OOR；可运行点慢约46%，源码撤回 | `runs/e245_dkv_q_gqa_interchange_resource/summary.md` |
| 08-05 15:53 PST | `e246` dKV accumulator多缓冲 | 与control差异≤0.06%，属性已撤回 | `runs/e246_dkv_no_acc_multibuffer_probe/summary.md` |
| 08-05 16:01 PST | `e247` causal稠密分支长端 | 32K/128K约慢0.8%–1.0%；源码再次撤回 | `runs/e247_dkv_dense_causal_long_probe/summary.md` |
| 08-05 16:06 PST | `e248` dKV scale常量化 | 与control基本相同；签名恢复 | `runs/e248_dkv_constexpr_scale_probe/summary.md` |
| 08-05 16:12 PST | `e249` FP16 dKV资源复扫 | BQ8略慢，s2/w4更慢；无晋级 | `runs/e249_fp16_tailwave_dkv_bq_resource/summary.md` |
| 08-05 16:17 PST | `e250` FP16 MoE dQ资源 | 全部候选慢，s3 OOR；原配置最优 | `runs/e250_fp16_moe_dq_resource/summary.md` |
| 08-05 16:22 PST | `e251` FP16 MoE forward资源 | w8仅快约0.5%；其余慢或OOR | `runs/e251_fp16_moe_forward_resource/summary.md` |
| 08-05 16:26 PST | `e252` dKV atomic PTX | 已为v8.f16/v4.f32；转测relaxed内存序 | `runs/e252_dkv_atomic_ptx_audit/summary.md` |
| 08-05 16:31 PST | `e253/e254` relaxed同卡A/B | FP16快约4%，BF16快约7%–8%；显存/正确性相同 | `runs/e253_dkv_relaxed_atomic_probe/summary.md` |
| 08-05 16:36 PST | `e255` relaxed production | 8卡边界/8K/packed正确；q1/H100/H200保持false | `runs/e255_relaxed_atomic_production_gate/summary.md` |
| 08-05 16:40 PST | `e256` relaxed压力 | BF16 scale16384、多seed/边界/8K均正确且显存稳定 | `runs/e256_relaxed_atomic_stress/summary.md` |
| 08-05 16:45 PST | `e257` relaxed后profile | dKV快约8.7%–8.9%，占比降至约43%；PTX与数值门禁通过 | `runs/e257_relaxed_atomic_post_profile/summary.md` |
| 08-05 16:49 PST | `e258` relaxed完整门禁 | CPU147；GPU197 + 7×8/8；compile/diff与三代隔离通过 | `runs/e258_relaxed_atomic_release_gate/summary.md` |
| 08-05 16:55 PST | `e259–e262` relaxed后split复扫 | FP16无晋级；BF16 q14初测快2.8%/3.8% | `runs/e262_relaxed_bf16_moe_qsplit_rescan/summary.md` |
| 08-05 17:00 PST | `e263` BF16 q14同卡确认 | E2B四卡+2.49%–2.62%；MoE四卡+3.77%–3.84% | `runs/e263_relaxed_bf16_q14_paired/summary.md` |
| 08-05 17:08 PST | `e264/e265` BF16 E2B q14边界 | raw32–34、45–52获益；中间区不晋级 | `runs/e265_relaxed_bf16_e2b_q14_boundary2/summary.md` |
| 08-05 17:14 PST | `e266` BF16 E2B q14长度族 | raw54–72快2.4%–3.5%；raw80+低于门槛 | `runs/e266_relaxed_bf16_e2b_q14_family/summary.md` |
| 08-05 17:20 PST | `e267` BF16 MoE q14长度族 | raw64–68快3.4%–3.8%；后续非单调衰减 | `runs/e267_relaxed_bf16_moe_q14_family/summary.md` |
| 08-05 17:27 PST | `e268/e269` BF16 E2B q14上边界 | 已测raw45–76均>2%；raw77+低于门槛 | `runs/e269_relaxed_bf16_e2b_q14_gapfill/summary.md` |
| 08-05 17:32 PST | `e270` BF16 MoE q14 tile尾消歧 | 同raw-grid收益不稳定；只保留精确2K/raw64候选 | `runs/e270_relaxed_bf16_moe_q14_gapfill/summary.md` |
| 08-05 17:38 PST | `e271` BF16 E2B q14最终补点 | raw45–76连续gate证据闭合 | `runs/e271_relaxed_bf16_e2b_q14_final_gapfill/summary.md` |
| 08-05 17:43 PST | `e272` BF16 q14 production | 8卡命中/边界正确；E2B 3.23×–4.20×、MoE2K 3.38× | `runs/e272_bf16_q14_production_gate/summary.md` |
| 08-05 17:48 PST | `e273` BF16 q14 release | CPU152；GPU202+3×8/8；scale16384正确、峰值不变 | `runs/e273_bf16_q14_release_gate/summary.md` |
| 08-05 18:03 PST | `e274` 当前BF16长度矩阵 | 2K–256K八点2.42×–4.20×；全正确，显存线性 | `runs/e274_bf16_q14_release_length_matrix/summary.md` |
| 08-05 18:08 PST | `e275` BF16 q14 post-profile | dKV仍约43%；E2B 796.7us、3.03waves，资源瓶颈未变 | `runs/e275_bf16_q14_post_profile/summary.md` |
| 08-05 18:15 PST | `e276` BF16 q14 dKV资源 | BQ8持平；BKV32/w4/s2均慢，16/16正确且显存相同 | `runs/e276_bf16_q14_dkv_resource_paired/summary.md` |
| 08-05 18:20 PST | `e277` BF16x2 relaxed微基准 | 8卡可编译且全有限；时延不优于FP32 relaxed | `runs/e277_bf16x2_relaxed_micro/summary.md` |
| 08-05 18:25 PST | `e278` BF16x2 relaxed真实dKV | 显存-14.1%，但慢1.5%–1.9%且1/8数值失败；源码撤回 | `runs/e278_bf16x2_relaxed_dkv_paired/summary.md` |
| 08-05 18:32 PST | `e279/e280` BF16 v4微基准失败 | 分别为地址未对齐、side-effect重复；由e281修复 | `runs/e280_bf16_v4_aligned_micro/summary.md` |
| 08-05 18:36 PST | `e281` BF16 v4微基准修复 | 8卡数值恢复；v4与FP32 relaxed时延持平 | `runs/e281_bf16_v4_aligned_retry/summary.md` |
| 08-05 18:42 PST | `e282/e283` BF16 v4 dKV编译失败 | 张量索引与局部函数不受支持；均在kernel前，由e284修复 | `runs/e283_bf16_v4_dkv_compile_retry/summary.md` |
| 08-05 18:47 PST | `e284` BF16 v4真实dKV | 显存-14.1%，但慢9%–10%且1/8数值失败；源码撤回 | `runs/e284_bf16_v4_dkv_compile_retry2/summary.md` |
| 08-05 18:50 PST | `e285` BF16 atomic撤回门禁 | 8张B200各varlen8/8通过 | `runs/e285_bf16_atomic_revert_gate/summary.md` |
| 08-05 18:56 PST | `e286/e287` GQA-head grid | E2B四卡+2.5%–2.7%；MoE+1.8%，显存/正确性相同 | `runs/e287_split_gqa_head_paired/summary.md` |
| 08-05 19:04 PST | `e288/e289` E2B正确边界 | e288格式修正；raw32–77相对真实production快2.4%–4.7% | `runs/e289_split_gqa_e2b_boundary_corrected/summary.md` |
| 08-05 19:10 PST | `e290` E2B上边界 | raw80–105显存不变；q1区FP32 scratch显存+16% | `runs/e290_split_gqa_e2b_upper/summary.md` |
| 08-05 19:18 PST | `e291/e292` head-grid/q1 BF16x2 | raw106–256快5.4%–37.6%，显存同q1、正确 | `runs/e292_split_gqa_q1_bf16x2_paired/summary.md` |
| 08-05 19:24 PST | `e293/e294` 上界与压力 | raw288–480约快1%–4.6%；scale16384正确 | `runs/e294_split_gqa_bf16x2_upper_stress/summary.md` |
| 08-05 19:32 PST | `e295/e296` 长端初测 | 32K–64K收益约1.2%–2.7%；显存/正确性相同 | `runs/e296_split_gqa_bf16x2_long_retry/summary.md` |
| 08-05 19:48 PST | `e297` 96K–256K独立会话 | 8/8 exit0；候选正确且显存相同，跨卡时延不用于gate | `runs/e297_split_gqa_bf16x2_long_independent/summary.md` |
| 08-05 15:05 PST | `e298` 长端同卡A/B | 32K–192K只快0.6%–1.3%，不扩长端gate | `runs/e298_split_gqa_bf16x2_long_paired/summary.md` |
| 08-05 15:10 PST | `e299–e301` 收益岛补点 | gate收敛为`[257,281)`、`[317,537)` | `runs/e301_split_gqa_bf16x2_island_gapfill/summary.md` |
| 08-05 15:14 PST | `e302` 真实production纠正 | raw106–223对q4快1.1%–4.7%且显存约-14% | `runs/e302_split_gqa_bf16x2_production_corrected/summary.md` |
| 08-05 15:19 PST | `e303` head-grid production | 8/8正确；2.47×–4.33×SDPA | `runs/e303_production_headgrid_gate/summary.md` |
| 08-05 15:22 PST | `e304` 高梯度/边界 | scale16384与三个回退边界8/8通过 | `runs/e304_production_boundary_scale_stress/summary.md` |
| 08-05 15:27 PST | `e305` packed/长端生产矩阵 | 8/8正确；2.42×–4.01×，显存线性到4.516GiB | `runs/e305_production_packed_long_matrix/summary.md` |
| 08-05 15:31 PST | `e306` head-grid post-profile | dKV仍占约42%；低occupancy/高no-eligible | `runs/e306_headgrid_post_profile/summary.md` |
| 08-05 15:35 PST | `e307` GQA head分组 | group1最优；group2/4/8变慢，源码撤回 | `runs/e307_headgrid_gqa_group_scan/summary.md` |
| 08-05 15:39 PST | `e308` BF16x2完整块直写 | 8/8慢0.19%–0.63%，源码撤回 | `runs/e308_bf16x2_full_block_atomic/summary.md` |
| 08-05 15:43 PST | `e309` q3 BF16x2 | 显存约-14%，但8/8慢约4%–5%，不晋级 | `runs/e309_headgrid_q3_bf16x2/summary.md` |
| 08-05 15:47 PST | `e310` q1资源扫描 | BQ8持平；BKV32/s2变慢，无winner | `runs/e310_headgrid_q1_resource_scan/summary.md` |
| 08-05 15:51 PST | `e311` raw106 dQ/forward NCU | forward 6.25% occupancy且9.87M spill；隔离回归正确 | `runs/e311_headgrid_raw106_other_ncu/summary.md` |
| 08-05 15:54 PST | `e312` dQ/forward资源 | w8噪声级；其余明显变慢，无winner | `runs/e312_raw106_other_resource/summary.md` |
| 08-05 15:58 PST | `e313` head-grid发布门禁 | CPU147；GPU197 + 7×8/8；三代合同与diff通过 | `runs/e313_headgrid_release_gate/summary.md` |
| 08-05 15:48 PST | `e314` raw281–316纠正A/B | 延迟-34.5%–34.8%、吞吐+52.6%–53.5%；显存逐字节一致，8/8正确 | `runs/e314_headgrid_gap_production_corrected/summary.md` |
| 08-05 15:49 PST | `e315` 断层合并production | raw280–537边界8/8正确；2.46×–2.59×SDPA，537正确回落 | `runs/e315_gap_promotion_production_gate/summary.md` |
| 08-05 15:50 PST | `e316` 大梯度与隔离 | scale16384、FP16、packed共8/8正确，非目标路径未接管 | `runs/e316_gap_scale_isolation/summary.md` |
| 08-05 15:54 PST | `e317` head-grid上边界 | raw537–544收益-0.03%–+0.16%，显存相同；不扩gate | `runs/e317_headgrid_upper_crossover/summary.md` |
| 08-05 15:56 PST | `e318` q3 BF16x2 split权衡 | 显存-14.12%，但吞吐慢1.62%–8.47%；两点数值失败，不晋级 | `runs/e318_q3_bf16x2_split_tradeoff/summary.md` |
| 08-05 15:57 PST | `e319` q2 BF16x2资源 | 仅w4在raw76/105双赢；吞吐+0.90%/+1.64%，显存-14.12% | `runs/e319_q2_bf16x2_resource_scan/summary.md` |
| 08-05 15:59 PST | `e320` q2+w4长度族 | raw80–105吞吐+1.13%–1.66%，全点显存-14.12%，8/8正确 | `runs/e320_q2_w4_bf16x2_family/summary.md` |
| 08-05 16:01 PST | `e321` q2+w4双seed | raw72–96吞吐+0.84%–1.46%，显存-14.12%；8/8正确 | `runs/e321_q2_w4_bf16x2_boundary_confirm/summary.md` |
| 08-05 16:02 PST | `e322` q2+w4 production | raw71/72/105/106、scale16384、FP16/packed 8/8正确 | `runs/e322_q2_w4_bf16x2_production_gate/summary.md` |
| 08-05 16:03 PST | `e323` q2+w4 release | CPU149；GPU199 + 7×8/8；三代合同与diff通过 | `runs/e323_q2_w4_bf16x2_release_gate/summary.md` |
| 08-05 16:11 PST | `e324` 最终单序列矩阵 | 2K–256K八点2.42×–4.33×；峰值0.041–4.516GiB，8/8正确 | `runs/e324_final_single_length_matrix/summary.md` |
| 08-05 16:13 PST | `e325` 最终packed矩阵 | 八种分布2.46×–4.02×；packed未误接管，8/8正确 | `runs/e325_final_packed_distribution_matrix/summary.md` |
| 08-05 16:15 PST | `e326` q2/w4 post-profile | dKV占40.4%；低occupancy/高no-eligible，DRAM非瓶颈 | `runs/e326_q2_w4_post_profile/summary.md` |
| 08-05 16:17 PST | `e327` q2/w4资源复扫 | BQ8持平；stages2/BKV32/w2变慢，无winner | `runs/e327_q2_w4_resource_followup/summary.md` |
| 08-05 16:20 PST | `e328` q2/w4桶首桶尾 | 吞吐+0.66%–1.64%、显存-14.12%–14.34%；scale16384正确 | `runs/e328_q2_w4_bin_tail_gate/summary.md` |
| 08-05 17:22 PST | `e329` q2/w4下边界 | raw68–71吞吐+0.69%–0.77%、显存约-14%；raw64一例dV失败 | `runs/e329_q2_w4_lower_boundary/summary.md` |
| 08-05 17:24 PST | `e330` raw68 production | `[68,106)`边界/scale/隔离8/8正确，raw68约0.075GiB | `runs/e330_q2_w4_lower_production/summary.md` |
| 08-05 17:26 PST | `e331` raw68 release | CPU150；GPU200 + 7×8/8；三代合同与diff通过 | `runs/e331_q2_w4_lower_release_gate/summary.md` |
| 08-05 17:28 PST | `e332` 短端FP32 q2/w4 | raw32–67全慢3.10%–7.38%，显存相同；不晋级 | `runs/e332_q2_w4_fp32_short_scan/summary.md` |
| 08-05 17:32 PST | `e333` packed前置基线 | 8/8正确；ragged active仅46.9%/28.6%，下一轴为紧凑block table | `runs/e333_packed_headgrid_scan/summary.md` |
| 08-05 17:36 PST | `e334` registry可达性 | B200新分段精确；FP16/MoE与H100/H200路径仍可达 | `runs/e334_registry_reachability_audit/summary.md` |
| 08-05 17:40 PST | `e335` 显存下限审计 | 长端稳定18.0625KiB/token；2K q3仍多约14% | `runs/e335_memory_floor_audit/summary.md` |
| 08-05 17:43 PST | `e336` packed block table设计 | prefix-block方案完成；待GPU A/B，不改production | `runs/e336_packed_block_table_design/summary.md` |
| 08-05 17:48 PST | `e337` NCU方向去重 | 局部资源/D-split轴已证伪；packed紧凑网格优先 | `runs/e337_profile_direction_audit/summary.md` |
| 08-05 17:50 PST | `e338` registry策略审计 | 80配置中14个策略配置，0违规；仅sm100 dKV | `runs/e338_registry_policy_invariant_audit/summary.md` |
| 08-05 17:52 PST | `e339` registry缓存开销 | 每角色约1.40µs，不是吞吐瓶颈 | `runs/e339_registry_cache_overhead/summary.md` |
| 08-05 17:56 PST | `e340` 公开文档审计 | 当前B200路径/门禁已同步；三代历史完整保留 | `runs/e340_public_docs_audit/summary.md` |
| 08-05 18:00 PST | `e341` BF16x2数值裕量 | 178个pass；dV max-abs最紧93.75%，不放宽门槛 | `runs/e341_bf16x2_correctness_margin_audit/summary.md` |
| 08-05 18:03 PST | `e342` q2/w4聚合置信度 | 26/26吞吐为正，中位+0.894%；显存中位-14.116% | `runs/e342_q2_w4_aggregate_confidence/summary.md` |
| 08-05 18:06 PST | `e343` 跨硬件selection fuzz | 14,592次resolve，0 ambiguity/空洞/策略泄漏 | `runs/e343_cross_hardware_selection_fuzz/summary.md` |
| 08-05 18:08 PST | `e344` 最终矩阵聚合 | 单序列geo2.899×、packed geo2.768×；16/16正确 | `runs/e344_final_matrix_aggregate/summary.md` |
| 08-05 18:10 PST | `e345` dKV Amdahl预算 | dKV快10%→full约+3.81%；过滤噪声级候选 | `runs/e345_dkv_amdahl_budget/summary.md` |
| 08-05 18:12 PST | `e346` production Pareto审计 | 晋级项吞吐不退、显存不增、数值通过 | `runs/e346_production_pareto_audit/summary.md` |
| 08-05 18:16 PST | `e347` 证据跟踪边界 | 5份顶层中文文档可提交；raw/runs仍忽略 | `runs/e347_evidence_tracking_audit/summary.md` |
| 08-05 18:19 PST | `e348` packed空网格消歧 | 实测32.4%≈理论33.34%；prefix-block降级 | `runs/e348_packed_empty_grid_disambiguation/summary.md` |
| 08-05 18:22 PST | `e349` dKV shared边界 | 双驻留需shared -42.9%；局部资源轴不可行 | `runs/e349_dkv_shared_residency_bound/summary.md` |
| 08-05 18:25 PST | `e350` q2/s1驻留计划 | 唯一未闭合局部点；待GPU配额恢复 | `runs/e350_q2_s1_residency_plan/summary.md` |
| 08-05 18:28 PST | `e351` 最终验收 | production通过；CPU/GPU/矩阵/证据/三代均闭环 | `runs/e351_final_acceptance_audit/summary.md` |

## Rules

- 原始日志不可修改；结论变化只更新本索引和 `results.md`。
- 每条 evidence 必须包含 commit/dirty diff、完整命令、环境、预期、实际、pass/fail 和相对路径。
- 失败被后续实验修复时保留 failure signature，并标记 `superseded_by=<new_id>`。
