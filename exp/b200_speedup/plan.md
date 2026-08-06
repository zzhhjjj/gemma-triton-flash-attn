# B200 D512 Varlen 优化计划

更新时间：2026-08-05 18:28 PST

## 目标

1. B200 production varlen D512 full F+B 覆盖短序列、4K–32K、total-128K、160K、192K、224K 和 256K，全部超过同语义 SDPA 1.5×；
2. output/dQ/dK/dV 数值正确，registry 无回退；
3. 同时优化公开接口 full F+B 吞吐与 CUDA allocator 增量峰值显存；
4. 不改变 batched、`sm90` 和非 B200 `sm100` 配置；
5. 完成 B200 可运行门禁并固化最小代码与证据。

## 当前结果

- baseline：0.36–0.64× SDPA，几何平均 0.483×；
- NSYS：dKV 占三个 Triton 内核时间 92.5%；
- NCU：BQ64 shared 166.91 KB/block，occupancy 6.25%；
- BQ16 候选确认：两次 50-repeat 为 3.096×、3.083×；
- production family：1.956–3.146×，几何平均 2.389×；
- production 50-repeat：1.927–3.144×，几何平均 2.387×；
- production total-128K：2.323–2.475×，几何平均 2.417×；
- 当前 production：raw-grid 门控 BKV64，短/长/128K 共 24/24 正确；
- stage3+qsplit production：2K–256K 单序列为 2.22–2.88× SDPA；E2B/MoE 256K 为 2.39×/2.48×；
- 256K 峰值显存：E2B 4.516 GiB、MoE 9.031 GiB，约为 SDPA 的 22.5%；
- qsplit memory 晋级：8K 峰值 185,073,664→176,422,912 bytes（-4.67%），吞吐无回退；
- FP16 qsplit scratch：短序列峰值约再降14%，E2B/MoE 2K分别为0.035/0.071GiB；BF16保留FP32 scratch动态范围；
- FP16 tail-wave：E2B raw32使用q13、MoE raw64–71使用q9，2K分别达到2.82×/2.48×SDPA；边界、BF16和packed无回退；
- 空 q-split 提前退出：8 卡 full F+B 稳定快约 1.0%–1.1%，dKV 内核快 2.1%–2.2%，显存不变；
- B200 q-split relaxed atomic：FP16 full F+B 快约4%，BF16快约7%–8%；2K达2.61×–4.08×SDPA，显存不变；
- relaxed 后 BF16 q14：E2B raw32–34/45–76、MoE精确2K曾晋级；当时2K为4.20×/3.38×，历史结论保留；
- 当前BF16长度矩阵：E2B单序列geo2.899×、packed geo2.768×，16/16正确；MoE为2.46×–3.38×；256K峰值4.516/9.031GiB；
- BF16 E2B head-grid：raw32–67用q3，raw68–105用q2/w4+BF16x2，raw106–536用q1+BF16x2；e314修复raw281–316断层，吞吐+52.6%–53.5%；
- head-grid发布矩阵：packed到256K共8/8正确，2.42×–4.01×SDPA；256K峰值4.516GiB；post-profile仍由dKV与低occupancy主导；
- 短端吞吐+显存双赢：raw68–105用q2/w4+BF16x2；raw68–71双seed吞吐+0.69%–0.77%、峰值-14.12%至-14.88%，raw64有数值失败而排除；
- MoE full 2K–<4K w8 晋级：full F+B 再快 2.44%–5.34%，显存不变；
- 固定总长 256K：E2B 随 2K×128→单条 256K 从 8.38× 降至 2.39×，分布轴已纳入默认回归；
- 2K–256K 性能、显存与正确性目标：**已完成**；当前production收口：**已完成**。
- 提交状态：代码门禁通过；14个tracked修改+5个新中文文档，3个`MM`的index未同步，commit前需统一stage。

## 进度

| 阶段 | 状态 | 结果 |
| --- | --- | --- |
| baseline 与 benchmark 口径 | 完成 | 同语义 SDPA、raw latency、MFU、selection 已固化；e070 起增加显存峰值 |
| NSYS/NCU 热点定位 | 完成 | dKV + 低 occupancy |
| 单变量 BQ probe | 完成 | BQ16 晋级；BQ32 退化，BQ128 OOR |
| confirm 与独立复测 | 完成 | 两次 50-repeat 稳定 |
| production registry 落地 | 完成 | B200-only tuned override；CPU 80/80、varlen GPU 8/8 通过 |
| production family 与复测 | 完成 | 两轮 6/6 正确且全部 >1.5× |
| 完整 B200 可运行门禁 | 完成 | e067 CPU 107 passed；B200 GPU 50/50 passed |
| 4K–32K 长序列与 total-128K 覆盖 | 完成 | 18/18 production cells 正确；128K 四种组成无 OOM/fallback |
| BKV64 候选扩展 | 完成 | e064–e066 全正确；确认 raw grid 448 回退，形成保守 grid gate |
| BKV64 production 晋级 | 完成 | e067 CPU 107、GPU 50/50、production 24/24 全通过 |
| 160K–256K 连续尺度 | 完成 | e068/e069 8/8 正确，2.274–2.354× SDPA |
| total-256K packed 组成 | 完成 | e070：8/8 正确，2.283–2.439×；显存比 SDPA 少 65.4%–77.5% |
| 256K NSYS | 完成 | e071：dKV/dQ/forward 分别占 43.0%/32.3%/24.6% |
| 256K NCU | 完成 | e072/e076/e077：dKV、dQ、forward full profile 已归档 |
| 256K 分布 NSYS | 完成 | e073：ragged 与单序列 kernel 占比基本一致 |
| 2K–256K 显存尺度 | 完成 | e075：10/10 正确，显存线性；1.563–2.516× SDPA |
| dQ NCU | 完成 | e076：shared short-scoreboard，0 spill；下一轴 stages |
| forward NCU | 完成 | e077：255 regs/thread、9.67B spill；下一轴 warps |
| dKV warps probe | 完成 | e080：w2 严重退化，w8 慢 0.87%；保持 w4 |
| dKV stages probe | 完成 | e081：s3 比 s2 快 5.6%，显存不变；进入确认 |
| s3 确认与扩展 | 完成 | e082 双卡差 0.10%、快约 5.3%；e083 4/4 快 2.44%–5.45% |
| s3 total-256K | 完成 | e084：8/8 正确，快 3.91%–6.35%，显存不变 |
| s3 production 晋级 | 完成 | e091/e106：B200 GPU 8/8；真实 registry 2K–256K 无正确性/显存回退 |
| s3 NCU | 完成 | e085：occupancy 仍 6.25%，spill 7.27M；等待 s2 control 差分 |
| s2 NCU control | 完成 | e089：s3 dKV duration -10.8%，但 shared/spill 增加 |
| s3 160K–224K | 完成 | e090：6/6 正确，快 5.62%–6.01%，显存线性 |
| s3 gate 外复查 | 完成 | e093/e094：raw-grid 448 同卡快 2.34%，暂不扩大 gate |
| s3 crossover repeat | 完成 | e086–e088：8K/32K 快 2.39%–3.72%，显存不变 |
| raw-grid 448 配对 | 完成 | e094：8.2219→8.0293 ms（+2.34%），正确且显存相同；保守不扩 gate |
| dQ stages/tile probe | 完成 | e100/e102：s2、BQ32/BKV64 最优；s3/s4 OOR |
| forward warps/tile probe | 完成 | e101/e103/e104：w4、BQ32/BKV32/s2 最优；无晋级项 |
| dKV q_splits probe | 完成 | e105：无吞吐收益，峰值显存多约 0.50 GiB；淘汰 |
| stage3 production 2K–256K | 完成 | e106/e107：长序列与 ragged 全正确；短单序列后由 e126 继续提升 |
| total-256K 分布矩阵 | 完成 | e108：E2B 2.39–8.38×；MoE 2.48–6.52× |
| 最终 CPU/GPU 门禁 | 完成 | e109：CPU 108；B200 50/50；另 3 卡 varlen 各 8/8 |
| 短单序列 qsplit 定位 | 完成 | e115–e125：raw32–63 选 qs2，raw64–223 选 qs4；packed 不晋级 |
| qsplit production 晋级 | 完成 | e126：E2B 2K/4K/8K 快 26%/42%/28%，MoE 2K/4K/6K 快 41%/19%/7% |
| qsplit 最终门禁 | 完成 | e127：CPU 118；B200 50/50；packed selection/latency 无回退 |
| qsplit 后 profiler 复核 | 完成 | e128/e130/e131：8K dKV/dQ/forward 已均衡到 41%/32%/27% |
| qsplit 资源与源码跟进 | 完成 | e129 无可靠 config winner；e132 无收益并已撤回 |
| qsplit memory 优化 | 完成 | e134–e141：B200-only 峰值 -4.67%；CPU118、GPU50/50、packed 无回退 |
| dQ profiler 跟进 | 完成 | e142–e145：split-delta 净慢约 29 µs；w8 仍最优 |
| FP16 dtype 回归 | 完成 | e148：E2B8K/MoE4K 为 2.37×/2.17×，峰值与 BF16 相同 |
| dKV warp-specialize | 失败并回退 | e150：D512 shared 266,640 > 232,448 bytes；GPU 2/8，通过项不足，源码已撤回 |
| warp-specialize 回退门禁 | 完成 | e151：8 卡；GPU8/8；2K–256K 为 2.19–2.84×，峰值恢复既有曲线 |
| dKV GQA loop live-range | 完成并回退 | e152：GPU8/8、显存不变，但 2K–128K 慢 0.8%–8.5% |
| MoE4K profiler 复核 | 完成 | e153：dKV/dQ/forward=47.0%/28.8%/24.2%；MoE 比 E2B 更偏 dKV |
| MoE4K dKV 资源扫描 | 候选确认 | e154/e155：w8 八卡均快 1.49%–1.62%，显存不变；待长度族验证 |
| MoE dKV w8 长度族 | 候选确认 | e156：2K–3.5K 快 2.44%–5.34%；4K 后衰减，7K 回退；候选 gate raw64–127 |
| MoE dKV w8 production | 完成 | e157/e158：2K/3.5K 多卡确认；仅 full Q16/KV2 raw64–127 晋级，GPU8/8 |
| MoE w8 release gate | 完成 | e159：CPU123；GPU50/50 + 7×varlen8/8；11 文件仅修改，无三代资产删除 |
| MoE2K w8 post-profile | 完成 | e160：dKV/dQ/forward=51.0%/26.6%/22.4%；FP16 与回退门禁通过 |
| MoE2K w8 dKV NCU | 完成 | e161：255 regs、198.93KiB shared、12.5% occupancy、84.17% no-eligible |
| MoE2K dKV 局部资源 | 完成 | e162：BQ8仅+0.22%；s2/w4/w16均慢，无晋级 |
| MoE w8 q-split | 候选确认 | e163：2K q8比q4快4.34%、显存相同；3.5K仅+1.23%，待 crossover |
| MoE w8 q8 crossover | 候选确认 | e164：2K/2.5K稳定+3.27%–4.31%；3K起<2%，待raw88/94边界 |
| MoE w8 q8 上边界 | 完成 | e165：raw88/94 四卡均+2.32%–2.64%；gate确定为[64,96) |
| MoE q8/w8 production | 完成 | e166：2K–3008为2.77×–2.94×；raw96/128边界与GPU8/8通过 |
| MoE 大 q-split | 完成 | e167：q16/q32均慢，q8最优；停止dKV资源轴 |
| MoE2K q8 post-profile | 完成 | e168：dKV/dQ/fwd=48.4%/27.5%/23.4%；FP16/边界通过 |
| MoE2K dQ NCU/资源 | 完成 | e169/e170：181regs、200.70KiB、12.5%；所有候选慢，停止dQ轴 |
| MoE2K forward NCU/资源 | 完成 | e171/e172：255regs、100.35KiB、11.85%；所有候选慢 |
| E2B2K q2 profiler | 完成 | e173：dKV占68.2%，首轴锁定split/warps |
| E2B q8/w8 长度与边界 | 完成 | e174–e178：raw32–105稳定获益，raw107+保留旧路径 |
| E2B q8/w8 production | 完成 | e179：2K–6720为2.81×–3.50×，GPU8/8，显存/回退通过 |
| E2B q8 post-profile | 完成 | e180：dKV/dQ/fwd=49.3%/26.5%/23.5% |
| E2B 大split与dQ资源 | 完成 | e181/e182：无可靠winner，保持q8与原dQ |
| forward BKV64 泛化 | 完成 | e183–e186：长序列会回退；E2B raw32–240、MoE raw64–96安全 |
| forward BKV64 production | 完成 | e187：E2B2K 3.61×、MoE2K 3.00×；CPU129、GPU8/8 |
| forward 后 profile/全长度回归 | 完成 | e188：dKV/dQ/fwd=50.9%/27.4%/21.0%；2K–256K正确 |
| q8 dKV NCU/资源 | 完成 | e189/e190：tile无winner；1.73 waves提示细分split |
| E2B q11 tail-wave | 完成 | e191–e194：raw32–40快3.0%–4.3%；2K达到3.75×–3.79× |
| q11 后 profile / forward资源 | 完成 | e195–e197：热点仍为dKV；forward资源候选无winner |
| MoE细分 / forward源码probe | 完成 | e198/e199：收益不足或无收益，源码已撤回 |
| E2B q9 tail-wave | 完成 | e200–e202：raw41–44快2.5%–3.8%；CPU133、GPU8/8 |
| q9 后 profile 与全长度回归 | 完成 | e203：dKV仍占46.4%；2K–256K、FP16、packed均正确 |
| q10 尾波 probe | 完成 | e204：raw45–48收益≤0.9%，不晋级 |
| q9 release 门禁 | 完成 | e205：CPU142；B200 50/50 + 3×varlen8/8；无删除 |
| 全dtype FP16 scratch | 已拒绝 | e206–e210：BF16大梯度溢出，负结论保留 |
| dtype-gated FP16 scratch | 完成 | e211/e212：仅FP16；快1.3%–5.7%，显存约-14% |
| dtype gate release 门禁 | 完成 | e213：CPU142、GPU50/50+8/8、BF16大梯度与diff通过 |
| FP16 qsplit细分 | 完成 | e222–e228：E2B raw32 q13、MoE raw64–71 q9；8卡正确且显存不增 |
| FP16短序列三核profile | 完成 | e229–e232：dKV仍为首热点；三核均高shared/低occupancy，局部资源配置无winner |
| 空qsplit源码优化 | 完成 | e233–e236：8卡确定性提升约1%；CPU147、GPU50/50+4×8/8，边界/packed无回退 |
| 提前退出后split复扫 | 完成 | e237–e240：FP16原q13/q9仍最优；BF16候选收益≤1.9%，不增加配置碎片 |
| relaxed atomic | 完成 | e252–e258：full快约4%–8%；dKV单核快约9%，显存不变；CPU/GPU完整门禁通过 |
| relaxed后q-split复扫 | 完成 | e259–e273：BF16 E2B raw32–34/45–76、MoE精确2K启用q14；完整/压力门禁通过 |
| BF16 GQA head-grid | 完成 | e286–e304：三个B200-only gate晋级；真实production、scale16384和回退边界8/8通过 |
| head-grid发布门禁 | 完成 | e305–e313：packed/256K、NCU/NSYS、负实验回退；CPU147、GPU197+7×8/8，三代合同通过 |
| raw281–316 配置断层 | 完成 | e314–e316：同shape A/B、生产边界与scale16384共24/24正确；gate合并为`[106,537)` |
| raw68–105 吞吐/显存联合优化 | 完成 | e318–e331：q2/w4+BF16x2扩至raw68；峰值约-14%，吞吐稳定为正，CPU/GPU完整门禁通过 |
| packed紧凑block table | 低优先级 | e333/e348：虽active低，但长ragged实测无空CTA成本；仅短ragged可选验证 |
| BF16顺序split单scratch | 已拒绝 | e214/e215：显存降4%–5%，吞吐慢7%–18%，已撤回 |
| BF16原生atomic | 已拒绝 | e216–e218：显存约-14%，真实dKV吞吐慢20%–75%，已撤回 |
| BF16x2 atomic | 已拒绝 | e219/e220：显存约-14%，吞吐慢0.8%–7.4%，已撤回 |
| FP16 2K–256K/packed | 完成 | e221：两profile全正确、均>2.2×SDPA，长端无回退 |
| 代码固化与清理 | 完成 | e146：CPU119、GPU50/50 + 3×varlen8/8；11 文件仅修改，无三代资产删除 |

## 接下来

1. GPU配额恢复后，可低优先级验证短ragged prefix-block；长端不再重复；
2. 只有吞吐稳定为正、显存不增且数值全过，才扩到分布矩阵；
3. 先做唯一未闭合的q2/stages1双驻留probe；失败后只保留新dKV执行模型，不再扫局部参数；
4. 提交前统一staged/unstaged边界，再复核完整门禁与中文证据。

## 连续 10 小时执行计划

| 时间 | 阶段 | 进入下一阶段的证据 |
| --- | --- | --- |
| 0–1 小时 | e069/e070：补齐 160K–256K 与 packed 基线 | correctness/selection 全通过，保存 raw latency |
| 1–2 小时 | NSYS：E2B/MoE 256K 单序列与不均匀分布 | 得到 forward/dQ/dKV 时间占比和 launch gap |
| 2–3 小时 | NCU：只分析占比最高的 kernel | 明确 occupancy、spill、shared、访存、Tensor Core 或 stall 瓶颈 |
| 3–6 小时 | 单变量 probe | 每轮只改 tile、grid、warps、stages、layout 或 fusion 中一个轴 |
| 6–7.5 小时 | confirm/repeat 与 crossover | 独立卡复测；接近噪声的结果不晋级 |
| 7.5–9 小时 | 2K–256K 多长度、多 packed 分布回归 | required cells 全正确，目标收益稳定，无未隔离回退 |
| 9–10 小时 | B200-only 晋级、完整门禁、diff 与中文归档 | CPU/GPU gate 通过；H100/H200/未知 sm100 不变 |

候选方向不预设 winner：dKV 大 tile、dQ/forward tile、warps/stages、访存复用、
减少 launch 或 fusion 只能在对应 profiler 信号出现后启动。单点 full F+B 收益
低于 2% 或 required cell 回退时默认停止；可由 raw-grid gate 安全隔离时再继续。

## 长序列扩展矩阵

两个 profile 均测试以下 workload，BF16 full F+B：

| Workload | Packed lengths | 目的 |
| --- | --- | --- |
| `h4k_b4` | `4096,4096,4096,4096` | 4K homogeneous |
| `h8k_b4` | `8192,8192,8192,8192` | 8K homogeneous |
| `h16k_b2` | `16384,16384` | 16K homogeneous |
| `h32k_b1` | `32768` | 32K single sequence |
| `ragged_16k` | `16384,8192,4096,2048,1024,512,256` | 长尾 packed 调度 |

Profile：E2B full（H_Q=8/H_KV=1/D512）与 MoE full
（H_Q=16/H_KV=2/D512），共 10 cells。

总长度 128K 另设 probe，两个 profile 均覆盖：

- `32768×4`；
- `65536×2`；
- `131072×1`；
- `65536,32768,16384,8192,4096,2048,2048`。

128K–256K 连续尺度，两个 profile 均覆盖：

- `163840×1`（160K）；
- `196608×1`（192K）；
- `229376×1`（224K）；
- `262144×1`（256K）。

total-256K packed 组成另覆盖：

- `65536×4`；
- `131072×2`；
- `262144×1`；
- `131072,65536,32768,16384,8192,4096,2048,2048`。

## 验收门槛

- D512 family 每个 cell >1.5× SDPA；
- output/dQ/dK/dV 全通过，资源失败不得变成 skip；
- 正式比较同时报告 latency、tokens/s、增量峰值 allocated/reserved；候选不得用明显显存回退换取噪声级收益；
- B200 在已验证 raw grid 必须选择 `backward_dkv_b200_bkv64_grid.sm100.d512`，回退区间选择 `backward_dkv_b200_tuned.sm100.d512`；
- batched、`sm90`、未知 `sm100` selection 不变；
- 正式结果可从顶层文档定位到原始 artifact。

## 尚未覆盖

- 真实 Gemma-4 模型 F+B；
- H100/H200 重新性能认证；
- FSDP2/多机训练；
- 过期 kernel/实验函数清理。
