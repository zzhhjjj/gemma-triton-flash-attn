# Kernel 优化项目
> 唯一入口。细节在 `context/`，经验在 `skills/`，此处只放状态与指针。

---

## 角色与启动

你是 kernel 性能优化助手：**数据驱动，不猜测，不跳步**。

每次会话启动，按序执行（不可跳过）：
1. 读本文件 → 确认当前阶段与下一步
2. 读 `context/index.md` → 获取知识地图
3. 读 `skills/skills.md` → 查可复用经验
4. 按需读具体 `context/<topic>.md`（**禁止全量加载**）
5. 向用户确认："当前任务是 [X]，计划 [Y]，请确认"

---

## 项目状态

**当前阶段**：**✅ Ship-ready（继续迭代中）** (2026-04-16)

**最终交付指标**（H100 80GB, BF16）：
| 场景 | Speedup vs SDPA | 备注 |
|------|-----------------|------|
| **Full-causal Fwd @ N=32K (D=512, GQA 8:1)** | **2.19×** | 191 vs 87 TFLOPS/s (primary plot) |
| Full-causal Fwd @ N=16K | 1.93× | |
| Full-causal Fwd @ N=4K | 1.63× | |
| **Full-causal Fwd+Bwd @ N=4K (D=512, GQA 8:1)** | **2.18×** | |
| Full-causal Fwd+Bwd @ N=16K | 1.96× | |
| Full-causal Fwd @ N=32K **D=128** GQA 4:1 | **1.31×** | 421 vs 322 TFLOPS/s — 首次打平 cuDNN/FA3! |
| Full-causal Fwd @ N=32K **D=256 SWA** slide=1024 | **18.3×** | 3.49 vs 64.04ms |
| Gemma-4-E2B E2E fwd @ N=16K | **4.51×** | 真实 5.1B 模型 35 层完整 stack |
| Gemma-4-E2B E2E fwd @ N=8K | **2.89×** | |
| Peak GPU memory @ N=16K | **−5.3 GB** (1.32× reduction) | vs SDPA |
| Max context on 80GB H100 | **32K** (vs SDPA 16K) | **2× 长上下文**，33.4 GB 峰值 |

**交付物**：
- `flash_attn/`: kernel 源码 + HF adapter + `__init__` 公开 API
- `tests/gemma4_integration/`: 3 个测试（adapter 单元 / 真实 Gemma-4-E2B E2E / memory）
- `README.md`: **成果 + 使用说明为主**（refactor 后），嵌入 3 张 benchmark 图
- `docs/`: integration / optimization_notes / architecture / api / tests 5 个技术文档
- `benchmarks/`: `run_final_benchmark.py` + `replot.py` + `results.json` + 3 PNG
- `pyproject.toml`: `pip install -e .` 即可用
- `context/baseline.md`: 全部量化数据及复现命令

**最近结论**：
- **[2026-04-16] Pack-GQA 风格 dKV kernel**（借鉴 `flash-attention/flash_attn/cute/pack_gqa.py`）：
  - Grid 从 `(N/BKV, B*H_Q)` 改为 `(N/BKV, B*H_KV)`，内层 `tl.static_range(GQA_RATIO)` 展开 Q heads
  - 单个 dk_acc/dv_acc 吸收所有 Q heads 的贡献 → 直接 store，**无 atomic、无 expand buffer、无 reduce kernel**
  - SWA Fwd+Bwd 大幅提升：N=8K 2.03×→**3.19×**(+57%), N=16K 3.89×→**5.93×**(+52%), N=32K 7.57×→**11.67×**(+54%)
  - Memory：免了 dk_expanded+dv_expanded（~1 GB @ N=32K Gemma-4-E2B）
  - 真实 Gemma-4-E2B 未回退（cos sim 0.9997 不变, E2E fwd 未变）
- **[2026-04-16] 两个失败 fusion 方向**（kernel 保留供参考）：
  - A.1 multi-head fusion forward: GS>1 慢 2-5×（GROUP_SIZE × BQ × D × 4 accumulator spill）
  - A.2 atomic fused dQ+dKV backward: 慢 6-8×（GQA_RATIO × N/BQ 路 atomic 竞争）
  - 共同根因：Triton 在 register / atomic 两端缺精细控制
  - **Pack-GQA 绕开两者**：accumulator 大小不变、竞争为零
- **[2026-04-16] 完整测试套件 + 真实 Gemma-4-E2B 集成验证**：
  - **HF integration API 打包**：`from gemma_triton_flash_attn import register_triton_attention`
    → 3 行接入 transformers 任何 GQA 模型，无需改 modeling_*.py
  - **Adapter 单元测试** (`test_adapter.py`): **24/24 PASS** 全部 cases cos sim > 0.999987
    - GQA ratios {1:1, 2:1, 4:1, 8:1} × {full causal, SWA slide=512} × {D=256, D=512} × {B=1, B=2}
    - 覆盖 SWA N≤slide (window 覆盖全序列) 和 N>slide (真实 window 截断)
  - **Gemma-4-E2B (5.1B) 真实模型 E2E** (`test_gemma4.py`):
    - 35 层（实测：7 full + 28 sliding），config: H_Q=8, H_KV=1, head_dim=256 (sliding) / 512 (full)
    - Call counter 验证：单次 forward 35 次 adapter 调用，SWA 和 full 路径都被触发
    - Logits 正确性 (N=1024 vs SDPA): cos sim **0.9997**, top-1 **100%**, top-5 overlap **5/5**
    - Forward throughput (真实模型完整 stack): N=2K **1.32x** → N=4K **1.95x** → N=8K **2.87x** → **N=16K 4.51x**
  - **Clean env**: torch 2.9.1 + transformers 5.5.4 + triton 3.5.1 (uv venv `/opt/tiger/flash_gemma`)
  - **Bug workaround**: transformers 5.5.4 `PACKAGE_DISTRIBUTION_MAPPING['flash_attn']` KeyError → `setdefault` 补丁包在 `patch_transformers_5_5_4_flash_attn_key()`
- **[2026-04-16] SWA 短 N 专项优化（三管齐下）**：
  1. **Fused delta Triton kernel**: `(do*o).float().sum()` 0.119ms → 0.046ms (**-61%**)
     - 原 `(do.float() * o.float()).sum()` 分配 2 个 fp32 临时张量；Triton kernel fp16 load + fp32 accumulate 无临时
  2. **dKV BQ=16→64 for D=256** (D=512 不变): N=4K dKV 3.27ms → 2.01ms (**-39%**)
     - D=256 register pressure 只有 D=512 一半，可以 4x 扩大 BQ
  3. **slide_size >= N → 0 normalization** (wrapper): 走 full-causal 路径省 window mask + NaN clamp
  - **综合效果（Fwd+Bwd vs SDPA full-causal）**:
    - N=1024: 0.65x → **0.73x** (+12%)
    - N=2048: 0.66x → **0.88x** (+33%)
    - N=4096: 0.90x → **1.23x** (+37%) **首次 N=4K 打平 SDPA**
    - N=8192: 1.48x → **1.96x** (+32%)
    - N=16384: 2.85x → **3.84x** (+35%)
    - N=32768: 5.49x → **7.57x** (+38%)
  - Forward 无回退（确认 N=1K-128K 数据与之前一致）
- **[2026-04-16] SWA 全范围 N=1K-120K benchmark 完成**（slide=1024, D=256）：
  - **Forward vs SDPA**: N=4K 1.32x → N=8K 2.10x → N=16K 3.49x → N=32K 6.97x → N=64K 14.43x → **N=128K 36.41x**
  - Block size sweep 确认：N=1K-64K 全范围最优 config 一致，无需 N-specific 调优
- E2E 训练 throughput 随 N 增长持续提升：1.09x@512 → 1.31x@2K → 1.44x@4K → 1.51x@8K

**当前瓶颈**：
> D=512 Full Attention 已逼近 HBM 带宽上限 (84-90%)；D=256 SWA Fwd+Bwd 现已 N≥4K 打平/超越 SDPA。
> 剩余不足：N≤2K 仍 0.73-0.88x（窗口覆盖全序列时 SWA ≈ full-causal，Triton 在短 N 输 cuDNN 约 10-25%）。

**下一步**：
> 1. ✅ [2026-04-16] 重新跑速度和内存的benchmark，对比 SDPA，生成 FLOPS/MEM vs SEQ LEN 图表
>    - 产物：`benchmarks/{results.json, flops_vs_sdpa.png, e2e_latency_vs_sdpa.png, memory_vs_sdpa.png}`
>    - **主图**：full-causal Gemma4 full config (D=512, GQA 8:1)，linear 坐标
>      - kernel fwd 2.19× @ N=32K (191 vs 87 TFLOPS/s)
>      - kernel fwd+bwd 2.18× @ N=4K, 稳定 ≥1.96×
>    - E2E 4.51× @ N=16K, 内存 1.32× reduction @ N=16K (SDPA OOM @ N=32K)
> 2. ✅ [2026-04-16] 缩小 Triton 和 FA2 在 D=128 的 gap：之前 0.88× vs SDPA，现在 1.31×
>    - exp2 替换 exp (1.13× raw)：所有 kernel（fwd + dQ + dKV packed）
>    - split-causal-loop：off-diagonal 跳过 mask，D=128 N=32K -33% 时间
>    - USE_SPLIT: tl.constexpr = (HEAD_DIM < 512)：D=512 上关掉避免 register spill
>    - bwd 所有 kernel 都已切到 exp2 + log2 LSE 中间形式
> 3. 剩余 gap 到 FA2/FA3：warp specialization + async TMA + cluster barrier（Triton 3.x 不支持）
> 2. ✅ [2026-04-16] README 重构：只放成果 + 使用说明，技术细节搬到 `docs/`
>    - `docs/integration.md`：HF adapter 机制 + 5.5.4 bug 补丁
>    - `docs/optimization_notes.md`：pack-GQA 成功 + 两条 dead-end
>    - `docs/architecture.md`：repo layout + kernel map + 设计选择
>    - `docs/api.md`：公开 API 参考
>    - `docs/tests.md`：测试矩阵
> 3. （空闲）未来方向：若迁移到 CuTeDSL 可借鉴 flash-attention 的 TMA + 2CTA，Triton 3.x 受限


已**显式跳过**（非 Gemma4 场景）：Persistent kernel、A100 tuning、Batch>2 sweep、
softcap/ALiBi/paged KV/varlen、PyPI 发布/CI/多平台支持。

### 已明确跳过（非 Gemma4 场景）
- ❌ Persistent kernel — 已 85-90% HBM BW 利用，收益 marginal
- ❌ A100 tuning — 目标平台 H100
- ❌ Batch > 2 benchmark sweep — Gemma4 长序列训练 B=1,2
- ❌ Softcap / ALiBi / paged KV / varlen — 非 Gemma4 特性
- ❌ PyPI 发布 / CI / 多平台支持 — 目前内部使用足够

---

## 禁止重复探索的结论

| 日期 | 结论 | 来源 |
|------|------|------|
| 2026-04-16 | HEAD_DIM=512 不需要 D-tiling：BLOCK_D=512 单次 tl.dot 比分块更快（1.23x vs 0.91x SDPA） | sweep 实测 |
| 2026-04-16 | num_warps=8 对大 HEAD_DIM 至关重要：同配置 4 warps 比 8 warps 慢 3.3x | sweep 实测 |
| 2026-04-16 | BLOCK_Q=64, BLOCK_KV=32 是 Gemma4 最优组合；BLOCK_KV=64/128 或 BLOCK_Q=128 会 OOM shared memory | sweep 实测 |
| 2026-04-16 | num_stages≥3 在 BLOCK_D=512 时 OOM shared memory；num_stages=2 足够 | sweep 实测 |
| 2026-04-16 | fp16 输入直传 tl.dot 走 tensor core，比原 kernel 的 .to(float32) 快 | 对比 _flash_attn_kernel |
| 2026-04-16 | Causal early termination: KV loop 只迭代到 (q_block_idx+1)*BLOCK_Q，speedup 随 N 增大（1.65x@4K→2.14x@32K） | causal benchmark |
| 2026-04-16 | BF16 和 FP16 性能无差异（<2%），无需分别调优 | bf16 benchmark |
| 2026-04-16 | MFU 稳定在 ~18-19%，TFLOPS ~187 不随 N 增长：确认 memory bandwidth bound | long seq benchmark |
| 2026-04-16 | HBM 带宽利用率 84-90%，已接近理论上限；瓶颈是 GQA 组内 K/V 重复 load | bandwidth analysis |
| 2026-04-16 | N=128K causal 正确且 2.81x speedup；Triton 比 SDPA 节省 5x GPU 内存（无 KV expand） | long seq + memory test |
| 2026-04-16 | Grid reorder（GQA interleave）无效：短 seq <2%，长 seq -3.6%；原因是 N≥32K 时 KV 超 L2(50MB) | grid reorder 实测 |
| 2026-04-16 | B=2 per-sample 比 B=1 快 4-28%：自然 SM 利用率效应，非 kernel 问题，无需特殊优化 | batch analysis |
| 2026-04-16 | Backward dQ kernel BQ=32/BKV=32、dKV kernel BQ=BKV=16 可正确运行 HEAD_DIM=512 | backward 实测 |
| 2026-04-16 | Backward 调优：dQ(32,64) -35%, dKV(16,32) -15%；dKV BQ=32 导致灾难性 spill(19ms) | backward sweep |
| 2026-04-16 | dKV 占 backward ~80% 时间（1.73ms vs dQ 0.43ms @N=1024），是主要优化目标 | kernel breakdown |
| 2026-04-16 | dKV split (per-Q-head + reduce) 大幅提升短 seq：N=512 1.55x→2.52x，N=1024 1.90x→2.41x | split dKV 实测 |
| 2026-04-16 | dKV split BQ=32 导致灾难性 spill（0.22x）；BQ=16 正常。dk/dv acc (BKV×D×4) ×2 是瓶颈 | split dKV sweep |
| 2026-04-15 | SWA (slide_size=1024) fwd+bwd 均正确（N≤4096 atol 通过）；Gemma4 Sliding 实际 config: H_KV=16, D=256（非 D=512） | SWA benchmark |
| 2026-04-15 | SWA kv_loop_start 保守下取整可产生全 mask KV 块，当 m_i=-inf 时 exp(-inf-(-inf))=nan；用 tl.maximum(-1e20) clamp 修复 | SWA NaN debug |
| 2026-04-15 | SWA 性能断点在 N=2*slide_size：N<2S 时不如 SDPA；N=4K→1.32x, N=8K→2.20x, N=16K→4.01x vs SDPA（D=256, S=1024） | SWA D=256 benchmark |
| 2026-04-15 | D=256 下 Triton full-causal kernel 慢于 SDPA 0.65-0.83x（D=512 则快 1.6x）：SDPA 对标准 D 值有专门优化 | D=256 causal benchmark |
| 2026-04-15 | D=256 最优 block: (BLOCK_Q=128, BLOCK_KV=64, 8 warps, stages=2)；shared memory 比 D=512 宽松，允许更大 block | D=256 sweep |
| 2026-04-16 | D=256 backward **warps=4 胜过 warps=8**（与 forward 及 D=512 相反）：dQ(64,64,w=4) -44%, dKV(16,32,w=4) -48% | SWA bwd sweep |
| 2026-04-16 | SWA Fwd+Bwd 调优后：N=4K 2.73x, N=8K 5.06x, **N=16K 10.0x** vs SDPA causal（D=256, slide=1024） | SWA tuned benchmark |
| 2026-04-16 | SWA block size 最优 config 在 N=1K-64K 全范围一致；无需 N-specific 调优（sweep 确认） | SWA N-range sweep |
| 2026-04-16 | SWA N=128K 极端测试：Fwd 36.41x, Fwd+Bwd 22.85x vs SDPA causal；完全支持超长上下文 | SWA 1K-120K benchmark |
| 2026-04-16 | delta 计算 `(do.float()*o.float()).sum()` 慢：fused Triton kernel -61% (0.119→0.046ms @N=1K)；fp16 load + fp32 accum 保精度无临时张量 | delta kernel 优化 |
| 2026-04-16 | dKV BQ=16→64 for D=256 (D=512 保持): -35~43% 跨 N=1K-16K；之前认为 BQ>16 spill 是 SWA 分支的 register overhead | dKV D=256 sweep |
| 2026-04-16 | SWA 短 N 瓶颈分解：N=1024 bwd = delta(46us) + dQ(182us) + dKV(307→224us) + reduce(53us)，dKV 主导 | bwd breakdown @N=1K |
| 2026-04-16 | SLIDE_SIZE=0 路径 register pressure 比 SLIDE_SIZE>0 少（no window mask、no NaN clamp），可用更大 BQ | 观察 |
| 2026-04-16 | SWA MFU 汇总：N=1K 9%（fwd+bwd），N≥4K 稳定 17-19%；Full D=512 MFU 6-9% 已 memory-bound 触顶 | MFU benchmark |
| 2026-04-16 | atomic_add dKV fuse（替代 expand+reduce）测试：GQA_RATIO-way 竞争导致 N≥2K 慢 3-5%，仅 N=1K 微量有用；保留为 opt-in constexpr | atomic_add 实验 |
| 2026-04-16 | GQA_RATIO=2 时 pairwise add (`a[0]+a[1]`) 比 `view+sum` 快 15%，已应用于 D=256 reduce path | reduce opt |
| 2026-04-16 | Gemma4 混合 stack (5 slide + 1 full, d=2048, 6 层) E2E 训练: N=1K 1.35x, N=4K 2.13x, N=8K 2.60x vs SDPA；10 步 loss 完全一致 | gemma4_e2e.py |
| 2026-04-16 | Package 化完成：`pip install -e .` → `from gemma_triton_flash_attn import flash_attn_gqa_train`；从 `/tmp` 导入测试通过 | pyproject.toml |
| 2026-04-16 | Gemma-4-E2B 真实模型端到端集成 (5.1B, 35 层): E2E fwd N=16K 4.51x vs SDPA；cos sim 0.9997, top-1 100% | test_gemma4.py |
| 2026-04-16 | transformers 5.5.4 `PACKAGE_DISTRIBUTION_MAPPING['flash_attn']` KeyError bug：在首个 import 前 `setdefault` 可绕过 | 5.5.4 bug |
| 2026-04-16 | Gemma-4-E2B 实际 config: H_Q=8, H_KV=1 (GQA 8:1), head_dim=256 (sliding) / 512 (full), sliding_window=512, 29 sliding + 6 full 共 35 层 | Gemma4 config 实测 |
| 2026-04-16 | transformers 5.5.4 `attention_interface` 约定：返回 `(B, N, H_Q, D)` 不是 `(B, H_Q, N, D)`，需要最终 transpose(1, 2) | transformers 集成 |
| 2026-04-16 | Gemma-4-E2B 每次 forward 确实会同时调用 full (7 层) + sliding (28 层) path，共 35 次 adapter invocation — SWA+GQA 均覆盖 | adapter call 计数 |
| 2026-04-16 | test_adapter.py 单元测试：GQA ratios {1:1,2:1,4:1,8:1} × {full causal, SWA} × D∈{256,512}，24/24 通过（cos sim >0.9999） | test_adapter.py |
| 2026-04-16 | Gemma-4-E2B forward 内存：N≤8K 与 SDPA 持平；N=16K Triton 省 5.3GB (22→16.7GB, 1.32x)；N=32K SDPA OOM，Triton 33.4GB 可运行 → 80GB H100 上 **2x 长上下文** | test_memory.py |
| 2026-04-16 | SDPA 在 Gemma-4-E2B 上 N<16K 走 flash-attn backend 不 materialize attn matrix，N≥16K 开始 materialize → memory 差距在此处出现 | 内存分析 |
| 2026-04-16 | SWA **backward-only** benchmark: N=8K **3.20x**, N=16K **5.87x**, **N=32K 11.63x** vs SDPA full-causal bwd (D=256, slide=1024)；比 forward-only 和 fwd+bwd 合并数字都更大，因为 bwd launch 开销固定、SWA 工作量 O(N·S) 优势随 N 放大 | SWA bwd-only benchmark |
| 2026-04-16 | Multi-head fusion (GROUP_SIZE>1 共享 K/V load) **无效**：GS=2 比 GS=1 慢 2×，GS=4 慢 3-5×，GS=8 shmem OOM。根因：(a) GROUP_SIZE × BQ × D × 4 accumulators 跨 KV loop 存活导致 register spill；(b) baseline 的 L2 已隐式做 K/V 复用，HBM 瓶颈不在此；kernel 保留为 `_flash_attn_gqa_grouped_kernel` 但 wrapper 不使用 | A.1 grouped sweep 实测 |
| 2026-04-16 | Fused dQ+dKV backward (atomic_add 到 fp32 dK/dV 共享 buffer) **慢 6-8×**。根因：GQA 8:1 下 256 个 program 竞争同一组 dK/dV tile，fp32 atomic 每次 bwd 产生 ~69M 原子操作（N=2K）。节省的 28% 重复 matmul 被原子竞争吞掉。Kernel 保留为 `_flash_attn_gqa_bwd_fused_kernel` 但不使用 | A.2 fused bwd 实测 |
| 2026-04-16 | **Pack-GQA 风格 dKV 成功**（借鉴 flash-attn/cute/pack_gqa.py）：grid=(N/BKV, B*H_KV)，内层 tl.static_range(GQA_RATIO) 展开 Q heads，单 accumulator 吸收 GQA 组贡献 → 无 atomic、无 expand、无 reduce。SWA fwd+bwd N=8K +57%, N=16K +52%, N=32K +54%。与 A.1/A.2 的差别：accumulator 大小不变（无 spill），写单个 tile（无竞争） | pack-gqa 实现 |

---

## 优化工作流

每次迭代必须按序执行，**禁止跳步，禁止同时改多个变量**：

```
① 测量基准  →  写入 context/baseline.md（必须附测量条件）
② 定位热点  →  更新 context/hotspots.md
③ 提出假设  →  "问题是 X，原因是 Y，预期改善 Z%"
④ 最小改动  →  只改一个变量
⑤ 对比测量  →  相同条件，结果写入 baseline.md
⑥ 记录结论  →  更新本文件；有复用价值则创建 skills/ 文件
```

**输出格式**（分析时）：`【假设】【依据】【行动】【预期结果】`
**输出格式**（汇报时）：`【改动】【基准对比 Δ%】【测量条件】【结论】【下一步】`

---

## 禁止行为

- 无 profiling 数据时断言热点
- 跳过基准测量直接提交优化
- 在本文件展开细节（细节放 `context/`）
- 结论不确定时用"应该/可能/大概"而不注明置信度

---

## 文档维护协议

| 触发条件 | 动作 |
|----------|------|
| 每次任务成功/失败 | 更新"最近结论"、"当前瓶颈"、"下一步行动", 更新 tutorial/flash_attn.md |
| 得到量化数据 | 写入 `context/baseline.md`（附测量条件，否则无效） |
| 发现可复用经验 | 创建 `skills/YYYY-MM-DD_<topic>.md`，更新 `skills/skills.md` |
| 完成 context 模块 | 更新该文件 + `context/index.md` |
| 确认无需再查 | 写入上方"禁止重复探索"表 |
| context 文件 > 200 行 | 拆分子文件，在 `index.md` 索引 |

---

## 文件地图

```
flash_attn/__init__.py    公开 API 入口（导出 attention_flash_gqa, flash_attn_gqa_train 等）
flash_attn/attention.py   flash attn kernel（含 MHA naive/opt + GQA fwd+bwd + SWA）
flash_attn/utils.py       benchmark 工具函数
flash_attn/gemma4_e2e.py  Gemma4-style mixed stack (full + sliding) E2E 训练 benchmark
tests/gemma4_integration/ 集成测试：test_adapter.py (单元) + test_gemma4.py (E2E 真实 Gemma-4-E2B)
pyproject.toml            package 元数据，`pip install -e .` 即可安装为 `gemma-triton-flash-attn`
requirements.txt          集成测试依赖 (torch 2.9.1 + transformers 5.5.4 + accelerate 等)
README.md                 包说明、性能总结、用法示例
.gitignore                排除 __pycache__, *.egg-info, dev docs 等
context/index.md          所有模块摘要，每次启动必读
context/baseline.md       量化基准数据（附测量条件）
context/arch.md           架构与关键路径
context/hotspots.md       profiling 结论与热点清单
skills/skills.md          可复用经验索引，先查这里
tutorial/flash_attn.md    Triton flash attention 完整教程（从原理到 SWA 实现）
```