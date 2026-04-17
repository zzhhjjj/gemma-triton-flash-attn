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

**当前阶段**：**✅ Ship-ready（继续迭代中）** (2026-04-17)

**最终交付指标**（H100 80GB, BF16）：
| 场景 | Speedup vs SDPA | 备注 |
|------|-----------------|------|
| **Full-causal Fwd @ N=32K (D=512, GQA 8:1)** | **2.18×** | 191 vs 87 TFLOPS/s (primary plot) |
| Full-causal Fwd @ N=16K | 1.91× | |
| Full-causal Fwd @ N=4K | 1.61× | |
| **Full-causal Fwd+Bwd @ N=2K (D=512, GQA 8:1)** | **2.94×** | 峰值，≥2.43× across all N |
| Full-causal Fwd+Bwd @ N=1K | 2.78× | |
| Full-causal Fwd+Bwd @ N=4K | 2.67× | |
| Full-causal Fwd+Bwd @ N=8K | 2.51× | |
| Full-causal Fwd+Bwd @ N=16K | 2.43× | |
| Full-causal Fwd @ N=32K **D=128** GQA 4:1 | **1.31×** | 421 vs 322 TFLOPS/s — 首次打平 cuDNN/FA3 |
| Full-causal Fwd @ N=32K **D=256 SWA** slide=1024 | **18.3×** | 3.49 vs 64.04ms |
| Gemma-4-E2B E2E fwd @ N=16K | **4.47×** | 真实 5.1B 模型 35 层完整 stack |
| Gemma-4-E2B E2E fwd @ N=8K | 2.86× | |
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
- **[2026-04-18] 长 N (8K-32K) E2E 扩展数据**：
  - **合成 Gemma4 stack 训练 fwd+bwd** (d_model=2048)：
    - N=8K: **3.12×**, N=16K: **3.82×**, N=32K: **4.39×** vs SDPA
    - N=32K 每 step 省 **4.19 秒**（5422 → 1234 ms）
  - **真实 Gemma-4-E2B forward-only** (5.1B, bf16, no_grad):
    - N=4K: 1.98×, N=8K: 2.90×, N=16K: 4.46×
    - **N=32K: SDPA OOM，Triton 892ms** ← Triton **独家支持 2× context**
    - 80GB H100 上最大 usable context: SDPA 16K vs Triton 32K
  - 真实 Gemma-4-E2B training fwd+bwd: **≥8K OOM**（5.1B 带 AdamW state 装不下 80GB），
    需要 activation checkpointing / 多卡。此处不测
  - Reproducer: `python flash_attn/gemma4_e2e.py bench`, `benchmarks/real_gemma4_fwdbwd.py`
- **[2026-04-17] E2E 训练 throughput 验证 — 所有 kernel 优化在真实训练 stack 里都生效**：
  - **合成 Gemma4 stack** (d_model=2048, 6 blocks, BF16, AdamW, ms/step training):
    - N=1K: 1.35× → **1.41×** (+4%)
    - N=2K: 1.68× → **1.85×** (+10%)
    - N=4K: 2.13× → **2.47×** (+16%)
    - N=8K: 2.60× → **3.11×** (+20%)
    - **长 N leverage 更大**：attention 占 step 比重越大，Triton 优化越显性
    - 正确性：N=512/1024/2048 loss trajectory **bit-exact** (0 rel diff vs SDPA)
  - **真实 Gemma-4-E2B (5.1B) 训练 fwd+bwd** (全 35 层 GQA 8:1):
    - N=512: 0.95× (linear proj dominate)
    - N=1K: **1.05×** (首次超越 SDPA 训练)
    - N=2K: **1.53×** (+131ms/step 节省)
    - N=4K: **2.28×** (+459ms/step 节省)
  - Forward-only 在真实模型上提升小（0.99-1.34×）是因为 Gemma-4-E2B 35 层 linear proj
    占 forward 主导。Triton 优化主要在 bwd，训练时才显性化
  - Reproducer: `flash_attn/gemma4_e2e.py both`, `benchmarks/real_gemma4_fwdbwd.py`
- **[2026-04-17] Allocation overhead 调查 + 小优化**（`benchmarks/alloc_overhead.py`）：
  - 动机：假设 Config B N=1K 短 N 场景下 alloc overhead 占 bwd 一大块
  - 量化测量：分离 "kernels_with_alloc" vs "kernels_no_alloc" vs "autograd_bwd"
  - **结论：alloc overhead 只有 10-15μs**，整个 bwd 的 3-5%，不值得重点攻击
  - **真正瓶颈是 Python autograd.Function overhead**：30-100μs，占 bwd 的 10-34%
    - Config B N=1K: **101 μs Python overhead**（35% of 296μs bwd！）
    - Config B N=4K: 36 μs（9% of bwd — 随 N 增长被 GPU work 掩盖）
    - 这部分需要 CUDA Graph 或 C++ 扩展，用户 explicit 说不做 CUDA Graph
  - 小优化：合并 `dk + dv` 为单个 5D `torch.empty` + `.zero_()`，省 1 次 allocator + 1 次 memset。
    量级 ~3-5μs，在 noise 以内，但**语义等价零成本**
  - 正确性保持：fp32 cos sim ≥ 0.9999998 × 5 shapes PASS
- **[2026-04-17] Config B 极端 starve 场景 (grid_at_BKV=64 ≤ 16) 加 BKV=64 rescue 路径**：
  - Config B (H_Q=8, H_KV=1) N=1K 下 raw_grid_at_BKV=32 = 32, QS=8 → 256 programs (2 waves)
  - sweep (`dkv_config_b_bkv64.py`) 发现更优：**BKV=64 BQ=128 w=8 s=1 + QS=8 → grid 128, 0.144ms vs BKV=32 QS=8 的 0.153ms (-6%)**
  - 原因：大 tile 每 program 工作量增加，原子 cost 被更好地摊销；1 wave 对大 tile 够用
  - Gate 分支：`if grid_at_BKV=64 >= 128 OR <= 16: 用 BKV=64`
    - Config A all N: 满足 ≥ 128 → BKV=64 + QS=1 (healthy grid)
    - Config B N=1K: 满足 ≤ 16 (raw=16) → BKV=64 + QS=8 (extreme starve rescue)
    - Config B N=2K/4K: 中间段 (raw=32/64) → 保留 BKV=32 小 tile + QS=4/2 (grid 256)
  - Q_SPLITS 目标也分化：BKV=64 target=128 (one wave)，BKV=32 target=256 (two waves)
  - 端到端 Config B N=1K：Triton fwd+bwd 0.407 → **0.391ms (-4%)**
  - 正确性：fp32 cos sim ≥ 0.99999982 × 5 shapes PASS
- **[2026-04-17] Config A dKV 短 N BKV=64 翻盘**（挖出隐藏 -23~35% dKV 时间）：
  - 诊断：2026-04-16 的 "短 N 用 BKV=32" 碎片是为 Config B 设计的，但误用到 Config A
  - Config A (H_Q=32, H_KV=16) @ N=1K, BKV=64 grid = 16 × 16 = **256 programs** = 健康
  - 同样 (BKV=64, BQ=128, w=8, s=1) 已在 N≥8K 被使用；短 N 也适用
  - `dkv_config_a_sweep.py` sweep 结论：
    - N=1K: 0.286 → **0.219ms (-23%)**
    - N=2K: 0.653 → **0.454ms (-30%)**
    - N=4K: 1.353 → **0.882ms (-35%)**
  - 新 gate：`grid_at_BKV=64 = cdiv(N, 64) × B × H_KV >= 128` → 用 big-tile config
    - Config A all N：满足 → (64, 128, 8, s=1)
    - Config B 短 N (H_KV=1)：不满足 → 保留 (32, 64, 4, s=2) + Q_SPLITS 救援
  - **Config A fwd+bwd vs SDPA 飞跃**：
    - N=1K: 0.74× → **0.83× (+12%)**
    - N=2K: 0.86× → **1.05× (+22%, 首次超越 SDPA)**
    - N=4K: 1.30× → **1.61× (+24%)**
  - Config B 路径完全不变（QS=8 仍然活跃）
  - 正确性：fp32 cos sim ≥ 0.99999988 × 8 shapes PASS
  - 翻案意义：2026-04-16 的"BKV=64 短 N starve SMs +10% regression" 结论只对 H_KV=1 成立
- **[2026-04-17] Q_SPLIT for pack-GQA dKV**（短 N 专项 · Config B 专攻）：
  - 问题：Gemma-4-E2B (H_Q=8, H_KV=1) 下 dKV grid = N/BKV × H_KV 只有 **32 programs @ N=1K = 24% H100 SM util**
    → dKV time N=1K→2K→4K = 0.273→0.270→0.293ms（几乎平，fixed overhead dominant）
  - **方案**：pack-GQA dKV kernel 加 `Q_SPLITS: tl.constexpr` + `tl.program_id(2)`
    - QS>1 时每个 KV 块的 Q loop 被切分到 Q_SPLITS 个 program
    - 输出从 `tl.store` 改为 `tl.atomic_add`（caller 预 zero dK/dV）
    - 启发式（根据 `dkv_qsplits_sweep.py` 实测）：
      - `raw_grid ≥ 256`: QS=1（正常 SM 饱和，避免 atomic 开销）
      - `raw_grid ≥ 128`: QS=2
      - `raw_grid ≥  64`: QS=4
      - `raw_grid <  64`: QS=8（例 Config B N=1K: 32→256 programs）
    - 目标：2-wave on 132 SMs (~256 programs)
  - **单 kernel 收益 (dKV only, Config B)**：
    - N=1K: 0.271 → **0.151ms (-44%)** @ QS=8
    - N=2K: 0.273 → **0.192ms (-30%)** @ QS=4
    - N=4K: 0.296 → **0.249ms (-16%)** @ QS=2
    - N=8K: 已饱和，QS=1 (0.360ms)
  - **fwd+bwd vs SDPA 累计提升（含前面 delta-fusion）**：
    - Config B (H_Q=8, H_KV=1) N=1K: **0.48× → 0.63× (+31%)**
    - Config B N=2K: **0.71× → 0.87× (+23%)**，N=4K: **1.44× → 1.65× (+15%)**
    - Config A (H_Q=32, H_KV=16) 不变（grid 健康，QS=1 保持）
  - **正确性**: fp32 cos sim ≥ 0.99999982 × 8 shapes 全 PASS；test_packed_dkv.py PASS
  - 约束：需要预 zero dk/dv buffer (`torch.zeros_like` 替代 `empty_like` when QS>1)
  - Skills: `skills/2026-04-17_q-split-dkv.md`
- **[2026-04-17] Delta fused into dQ kernel**（短 N 专项，针对 N≤2K SWA gap）：
  - 根因分析（`benchmarks/profile_short_n.py` + `diag_short_n.py`）：
    - N=1K 短 N: Triton fwd **已与 cuDNN 打平**（0.115 vs 0.114ms），gap 全在 bwd
    - SDPA 短 N 用 **cuDNN**（不是 FA2，FA2 慢 40%），我们真正要追的是 cuDNN
    - dKV 占 47-55% bwd 时间，但 Config A grid 健康（388% SM util），问题在 kernel 本身不敌 cuDNN
    - Config B (Gemma-4-E2B 8:1) N=1K 时 dKV grid 仅 32 programs = **24% H100 SMs starved**
    - launch overhead floor ~19μs/kernel × 4 kernels = 76μs ≈ 12-14% of fwd+bwd @ N=1K
  - **优化**：delta kernel 融合进 dQ prologue（dQ 反正要 load `dO`，顺手 load `O` 算 `rowsum(dO*O)`）
    - 新 `STORE_DELTA: tl.constexpr` flag；dQ kernel 加 `O_ptr` + 4 stride
    - 省掉整个 `_delta_kernel` launch（N × B × H_Q 个 tiny program launch-bound）
    - 新增成本：dQ 加载 `O` ~23μs；净省 ~23μs
  - **结果（fwd+bwd vs SDPA, D=256 H100）**:
    - Config A (H_Q=32, H_KV=16) N=1K: 0.71× → **0.74×** (-3.8%)
    - Config A N=2K: 0.84× → **0.86×**, Config A N=4K: 1.26× → **1.30×**
    - Config B (Gemma-4-E2B 8:1) N=1K: 0.48× → **0.51×** (-5.9%)
    - Config B N=2K: 0.71× → **0.75×**, Config B N=4K: 1.44× → **1.49×**
    - Config B overhead%: 18.6% → **13.2%** @ N=1K
  - 正确性：flash_attn_gqa_train vs SDPA (fp32 cos sim) **all PASS ≥0.99999988**
    - 覆盖 {D=256, D=512} × {GQA 2:1, 8:1, 8:1} × {N=1K, 2K, 4K}
  - Skill: `skills/2026-04-17_fp16-cossim-pitfall.md` — fp16 cos sim 会误报（同一数据 0.98 vs fp32 0.9999），校验必须 `.float()` 后再 cos sim
- **[2026-04-17] D=512 full-causal fwd+bwd 详细 profiling + dKV 四路攻坚全失败**：
  - Profiling (`benchmarks/profile_n8k.py`) @ N=4K/8K/16K 三档稳定 hotspot rank：
    - **dKV 53-54%** (MFU 6%) > dQ 27-28% (MFU 12%) > fwd 18-19% (MFU 17-19%) > delta <1%
    - Fwd+Bwd vs SDPA 反常递减：2.64× (N=4K) → 2.49× (N=8K) → 2.41× (N=16K)，dKV 拖的
  - Register spill 量化 (`benchmarks/dump_kernel_regs.py`) 发现烟枪：
    - dKV: **302 spills**, 255 regs (cap), 163KB shmem, w=4
    - dQ: **0 spills**, 189 regs, 196KB shmem, w=8
    - fwd: 4 spills, 255 regs, 192KB shmem, w=8
  - 四次针对性实验全失败（`benchmarks/dkv_stages_sweep.py` + `dkv_split_bench.py`）：
    1. num_stages ∈ {1, 3}: **OOM shmem** (s=1 需要 294KB, s=3 需要 298KB) 
       — 反直觉，Triton pipeliner 在 s=2 才做 buffer overlapping
    2. BQ=64→32 sweep: 找到**唯一 0-spill 配置** (BQ=32 w=4 s=1)，但**慢 19%**
       — pipelining 比消除 spill 值钱
    3. GQA loop `tl.static_range` → `tl.range`: 慢 20%，spill 几乎没变 (300 vs 302)
       — unroll 是 performance-positive，代码复制不是 spill 根因
    4. split 为独立 dV + dK kernel: spills 大减 (302 → 46/92) 但**总时间 +35-36%**
       — compute sharing (scores+p 共用) 比消 spill 值钱
  - 结论：`baseline (BQ=64, BKV=16, w=4, s=2, static_range, packed)` 在 5D 空间
    (block×warps×stages×unroll×structure) **局部最优且已到 Triton 3.x 硬限**
  - 19% 的 dKV 空间是**算法 live state 硬限**（4 matmul + 2 fp32 accumulator + GQA=8 unroll），
    对应 claude.md 之前写的"FA3 warp spec / TMA / cluster barrier"同一件事——Triton 语义不给
  - 代码：留了 `_flash_attn_gqa_bwd_dv_only_kernel` / `_flash_attn_gqa_bwd_dk_only_kernel`
    on source 作 reference（同 A.1/A.2 风格）；autograd hot path 不变
- **[2026-04-17] Pack-GQA dKV D=512 block size re-sweep**（full-causal bwd 主攻）:
  - Breakdown profile 显示 dKV 吃 fwd+bwd 62-69%，speedup 随 N 下降 (2.30× → 1.93×)
  - 旧默认 `(BKV=32, BQ=16, w=8)` 沿用 split kernel 的常量，但 pack-GQA register / shmem 模型不同
  - **Pack-GQA accumulator 只由 BKV 驱动**（dk_acc/dv_acc = BKV×D×fp32），BQ 不进 accumulator
  - Sweep 36 configs @ N=4K → **(BKV=16, BQ=64, w=4) 胜**：
    - BKV 减半让 shmem 预算给大 BQ；BQ=64 使 inner Q loop 迭代数减 4×
    - dKV @ N=4K: 12.90 → **9.30ms** (-28%)；@ N=32K: 798 → **563ms** (-29%)
    - Fwd+Bwd vs SDPA: N=4K 2.18× → **2.66×**, N=16K 1.96× → **2.42×**, N=32K 2.31× → **2.40×**
  - 旧禁忌 "BQ=32 灾难 spill" 只对 per-Q-head split 成立，pack-GQA 下无效
  - 正确性 8 shape × {causal, non-causal} 全 PASS
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
> D=512 Full Attention 已逼近 HBM 带宽上限 (84-90%)；D=256 SWA Fwd+Bwd 全 N 范围打平/超越 SDPA。
> Config A (Gemma4 syn 2:1) 现状：**N=1K 0.83×, N=2K 1.05× (beats SDPA), N=4K 1.61×**
> Config B (Gemma-4-E2B 8:1) 现状：**N=1K 0.67×, N=2K 0.88×, N=4K 1.65×**
> 剩余 N=1K 的 Config B 0.67× 是 SM-occupancy 硬限（H_KV=1 raw_grid=32 @ N=1K，
> 即便 QS=8 也只到 256 programs = 2 waves），Triton 3.x 无法进一步优化。
> **fwd 已全面打平 cuDNN；N≥2K 所有 config 均超越 cuDNN**。
>
> **[2026-04-17 更新] dKV 剩余 19% gap 确认为 Triton 3.x 硬限**：fwd+bwd hotspot 53-54% 来自 dKV，其 302 个 register spill 在 block/warps/stages/unroll/structure 5D 空间内均不可消除（四次实验全败）。对应 FA3 warp specialization + async TMA + cluster barrier——Triton 3.x 不提供。结论：Full Attention 攻坚已到天花板，新增知识见"禁止重复探索"2026-04-17 六行。

**下一步**：
> 1. ✅ [2026-04-16] 重新跑速度和内存的benchmark，对比 SDPA，生成 FLOPS/MEM vs SEQ LEN 图表
>    - 产物：`benchmarks/{results.json, flops_vs_sdpa.png, e2e_latency_vs_sdpa.png, memory_vs_sdpa.png}`
>    - **主图**：full-causal Gemma4 full config (D=512, GQA 8:1)，linear 坐标
>      - kernel fwd 2.18× @ N=32K (191 vs 87 TFLOPS/s)
>      - kernel fwd+bwd 2.94× @ N=2K (峰值), 所有 N 都 ≥2.43×
>    - E2E 4.47× @ N=16K, 内存 1.32× reduction @ N=16K (SDPA OOM @ N=32K)
> 2. ✅ [2026-04-16] 缩小 Triton 和 FA2 在 D=128 的 gap：之前 0.88× vs SDPA，现在 1.31×
>    - exp2 替换 exp (1.13× raw)：所有 kernel（fwd + dQ + dKV packed）
>    - split-causal-loop：off-diagonal 跳过 mask，D=128 N=32K -33% 时间
>    - USE_SPLIT: tl.constexpr = (HEAD_DIM < 512)：D=512 上关掉避免 register spill
>    - bwd 所有 kernel 都已切到 exp2 + log2 LSE 中间形式
> 3. 剩余 gap 到 FA2/FA3：warp specialization + async TMA + cluster barrier（Triton 3.x 不支持）
>    - [2026-04-17 确认] dKV 层面的 19% gap 已用四次实验定量证实属于这一类（见"禁止重复探索"表 2026-04-17 条目）
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
| 2026-04-17 | **Pack-GQA 的 register 模型**：dk_acc/dv_acc = BKV×D×fp32×2，**BQ 不进 accumulator**（只影响 Q/dO tile，每次 iter reload）。旧 split kernel "BQ=32 灾难 spill" 结论不适用。D=512 pack-GQA 最优 (BKV=16, BQ=64, w=4)：BKV 减半腾 shmem 给 BQ=64 → inner Q loop -4×，dKV -28~38% | pack-GQA block sweep |
| 2026-04-17 | D=512 full-causal bwd 占比：delta 0.1-2% / dQ 25-28% / dKV 54-58%（新默认后），dKV 仍是最大头但不再压倒性。delta 已被 fused kernel 压榨到可忽略，dQ 未见明显空间 | breakdown 新默认 |
| 2026-04-17 | Pack-GQA bwd num_warps=4 胜过 8（与 dKV compute 相对轻 + register 压力低相符，D=256 bwd 同结论）| sweep 实测 |
| 2026-04-17 | dQ kernel sweep @ D=512 确认当前默认 (BQ=32, BKV=64, w=8, s=2) 已是 local optimum：Top-5 间距 6-28%，N=16K 验证仍第一 | dQ sweep |
| 2026-04-17 | dQ kernel warps=8 胜过 4（与 dKV 相反）：dq_acc 只一个 (BQ×D×fp32=64KB)，register 预算宽松，8 warps 更好 hide memory latency | dQ sweep |
| 2026-04-17 | Split-causal for dKV packed kernel 在 D=512 **dead-end**：两阶段代码复制 shmem 从 232KB 推到 295KB → 全配置 OOM。BQ=16 baseline 13.2ms，即便 split -15% 仍 ~11ms > 当前 9.36ms。BQ=32 baseline 178-305ms 灾难，split 无救。与 forward USE_SPLIT=(HEAD_DIM<512) 结论一致 | split-causal 实验 |
| 2026-04-17 | D=512 full-causal fwd+bwd hotspot rank 跨 N=4K/8K/16K 全稳定：dKV 53-54% > dQ 27-28% > fwd 18-19% > delta <1%。Fwd+Bwd vs SDPA 随 N 反常递减 (2.64× → 2.41×) 由 dKV 拖 | profile_n8k.py |
| 2026-04-17 | dKV packed kernel **302 register spills**（255 reg cap）@ D=512 默认 config；dQ 0 spills，fwd 4 spills。dKV MFU 6% 是 dQ 的一半，差距来自算法 live state 硬限（4 matmul + 2 fp32 accumulator + GQA=8 unroll） | dump_kernel_regs.py |
| 2026-04-17 | dKV num_stages ∈ {1, 3} 在 D=512 默认 block 下 OOM shmem：s=1 需要 294KB，s=3 需要 298KB（vs s=2 的 163KB）。反直觉：Triton pipeliner 在 s=2 做 buffer overlapping，单 stage 反而每 buffer 独立分配。只能 s=2 | dkv_stages_sweep.py |
| 2026-04-17 | dKV BQ=64→32 sweep：唯一 0-spill 配置 (BQ=32, BKV=16, w=4, s=1) 慢 19% vs baseline。**pipelining 比消除 spill 值钱**。BQ=32 配 s≥2 灾难（Triton 退到 32 regs/thread + 1000+ spills 的 fallback 模式） | dkv_stages_sweep.py |
| 2026-04-17 | dKV GQA loop `tl.static_range` → `tl.range` (dynamic)：慢 18-20%，spill 300 vs 302 基本不变，shmem 也不变。**unroll 不是 spill 根因**，且 Triton 的 constant folding / pipeline scheduling 在展开码上 performance-positive | dkv_stages_sweep.py |
| 2026-04-17 | 将 dKV packed 拆成独立 dV-only + dK-only kernel：spills 大减 (302 → 46/92)，但总时间 **+35-36%**（N=4K 9.27→12.49ms, N=8K 35.48→48.39ms）。dV 和 dK 都要重算 scores+p，**compute sharing 比消 spill 更值**。packed kernel 保留为 hot path，两个 split kernel 作 reference 存于源码 | dkv_split_bench.py |
| 2026-04-17 | **dKV 攻坚四路全失败**：block/warps/stages/unroll/structure 5D 空间穷尽，`packed (BQ=64, BKV=16, w=4, s=2, static_range)` 是局部最优。dKV 剩余 19% gap 是算法 live state 硬限，需 warp specialization + async TMA + cluster barrier（Triton 3.x 不支持，对应 claude.md 早已标注的 FA3 gap） | 4 次实验汇总 |
| 2026-04-17 | **短 N 瓶颈全在 bwd**：fwd @ N=1K D=256 与 cuDNN 完全打平（0.115 vs 0.114ms）；SDPA 短 N 用 cuDNN（FA2 慢 40%）。短 N gap 的根因是 bwd 4-launch 堆叠 + dKV kernel 效率输 cuDNN 30-50% | `profile_short_n.py` + `diag_short_n.py` |
| 2026-04-17 | **Delta fused into dQ prologue**：STORE_DELTA constexpr + O_ptr 加进 dQ 签名，省一次 launch (N × B × H_Q tiny programs)。净省 ~23μs = 3-6% fwd+bwd @ N≤4K。正确性 fp32 cos sim ≥0.99999988 全 PASS | benchmark + test |
| 2026-04-17 | **Config B (Gemma-4-E2B 8:1) 短 N dKV SM starvation**：H_KV=1 导致 grid = N/BKV 个 program，N=1K 仅 32 programs = 24% H100 SMs。dKV 时间 N=1K→2K 几乎不变（0.273→0.270ms），证明 fixed overhead dominant，不是 compute | `diag_short_n.py` |
| 2026-04-17 | **fp16 cos sim 会误报**：同一数据 fp16 accumulated cos sim 0.98，fp32 accumulated 0.9999999。校验 Triton kernel 正确性必须先 `.float()` 再 cos sim；否则 fp16 dot product 精度损失会假装大误差 | 调试 |
| 2026-04-17 | **Q_SPLIT (split-Q + atomic_add) 对低 H_KV 短 N 夺回 SM occupancy**：Config B (H_KV=1) dKV 单 kernel 收益 -44% @ N=1K (QS=8), -30% @ N=2K (QS=4), -16% @ N=4K (QS=2)。Config A (H_KV=16) 正常 QS=1，grid 健康无需切分 | dkv_qsplits_sweep.py |
| 2026-04-17 | **Q_SPLITS 启发式目标 = 2-wave on 132 SMs (~256 programs)**：raw_grid≥256→QS=1, ≥128→QS=2, ≥64→QS=4, 否则 QS=8。比"grid≥132→QS=1"更贴近实测最优（@raw=32 QS=8 胜 QS=4 约 10%） | 实测 |
| 2026-04-17 | **旧"atomic_add dKV fuse 失败"结论在短 N + 低 H_KV 场景反转**：2026-04-16 测 Gemma4 (H_KV=16) 下 atomic 竞争让 fwd+bwd 慢 3-5% → 拒绝；但 Gemma-4-E2B (H_KV=1) N=1K 时 grid starvation 严重过 atomic 代价，Q_SPLIT=8 净赚 44%。**关键是 grid-gated (raw_grid<256)，不要全局启用** | Q_SPLIT 实测 |
| 2026-04-17 | **dKV 结构类实验：bwd launch 削减不是关键，structural 增加 grid 才是**：delta-fusion 省 1 launch 只拿 3-6%；Q_SPLIT 增加 grid 到 2-wave 可拿 15-44%。结论：launch overhead 不是短 N bwd 主因，SM occupancy 才是 | profile_short_n + dkv_qsplits_sweep |
| 2026-04-17 | **Config A 短 N BKV=32 碎片是误配**：2026-04-16 的"短 N BKV=64 starves SMs"对 Config B (H_KV=1) 成立，被错误推广到 Config A。实际 Config A H_KV=16 在 BKV=64 时 grid = N/64 × 16 永远健康（N=1K: 256 programs）。修复后 dKV 减 23-35%，Config A N≥2K 超越 SDPA | dkv_config_a_sweep.py |
| 2026-04-17 | **Gate 形式：grid_at_BKV=64 >= 128 是 config 选择的正确判据**：比 `N<8K` 单维判断更精确。Config A all N 满足 → big-tile；Config B 短 N 不满足 → small-tile + QS 救援。综合两个 config 的需求 | dkv_config_a_sweep 分析 |
| 2026-04-17 | **coupling BQ=128 w=8 with QS>1 在 N=1K 赢 7% 但 N=2K/4K 退步 10%**：extrapolation 失败，撤回。教训：不要从单一数据点推广到全范围；应在所有目标 N 都验证 | profile_short_n 回退实测 |
| 2026-04-17 | **Allocation overhead 只有 10-15μs**，是 bwd 时间的 3-5%，不是瓶颈。别再投资在这里 | alloc_overhead.py |
| 2026-04-17 | **Python autograd.Function overhead 才是 bwd 最大剩余 gap (Config B N=1K: 101μs = 35% of bwd)**。要攻击得用 CUDA Graph（改 API）或 C++ 扩展（移植成本高）。Triton 3.x 无法进一步压 | alloc_overhead.py 定量 |
| 2026-04-17 | **Python overhead 在短 N 显著，长 N 被 GPU work 掩盖**：Config B N=1K 101μs vs N=4K 36μs。原因：kernel launch CPU 部分在短 N 与 GPU 部分同量级，无法 overlap | alloc_overhead.py 观察 |
| 2026-04-17 | **Python overhead 分解**（Config B N=1K）：kernel launch 2 次 Python wrapper ~40μs（可被 GPU work 掩盖）、torch.autograd.Function 框架 ~80μs（**不可掩盖**，CPU-exclusive）、alloc 20μs。80μs autograd 开销无法在 Triton 3.x + 不改 API 的前提下优化 | triton_launch_overhead.py 量化 |
| 2026-04-17 | **2 个 bwd kernel 背靠背启动比孤立启动快 -36μs**：证明 Python launch overhead 可以被 GPU work overlap 掩盖（pipeline），不是瓶颈。瓶颈是 autograd.Function 纯 CPU 框架路径 | 实测 |

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
flash_attn/utils.py       benchmark 工具函数（benchmark_fn 等）
flash_attn/gemma4_e2e.py  Gemma4-style mixed stack (5 slide + 1 full) 合成 E2E 训练 benchmark

tests/
├── README.md             测试 tiered workflow
├── test_packed_dkv.py    ✅ 核心 dKV 正确性（每次 commit 必跑）
├── gemma4_integration/   真实 Gemma-4-E2B 集成测试（需要 clean env）
│   ├── test_adapter.py       HF adapter 单元测试 (24/24 cases)
│   ├── test_gemma4.py        真实模型 E2E forward
│   └── test_memory.py        SDPA vs Triton 显存对比
└── legacy/               失败实验保留（不在 hot path）
    ├── test_fused_backward.py  A.2 atomic-fused（净负，未用）
    └── test_grouped_forward.py A.1 multi-head fusion（spill，未用）

benchmarks/
├── README.md                   分类 + commit workflow + 测量注意事项
├── attn_only_all_n.py          ⭐ 典藏 kernel-only 全 N 全 config sweep (支持 --quick)
├── real_gemma4_fwdbwd.py       ⭐ 真实 Gemma-4-E2B training throughput
├── run_final_benchmark.py      ⭐ Release bench + 生成图
├── replot.py                   从 results.json 重绘
├── profile_*.py                Profiling (短/长 N, SWA, D=512)
├── bwd_breakdown.py / diag_short_n.py / dump_kernel_regs.py
├── alloc_overhead.py / triton_launch_overhead.py
└── archive/                    已落地的 sweep 脚本（除非 kernel 结构变，一般不跑）
    ├── dkv_*_sweep.py, dq_*_sweep.py  config 调优
    ├── dkv_stages_sweep.py, dkv_split_bench.py  FAILED 实验
    └── swa_e2e_bench.py        N-gate 开发 bench

pyproject.toml            package 元数据，`pip install -e .` 即可安装为 `gemma-triton-flash-attn`
requirements.txt          集成测试依赖 (torch 2.9.1 + transformers 5.5.4 + accelerate 等)
README.md                 包说明、性能总结、用法示例
context/index.md          所有模块摘要，每次启动必读
context/baseline.md       量化基准数据（附测量条件）
context/arch.md           架构与关键路径
context/hotspots.md       profiling 结论与热点清单
skills/skills.md          可复用经验索引，先查这里
tutorial/flash_attn.md    Triton flash attention 完整教程（从原理到 SWA 实现）
```

## Commit workflow（tiered）

每次动 kernel 代码按这 4 层跑。详见 `tests/README.md` + `benchmarks/README.md`。

| Tier | 何时跑 | 预期时间 | 命令 |
|------|--------|----------|------|
| 0. Smoke | 每次 commit | <1 min | `python tests/test_packed_dkv.py` |
| 1. Perf 回归 | 碰 kernel perf 的 commit | 2-3 min | `python benchmarks/attn_only_all_n.py --quick` |
| 2. Full bench | Major refactor / 发版前 | ~20 min | `python benchmarks/attn_only_all_n.py` + `python benchmarks/run_final_benchmark.py` |
| 3. 真实模型集成 | 发版前 | 10+ min | clean env + `tests/gemma4_integration/*.py` + `benchmarks/real_gemma4_fwdbwd.py` |

**硬性规则**：
- Tier 0 必须 PASS 才能 commit
- Tier 1 如果任何 config N 回退 >5% 必须先 investigate
- 跑出的 JSON (`*.json`) 保留在 `benchmarks/` 根，作为"上次的数字"对比基准
- Tier 2 跑完要更新 `context/baseline.md` 的表格（如果数字改了）

## Shell 命令规范
### 禁止使用的组合符
不要在 Bash 命令里使用以下符号，它们会触发权限弹窗中断工作流：
- 管道 `|`
- 逻辑连接 `&&` `||`
- 分号串联 `;`
- 重定向 `>` `>>` `<`（简单写入除外）

原因：Claude Code 的权限系统对组合命令按整条字面量匹配，即使各段单独已授权，组合后仍会重新询问。

### ❌ 不要这样
```bash
ls /tmp | wc -l
ps aux | grep python
cat file.txt | head -20
du -sh * | sort -h
echo "done" && rm /tmp/x
```

### ✅ 改用 Python 单行脚本
```bash
python -c "import os; print(len(os.listdir('/tmp')))"
python -c "import subprocess; print(subprocess.run(['ps','aux'],capture_output=True,text=True).stdout)" 
python -c "print(open('file.txt').read()[:2000])"
```

### ✅ 或用单命令自带的参数
```bash
find /tmp -maxdepth 1 | wc -l         # 如果必须管道，这种简单的可加白名单
ps -C python -o pid,cmd                # ps 自带过滤，不用 grep
head -20 file.txt                      # head 直接读文件
du -sh --sort=size *                   # 部分工具自带排序
```

### ✅ 或拆成两步，中间落盘
```bash
ps aux > /tmp/ps.out
# 然后用 Read 工具读 /tmp/ps.out
```