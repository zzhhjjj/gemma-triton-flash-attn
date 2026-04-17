# Hotspots 与 Profiling 结论

> 需要 nsys/ncu profiling 数据支撑。目前为定性分析，待量化确认。

---

## 已知热点（定性，待 profiling 确认）

### GQA kernel (`_flash_attn_gqa_kernel`)

1. **KV loop 中的 tl.dot（score + V accum）**
   - 两个 dot 占每次迭代的主要 compute 和 register 压力
   - score dot: (BLOCK_Q, BLOCK_D) @ (BLOCK_D, BLOCK_KV) → (BLOCK_Q, BLOCK_KV)
   - V dot: (BLOCK_Q, BLOCK_KV) @ (BLOCK_KV, HEAD_DIM) → (BLOCK_Q, HEAD_DIM)

2. **V block 全量 load**
   - (BLOCK_KV, HEAD_DIM) = (32, 512) = 32KB/tile，每 KV iteration 一次
   - 长序列时总 V 读取: N/BLOCK_KV × 32KB = N × 1KB per program

3. **acc 的 alpha rescale**
   - `acc = acc * alpha[:, None]`：(BLOCK_Q, HEAD_DIM) fp32 的 element-wise multiply
   - 每 KV iteration 一次，HEAD_DIM=512 时较重

### 短序列 overhead（N=128）

- N=128, BLOCK_Q=64 → 只有 2 个 Q block → grid dim0 = 2
- B*H_Q = 32 → 总 programs = 64
- H100 有 132 SM：GPU utilization 只有 ~48%
- Kernel launch overhead 占比大

### Memory bandwidth bound（定量确认 ✓）

**HBM 带宽利用率**：84-90%（N≥4096），接近理论上限 3.35 TB/s

| Config | BW Util | Arithmetic Intensity |
|--------|---------|---------------------|
| N=1024 causal | 59% | 54 F/B |
| N=4096 causal | 84% | 61 F/B |
| N=16K causal  | 90% | 63 F/B |

- Ridgeline = 295 F/B → 63 F/B 远低于此，确认 **memory-bound**
- MFU 18.9% 不是因为 kernel 低效，而是 attention 的 arithmetic intensity 天然低

**GQA K/V 重复加载是主要 traffic 来源**：
- GQA 8:1 意味着 8 个 Q head 独立加载相同的 K/V 数据
- K+V traffic 占总 traffic ~95%
- 完美 L2 复用可降 K+V traffic ~7/8 → total traffic 降 ~4x → MFU 可达 ~35%
- H100 L2 = 50MB, KV per head = 2×N×D×2 bytes
  - N=4096: 8MB/head → fits L2 ✓
  - N=16K: 32MB/head → marginal
  - N=64K: 128MB/head → 远超 L2

**优化方向**：
1. Grid reorder: 确保同 KV group 的 Q heads 连续调度，maximise L2 temporal locality
2. Persistent kernel: 一个 program 处理同 KV group 的多个 Q heads
3. Multi-head per program: 一个 program 内处理 GQA_RATIO Q heads（register pressure 限制）

---

## 已排除的误报

| 假设 | 排除原因 |
|------|----------|
| BF16 比 FP16 慢 | 实测 <2% 差异，H100 tensor core 两者 throughput 一致 |
| D-tiling 能提升性能 | BLOCK_D=512（无 tiling）比 BLOCK_D=128（4次 tiling）快 35%+ |
| dKV spill 可通过 block/warps/stages 调优消除 | **4 次实验全败 (2026-04-17)**，详见下方"dKV 四次攻坚"section |
| dKV spill 由 GQA loop code duplication 引起 | `tl.range` (dynamic) 对比 `tl.static_range` (unrolled)：spills 300 vs 302，基本不变。unroll 不是 spill 根因 |
| split dK/dV 能 beat packed | spills 大减 (302 → 46/92) 但总时间 +35-36%。compute sharing (scores+p) 比消 spill 更值 |
| atomic_add dKV 永远赔本 | **2026-04-17 翻案**：Config A (H_KV=16) 下确实慢 3-5%，但 Config B (H_KV=1 raw_grid=32) 下 Q_SPLIT=8 dKV -44%。grid-gated 启用才是对的 |
| Config A 短 N 必须用 BKV=32 | **2026-04-17 翻案**：原先结论基于 "BKV=64 短 N grid starves SMs"，但那对 H_KV=1 成立；Config A H_KV=16 时 grid@BKV=64 = N/64 × 16 ≥ 256 健康。改用 BKV=64 BQ=128 w=8 s=1 后 dKV 减 23-35%，Config A N≥2K beats SDPA |

---

## Profiling 状态

- [x] nsys timeline: kernel 占 99.2% GPU 时间，无 overhead（确认 2026-04-16）
- [x] Memory vs compute bound: 定量确认 memory-bound，BW util 84-90%（理论分析 2026-04-16）
- [x] Arithmetic intensity: ~63 F/B << ridgeline 295 F/B（理论计算 2026-04-16）
- [x] **D=512 fwd+bwd hotspot rank**（2026-04-17，`benchmarks/profile_n8k.py`）：dKV 53-54% > dQ 27-28% > fwd 18-19% > delta <1%，跨 N=4K/8K/16K 稳定
- [x] **Register spill / shmem 量化**（2026-04-17，`benchmarks/dump_kernel_regs.py`）：从 CompiledKernel.n_regs / n_spills / metadata.shared 直接读，无需 ncu
  - fwd: 255 regs, 4 spills, 192KB
  - dQ:  189 regs, **0** spills, 196KB
  - dKV: 255 regs, **302** spills, 163KB (烟枪)
  - delta: 19 regs, 0 spills
- [ ] ncu roofline（ncu 不可用，需安装）
- [ ] L2 cache hit rate for K/V across GQA Q heads（关键：决定 GQA sharing 优化收益上限）

---

## dKV 四次攻坚（2026-04-17，全失败，已固化）

> 详细数据见 `context/baseline.md` "D=512 Full-Causal Profiling Breakdown + dKV 4-Way Dead End"。
> 此处只列结论，防止未来重跑。

| Exp | 方向 | 结论 | 脚本 |
|-----|------|------|------|
| 1 | num_stages ∈ {1, 3} | OOM shmem (294/298KB)；只能 s=2 | `dkv_stages_sweep.py` |
| 2 | BQ=64→32 | 找到唯一 0-spill 配置但慢 19%；BQ=32 s≥2 灾难（Triton fallback） | `dkv_stages_sweep.py` |
| 3 | `tl.range` 代 `tl.static_range` | 慢 20%，spill 不变（unroll 不是 spill 根因） | `dkv_stages_sweep.py` |
| 4 | split dV + dK 独立 kernel | spills 大减但总时间 +35-36%（compute sharing 更值） | `dkv_split_bench.py` |

**总结论**：baseline (packed, BQ=64, BKV=16, w=4, s=2, static_range) 在 block × warps × stages × unroll × structure **5D 空间内局部最优**，dKV 19% gap 是 Triton 3.x 算法 live state 硬限，对应 FA3 的 warp specialization + async TMA + cluster barrier（Triton 不支持）。

**保留在 source**（同 A.1/A.2 风格，供未来迁移 CuTeDSL 参考）：
- `_flash_attn_gqa_bwd_dv_only_kernel` — dV-only 单独跑快 (5.54ms @ N=4K vs packed 9.27ms)
- `_flash_attn_gqa_bwd_dk_only_kernel` — dK-only 单独跑快 (6.95ms @ N=4K)
