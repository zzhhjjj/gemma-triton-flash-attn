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

---

## Profiling 状态

- [x] nsys timeline: kernel 占 99.2% GPU 时间，无 overhead（确认 2026-04-16）
- [x] Memory vs compute bound: 定量确认 memory-bound，BW util 84-90%（理论分析 2026-04-16）
- [x] Arithmetic intensity: ~63 F/B << ridgeline 295 F/B（理论计算 2026-04-16）
- [ ] ncu roofline（ncu 不可用，需安装）
- [ ] register spill 量化（ncu --metrics launch__registers_per_thread）
- [ ] shared memory 实际使用量
- [ ] L2 cache hit rate for K/V across GQA Q heads（关键：决定 GQA sharing 优化收益上限）
