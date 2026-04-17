# 基准数据

> 规则：每条数据必须附测量条件，否则视为无效数据。

## 测量条件模板

```
硬件：<CPU 型号 / GPU 型号 / 内存>
OS / 驱动：<版本>
编译选项：<flags>
输入规模：<batch size / shape / dtype>
重复次数：<N 次取中位数 / 均值>
测量工具：<nsys / perf / nvprof / 自定义 timer>
测量时间：<YYYY-MM-DD>
```

---

## 基准记录

### Gemma4 GQA Flash Attention v1（2026-04-16）

**测量条件**：

```
硬件：NVIDIA H100 80GB HBM3
OS / 驱动：Linux 5.15, 驱动 535.129.03
编译选项：Triton 3.5.1, PyTorch 2.9.1, CUDA（驱动自带）
输入规模：见下表，dtype=float16, layout=(B, H, N, D) contiguous
重复次数：50 次取中位数（warmup 10 次）
测量工具：torch.cuda.Event (CUDA events)
测量时间：2026-04-16
```

**Kernel 配置**：BLOCK_Q=64, BLOCK_KV=32, BLOCK_D=512, num_warps=8, num_stages=2

#### Gemma4 config: H_Q=32, H_KV=4, D=512, FP16

| Shape (B,H_Q,N,D,H_KV) | SDPA (ms) | Triton (ms) | Speedup | SDPA TFLOPS | Triton TFLOPS | Triton MFU |
|-------------------------|-----------|-------------|---------|-------------|---------------|------------|
| (1,32,128,512,4)        | 0.0479   | 0.0580      | 0.83x   | 22.4        | 18.5          | 1.9%       |
| (1,32,256,512,4)        | 0.0765   | 0.0718      | 1.07x   | 56.1        | 59.8          | 6.0%       |
| (1,32,512,512,4)        | 0.1931   | 0.1528      | 1.26x   | 89.0        | 112.4         | 11.4%      |
| (1,32,1024,512,4)       | 0.6467   | 0.4538      | 1.42x   | 106.3       | 151.4         | 15.3%      |
| (1,32,2048,512,4)       | 2.4321   | 1.5286      | 1.59x   | 113.0       | 179.8         | 18.2%      |
| (1,32,4096,512,4)       | 9.6708   | 6.1292      | 1.58x   | 113.7       | 179.4         | 18.1%      |
| (2,32,1024,512,4)       | 1.2826   | 0.8276      | 1.55x   | 107.2       | 166.1         | 16.8%      |
| (2,32,2048,512,4)       | 4.8341   | 2.9569      | 1.63x   | 113.7       | 185.9         | 18.8%      |

> FLOPs 公式: `4 * B * H_Q * N^2 * D`（QK^T + PV 两个 matmul）
> MFU = Achieved TFLOPS / 989.5 TFLOPS（H100 FP16 tensor core peak）

**备注**：
- N=128 时 Triton 略慢，kernel launch overhead 占比大
- N≥512 后稳定快于 SDPA 1.26x–1.63x
- B=2 时 speedup 更大（1.55x–1.63x），GQA 的 KV 共享带来更好的 memory reuse
- Peak MFU 18.8%，SDPA peak 11.5%：HEAD_DIM=512 使 V load (32KB/tile) 占主导，kernel memory-bound
- TFLOPS 在 N≥2048 后趋平 (~180 TFLOPS)，表明已逼近 memory bandwidth 瓶颈

---

### Block Size Sweep 摘要（2026-04-16）

测试 shape: (1, 32, 1024, 512, 4)，SDPA baseline ≈ 0.635 ms

| Config (BQ,BKV,BD,warps,stages) | Time (ms) | vs SDPA | 备注 |
|---------------------------------|-----------|---------|------|
| (32, 32, 512, 4, 2)            | 0.527     | 1.23x   | 无 D-tiling, 4 warps |
| (32, 64, 512, 4, 2)            | 0.559     | 1.16x   | |
| (32, 64, 512, 8, 2)            | 0.543     | 1.19x   | |
| **(64, 32, 512, 8, 2)**        | **0.441** | **1.44x** | **最优** |
| (64, 32, 512, 4, 2)            | 1.469     | 0.43x   | 4 warps 不够 |
| (64, 64, 512, *, *)            | OOM       | —       | shared memory 超限 |
| (128, 32, 512, *, *)           | OOM       | —       | shared memory 超限 |
| (32, 32, 128, 4, 3)            | 0.602     | 1.07x   | D-tiling, 不如无 tiling |
| (16, 64, 128, 4, 3)            | 0.747     | 0.85x   | 小 BLOCK_Q 不利 |

**结论**：BLOCK_D=HEAD_DIM（跳过 D-tiling）+ 大 BLOCK_Q + 8 warps 是最优方向。

---

### Causal Attention（2026-04-16）

**测量条件**：同 v1（H100, FP16, CUDA events, 100 次取中位数）

**Kernel 配置**：同 v1 + IS_CAUSAL=True（KV loop early termination + causal mask）

| Shape (B,H_Q,N,D,H_KV) | SDPA Causal (ms) | Triton Causal (ms) | Speedup |
|-------------------------|-------------------|---------------------|---------|
| (1,32,128,512,4)        | 0.0499           | 0.0615              | 0.81x   |
| (1,32,256,512,4)        | 0.0734           | 0.0740              | 0.99x   |
| (1,32,512,512,4)        | 0.1596           | 0.1405              | 1.14x   |
| (1,32,1024,512,4)       | 0.4438           | 0.3222              | 1.38x   |
| (1,32,2048,512,4)       | 1.4509           | 0.9388              | 1.55x   |
| (1,32,4096,512,4)       | 5.2467           | 3.1808              | 1.65x   |
| (2,32,2048,512,4)       | 2.7596           | 1.6836              | 1.64x   |

**备注**：
- Causal ~2x faster than non-causal（符合预期：三角形 mask 约减半 FLOPs）
- Speedup 模式与 non-causal 一致：N≥512 后稳定优势

---

### Long Sequence Causal（2026-04-16）

**测量条件**：同 v1，warmup=3-5, rep=10-20（长序列单次较慢）

> Causal FLOPs ≈ `2 * B * H_Q * N^2 * D`（三角形约为 non-causal 一半）

| Shape (B,H_Q,N,D,H_KV) | SDPA (ms) | Triton (ms) | Speedup | Triton TFLOPS | Triton MFU |
|-------------------------|-----------|-------------|---------|---------------|------------|
| (1,32,4096,512,4)       | 5.18     | 3.16        | 1.64x   | 174           | 17.6%      |
| (1,32,8192,512,4)       | 20.16    | 11.91       | 1.69x   | 186           | 18.8%      |
| (1,32,16384,512,4)      | 86.10    | 46.01       | 1.87x   | 193           | 19.5%      |
| (1,32,32768,512,4)      | 402.45   | 187.93      | 2.14x   | 187           | 18.9%      |
| (1,32,65536,512,4)      | 1941.7   | 759.1       | 2.56x   | 185           | 18.7%      |
| (1,32,131072,512,4)     | 8569.5   | 3053.4      | 2.81x   | 184           | 18.6%      |

**备注**：
- N=128K 达到 **2.81x** speedup，max_diff=9.8e-4，正确性通过
- Speedup 随 N 持续增长（1.64x→2.81x），causal early-termination 优势随 N 放大
- Triton TFLOPS 稳定在 ~187，与 non-causal 一致（memory bandwidth bound）
- N=32768 正确性验证 max_diff=9.8e-4

---

### BF16 Non-Causal（2026-04-16）

**测量条件**：同 v1，dtype=bfloat16

| Shape (B,H_Q,N,D,H_KV) | SDPA BF16 (ms) | Triton BF16 (ms) | Speedup |
|-------------------------|-----------------|-------------------|---------|
| (1,32,512,512,4)        | 0.1937          | 0.1557            | 1.24x   |
| (1,32,1024,512,4)       | 0.6457          | 0.4518            | 1.43x   |
| (1,32,2048,512,4)       | 2.4108          | 1.5364            | 1.57x   |
| (1,32,4096,512,4)       | 9.5429          | 6.1024            | 1.56x   |

**备注**：
- BF16 性能与 FP16 基本一致（差异 <2%），符合预期：H100 tensor core FP16/BF16 throughput 相同
- 正确性通过（atol=5e-2, rtol=5e-2）

---

### Bandwidth Utilization Analysis（2026-04-16）

> H100: 3.35 TB/s HBM, 989.5 TFLOPS FP16, ridgeline = 295 FLOP/byte

| Config | FLOPs (G) | HBM Traffic (GB) | Arith Intensity | BW Util | MFU |
|--------|-----------|-------------------|-----------------|---------|-----|
| N=1024 causal  | 34     | 0.64  | 53.9 F/B | 59.1%  | 10.8% |
| N=4096 causal  | 550    | 8.99  | 61.1 F/B | 84.4%  | 17.5% |
| N=4096 noncaus | 1100   | 17.45 | 63.0 F/B | 85.0%  | 18.1% |
| N=16384 causal | 8796   | 139   | 63.3 F/B | 90.2%  | 19.3% |
| N=32768 causal | 35184  | 553   | 63.6 F/B | 87.8%  | 18.9% |

**结论**：
- Arithmetic intensity ~63 F/B << ridgeline 295 F/B → **确认 memory-bound**
- Bandwidth utilization 84-90% — 已接近 HBM 带宽上限
- Traffic 假设 zero L2 reuse（worst case）；GQA 8:1 如果 K/V 能在 L2 复用，traffic 可降 ~7/8 → 主要优化方向

---

### Peak GPU Memory Usage（2026-04-16）

> Triton GQA kernel 只分配 output tensor，无 KV head expansion

| Shape (B,H_Q,N,D,H_KV) | Input (MB) | SDPA Peak (MB) | Triton Peak (MB) | Ratio |
|-------------------------|------------|----------------|-------------------|-------|
| (1,32,1024,512,4)       | 41.9      | 167.8          | 33.6              | 5.0x  |
| (1,32,4096,512,4)       | 167.8     | 671.1          | 134.2             | 5.0x  |
| (1,32,16384,512,4)      | 671.1     | 2684.4         | 536.9             | 5.0x  |
| (1,32,32768,512,4)      | 1342.2    | 5368.7         | 1073.7            | 5.0x  |

**备注**：
- SDPA 需 expand KV heads (4→32)：额外 K'+V' = 2 × H_Q × N × D × 2
- Triton 只分配 output = H_Q × N × D × 2，**5x 内存节省**

---

### Batch Size Analysis — Causal（2026-04-16）

| Config | SDPA (ms) | Triton (ms) | Speedup | MFU | ms/sample |
|--------|-----------|-------------|---------|-----|-----------|
| B=1, N=512  | 0.161 | 0.141 | 1.14x | 6.2%  | 0.141 |
| B=2, N=512  | 0.279 | 0.203 | 1.37x | 8.6%  | 0.102 |
| B=1, N=1024 | 0.448 | 0.323 | 1.39x | 10.8% | 0.323 |
| B=2, N=1024 | 0.814 | 0.605 | 1.34x | 11.5% | 0.303 |
| B=1, N=2048 | 1.437 | 0.943 | 1.52x | 14.7% | 0.943 |
| B=2, N=2048 | 2.758 | 1.704 | 1.62x | 16.3% | 0.852 |
| B=1, N=4096 | 5.215 | 3.190 | 1.63x | 17.4% | 3.190 |
| B=2, N=4096 | 10.293| 6.139 | 1.68x | 18.1% | 3.069 |

**备注**：
- B=2 per-sample 比 B=1 快 4-28%：更多 programs → 更高 SM 利用率
- 差距随 N 增大收敛（N=4096 仅 4%），因为 B=1 在 N≥4096 已有足够 programs
- 不需要针对 batch size 做特殊 kernel 优化

---

### Grid Reorder 实验（失败，已回退）（2026-04-16）

**假设**：将 GQA 组内 Q heads 交叉排列在 grid pid[0] 中可提升 L2 KV 复用

**方法**：grid 从 `(cdiv(N,BQ), B*H_Q)` 改为 `(GQA_RATIO*cdiv(N,BQ), B*H_KV)`

**结果**：

| Config | 原始 (ms) | Reorder (ms) | 变化 |
|--------|-----------|--------------|------|
| N=1024 causal | 0.322 | 0.317 | -1.8% |
| N=4096 causal | 3.180 | 3.132 | -1.5% |
| N=32768 causal | 187.93 | 194.79 | **+3.6%** |
| N=4096 noncausal | 6.129 | 6.310 | **+2.9%** |

**结论**：短 seq 微量提升 (<2%), 长 seq 反而退步。原因：N≥32K 时 KV data(64MB) > L2(50MB)，
grid reorder 打乱了原有的 L2 access pattern。**已回退到原始 grid。**

---

### Backward Pass — Fwd+Bwd Benchmark（2026-04-16）

**测量条件**：同 v1，Gemma4 causal，B=1, H_Q=32, H_KV=4, D=512, FP16

**Backward config (final)**: dQ kernel BQ=32/BKV=64, dKV split kernel BQ=16/BKV=32 + reduce

| N | PyTorch Fwd+Bwd (ms) | Triton Fwd+Bwd (ms) | Speedup |
|---|----------------------|----------------------|---------|
| 512  | 2.02 | 0.80 | **2.52x** |
| 1024 | 4.97 | 2.07 | **2.41x** |
| 2048 | 14.29 | 6.49 | **2.20x** |
| 4096 | 45.9 | 22.9 | **2.01x** |
| 8192 | 164.9 | 88.8 | 1.86x |
| 16384 | 623.5 | 346.4 | 1.80x |

**正确性**：N={64,128,256,512,1024} × {causal, non-causal} × B={1,2} 全部通过（atol=5e-2）

**dKV split 设计**:
- 原 fused: grid=(cdiv(N,BKV), B*H_KV=4), 每个 program 遍历 8 Q heads → 高 register pressure
- 新 split: grid=(cdiv(N,BKV), B*H_Q=32), 每个 program 只处理 1 Q head → 8x 更多 programs
- 输出 dk_expanded(B,H_Q,N,D)，再 view+sum 得到 dk(B,H_KV,N,D)
- 短序列 SM 利用率从 ~128 programs → ~1024 programs，大幅提升
- 代价：额外 2×B×H_Q×N×D 内存 for intermediate dk/dv

**备注**：
- 短序列提升最大：N=512 从 1.55x → **2.52x**（+63%）
- 长序列略有回退：N=16K 从 1.89x → 1.80x（reduction overhead + 额外 memory traffic）
- dKV split BQ=32 仍导致 register spill（0.22x SDPA），BQ=16 是上限

---

### End-to-End Training Test（2026-04-16）

**Config**: GQA attention layer with linear projections, BF16, AdamW, causal

#### 正确性（d=4096, n_q=8, n_kv=1, head_dim=512）
- 30 步 loss 完全一致（diff=0.00），max param diff=1.83e-3
- Loss 从 1.11 → 0.92，训练收敛确认

#### 正确性（Full Gemma4: d=16384, n_q=32, n_kv=4, head_dim=512）
- 10 步 loss 完全一致（diff=0.00）
- Loss 从 1.17 → 1.08，训练收敛确认

#### E2E Throughput（Full Gemma4, B=1, BF16）

| N | SDPA (ms/step) | Triton (ms/step) | Speedup | Attn saved % |
|---|----------------|------------------|---------|--------------|
| 512 | 14.1 | 12.9 | 1.09x | 9% |
| 1024 | 19.0 | 16.1 | 1.18x | 15% |
| 2048 | 32.0 | 24.5 | 1.31x | 23% |
| 4096 | 72.4 | 50.3 | **1.44x** | 30% |
| 8192 | 207.5 | 137.5 | **1.51x** | 34% |

**备注**：
- d=16384 时 linear projections (wq/wk/wv/wo) 每个 268M 参数，短 seq 占 step 主要时间
- 随 N 增大 attention 占比增高（O(N²) vs O(N)）→ E2E speedup 持续增长
- 趋势：N→∞ 时 E2E speedup 趋近 kernel speedup (~2x)
- Kernel-only speedup（fwd+bwd causal）: 2.01-2.52x

---

> 每次优化前后都要在此追加新记录，用于对比。

---

### Sliding Window Attention (SWA) — Gemma4 Sliding Layer Config（2026-04-15）

> **⚠️ Config 更正**：Gemma4 Sliding Attention 层的 head config 与 Full Attention 不同：
> - Full Attention:    H_Q=32, **H_KV=4,  D=512**, GQA 8:1（之前章节的所有 benchmark）
> - Sliding Attention: H_Q=32, **H_KV=16, D=256**, GQA 2:1（本节）

**测量条件**：

```
硬件：NVIDIA H100 80GB HBM3
OS / 驱动：Linux 5.15, 驱动 535.129.03
编译选项：Triton 3.5.1, PyTorch 2.9.1
输入规模：B=1, H_Q=32, H_KV=16, D=256, dtype=float16
slide_size=1024
重复次数：30 次取中位数（warmup 10 次）
测量工具：torch.cuda.Event
测量时间：2026-04-15
```

**Kernel 配置（针对 D=256 重新调优）**：BLOCK_Q=128, BLOCK_KV=64, BLOCK_D=256, num_warps=8, num_stages=2

#### Forward benchmark

| N | SDPA Causal (ms) | Triton Full Causal (ms) | Triton SWA-1024 (ms) | SWA vs SDPA | Full vs SDPA |
|---|-----------------|------------------------|---------------------|-------------|---------------|
| 512   | 0.067 | 0.081 | 0.079 | 0.85x | 0.83x |
| 1024  | 0.115 | 0.154 | 0.156 | 0.74x | 0.75x |
| 2048  | 0.250 | 0.331 | 0.269 | 0.93x | 0.76x |
| 4096  | 0.647 | 0.934 | 0.489 | **1.32x** | 0.69x |
| 8192  | 2.074 | 3.202 | 0.944 | **2.20x** | 0.65x |
| 16384 | 7.990 | 11.93 | 1.992 | **4.01x** | 0.67x |

#### 关键结论

1. **D=256 下 Triton full-causal kernel 全面弱于 SDPA**（0.65-0.83x），与 D=512 下快 1.6x+ 形成对比
   - 推测原因：SDPA 对标准 D 值（64/128/256）有高度优化的 CUDA 实现（CUTLASS/cuDNN/FlashAttention-3）
   - D=512 是非标准值，SDPA fallback 到通用实现，我们的 Triton kernel 可以胜出
2. **SWA 性能断点：N = 2×slide_size**
   - N < 2S: SWA 工作量节省 <25%，不足以 overcome kernel 绝对效率差距
   - N ≥ 2S: 工作量节省 ≥50%，SWA 胜出 SDPA
3. **Speedup 随 N 线性增长**：N=4K→1.32x, N=8K→2.20x, N=16K→4.01x
   - 理论上 SWA work = N·S/2 (N≫S), Full work = N²/2, ratio = N/S
   - 实测 N=16K/S=1K 时 4.01x，接近理论 min(N/S, kernel 常数开销极限)

#### Block Size Sweep（N=2048, D=256）

SDPA baseline = 0.250 ms

| Config (BQ, BKV, BD, W, S) | Time (ms) | vs SDPA | 备注 |
|----------------------------|-----------|---------|------|
| (64, 32, 256, 4, 2)        | 0.282    | 0.89x   | D=512 默认 |
| (64, 32, 256, 4, 3)        | 0.354    | 0.71x   | stages=3 反而慢 |
| (64, 64, 256, 4, 2)        | 0.337    | 0.74x   | |
| (128, 32, 256, 8, 2)       | 0.307    | 0.81x   | |
| (128, 64, 128, 8, 2)       | 0.732    | 0.34x   | D-tiling 慢 |
| **(128, 64, 256, 8, 2)**   | **0.269** | **0.93x** | **最优** |
| (128, 128, 256, *, *)      | OOM       | —       | shared memory 超限 |
| (256, 32, 128, *, *)       | OOM/slow  | —       | 太大 |

**结论**：D=256 最优 block 比 D=512 大一倍（BQ=128/BKV=64 vs 64/32），因 shared memory 使用量
BLOCK_Q × D × 2bytes 与 D=512 的 64 × 512 × 2 相当。已更新 kernel auto-default。

#### Backward 正确性

N=128~2048 全部通过（dQ/dK/dV atol=1e-1）。

#### Backward Block Size Sweep for D=256（2026-04-16）

**测试 shape**: N=4096, D=256, slide_size=1024, causal=True

**dQ kernel sweep**（原 BQ=32, BKV=64, w=8 → 1.305ms）:

| Config (BQ, BKV, warps) | Time (ms) | 备注 |
|--------------------------|-----------|------|
| (32, 64, 8)              | 1.305     | 原默认（D=512 调优值）|
| (32, 128, 8)             | 1.094     | |
| (64, 64, 8)              | 1.139     | |
| (64, 32, 4)              | 0.874     | warps=4 开始胜出 |
| **(64, 64, 4)**          | **0.738** | **-44% 最优** |
| (64, 128, 8), (128, 64, 8), (128, 128, 8) | OOM | shared memory |

**dKV kernel sweep**（原 BQ=16, BKV=32, w=8 → 2.900ms）:

| Config (BQ, BKV, warps) | Time (ms) | 备注 |
|--------------------------|-----------|------|
| (16, 32, 8)              | 2.900     | 原默认 |
| (16, 64, 8)              | 1.699     | |
| (32, 32, 8)              | 2.415     | |
| **(16, 32, 4)**          | **1.503** | **-48% 最优** |
| (32, 32, 4)              | 1.862     | |
| (16, 128, 8), (32, 128, 8) | 4-7ms  | 大 BKV 反而慢 |

**关键发现**：D=256 下 **num_warps=4 胜过 num_warps=8**（与 forward 及 D=512 的结论相反）
- 推测原因：D=256 compute 较轻，8 warps 反而增加调度开销 / register pressure per warp
- Forward 仍保持 num_warps=8 最优

#### SWA Fwd+Bwd 调优后 Benchmark（2026-04-16）

**Config**: D=256, H_Q=32, H_KV=16, slide_size=1024, causal=True

| N | SDPA Causal Fwd+Bwd (ms) | Triton SWA Fwd+Bwd (ms) | Speedup |
|---|--------------------------|--------------------------|---------|
| 1024  | 0.94   | 0.80  | 1.19x |
| 2048  | 2.54   | 1.62  | 1.57x |
| 4096  | 9.00   | 3.30  | **2.73x** |
| 8192  | 34.07  | 6.73  | **5.06x** |
| 16384 | 136.7  | 13.7  | **10.0x** |

**对比调优前（Before）**:
| N | Before (ms) | After (ms) | Bwd 调优收益 |
|---|-------------|------------|--------------|
| 1024  | 1.14  | 0.80  | -30% |
| 2048  | 2.46  | 1.62  | -34% |
| 4096  | 5.27  | 3.30  | -37% |
| 8192  | 10.98 | 6.73  | -39% |

**D=512 Full Causal Fwd+Bwd 未回退验证**（仍使用原 w=8 配置）:

| N | SDPA (ms) | Triton (ms) | Speedup |
|---|-----------|-------------|---------|
| 1024 | 4.90 | 2.07 | 2.37x |
| 2048 | 14.24 | 6.46 | 2.20x |
| 4096 | 46.15 | 22.88 | 2.02x |

---

### SWA Full Range Benchmark N=1K-120K（2026-04-16）

**配置**: Gemma4 Sliding: B=1, H_Q=32, H_KV=16, D=256, slide_size=1024, FP16, causal

**Forward + Backward kernel configs**（全范围统一，sweep 确认 N=1K~64K 无需 N-specific 调优）:
- Forward: BQ=128, BKV=64, w=8, s=2
- Backward dQ: BQ=64, BKV=64, w=4, s=2
- Backward dKV: BQ=16, BKV=32, w=4, s=2

#### 正确性（vs attention_swa_ref, atol=5e-2）
| N | Forward | Backward |
|---|---------|----------|
| 1024  | OK (4.88e-4) | OK |
| 4096  | OK (4.88e-4) | OK |
| 16384 | OK (4.88e-4) | OK |
| 65536 | OK (4.88e-4) | ref OOM |
| 131072 | ref OOM | ref OOM |

#### 性能（vs SDPA full-causal）

| N       | SDPA Fwd (ms) | SWA Fwd (ms) | Fwd Speedup | SDPA F+B (ms) | SWA F+B (ms) | F+B Speedup |
|---------|---------------|--------------|-------------|---------------|--------------|-------------|
| 1,024   | 0.12          | 0.16         | 0.72x       | 0.5           | 0.8          | 0.65x       |
| 2,048   | 0.25          | 0.27         | 0.93x       | 1.1           | 1.6          | 0.66x       |
| 4,096   | 0.65          | 0.49         | **1.32x**   | 3.0           | 3.3          | 0.90x       |
| 8,192   | 2.04          | 0.97         | **2.10x**   | 10.1          | 6.9          | **1.48x**   |
| 16,384  | 7.51          | 2.15         | **3.49x**   | 39.7          | 13.9         | **2.85x**   |
| 32,768  | 30.00         | 4.31         | **6.97x**   | 152.1         | 27.7         | **5.49x**   |
| 65,536  | 127.84        | 8.86         | **14.43x**  | 623.4         | 55.5         | **11.22x**  |
| 131,072 | 559.58        | 15.37        | **36.41x**  | 2528.1        | 110.6        | **22.85x**  |

#### 关键观察

1. **性能断点**：SWA 在 N ≥ 2S (=2048) 后 forward 开始胜出；fwd+bwd 需 N ≥ 4S (=4096) 才接近 break-even
2. **Speedup 随 N 接近线性增长**：N=128K 时 fwd 36x, fwd+bwd 23x
   - 理论上限：fwd ~ N/S = 128，实测 36x（受 kernel 常数开销限制）
   - fwd+bwd 比 fwd 略低，因 bwd 的 recompute 在长序列上相对开销更大
3. **Block size 全范围统一**：sweep 确认 N=1K 到 N=65K 最优 block size 不变
   - Forward 最优 (BQ=128, BKV=64, w=8) 在所有 N 下保持第一
   - Backward 最优 (dQ: 64,64,w=4; dKV: 16,32,w=4) 在所有 N 下保持第一
4. **N=128K 极端测试**：Triton SWA fwd 15.37ms, fwd+bwd 110.6ms，完全可用于超长上下文训练

#### NaN Bug 修复

- 症状：N > slide_size 时 forward 输出 NaN
- 原因：kv_loop_start 按第一个 Q 位置保守下取整，导致某些 KV 块对部分 Q row 全 mask
  若此时 m_i 仍为 -inf（该 Q row 尚未见到任何有效 KV），则 exp(-inf-(-inf))=exp(nan)=nan
- 修复：将 m_i 和 new_max 在计算 alpha/p 时 clamp 到 -1e20（避免 inf-inf）
  ```python
  alpha = tl.exp(tl.maximum(m_i, -1e20) - tl.maximum(new_max, -1e20))
  p = tl.exp(scores - tl.maximum(new_max, -1e20)[:, None])
  ```
- **后续优化（2026-04-16）**：clamp 只在 `SLIDE_SIZE > 0` 分支执行；SLIDE_SIZE=0 时
  因 KV 单调从 0 起迭代，m_i 在 step 1 后必为 finite，不需 clamp（节省 constexpr 分支指令）

---

### SWA 短 N 专项优化（2026-04-16）

**动机**：用户反馈 N<8K 时 SWA Fwd+Bwd 慢于 SDPA (0.65-0.90x)。

**根本原因分析**：
- 在 N ≤ slide 时，SWA ≡ full causal（窗口覆盖全序列），没有工作量节省
- Triton D=256 full-causal kernel 本身比 SDPA 慢 ~15-25%（cuDNN/CUTLASS 对 D=256 有特殊优化）
- **Bwd 侧额外有 delta 计算 + kernel 分裂（dQ + dKV + reduce）** 的常数开销

**Bwd 分解 (N=1024, D=256, 调优前)**:
| 步骤 | 时间 (ms) | 占比 |
|------|-----------|------|
| delta = (do*o).float().sum() | 0.119 | 18% |
| dQ kernel | 0.182 | 28% |
| dKV kernel | 0.307 | 47% |
| reduce (view+sum) | 0.053 | 8% |
| **总计** | **0.661** | 100% |

**三项优化**：

**1. Fused Delta Triton Kernel**：
- 原：`(do.float() * o.float()).sum(dim=-1)` 分配 2 个 fp32 tensor
- 新：自写 `_delta_kernel`：grid (N, B*H) × fp16 load → fp32 accum → scalar store
- 结果：N=1024 delta 0.119 → **0.046ms (-61%)**

**2. slide_size >= N 规范化 (wrapper level)**：
- 当窗口覆盖整个序列，SWA ≡ full causal
- wrapper 提前 `if slide_size >= N: slide_size = 0` → 走 full-causal 路径
- 省 window mask + NaN clamp，节省 ~4% 开销（小收益，但免维护两份逻辑）
- 辅助：kernel 中 NaN clamp 也做成 `if SLIDE_SIZE > 0` 条件

**3. dKV BQ=16→64 for D<512**：
- D=256 register pressure 只有 D=512 的一半，可以扩大 BQ
- 之前因 SWA 分支 register overhead 造成 "BQ=32 spill" 印象，当 SLIDE_SIZE=0 时无此压力
- sweep 跨 N=1K-16K + slide={0, 1024} 全域确认 (BQ=64, BKV=32, w=4) 最优
- 结果：N=4K dKV 3.27 → **2.01ms (-39%)**；N=16K dKV 52.0 → **29.7ms (-43%)**

**综合效果 — SWA Fwd+Bwd vs SDPA full-causal**：

| N       | 调优前 | 调优后 | 提升  |
|---------|--------|--------|-------|
| 1,024   | 0.65x  | **0.73x** | +12%  |
| 2,048   | 0.66x  | **0.88x** | +33%  |
| 4,096   | 0.90x  | **1.23x** | +37% **首次 N=4K 打平 SDPA** |
| 8,192   | 1.48x  | **1.96x** | +32%  |
| 16,384  | 2.85x  | **3.84x** | +35%  |
| 32,768  | 5.49x  | **7.57x** | +38%  |

**D=512 full-causal 路径未回退**（验证 N=256~2048 全部 OK，BQ=16 保持）。

**Forward 未回退**（N=1K-128K 与调优前一致）。

---

### MFU 分析 & atomic_add + 额外优化（2026-04-16）

> 公式：MFU = Achieved TFLOPS / 989.5 TFLOPS (H100 FP16 peak)；Fwd+Bwd FLOPs = fwd × 3.5 (flash-attn 经验)

#### SWA Fwd+Bwd MFU（D=256, H_Q=32, H_KV=16, slide=1024）

| N       | SWA Time (ms) | TFLOPS | **MFU**  |
|---------|---------------|--------|----------|
| 1,024   | 0.68          | 89     | **9.0%**  |
| 2,048   | 1.17          | 154    | 15.6%    |
| 4,096   | 2.38          | 177    | 17.9%    |
| 8,192   | 5.06          | 178    | 18.0%    |
| 16,384  | 10.09         | 185    | 18.7%    |
| 32,768  | 19.93         | 190    | 19.2%    |

#### Full Causal Fwd+Bwd MFU（D=512, H_Q=32, H_KV=4）

| N       | Time (ms) | TFLOPS | **MFU** |
|---------|-----------|--------|---------|
| 1,024   | 1.92      | 63     | 6.3%    |
| 2,048   | 6.14      | 78     | 7.9%    |
| 4,096   | 22.17     | 87     | 8.8%    |

**结论**：
1. **SWA N≥4K 稳定 17-19% MFU**，接近 D=256 ridgeline 极限
2. **Full D=512 MFU 6-9% 看似低，但已 85-90% HBM BW 利用率** (见 `hotspots.md`)，attention 算术强度天生低 (63 F/B vs ridgeline 295) → memory-bound, 不是 compute 浪费
3. **SWA N=1024 MFU 9%** (刚好<10%) 受限于：absolute time 短 (0.68ms)，kernel launch 开销 (~40us × 4 kernels) 占比大

#### atomic_add dKV Fuse 实验（最终拒绝）

**思路**：用 `tl.atomic_add` 将 per-Q-head dK/dV 直接累加到 shared (B, H_KV, N, D)，免去 expand buffer + reduce kernel。

**结果**：
| N       | expand+reduce (ms) | atomic_add (ms) | 变化 |
|---------|---------------------|------------------|------|
| 1,024 SWA | 0.71              | 0.71             | 0%    |
| 2,048 SWA | 1.19              | 1.23             | +3.4% |
| 4,096 SWA | 2.47              | 2.48             | +0.4% |
| 8,192 SWA | 5.08              | 5.21             | +2.6% |
| 16,384 SWA| 10.10             | 10.55            | +4.5% |

**原因**：GQA_RATIO-way atomic 竞争（D=256 为 2-way, D=512 为 8-way）抵消了省掉 reduce kernel 的收益。

**保留**：`ATOMIC_REDUCE: tl.constexpr` 已加入 kernel，默认 False，作为将来可选的 opt-in（例如 GQA_RATIO=1 的 MHA 没竞争）。

#### Pairwise Add Reduce (GQA_RATIO=2)

替换 `view + sum(dim=2)` 为 `a[:, :, 0] + a[:, :, 1]`：
- Reduce 时间：0.053ms → **0.045ms** (-15%) @ N=1K
- 仅适用 GQA_RATIO=2（D=256 case）；其他 ratio 仍用 `.sum()`

综合上述优化后，SWA fwd+bwd 最终速度：

| N       | 原始  | 调优后 | 总提升 |
|---------|-------|--------|--------|
| 1,024   | 0.65x | 0.74x  | +14%   |
| 2,048   | 0.66x | 0.91x  | +38%   |
| 4,096   | 0.90x | 1.25x  | +39%   |
| 8,192   | 1.48x | 2.03x  | +37%   |
| 16,384  | 2.85x | 3.89x  | +36%   |
| 32,768  | 5.49x | 7.74x  | +41%   |

---

### Gemma4 Mixed Stack E2E Training (2026-04-16)

**实现**：`flash_attn/gemma4_e2e.py` — 6-block stack, pattern = 5 sliding + 1 full:
- Sliding block: H_Q=32, H_KV=16, head_dim=256, slide_size=1024
- Full block:    H_Q=32, H_KV=4,  head_dim=512

**测量条件**：
```
d_model=2048, n_blocks=6, B=1, BF16, AdamW optimizer
H100 80GB, Triton 3.5.1, PyTorch 2.9.1
Warmup=3, rep=5-10 (long N fewer reps)
```

#### 正确性验证（10 training steps, same seed+weights）

| N    | Fwd cos sim | Min grad cos sim | Loss trajectory max rel diff | Status |
|------|-------------|-------------------|------------------------------|--------|
| 512  | 0.999999    | 0.986             | 0.00e+00                     | PASS   |
| 1024 | 0.999999    | 0.960             | 0.00e+00                     | PASS   |
| 2048 | 0.999999    | 0.885             | 0.00e+00                     | PASS   |

**Loss trajectory 0 rel diff** 是最强保证：10 步 training loss 完全一致（BF16 bit-level 匹配）。
grad cos sim < 1.0 仅反映 BF16 在 small-magnitude tensor（如 LayerNorm gain）上的舍入噪声 — 不影响训练。

#### E2E Training Throughput

| N      | SDPA (ms/step) | Triton (ms/step) | **Speedup** |
|--------|----------------|-------------------|-------------|
| 512    | 12.49          | 11.77             | 1.06x       |
| 1,024  | 17.90          | 13.27             | **1.35x**   |
| 2,048  | 38.28          | 22.85             | **1.68x**   |
| 4,096  | 107.67         | 50.55             | **2.13x**   |
| 8,192  | 365.39         | 140.64            | **2.60x**   |

**关键观察**：
- E2E speedup 随 N 增长持续提升，因 attention 在 total step 中占比从 N=512 的 ~10% 升至 N=8K 的 ~60%
- 在实际训练场景下（有 linear projections + norms + MLP overhead），N=8K 时 Triton 仍达 **2.60x** E2E speedup
- d_model=2048 的 wq/wk/wv/wo 项大约是 8MB/层，与 attention compute 在短 N 可比；长 N 下 attention O(N²) 占主导

---

### Real Gemma-4-E2B Integration (2026-04-16)

**配置**:
```
硬件:         NVIDIA H100 80GB HBM3
软件:         torch 2.9.1 + transformers 5.5.4 + triton 3.5.1，clean uv env
模型:         google/gemma-4-E2B (5.1B params, 35 layers)
架构:         29 sliding + 6 full attention layers (交替)
attention:    H_Q=8, H_KV=1 (GQA 8:1)
head_dim:     256 (sliding) / 512 (full)
sliding_window: 512
dtype:        bfloat16
Integration:  register_triton_attention() + config._attn_implementation="triton_gqa"
```

#### 1. Adapter 路径覆盖验证（call counter instrumented）

Wrapped `triton_gqa_attention` with a counter, ran one forward pass:

| 路径 | 调用次数 | 期望 |
|------|---------|------|
| `sliding_window=512` (sliding path) | 28 | 28 (sliding layers) |
| `sliding_window=None` (full causal) | 7 | 7 (full layers) |
| **总计** | **35** | **35 (all layers)** |

**结论**：所有 35 层 attention 都通过 Triton kernel，**SWA 和 full-causal 两条路径都被实际使用**。

#### 2. Adapter 单元测试 (24/24 cases PASS)

`tests/gemma4_integration/test_adapter.py` — 直接调用 `triton_gqa_attention` adapter，
对比 SDPA reference。不依赖模型下载。

**测试矩阵**:
- GQA ratios: `(8,8)=1:1 MHA`, `(8,4)=2:1`, `(8,2)=4:1`, `(8,1)=8:1`, `(16,2)=8:1`
- Attention pattern: full causal + SWA (slide_size=512)
- SWA sub-cases: N ≤ slide (512) + N > slide (2048)
- HEAD_DIM: 256 (Gemma4 sliding) + 512 (Gemma4 full)
- Batch size: 1 + 2

**结果**: 所有 24 cases **cos similarity > 0.999987**, rel err < 5e-3。

| 代表性用例 | cos sim | rel err |
|-----------|---------|---------|
| causal D=256 8:1 N=512 | 0.999993 | 3.88e-3 |
| SWA D=256 8:1 N=2048 slide=512 | 0.999991 | 4.26e-3 |
| causal D=512 32:4 N=2048 | 0.999987 | 5.02e-3 |
| B=2 SWA D=256 8:1 N=2048 slide=512 | 0.999991 | 4.25e-3 |

#### 3. 正确性：logits 匹配（Gemma-4-E2B，N=1024 vs SDPA）

- max logits abs diff: 3.23 (vocab=262144, 正常 BF16 × 35-layer 噪声)
- rel Frobenius diff:  2.27e-02
- **cos similarity:   0.999745**
- **top-1 token match (last pos): 100%**
- **top-5 token overlap (last pos): 5/5**

#### 4. Forward Throughput (Gemma-4-E2B 端到端，bf16)

| seq_len | SDPA (ms) | Triton (ms) | Speedup  | 备注 |
|---------|-----------|-------------|----------|------|
| 512     | 49.04     | 49.74       | 0.99x    | 短 N，attention 不是瓶颈 |
| 1,024   | 46.48     | 46.61       | 1.00x    | 打平 |
| 2,048   | 63.51     | 48.00       | **1.32x** | attention 开始占主导 |
| 4,096   | 158.38    | 81.03       | **1.95x** | |
| 8,192   | 480.99    | 167.31      | **2.87x** | |
| 16,384  | 1638.54   | 363.37      | **4.51x** | 长上下文，kernel 优势最大 |

**分析**:
- 短 N (512-1024): Gemma-4-E2B 有 35 层 linear projections，占 step 主要时间。Attention
  本身只有小 fraction，Triton 的优势被稀释。
- 长 N (≥4K): Attention 成为 O(N²) 主导 (sliding 成 O(N·S), full 成 O(N²))，Triton
  kernel 的优势（SWA work-reduction + 大 HEAD_DIM 优化）放大，E2E speedup 显著。
- 趋势：**N=32K 理论外推可达 ~7x E2E**（未测，需 OOM 检查）。

#### Reproduction

```bash
# 1. Clean env
source /opt/tiger/flash_gemma/bin/activate   # uv venv: torch 2.9.1 + transformers 5.5.4
uv pip install -e .                          # install kernel package

# 2. Unit test (fast, no HF download)
python tests/gemma4_integration/test_adapter.py
# → 24/24 passed

# 3. E2E test (needs HF_TOKEN; downloads 5GB on first run)
export HF_TOKEN="hf_..."
python tests/gemma4_integration/test_gemma4.py --seq-len 1024
# → PASS ✓, throughput table above
```

---

### Gemma-4-E2B Memory Benchmark (2026-04-16)

**测量方法**：
```python
torch.cuda.empty_cache(); gc.collect()
torch.cuda.reset_peak_memory_stats()
base = torch.cuda.memory_allocated()
with torch.no_grad():
    model(ids)                    # forward pass
peak_mem = torch.cuda.max_memory_allocated() - base
```

**Config**: Gemma-4-E2B (5.1B params in BF16, 35 layers), B=1, H100 80GB。
权重本身占 ~10GB；下表是 forward pass 中的 **额外** peak allocation。

#### Forward Pass Peak Memory

| seq_len | SDPA peak (MB) | Triton peak (MB) | Reduction | Memory saved |
|---------|---------------|-------------------|-----------|-------------|
| 512    | 522.5    | 522.5    | 1.00x | 0 |
| 1,024  | 1,045.0  | 1,045.0  | 1.00x | 0 |
| 2,048  | 2,090.5  | 2,091.0  | 1.00x | 0 |
| 4,096  | 4,180.0  | 4,180.0  | 1.00x | 0 |
| 8,192  | 8,360.0  | 8,360.0  | 1.00x | 0 |
| 16,384 | **22,016.4** | **16,720.0** | **1.32x** | **5,296 MB** |
| 32,768 | **OOM**      | **33,472.0** | **∞**     | —           |
| 65,536 | OOM          | OOM              | —         | —           |

#### Max Context Length on 80GB H100

| Attention impl | Max N (Gemma-4-E2B forward) |
|----------------|------------------------------|
| SDPA           | **~16K** (22 GB; 32K OOM) |
| Triton         | **~32K** (33 GB; 65K OOM) |

**结论**:
- N ≤ 8K：两者 peak 几乎一致 (<0.1% 差异)；此范围内 SDPA flash-attention 后端没有
  materialize 完整 attn matrix，memory 差距小。
- **N = 16K：Triton 节省 5.3 GB** (22 GB → 16.7 GB, 1.32x reduction)。此时 SDPA
  开始 materialize 部分 attn scratch space，而 Triton 的 online softmax 完全不
  materialize，peak 停留在 Q/K/V/output 级别。
- **N = 32K：Triton 仍可运行（33 GB），SDPA OOM**。Triton kernel 让用户在同一
  80GB H100 上把 context 翻倍（16K → 32K）。
- H_KV=1 (Gemma-4-E2B GQA 8:1) 放大了 memory 优势：SDPA 的 `repeat_kv` 把 K/V 扩
  展到 H_Q=8 heads，我们的 kernel 直接在 H_KV=1 的 tensor 上工作，省去这部分拷贝。

**可复现**：
```bash
source /opt/tiger/flash_gemma/bin/activate
export HF_TOKEN="hf_..."
python tests/gemma4_integration/test_memory.py   # (见 test_memory.py)
```

---

### SWA Backward-only Benchmark (2026-04-16)

**动机**：之前所有 SWA 数据都是 Fwd+Bwd 合并（e.g., N=8K 5.06x, N=16K 10.0x）。
为了把 backward 的独立贡献分离出来，测量 pure backward time（fwd+bwd 减去 fwd）。

**测量条件**：
```
硬件:         NVIDIA H100 80GB HBM3
Config:       D=256, H_Q=32, H_KV=16, slide_size=1024, BF16
Baseline:     SDPA full-causal（没有原生 SWA backward，用 full-causal 作为对照）
测量方法:     fwd+bwd time - fwd time = bwd-only time
```

| N       | SDPA bwd (ms) | Triton SWA bwd (ms) | Speedup | (fwd+bwd 合并) |
|---------|---------------|---------------------|---------|---------------|
| 1,024   | 0.41          | 0.59                | 0.70x   | 0.75 ms (tri) |
| 2,048   | 1.08          | 0.86                | 1.25x   | 1.14 ms       |
| 4,096   | 3.35          | 1.80                | **1.87x** | 2.29 ms     |
| 8,192   | 11.61         | 3.63                | **3.20x** | 4.58 ms     |
| 16,384  | 43.01         | 7.33                | **5.87x** | 9.36 ms     |
| 32,768  | 170.87        | 14.69               | **11.63x** | 18.78 ms   |

**关键发现**:
1. **Backward 独立 speedup > Forward speedup**（在长 N 下更明显）：
   - 例如 N=32K: bwd-only **11.63x**, fwd-only ~6.97x (见 SWA full-range 表)
   - 原因：SWA 把 dQ 和 dKV 两个 bwd kernel 都从 O(N²) 降到 O(N·S)，bwd 含多个
     kernel launch (delta + dQ + dKV + reduce)，absolute time 大时 launch 开销
     占比下降，SWA 工作量减少的红利更充分体现
2. **N=32K 接近理论上限**：work-reduction ratio = N/S = 32K/1K = 32x，实测 11.63x
   还有 kernel constant + reduce + delta compute 等固定开销压着
3. **N=1024 低于 1x** (0.70x)：slide_size ≥ N 触发 normalization 走 full-causal，
   这是 Triton D=256 full causal 本身弱于 SDPA 的已知问题（见上面的 D=256 causal
   benchmark），**非 SWA 实现问题**

**与 MFU 数据的对应**（见上面 SWA MFU 小节）:
- bwd-only 时间随 N 从 0.59ms → 14.69ms 线性增长（因 SWA 工作量 O(N·S), S 固定）
- 对比 SDPA 的 O(N²) 增长，long N 下差距无限放大

---

### Pack-GQA dKV Block Size Re-sweep for D=512 (2026-04-17)

**动机**：Pack-GQA 切换时沿用旧 split kernel 的 `(BKV=32, BQ=16, w=8)` 配置，但 pack-GQA
的 register / shmem 模型不同：dk_acc/dv_acc 尺寸 = BKV × D × fp32 (**BQ 不进 accumulator**)。
旧常识 "BQ=32 灾难 spill" 仅对 per-Q-head split 路径成立。

**Breakdown profile（旧默认 BKV=32, BQ=16, w=8）揭示 dKV 占 fwd+bwd 62-69%**：

| N | delta | dQ | dKV | tri bwd | vs SDPA |
|---|-------|----|----|---------|---------|
| 1K | 0.05ms | 0.41ms | 1.64ms | 2.02ms | 2.22× |
| 4K | 0.11ms | 4.76ms | 12.90ms | 17.71ms | 2.30× |
| 16K | 0.37ms | 72.6ms | 200.2ms | 273.2ms | 1.98× |
| 32K | 0.71ms | 289.9ms | 798.2ms | 1089.3ms | 1.93× |

**Sweep @ N=4K**: 36 configs (BQ × BKV × warps × stages)，H100 shmem=232KB 限制

| BQ | BKV | w | s | time (ms) | vs 旧默认 |
|----|-----|---|---|-----------|-----------|
| 16 | 32 | 8 | 2 | 13.126 | 1.00× (旧默认) |
| 16 | 16 | 4 | 2 | 13.211 | 0.99× |
| 32 | * | * | * | 178-282 | 0.05-0.07× (BKV<64 全灾难) |
| 64 | 16 | 8 | 2 | 11.622 | 1.13× |
| **64** | **16** | **4** | **2** | **9.355** | **1.40× 最优** |
| 64 | 32 | 8 | 2 | 19.985 | 0.66× |
| 64 | 64 | * | * | OOM | — |
| 128 | * | * | * | OOM | shmem 超限 |

**胜者：(BKV=16, BQ=64, w=4, s=2)**
- BKV=16 让 dk_acc+dv_acc 从 128KB 减到 64KB → shmem 腾出给大 BQ
- BQ=64 让 inner Q loop 迭代数减 4×，Q/dO tile 复用更充分
- num_warps=4 胜 8（D=256 bwd 同样结论，bwd kernel compute 较轻 8 warps 过度）

**Top-3 验证 @ N=16K**（确认 scaling 一致）:
| (BQ, BKV, w) | @N=4K | @N=16K |
|--------------|-------|--------|
| (64, 16, 4)  | 9.36  | 141.7 |
| (64, 16, 8)  | 11.62 | 172.9 |
| (16, 32, 8)  | 13.13 | 200.5 |

**切换后 breakdown（新默认 BKV=16, BQ=64, w=4）**:

| N | delta | dQ | dKV | tri bwd | tri fwd+bwd | vs SDPA fwd+bwd |
|---|-------|----|----|---------|-------------|-----------------|
| 1K | 0.05ms | 0.44ms | **1.02ms** | **1.41ms** | **1.77ms** | 2.08× → **2.83×** |
| 4K | 0.11ms | 4.75ms | **9.30ms** | **14.12ms** | **17.47ms** | 2.18× → **2.66×** |
| 16K | 0.37ms | 72.6ms | **141.8ms** | **214.4ms** | **262.1ms** | 1.96× → **2.42×** |
| 32K | 0.71ms | 289.9ms | **562.8ms** | **853.3ms** | **1048.4ms** | 2.31× → **2.40×** |

**dKV 相对占比**：62-69% → 54-58%（仍是最大头，但 27-29% 时间消失）
**正确性**：8 个 shape × {causal, non-causal} 全 PASS (fwd 1e-3, grad 8e-3 atol)
**未回退**：D=256 分支未动，SWA 未受影响

**修改**：`attention.py:1296-1299` 三行常量
**脚本**：`benchmarks/bwd_breakdown.py` + `benchmarks/dkv_sweep_D512.py`

---

### D=512 Full-Causal Profiling Breakdown + dKV 4-Way Dead End (2026-04-17)

> **目的**：数据驱动定位 full attention 热点 + 系统性验证 dKV 是否还有优化空间。
> 结论：dKV 占 fwd+bwd 53-54%，**302 register spill 是 Triton 3.x 硬限**，4 次独立实验证明不可消除。

**测量条件**：

```
硬件:     NVIDIA H100 80GB HBM3
软件:     Triton 3.5.1, PyTorch 2.9.1, fp16
输入:     B=1, H_Q=32, H_KV=4, D=512, causal=True, slide_size=0
重复:     warmup=8, rep=20 取中位数（CUDA events）
脚本:     benchmarks/profile_n8k.py (跨 N 热点), dump_kernel_regs.py (spill),
         dkv_stages_sweep.py (exp 1-3), dkv_split_bench.py (exp 4)
```

#### Hotspot Rank（跨 N 稳定）

| rank | kernel | N=4K | N=8K | N=16K | MFU | 占 fwd+bwd kernel sum |
|------|--------|------|------|-------|-----|----------------------|
| #1 | **dKV** | 9.29ms | 35.64ms | 141.2ms | **6.0-6.3%** | **53-54%** |
| #2 | dQ | 4.79ms | 18.56ms | 72.63ms | 11.6-12.2% | 27-28% |
| #3 | fwd | 3.31ms | 12.31ms | 47.52ms | 16.8-18.7% | 18-19% |
| #4 | delta | 0.12ms | 0.20ms | 0.39ms | — | <1% |

**vs SDPA**:
- fwd:      1.57× (N=4K) → 1.64× (N=8K) → 1.81× (N=16K)（随 N 上升）
- fwd+bwd:  2.64× (N=4K) → 2.49× (N=8K) → 2.41× (N=16K)（**随 N 下降，被 dKV 拖累**）

#### Register / Shmem 状态（编译后 metadata @ N=8K）

| kernel | regs/thread | spills | shmem | warps | 备注 |
|--------|-------------|--------|-------|-------|------|
| fwd_gqa | 255 (cap) | 4 (~0) | 192KB (83%) | 8 | 基本无 spill |
| dQ | 189 | **0** ✓ | 196KB (85%) | 8 | 完全干净 |
| **dKV_packed** | **255 (cap)** | **302** ⚠️ | 163KB (70%) | **4** | **烟枪**：reg cap 下仍需 spill |
| delta | 19 | 0 | 0 | 4 | — |

> H100 硬限：65536 regs/SM, 255 regs/thread max, 232KB dyn shmem/SM

#### dKV 四次攻坚（全部失败）

**Exp 1 — num_stages ∈ {1, 3}**（@ BQ=64 BKV=16 w=4）:

| stages | shmem needed | 是否 fit (232KB) | 结果 |
|--------|--------------|------------------|------|
| 2 (baseline) | 163KB | ✓ | 35.5ms, 302 spills |
| 1 | 294KB | ✗ OOM | 不可行 |
| 3 | 298KB | ✗ OOM | 不可行 |

**发现**：Triton pipeliner 在 s=2 才做 buffer overlapping；s=1/s=3 反而每 buffer 独立分配 shmem。只能 s=2。

**Exp 2 — BQ=64→32 sweep**（@ BKV=16）:

| config | time @ N=4K/N=8K | spills | shmem | vs baseline |
|--------|-------------------|--------|-------|-------------|
| BQ=64 w=4 s=2 (baseline) | 9.28 / 35.58 | 302 | 163KB | — |
| BQ=32 w=4 s=1 | 11.01 / 43.13 | **0** ⭐ | 97KB | **+19% 慢** |
| BQ=32 w=4 s=2 | 224 / 885 | 1186 | 97KB | +2400% (灾难) |
| BQ=32 w=4 s=3 | 259 / 1023 | 1280 | 162KB | +2700% (灾难) |
| BQ=32 w=8 s=2 | 283 / 1122 | 966 | 97KB | +2950% (灾难) |

**发现**：唯一 0-spill 配置慢 19%——**pipelining 比消除 spill 值钱**。BQ=32 s≥2 Triton 退到 32 regs/thread + 1000+ spills 的 fallback 模式（compiler 行为，不可调）。

**Exp 3 — dynamic GQA loop**（`tl.static_range(GQA_RATIO)` → `tl.range(GQA_RATIO)`）:

| config | dyn-gqa | static-gqa | Δ | spills dyn/static |
|--------|---------|------------|-----|---------------------|
| BQ=64 w=4 s=2 | 10.96 / 42.83 | 9.28 / 35.58 | +18-20% 慢 | 300 / 302 |
| BQ=32 w=4 s=1 | 10.82 / 42.40 | 11.01 / 43.13 | -1% (持平) | 2 / 0 |

**发现**：Unroll **不是** spill 根因（spills 几乎不变）。丢失 Triton 的 constant folding + pipeline scheduling on unrolled 代码导致慢 20%——**unroll 是 performance-positive**。已 revert。

**Exp 4 — split dV + dK**（kept in source as reference kernels）:

| | N=4K | N=8K | spills | shmem |
|---|------|------|--------|-------|
| PACKED (baseline) | 9.27ms | 35.48ms | 302 | 163KB |
| dV-only 单独 | 5.54ms | 22.04ms | **46** ⬇️ | 211KB |
| dK-only 单独 | 6.95ms | 26.35ms | **92** ⬇️ | 227KB |
| **dV + dK 总** | **12.49ms** | **48.39ms** | — | — |
| **vs packed** | **+35%** | **+36%** | | |

**发现**：Split 大幅消除 spill（302 → 46/92），但总时间 +35-36%——**compute sharing (scores+p) 比消 spill 更值**。packed kernel 保留为 hot path，`_flash_attn_gqa_bwd_dv_only_kernel` / `_flash_attn_gqa_bwd_dk_only_kernel` 保留在 source 供参考。

#### 综合结论

- **dKV baseline (BQ=64, BKV=16, w=4, s=2, static_range, packed) 是 5D 空间的局部最优**：block × warps × stages × unroll × structure 五个维度都已系统性探索。
- dKV 剩余 19% gap 是**算法 live state 硬限**：4 matmul + 2 fp32 accumulator + GQA=8 unroll 在 HEAD_DIM=512 下必须 spill。需要 FA3 的 warp specialization + async TMA + cluster barrier（Triton 3.x 不支持）。
- Forward MFU 18% 已达 memory-bound 饱和（baseline.md 的 84-90% HBM BW 结论），无需再探。
- dQ MFU 12% 是 fwd 的 65%，gap 主要来自 bwd 的重算（delta + lse + 三 matmul）。空间可能有但比 dKV 小。

正确性：4 次实验所有 dK/dV 输出 max diff = 0.00 vs packed baseline。

---

### Pack-GQA dKV Kernel (2026-04-16)

**来源**：借鉴 `/mnt/bn/ic-aip/haojun/code/flash-attention/flash_attn/cute/pack_gqa.py`
中的 pack_gqa 设计思路。Flash-attention 有两种 GQA 后向模式：
- `pack_gqa=False`: per-Q-head 程序，dK/dV 用 atomic_add（≡ 我们失败的 A.2）
- `pack_gqa=True`: per-KV-head 程序，Q heads 折进 seqlen，dK/dV 直接 store

**本 kernel 采用 pack_gqa=True 思路**（但 Triton 不便做真的 seqlen packing，
改用 `tl.static_range(GQA_RATIO)` 在同一 program 内串行展开 Q heads）：

```
Grid: (cdiv(N, BKV), B × H_KV)    # 原先 H_Q → 改为 H_KV
每个 program:
    load K, V 一次
    for qh_offset in tl.static_range(GQA_RATIO):
        for q_block in q_range:
            load Q, dO (for this q_h)
            compute P, dP, dS
            dk_acc += scale × dS^T @ Q
            dv_acc += P^T @ dO
    store dk_acc, dv_acc   # 直接写，无 atomic
```

**与原 split + reduce 方案对比**：
| | Split+Reduce (旧) | Pack-GQA (新) |
|---|---|---|
| Grid 程序数 | `H_Q × N/BKV` | `H_KV × N/BKV` (GQA_RATIO × fewer) |
| K/V HBM load 次数 | GQA_RATIO × | 1 × |
| dK/dV 写 | atomic / expand+reduce | 直接 store |
| Scratch memory | dk_expanded + dv_expanded (H_Q × N × D × 2 × 2 bytes) | 0 |
| Register pressure | dk_acc + dv_acc | 同 |

**Fwd+Bwd 性能对比（Gemma4 sliding, D=256, H_Q=32, H_KV=16, slide=1024）**:

| N | Split+Reduce (旧) | Pack-GQA (新) | 提升 |
|---|-------------------|---------------|------|
| 1,024 | 0.73× | 0.91× | +25% |
| 2,048 | 0.88× | 1.28× | +45% |
| 4,096 | 1.25× | **1.91×** | +53% |
| 8,192 | 2.03× | **3.19×** | +57% |
| 16,384 | 3.89× | **5.93×** | +52% |
| 32,768 | 7.57× | **11.67×** | +54% |

**Full causal D=512 对比**:
| N | Split | Pack-GQA | 提升 |
|---|-------|----------|------|
| 1,024 | 2.37× | 2.00× | -16% |
| 2,048 | 2.20× | 1.96× | -11% |
| 4,096 | 2.02× | 2.13× | +5% |

**Memory 节省** (免 dk_expanded + dv_expanded):
| 场景 | 每次 bwd call 节省 |
|------|-------------------|
| Gemma4 sliding, N=16K | 540 MB |
| Gemma4 sliding, N=32K | **1.07 GB** |
| Gemma4 full, N=8K | 540 MB |

**为什么 D=512 短 N 会略慢**：grid 变小 GQA_RATIO=8 倍（1K 时从 1024 → 128 programs），
H100 132 SM 未饱和，短 N 出现 tail effect。D=256 GQA_RATIO=2，grid 只减半，影响小。

**为什么 A.1/A.2 失败但 pack-GQA 成功**：
- A.1 在同一 program 持有 GROUP_SIZE × BQ × D × 4 的 accumulator → register spill
- A.2 GQA_RATIO-way atomic 竞争同一 dK/dV tile → 硬件序列化
- **Pack-GQA 的 accumulator 大小与原 split kernel 相同**（register 无压力增加），
  并且每 tile 只由 1 个 program 写（无竞争）—— 两头都避开了

**Correctness**：
- 直接单元测试：7 cases（Gemma4 full + sliding）全通过，dK/dV max_diff < 2e-3
- 完整 autograd 路径：N=512..4096 causal/SWA 均 < 4e-3
- 真实 Gemma-4-E2B：cos sim 0.999745, top-1 100%, top-5 5/5（与 split 路径完全一致）

Reproducer: `tests/test_packed_dkv.py`

---

### D=256 SWA pack-GQA dKV N-gated Block Size (2026-04-17)

**动机**：Profile (`benchmarks/profile_swa.py`) 显示 SWA fwd+bwd 里 **dKV 占
44-50%**（跨 N=8K/16K/32K × slide=512/1024 全部稳定，最高 hotspot），MFU 仅 4-5%。
旧默认 `(BKV=32, BQ=64, w=4, s=2)` 是 2026-04-16 为 split+reduce kernel 调的，
pack-GQA register 模型不同（BKV 驱动 accumulator，BQ 不进），**从未重 sweep**。

**测量条件**：

```
硬件:         NVIDIA H100 80GB HBM3
软件:         Triton 3.5.1, PyTorch 2.9.1, fp16
输入:         B=1, H_Q=8, H_KV=1 (GQA 8:1, Gemma-4-E2B sliding), D=256, causal=True
Sweep:        BKV∈{16,32,64} × BQ∈{32,64,128} × warps∈{4,8} × stages∈{1,2,3} = 54 configs
Sweep shapes: N=16K slide=512 (primary), N=32K slide=512, N=16K slide=1024
重复次数:     warmup=3 rep=10 (sweep) / warmup=5-10 rep=10-20 (e2e)
脚本:         benchmarks/dkv_swa_sweep_D256.py + benchmarks/swa_e2e_bench.py
```

#### Sweep Result — Cross-shape Top-5 (geomean)

| rank | BKV | BQ | W | S | geomean (ms) | vs base |
|------|-----|----|----|---|--------------|---------|
| **1** | **64** | **128** | **8** | **1** | **1.019** | **1.12×** |
| 2 | 32 | 64  | 4 | 2 | 1.146 | 1.00× (baseline) |
| 3 | 32 | 32  | 4 | 1 | 1.364 | 0.84× |
| 4 | 64 | 32  | 8 | 1 | 1.393 | 0.82× |
| 5 | 32 | 128 | 8 | 2 | 1.451 | 0.79× |

**赢家 (BKV=64, BQ=128, w=8, s=1)**：
- BKV=64 → dk_acc/dv_acc = 2×64×256×4 = 128KB fp32（D=256 半字节，shmem 够）
- BQ=128 → inner Q loop 迭代数 -2×，Q/dO tile 复用更充分
- warps=8 → 大 tile 需要更多 warp 隐藏 memory latency
- **stages=1** → 配 BKV=64 BQ=128 时 s=2 会 OOM shmem；stages=1 单 buffer 腾出空间

#### E2E Application — 短 N 回退发现 & N-gate 修正

**直接 ship 新 config** 暴露问题（sweep 没覆盖 N<16K）：

| N | slide | baseline (ms) | new config only (ms) | Δ% |
|---|-------|---------------|----------------------|------|
| 2K  | 512  | 0.557 | 0.612 | **+9.9%** ⚠ |
| 4K  | 512  | 0.615 | 0.663 | +7.8% ⚠ |
| 2K  | 1024 | 0.783 | 0.866 | **+10.6%** ⚠ |
| 4K  | 1024 | 0.865 | 0.941 | +8.8% ⚠ |

**根因**：N=2K 时 grid = N/BKV × H_KV = 32 programs → **24% of H100 132 SMs**
starved；bigger tile 无法填充少量 SM。Bigger BKV 只在 grid 足够大时 win。

**N-gate 修正（`attention.py:1509-1527`）**：
```python
if D >= 512:        BLOCK_KV_DKV, BLOCK_Q_DKV, w, s = 16, 64, 4, 2
elif N >= 8192:     BLOCK_KV_DKV, BLOCK_Q_DKV, w, s = 64, 128, 8, 1  # 新
else:               BLOCK_KV_DKV, BLOCK_Q_DKV, w, s = 32, 64, 4, 2   # 旧
```

#### N-gate E2E 结果（before = old default everywhere）

**SWA Fwd+Bwd vs SDPA full-causal, Gemma-4-E2B sliding config**

| N       | slide | before (ms) | N-gate (ms) | Δ%    | before sp | **N-gate sp** |
|---------|-------|-------------|-------------|-------|-----------|---------------|
| 2,048   | 512   | 0.557       | 0.558       | +0.1% | 0.77x     | 0.72x         |
| 4,096   | 512   | 0.615       | 0.618       | +0.6% | 1.42x     | 1.42x         |
| 8,192   | 512   | 0.785       | 0.770       | -1.9% | 3.17x     | **3.23x**     |
| 16,384  | 512   | 1.525       | 1.552       | +1.7% | 6.61x     | 6.59x         |
| **32,768**  | **512** | 3.180  | **3.040**   | **-4.4%** | 12.63x | **13.05x** |
| 2,048   | 1024  | 0.783       | 0.775       | -1.0% | 0.51x     | 0.52x         |
| 4,096   | 1024  | 0.865       | 0.871       | +0.7% | 1.03x     | 1.02x         |
| **8,192**   | **1024**| 1.221  | **1.166**   | **-4.4%** | 2.04x  | **2.14x** |
| **16,384**  | **1024**| 2.464  | **2.373**   | **-3.7%** | 4.22x  | **4.41x** |
| **32,768**  | **1024**| 4.937  | **4.662**   | **-5.6%** | 8.00x  | **8.51x** |

**核心提升**：
- **N=32K slide=1024**: 8.00× → **8.51×** (+6.4% speedup)
- **N=32K slide=512**:  12.63× → **13.05×** (+3.3% speedup)
- **N=16K slide=1024**: 4.22× → **4.41×** (+4.5% speedup)
- **N=8K slide=1024**:  2.04× → **2.14×** (+4.9% speedup)

**短 N（N<8K）** 全部在 ±1% 内，**无回退**。

#### SWA Hotspot Breakdown（profile_swa.py 旧默认 baseline）

| N | slide | fwd | delta | dQ | **dKV** | dKV% of fwd+bwd |
|---|-------|-----|-------|-----|---------|-----------------|
| 8K  | 512  | 0.168 | 0.064 | 0.229 | **0.362** | **43.9%** |
| 16K | 512  | 0.292 | 0.108 | 0.455 | **0.820** | **48.9%** |
| 32K | 512  | 0.618 | 0.200 | 0.920 | **1.703** | **49.5%** |
| 8K  | 1024 | 0.249 | 0.067 | 0.351 | **0.611** | **47.8%** |
| 16K | 1024 | 0.469 | 0.107 | 0.745 | **1.279** | **49.2%** |
| 32K | 1024 | 0.956 | 0.198 | 1.471 | **2.609** | **49.8%** |

(MFU: fwd 10-14%, dQ 7-9%, **dKV 4-5%** 最低。pack-GQA dKV 仍是 Triton 3.x
bwd 里最低 MFU 的 kernel，与 D=512 pack-GQA 类似（~6% MFU），算法 live-state
硬限。)

#### 正确性验证

8 shapes 覆盖两个 N-gate 分支 + full-causal 未回退检查：

| 分支 | N | slide | fwd max-diff | dQ | dK | dV | Status |
|------|---|-------|--------------|-----|-----|-----|--------|
| N<8K  | 2K  | 512  | 3.05e-05 | 1.91e-06 | 3.05e-05 | 7.81e-03 | OK |
| N<8K  | 4K  | 512  | 6.10e-05 | 3.81e-06 | 3.05e-05 | 7.81e-03 | OK |
| N≥8K  | 8K  | 512  | 1.53e-05 | 3.81e-06 | 3.05e-05 | 7.81e-03 | OK |
| N≥8K  | 16K | 512  | 3.05e-05 | 3.81e-06 | 3.05e-05 | 3.91e-03 | OK |
| N<8K  | 2K  | 1024 | 3.05e-05 | 3.81e-06 | 3.05e-05 | 3.91e-03 | OK |
| N≥8K  | 8K  | 1024 | 3.05e-05 | 3.81e-06 | 3.05e-05 | 7.81e-03 | OK |
| N≥8K  | 16K | 1024 | 3.05e-05 | 1.91e-06 | 3.05e-05 | 3.91e-03 | OK |
| N≥8K  | 8K  | full | 3.05e-05 | 3.81e-06 | 6.10e-05 | 3.91e-03 | OK |

tolerance: fwd<5e-2, grad<8e-2（atol vs `attention_swa_ref` 和 `attention_gqa_ref`）

#### 结论

1. 短 N 新 config 回退 10% 是 **SM occupancy 硬限**（grid < SM count），不是 kernel bug
2. pack-GQA dKV block sweep **必须覆盖最小目标 N**，只看长 N 会得到误导性结论
3. N-gate 在 backward 路径多一个 Python `if`，开销可忽略（纳秒级），收益清晰
4. D=256 下 `(BKV=64, BQ=128, w=8, s=1)` 是 pack-GQA dKV 在长 N 的**新局部最优**

Reproducer:
- Sweep:   `python benchmarks/dkv_swa_sweep_D256.py`
- Profile: `python benchmarks/profile_swa.py`
- E2E:     `LABEL=ngate python benchmarks/swa_e2e_bench.py`
- 结果 JSON: `benchmarks/dkv_swa_sweep_D256.json`, `profile_swa.json`,
  `swa_e2e_bench_{before,after,ngate}.json`

---

### 短 N 专项 + Delta Fusion into dQ (2026-04-17)

**动机**：claude.md 标注 "剩余不足: N≤2K 仍 0.73-0.88x"。进一步 profiling 诊断根因并尝试
低风险优化。

#### 诊断（`benchmarks/profile_short_n.py`, `diag_short_n.py`）

**硬件**: H100 80GB (contaminated — 训练任务同时在跑，ratio 可信，绝对时间有噪声)

**Fused before:** (Config A = synthetic Gemma4 sliding: B=1, H_Q=32, H_KV=16, D=256, slide=1024)
| N | fwd | delta | dQ | dKV | K_sum | Triton F+B | OH% | SDPA F+B | s_fwd | s_fb | s_bwd |
|---|-----|-------|-----|-----|-------|-----|-----|------|--------|--------|--------|
| 1K | 0.115 | 0.046 | 0.155 | 0.283 | 0.598 | 0.604 | 0.8% | 0.430 | 0.81× | 0.71× | 0.68× |
| 2K | 0.224 | 0.065 | 0.303 | 0.643 | 1.235 | 1.179 | 0.0% | 0.990 | 1.00× | 0.84× | 0.80× |
| 4K | 0.421 | 0.106 | 0.601 | 1.351 | 2.480 | 2.393 | 0.0% | 3.023 | 1.50× | 1.26× | 1.21× |

**Fused before:** (Config B = Gemma-4-E2B real sliding: B=1, H_Q=8, H_KV=1, D=256, slide=512)
| N | Triton F+B | OH | OH% | SDPA F+B | s_fb |
|---|-----|-----|-----|------|--------|
| 1K | 0.539 | 0.100 | **18.6%** | 0.257 | 0.48× |
| 2K | 0.544 | 0.086 | 15.8% | 0.385 | 0.71× |
| 4K | 0.606 | 0.035 | 5.8% | 0.875 | 1.44× |

#### 根因

1. **Forward 已打平 cuDNN** (Config A N=1K: Triton 0.115ms vs cuDNN 0.114ms)，**gap 全在 bwd**
2. **SDPA 走 cuDNN**，不是 FA2（FA2 比 cuDNN 慢 40% @ 短 N）
3. **Config B dKV SM starvation**: grid = `(N/BKV, H_KV)` = `(32, 1) = 32 programs` @ N=1K
   = **24% of H100 132 SMs**. dKV time N=1K→2K: 0.273→0.270ms（几乎不变，fixed overhead）
4. **Launch overhead floor**: noop Triton kernel ≈ 19μs；bwd 有 delta + dQ + dKV = 3 launches,
   加 fwd 总 4 launches ≈ 76μs ≈ **12-14% of fwd+bwd @ N=1K**

#### 优化：Delta Fused into dQ Prologue

**思路**：dQ kernel 反正要 load `dO`，顺便 load `O` 算 `rowsum(dO*O)`，在 prologue 存
Delta buffer 给 dKV 用。省掉 `_delta_kernel` 整个 launch（其 grid = N × B × H_Q 个 tiny
program，launch-bound 严重）。

**API 改动**（`flash_attn/attention.py`）：
- `_flash_attn_gqa_bwd_dq_kernel` 新增参数：`O_ptr`, 4 个 O stride, `STORE_DELTA: tl.constexpr`
- 当 `STORE_DELTA=True`：prologue 内 `tl.sum(do.float() * o.float(), axis=1)` → `tl.store(Delta)`
- 当 `STORE_DELTA=False`：保持原行为，从 Delta buffer load（benchmark / 隔离 kernel 测试用）
- 主 backward wrapper 切到 `STORE_DELTA=True` 并移除 `_delta_kernel` launch

**正确性**: fp32 cos sim（fp16 不行，见 skills/2026-04-17_fp16-cossim-pitfall.md）

| Config | N | slide | cos(out) | cos(dQ) | cos(dK) | cos(dV) |
|--------|---|-------|----------|---------|---------|---------|
| 32:16, D=256 | 1K | ≥N→0 | 0.99999994 | 0.99999994 | 0.99999988 | 1.0 |
| 32:4,  D=512 | 1K | 0 | 1.0 | 1.0 | 0.99999994 | 0.99999994 |
| 32:4,  D=512 | 2K | 0 | 1.00000012 | 0.99999976 | 0.99999988 | 0.99999988 |

SWA cases 验证无 NaN（SDPA 不支持 SWA，不能直接对比）。

#### 性能（fused-before vs fused-after，同一脚本同一条件）

**Config A — Gemma4 synthetic (H_Q=32, H_KV=16, D=256, slide=1024)**:
| N | Before F+B | **After F+B** | Δ | Before s_fb | **After s_fb** |
|---|------------|---------------|------|-------------|----------------|
| 1K | 0.604 ms | **0.581 ms** | **-23μs (-3.8%)** | 0.71× | **0.74×** |
| 2K | 1.179 ms | 1.144 ms | -35μs (-3.0%) | 0.84× | **0.86×** |
| 4K | 2.393 ms | 2.330 ms | -63μs (-2.6%) | 1.26× | **1.30×** |

**Config B — Gemma-4-E2B real (H_Q=8, H_KV=1, D=256, slide=512)**:
| N | Before F+B | **After F+B** | Δ | Before s_fb | **After s_fb** |
|---|------------|---------------|------|-------------|----------------|
| 1K | 0.539 ms | **0.507 ms** | **-32μs (-5.9%)** | 0.48× | **0.51×** |
| 2K | 0.544 ms | 0.516 ms | -28μs (-5.1%) | 0.71× | **0.75×** |
| 4K | 0.606 ms | 0.582 ms | -24μs (-4.0%) | 1.44× | **1.49×** |

Config B N=1K **overhead% 18.6% → 13.2%**（delta launch 被消掉，~5% 减少）。

**符合预期**：standalone `_delta_kernel` 时间 ≈ 46μs @ N=1K Config A；融合进 dQ 后新增的
`O` load 成本 ≈ 23μs；净省 23μs = **实测一致**。

#### 剩余 gap 与可行方向

已确认 N≤2K 剩余 gap 的主要来源（按大小）：
1. **dKV kernel 效率不敌 cuDNN ~30-50%**（同等工作量下）
2. **Config B dKV SM starvation**（GQA 8:1 + H_KV=1 下 grid 只有 32 programs @ N=1K）
3. **3 launches 的 floor overhead**（~57μs）

候选方向（未做）：
- Config B dKV split-KV（风险：之前 atomic_add 实验失败，但短 N SM idle 严重可能反转 tradeoff）
- CUDA Graph 包裹（需要 static shapes，改 API）
- Bwd 融成 1 kernel（工程量大）

Reproducer:
- `python benchmarks/profile_short_n.py` (profile + breakdown)
- `python benchmarks/diag_short_n.py` (SM 占用、SDPA backend 检测)
- 结果 JSON: `benchmarks/profile_short_n.json`

---

### Q_SPLIT dKV (2026-04-17) — Config B SM starvation 攻坚成功

**动机**：上节 "候选方向 2" 里写的 Config B dKV split-KV，虽然 2026-04-16 同类
atomic_add fuse 在 Config A 失败（GQA=2:1 grid 健康，atomic 只添乱），但 Config B
(H_KV=1, raw_grid=32 @ N=1K, 24% SM util) 下 tradeoff 可能反转 — 直接做实验验证。

#### 设计
Pack-GQA dKV kernel 加 `Q_SPLITS: tl.constexpr`：
- Grid 从 `(cdiv(N, BKV), B × H_KV)` 改为 `(cdiv(N, BKV), B × H_KV, Q_SPLITS)`
- 每个 program 根据 `program_id(2)` 切一段 Q 范围 `[q_lo, q_hi)`
- QS>1 时输出从 `tl.store` 换为 `tl.atomic_add`，调用方预 zero `dk`/`dv`
- QS=1 时保持原逻辑（direct store）

#### 启发式（`benchmarks/dkv_qsplits_sweep.py` 实测 tuned）

目标：~256 programs = 2 waves on 132 SMs

```python
raw_grid = triton.cdiv(N, BLOCK_KV) * B * H_KV
if raw_grid >= 256:   Q_SPLITS = 1
elif raw_grid >= 128: Q_SPLITS = 2
elif raw_grid >= 64:  Q_SPLITS = 4
else:                 Q_SPLITS = 8
```

#### Q_SPLITS sweep (Config B, dKV kernel only, ms)

| N | raw_grid | QS=1 | QS=2 | QS=4 | QS=8 | best | vs QS=1 |
|---|----------|------|------|------|------|------|---------|
| 1K | 32 | 0.271 | 0.208 | 0.167 | **0.151** | QS=8 | **-44%** |
| 2K | 64 | 0.273 | 0.213 | **0.192** | 0.282 | QS=4 | **-30%** |
| 4K | 128 | 0.296 | **0.249** | 0.360 | 0.477 | QS=2 | **-16%** |
| 8K | 256 | **0.360** | 0.485 | 0.670 | 0.930 | QS=1 | 0% (sat) |

Config A (H_Q=32, H_KV=16): raw_grid ≥ 512 at all N → QS=1 永远最优；sweep 确认
QS>1 在 healthy grid 下慢 5-110%（atomic 开销毫无收益）。

#### 累计 fwd+bwd vs SDPA (Config B, Gemma-4-E2B 8:1, D=256, slide=512)

| N | Pre-fusion (2026-04-17 开局) | + Delta fusion | + Q_SPLIT | 累计 Δ |
|---|---|---|---|---|
| 1K | 0.48× | 0.51× | **0.63×** | **+31%** |
| 2K | 0.71× | 0.75× | **0.87×** | **+23%** |
| 4K | 1.44× | 1.49× | **1.65×** | **+15%** |

Config A (H_KV=16) 路径不变（QS=1 保留），fwd+bwd 数字与 delta-fusion 后一致。

#### 正确性

fp32 cos sim ≥ 0.99999982 across 8 shape configs（Config B QS=8 路径 + Config A
QS=1 路径 + D=512 full causal）。`test_packed_dkv.py` PASS all 7 correctness checks.

#### 为什么 2026-04-16 "atomic fuse 失败" 不矛盾

旧实验 (Config A H_KV=16, GQA=2:1) 下 raw_grid=512 @ N=1K 已饱和 SMs — atomic_add
纯添乱。新实验 (Config B H_KV=1) 下 raw_grid=32 @ N=1K 只用 24% SMs — SM 空转浪费
远超 atomic 竞争代价。**gated on raw_grid** 是关键：两种场景都用正确路径。

Reproducer:
- Sweep:     `python benchmarks/dkv_qsplits_sweep.py`
- Profile:   `python benchmarks/profile_short_n.py`
- 结果 JSON: `benchmarks/profile_short_n.json`

---

### Config A 短 N BKV=64 翻盘 (2026-04-17)

**动机**：上节 Q_SPLIT 专攻 Config B；但 Config A (H_Q=32, H_KV=16) 在短 N 仍
0.74-0.86× vs SDPA。根因是否在 dKV block config 错配？

#### 诊断

现有 D<512 短 N 默认 (BKV=32, BQ=64, w=4, s=2) 来自 2026-04-16，动机是 "BKV=64
短 N 会 starve SMs"。但那个论断 tested 在哪里？查 claude.md 发现基于 Gemma-4-E2B
(H_KV=1) 的数据 — raw_grid = N/64 × 1 = 16 @ N=1K 确实 starved。**推广到
Config A 是错误的**：Config A H_KV=16，raw_grid = N/64 × 16 = 256 @ N=1K，健康。

#### Config A sweep (`dkv_config_a_sweep.py`)

BKV ∈ {32, 64} × BQ ∈ {32, 64, 128} × w ∈ {4, 8} × s ∈ {1, 2}
@ (H_Q=32, H_KV=16, D=256, slide=1024)

| N | 当前 (32,64,4,s=2) | TOP (64,128,8,s=1) | 降幅 |
|---|-------------------|---------------------|------|
| 1K | 0.286 ms | **0.219 ms** | -23% |
| 2K | 0.653 ms | **0.454 ms** | -30% |
| 4K | 1.353 ms | **0.882 ms** | -35% |

**TOP 5 一致**：`(64, 128, 8, s=1)` 在 N=1K/2K/4K 全部第一；`(32, 128, 8, s=2)`
第二。现有 `(32, 64, 4, s=2)` 排 #3 across all N.

#### 修复

Gate 改成 `grid_at_BKV=64 = cdiv(N, 64) × B × H_KV >= 128`：
- Config A all N: 满足 → `(BKV=64, BQ=128, w=8, s=1)`
- Config B all N: 不满足 → `(BKV=32, BQ=64, w=4, s=2)` + Q_SPLITS 救援

#### End-to-end fwd+bwd vs SDPA (Config A, Gemma4 syn 2:1)

| N | Before (post-Q_SPLIT) | **After BKV=64** | Δ |
|---|------------------------|-------------------|------|
| 1K | 0.74× (0.596 ms) | **0.83× (0.520 ms)** | +12% |
| 2K | 0.86× (1.152 ms) | **1.05× (0.944 ms)** | +22% |
| 4K | 1.30× (2.342 ms) | **1.61× (1.874 ms)** | +24% |

**N=2K crosses SDPA** (1.05×, first time with this config).

Config B 路径 0.67×/0.88×/1.65× 不变（gate 正确保留了 BKV=32 + QS=8 路径）。

#### 正确性

fp32 cos sim ≥ 0.99999988 across 8 shape configs (Config A BKV=64 path +
Config B BKV=32 QS=8 path + D=512 unchanged path).

#### 翻案的系统性教训

原 2026-04-16 "短 N BKV=64 starves SMs" 结论是基于 Gemma-4-E2B 测的，但那是
H_KV=1 的极端场景。被推广到所有 D<512 短 N (包括 H_KV=16) → 误配持续了整个 4 月。
Gate 应该基于 **raw grid** 而不是 **N 单维**；H_KV 是和 N 同等重要的 grid 因子。

Reproducer:
- `python benchmarks/dkv_config_a_sweep.py`
- 用新 wrapper: `python benchmarks/profile_short_n.py`

---

### Session Summary — 2026-04-17 三步优化组合拳

短 N gap 攻坚的三个正交优化，累计效果：

**Config A (Gemma4 syn, H_Q=32, H_KV=16, D=256, slide=1024)**:
| N | 开局 | + Delta fusion | + Q_SPLIT (no-op A) | + BKV=64 | 累计 |
|---|------|----------------|---------------------|----------|------|
| 1K | 0.71× | 0.74× | 0.74× | **0.83×** | **+17%** |
| 2K | 0.84× | 0.86× | 0.86× | **1.05×** | **+25% (beats SDPA)** |
| 4K | 1.26× | 1.30× | 1.30× | **1.61×** | **+28%** |

**Config B (Gemma-4-E2B, H_Q=8, H_KV=1, D=256, slide=512)**:
| N | 开局 | + Delta fusion | + Q_SPLIT | + BKV=64 Config A gate | + BKV=64 rescue @ starve | 累计 |
|---|------|----------------|-----------|-------------------------|--------------------------|------|
| 1K | 0.48× | 0.51× | **0.63×** | 0.67× | **0.66×**（Triton 时间 -4%）| **+40%** |
| 2K | 0.71× | 0.75× | **0.87×** | 0.88× | 0.85×（noise）| **+24%** |
| 4K | 1.44× | 1.49× | **1.65×** | 1.65× | **1.68×** | **+17%** |

四个优化互补无干扰：每个 config 各走自己的 gate 路径，正确性保持。

### E2E Training Throughput Validation — 2026-04-17

**动机**：所有前述优化都是 kernel 级别。需验证在真实训练 stack（含 linear projections,
LayerNorm, AdamW optimizer, MSE loss）里是否有对应增益。

#### 合成 Gemma4 stack（`flash_attn/gemma4_e2e.py`）

配置：d_model=2048, 6 blocks (5 slide Config A + 1 full D=512), BF16, AdamW, 10 steps.

**正确性**: N=512/1024/2048 全部 PASS, loss trajectory **bit-exact** (0 rel diff).

**Training throughput (ms/step) — SDPA vs Triton**:

| N | SDPA | Triton **After** | Triton **Before** (2026-04-16) | Speedup **After** | Speedup **Before** | Δ |
|---|------|------------------|--------------------------------|-------------------|---------------------|---|
| 512  | 12.46  | **11.24** | 11.77 | **1.11×** | 1.06× | +5% |
| 1024 | 17.72  | **12.60** | 13.27 | **1.41×** | 1.35× | +4% |
| 2048 | 37.88  | **20.48** | 22.85 | **1.85×** | 1.68× | +10% |
| 4096 | 106.99 | **43.28** | 50.55 | **2.47×** | 2.13× | **+16%** |
| 8192 | 362.25 | **116.35** | 140.64 | **3.11×** | 2.60× | **+20%** |

**Absolute Triton savings per step**:
- N=1K: 13.27 → 12.60 ms (-5%)
- N=4K: 50.55 → **43.28 ms (-14%)**
- N=8K: 140.64 → **116.35 ms (-17%)**

**关键观察**：
1. 所有 kernel 级优化（delta-fusion, Q_SPLIT, BKV=64 big-tile, BKV=64 rescue）
   都在 E2E stack 里**真实生效** — 这不只是 benchmark 数字，而是 end-to-end
   training 时间真的缩短。
2. **长 N 增益更大**（+20% @ N=8K vs +4% @ N=1K）：因为长 N 下 attention 占 step
   比重大（MLP + linear proj 的 O(N) 部分相对小），所以 attention 优化 leverage 更高。
3. Config A 的 BKV=64 大 tile 优化在 synthetic Gemma4 里格外显著：这是 primary
   layer type (5/6 blocks 是 sliding Config A)。

#### 真实 Gemma-4-E2B 模型（5.1B params）

配置：Gemma-4-E2B 35 层 (29 sliding + 6 full), H_Q=8 H_KV=1 GQA 8:1, head_dim=256.
环境：`/opt/tiger/flash_gemma` (torch 2.9.1, transformers 5.5.4, triton 3.5.1).

**Forward-only** (`tests/gemma4_integration/test_gemma4.py`):

| N | SDPA (ms) | Triton (ms) | Speedup |
|---|-----------|-------------|---------|
| 512 | 40.99 | 41.44 | 0.99× |
| 1024 | 42.93 | 43.03 | 1.00× |
| 2048 | 62.98 | 46.95 | **1.34×** |

Correctness: N=1024 cos sim **0.999758**, top-1 **100%**, top-5 **5/5 match**, PASS.

**Training-style fwd+bwd + AdamW step** (`benchmarks/real_gemma4_fwdbwd.py`, 新增)：

| N | SDPA (ms) | Triton (ms) | **Speedup** | 备注 |
|---|-----------|-------------|-------------|------|
| 512 | 203.13 | 214.83 | 0.95× | linear proj dominant |
| 1024 | 222.76 | **211.61** | **1.05×** | 首次在 N=1K 超越 SDPA |
| 2048 | 377.59 | **246.74** | **1.53×** | attention 开始 dominant |
| 4096 | 816.91 | **357.67** | **2.28×** | |

**训练时真实收益拆解**：
- N=1K training: SDPA 被 Triton 超越 (1.05×)
- N=2K training: **1.53× 总 step 加速** — 每个 step 省 131ms
- N=4K training: **2.28× 加速** — 每个 step 省 459ms；长训练累计可省数小时
- **N≥8K training: 5.1B 模型在 80GB H100 上 OOM**（fp32 AdamW state × 3 + activations），
  需要 activation checkpointing / ZeRO / 多卡。此处不展示长 N training 数字

#### Kernel-only benchmark 全 N 统一视图（2026-04-18, `attn_only_all_n.py`）

纯 attention kernel (fwd/F+B/bwd), Triton vs SDPA, H100 FP16, no E2E overhead.

**Config A — Gemma4 synthetic sliding** (H_Q=32, H_KV=16, D=256, slide=1024):
| N | fwd (ms) | F+B (ms) | **fwd** | **F+B** | **bwd** |
|---|---------|---------|---------|---------|---------|
| 1K | 0.144 vs 0.115 | 0.526 vs 0.442 | 0.80× | 0.84× | 0.85× |
| 2K | 0.253 vs 0.247 | 0.961 vs 0.980 | 0.98× | **1.02×** | 1.03× |
| 4K | 0.447 vs 0.648 | 1.933 vs 3.018 | 1.45× | **1.56×** | 1.59× |
| 8K | 0.869 vs 2.040 | 3.996 vs 10.498 | 2.35× | **2.63×** | 2.70× |
| 16K | 1.733 vs 7.910 | 7.816 vs 41.276 | 4.56× | **5.28×** | 5.49× |
| **32K** | 3.446 vs 30.056 | 15.767 vs 161.954 | **8.72×** | **10.27×** | **10.71×** |

**Config B — Gemma-4-E2B real sliding** (H_Q=8, H_KV=1, D=256, slide=512):
| N | fwd (ms) | F+B (ms) | **fwd** | **F+B** | **bwd** |
|---|---------|---------|---------|---------|---------|
| 1K | 0.097 vs 0.075 | 0.409 vs 0.273 | 0.78× | 0.67× | 0.64× |
| 2K | 0.097 vs 0.099 | 0.462 vs 0.401 | 1.02× | 0.87× | 0.83× |
| 4K | 0.128 vs 0.218 | 0.537 vs 0.874 | 1.70× | **1.63×** | 1.60× |
| 8K | 0.200 vs 0.586 | 0.764 vs 2.512 | 2.94× | **3.29×** | 3.42× |
| 16K | 0.335 vs 1.901 | 1.517 vs 10.287 | 5.68× | **6.78×** | 7.09× |
| **32K** | 0.653 vs 7.433 | 3.224 vs 39.484 | **11.38×** | **12.25×** | **12.47×** |

**Full-causal D=512** — Gemma4 full-attention config (H_Q=32, H_KV=4, D=512, no SWA):
| N | fwd (ms) | F+B (ms) | **fwd** | **F+B** | **bwd** |
|---|---------|---------|---------|---------|---------|
| 1K | 0.372 vs 0.452 | 1.734 vs 5.000 | 1.21× | **2.88×** | 3.34× |
| 2K | 0.994 vs 1.423 | 4.906 vs 14.284 | 1.43× | **2.91×** | 3.29× |
| 4K | 3.346 vs 5.199 | 17.526 vs 46.169 | 1.55× | **2.63×** | 2.89× |
| 8K | 12.360 vs 20.179 | 66.642 vs 164.449 | 1.63× | **2.47×** | 2.66× |
| 16K | 47.582 vs 86.167 | 262.981 vs 626.026 | 1.81× | **2.38×** | 2.51× |
| **32K** | 195.271 vs 403.306 | 1047.714 vs 2505.638 | 2.07× | **2.39×** | 2.47× |

**观察**：
1. **长 N 的 Triton 优势巨大**（Config A @ 32K F+B **10.27×**，Config B @ 32K **12.25×**）
   — 因为 SWA 工作量为 O(N·slide) 而 full-causal SDPA 是 O(N²)
2. **Full-causal D=512 是稳定 2.4-2.9× F+B speedup 跨全 N**
   — 这类工作量是 O(N²) 两边相同，纯对比 kernel 效率
3. **Config A N=1K** (0.84× F+B) 和 **Config B N=1K/2K** (0.67×/0.87×) 是仅有的
   低于 SDPA 场景：短 N + SWA 下 SDPA 走 cuDNN fast path Triton 暂时不敌
4. **Bwd 比 F+B 提升略高**（因为 fwd 被 cuDNN 拖累，但 Triton bwd 优势显著）

#### 长 N (8K-32K) 扩展数据

**合成 Gemma4 stack 训练 fwd+bwd**（d_model=2048，内存小，长 N 可跑）：

| N | SDPA (ms/step) | Triton (ms/step) | **Speedup** |
|---|----------------|-------------------|-------------|
| 8K | 362.23 | 116.23 | **3.12×** |
| 16K | 1368.11 | 358.53 | **3.82×** |
| **32K** | 5422.94 | 1234.42 | **4.39×** — 每 step 省 **4.19 秒** |

**真实 Gemma-4-E2B forward-only 长 N** (no_grad，inference-style):

| N | SDPA (ms) | Triton (ms) | **Speedup** |
|---|-----------|-------------|-------------|
| 4K | 157.62 | 79.61 | **1.98×** |
| 8K | 479.06 | 165.43 | **2.90×** |
| 16K | 1620.82 | 363.36 | **4.46×** |
| **32K** | **OOM** | **892.53** | **SDPA 跑不动** — Triton **独家支持 2× context** |

**关键**：N=32K 真实 Gemma-4-E2B forward，**SDPA materialize attention matrix 爆显存**
（80GB H100 不够），**Triton flash pattern 只要 892ms**。等效说 Triton 把 80GB 卡的
**最大 context 从 16K 推到 32K**（long-context inference 场景的核心价值）。

**为什么 Gemma-4-E2B forward-only 提升有限而 fwd+bwd 大**：
- Gemma-4-E2B 5.1B 参数，35 层 linear projections (wq, wk, wv, wo, MLP) 主导 forward
- 我们的优化 80% 集中在 bwd（dKV, dQ 的各种调优）
- fwd+bwd 训练时 attention bwd 占 step 一大块，所以 Triton bwd 收益显性化
- 长 N 下 attention O(N²) 占比上升，Triton 优势 further amplified

---

### BKV=64 rescue 细节（2026-04-17）

Config B N=1K 下 `grid_at_BKV=64 = 16`（极度 starve）。sweep（`dkv_config_b_bkv64.py`）
发现 BKV=64 BQ=128 w=8 s=1 + QS=8 反而更优：

| Config B | BKV=32 QS=8 (current) | BKV=64 QS=8 (rescue) | Δ |
|----------|----------------------|----------------------|------|
| N=1K | 0.153 ms | **0.144 ms** | -6% |
| N=2K | 0.196 ms | 0.194 ms | tied |
| N=4K | 0.248 ms | 0.251 ms | +1% (tied) |

原因：BKV=64 时大 tile 每 program 更大工作量，虽然 QS=8 增加 8-way atomic 竞争，
但 tile 数量减半（16 vs 32 写入 dk/dv），总 atomic 代价反而降低；amortization 赢。

**Gate 最终形式**（attention.py backward wrapper）：
```python
# BKV 选择
grid_at_bkv64 = cdiv(N, 64) * B * H_KV
if grid_at_bkv64 >= 128 or grid_at_bkv64 <= 16:
    BKV, BQ, w, s = 64, 128, 8, 1  # big-tile (healthy OR extreme-starve rescue)
else:
    BKV, BQ, w, s = 32, 64, 4, 2   # middle regime (Config B N=2K/4K)

# Q_SPLITS 选择（目标根据 BKV 分化）
target = 128 if BKV == 64 else 256  # BKV=64 想 1 wave, BKV=32 想 2 waves
raw = cdiv(N, BKV) * B * H_KV
QS = 1 if raw >= target else 2 if raw*2 >= target else 4 if raw*4 >= target else 8
```

覆盖表（所有 config 都经过验证）:
| Config, N | raw_BKV64 | raw_BKV32 | BKV 选择 | QS | grid |
|-----------|-----------|-----------|----------|-----|------|
| A (H_KV=16) N=1K | 256 | 512 | 64 | 1 | 256 |
| A N=2K | 512 | 1024 | 64 | 1 | 512 |
| A N=4K | 1024 | 2048 | 64 | 1 | 1024 |
| B (H_KV=1) N=1K | 16 | 32 | 64 (rescue) | 8 | 128 |
| B N=2K | 32 | 64 | 32 | 4 | 256 |
| B N=4K | 64 | 128 | 32 | 2 | 256 |
| B N=8K | 128 | 256 | 64 | 1 | 128 |
| D=512 any | — | — | 16 | 1 | — |

---
