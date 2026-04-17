# Pack-GQA kernel 的 register/shmem 模型与 split kernel 不同

**适用场景**：调整 pack-GQA backward kernel 的 block size，或任何 per-KV 程序
将多个 Q heads 折进同一 program 的 Triton kernel。

## 关键模型差异

| 项 | Split kernel（per-Q-head） | Pack-GQA kernel（per-KV-head） |
|---|---|---|
| Grid | `(N/BKV, B × H_Q)` | `(N/BKV, B × H_KV)` (GQA_RATIO× fewer programs) |
| Q heads | 每 program 处理 1 个 Q head | `tl.static_range(GQA_RATIO)` 循环展开 |
| **dk_acc / dv_acc 大小** | `BKV × D × fp32 × 2` | **同样 `BKV × D × fp32 × 2`**（不因 GQA_RATIO 增大）|
| BQ 的作用 | Q 入 Q block，影响 local accumulator / ds tile | **Q 每 iter reload，不进 accumulator** |
| Shmem 主要瓶颈 | Q + K/V 块 + 可能的 dK/dV buffer | **K/V 块 + accumulator 主导** |

## 反直觉结论：大 BQ 在 pack-GQA 下可能可行

Split kernel 下 "BQ=32 灾难 spill" 是因为 dK/dV buffer (expand 版本) 或 Q acc 随 BQ 线性增长。
Pack-GQA 下 dk_acc/dv_acc 只由 BKV 决定，大 BQ 只增加 Q/dO tile 本身的 shmem 占用（一次 tile + 临时寄存器），不会在 iter 间持有。

**实测 D=512 Gemma4 full-causal pack-GQA dKV (2026-04-17 sweep)**:
- 旧默认 (BKV=32, BQ=16, w=8) → 13.13ms @ N=4K
- 新最优 (BKV=16, BQ=64, w=4) → **9.36ms @ N=4K (-29%)**
- BQ=32 几乎所有 config 都 150-300ms（退化，但原因不是 spill，可能是 inner-loop shape 不匹配 tensor core mma）
- BQ=128 全部 shmem OOM（Q tile 太大）

## 调优套路

1. **先减 BKV** 腾 shmem 预算（BKV=32 → 16，dk/dv accumulator 减半）
2. **后增 BQ** 减 inner loop 迭代数（Q/dO tile 更大，矩阵乘利用率更高）
3. **warps=4 常胜 warps=8**（pack-GQA bwd compute 较轻，warps=8 过度调度）
4. **stages=2 足矣**（stages=3 经常 shmem OOM 或略慢）

## 何时不适用

- 如果 GQA_RATIO=1（即 MHA），pack-GQA 退化为 split，BKV↑ 可能更合算（K/V 加载一次跨更大 N tile）
- D<256 时 shmem 不是主要约束，这套 trade-off 不成立

## 关联

- baseline.md 的 "Pack-GQA dKV Block Size Re-sweep for D=512 (2026-04-17)"
- attention.py `_flash_attn_gqa_bwd_dkv_packed_kernel` + autograd backward 调用点
