# Flash Attention in Triton: 从原理到实现

> 这份教程基于本项目的实验过程总结。我是 Triton 初学者，这里记录的是我在实现
> Gemma4 GQA Flash Attention 过程中学到的核心知识。

---

## 目录

1. [为什么需要 Flash Attention](#1-为什么需要-flash-attention)
2. [核心思路：Online Softmax + Tiling](#2-核心思路online-softmax--tiling)
3. [Triton 基础：如何写一个 kernel](#3-triton-基础如何写一个-kernel)
4. [实现 Flash Attention Forward](#4-实现-flash-attention-forward)
5. [GQA 扩展：Grouped Query Attention](#5-gqa-扩展grouped-query-attention)
6. [Causal Masking 与 Early Termination](#6-causal-masking-与-early-termination)
7. [Sliding Window Attention (SWA)](#7-sliding-window-attention-swa)
8. [Backward Pass：梯度计算](#8-backward-pass梯度计算)
9. [性能调优：从 1x 到 2x](#9-性能调优从-1x-到-2x)
10. [坑与经验总结](#10-坑与经验总结)

---

## 1. 为什么需要 Flash Attention

标准 attention 计算如下：

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(D)) @ V
```

问题：**中间矩阵 `A = Q @ K^T` 的大小是 (N, N)**。当 N=4096 时，A 需要 4096×4096×4 bytes = 64MB。对每个 (batch, head) 都要这样，GPU 内存很快耗尽。

更深的问题：现代 GPU（H100）的计算速度（989 TFLOPS）比内存带宽（3.35 TB/s）快得多。
计算 ridgeline = 989/3.35 ≈ 295 FLOP/byte。Attention 的 arithmetic intensity 只有 ~63 FLOP/byte，
远低于 ridgeline，所以 **attention 是 memory-bound**，而不是 compute-bound。

Flash Attention 的解法：**不物化 N×N 矩阵，把 Q/K/V 分块，在 SRAM（shared memory）里流式计算**。

---

## 2. 核心思路：Online Softmax + Tiling

### Softmax 的分块挑战

Softmax 需要全局 sum：`p_i = exp(x_i - max) / sum(exp(x_j - max))`。
如果分块计算，怎么保证 max 和 sum 是全局正确的？

### Online Softmax 算法

核心：维护 **running max `m`** 和 **running sum `l`**，每处理一个 KV 块就更新。

```
初始化: m = -inf, l = 0, acc = 0

对每个 KV 块 (k_i, v_i):
    s_i = Q @ k_i^T / sqrt(D)         # (BLOCK_Q, BLOCK_KV) scores
    m_new = max(m, max(s_i, axis=1))   # 更新 running max

    # 修正之前的 acc（因为 max 变了，exp 的基准变了）
    alpha = exp(m - m_new)             # 缩放因子 < 1
    l = l * alpha + sum(exp(s_i - m_new), axis=1)
    acc = acc * alpha + exp(s_i - m_new) @ v_i

    m = m_new

# 最终归一化
output = acc / l
```

**关键洞察**：每次更新时，旧的 `acc` 需要乘以 `alpha = exp(m_old - m_new)` 来"修正"基准变化。
这个 rescale 是 Flash Attention 的灵魂。

### 内存效率

每个 program 只在寄存器/SRAM 中保持：
- Q block: (BLOCK_Q, D) — 一次性 load
- 当前 K/V block: (BLOCK_KV, D) — 流式 load
- 累加器 acc: (BLOCK_Q, D)
- m, l: (BLOCK_Q,)

总计: O(BLOCK_Q × D)，与 N 无关！

---

## 3. Triton 基础：如何写一个 kernel

### Grid 与 Program ID

Triton 的 kernel 以 **program** 为粒度并行。每个 program 处理数据的一个 tile。

```python
@triton.jit
def my_kernel(..., N_ROWS: tl.constexpr, BLOCK: tl.constexpr):
    # 每个 program 知道自己的 ID
    pid = tl.program_id(0)  # 第 0 维的 ID

    # 计算这个 program 负责的数据范围
    row_start = pid * BLOCK
    offsets = row_start + tl.arange(0, BLOCK)
    mask = offsets < N_ROWS

    # Load / compute / store
    data = tl.load(ptr + offsets, mask=mask, other=0.0)
    ...
    tl.store(out_ptr + offsets, result, mask=mask)
```

### 2D Grid

Flash Attention 通常用 2D grid：

```python
grid = (triton.cdiv(N, BLOCK_Q),   # dim0: Q blocks
        B * H)                      # dim1: (batch, head) pairs
```

在 kernel 里：
```python
q_block_idx = tl.program_id(0)   # 哪个 Q block
bh_idx = tl.program_id(1)        # 哪个 (batch, head)
h_idx = bh_idx % N_HEADS
b_idx = bh_idx // N_HEADS
```

### tl.constexpr

Block size 必须是 `tl.constexpr`，因为 Triton 需要在编译时知道 tensor 形状来做优化。
Python 整数传进去时会自动当成 constexpr。

```python
@triton.jit
def kernel(..., BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr):
    # 可以用 BLOCK_Q 创建 arange
    offsets = tl.arange(0, BLOCK_Q)  # 必须是 constexpr
```

### 2D Load

```python
q_offsets = q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)  # (BLOCK_Q,)
d_offsets = tl.arange(0, HEAD_DIM)                          # (HEAD_DIM,)

# 2D pointer
q_ptrs = q_ptr + q_offsets[:, None] * stride_n + d_offsets[None, :] * stride_d
# shape: (BLOCK_Q, HEAD_DIM)

q_block = tl.load(q_ptrs, mask=mask[:, None], other=0.0)
```

### tl.dot

矩阵乘法：
```python
scores = tl.dot(q_block, tl.trans(k_block))  # (BLOCK_Q, BLOCK_KV)
```

**重要**：`tl.dot` 要求输入是 fp16/bf16（走 tensor core）。fp32 输入会 fallback 到 CUDA core，慢很多。
本项目实测：fp16 → tl.dot 比 fp32 → tl.dot 快 ~20%（HEAD_DIM=512）。

---

## 4. 实现 Flash Attention Forward

完整 forward kernel（简化版）：

```python
@triton.jit
def flash_attn_forward(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qn, stride_qd,  # Q 的 stride（简化：只写 1 个 head）
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_on, stride_od,
    SEQ_LEN, scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    q_block_idx = tl.program_id(0)  # 每个 program 处理一个 Q block

    q_offsets = q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < SEQ_LEN
    d_offsets = tl.arange(0, HEAD_DIM)

    # Load Q block 到寄存器（持久化，不在 KV loop 内重复 load）
    q_block = tl.load(
        Q_ptr + q_offsets[:, None] * stride_qn + d_offsets[None, :] * stride_qd,
        mask=q_mask[:, None], other=0.0
    )

    # Online softmax 状态
    m_i = tl.full([BLOCK_Q], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    # 遍历所有 KV 块
    for kv_start in range(0, SEQ_LEN, BLOCK_KV):
        kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offsets < SEQ_LEN

        # Load K, V
        k_block = tl.load(
            K_ptr + kv_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd,
            mask=kv_mask[:, None], other=0.0
        )
        v_block = tl.load(
            V_ptr + kv_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd,
            mask=kv_mask[:, None], other=0.0
        )

        # Compute scores: Q @ K^T
        scores = tl.dot(q_block, tl.trans(k_block)) * scale  # (BLOCK_Q, BLOCK_KV)

        # Mask padding
        scores = tl.where(kv_mask[None, :], scores, -float("inf"))

        # Online softmax update
        block_max = tl.max(scores, axis=1)          # (BLOCK_Q,)
        new_max = tl.maximum(m_i, block_max)
        alpha = tl.exp(m_i - new_max)               # rescale factor
        p = tl.exp(scores - new_max[:, None])       # (BLOCK_Q, BLOCK_KV)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_block.dtype), v_block)

        m_i = new_max

    # 归一化并写出
    acc = acc / l_i[:, None]
    tl.store(
        O_ptr + q_offsets[:, None] * stride_on + d_offsets[None, :] * stride_od,
        acc, mask=q_mask[:, None]
    )
```

---

## 5. GQA 扩展：Grouped Query Attention

Gemma4 config: **H_Q=32 个 Q head，H_KV=4 个 KV head**（GQA ratio = 8）。

### 为什么用 GQA？

每个 KV cache 是 (N, D) per head。H_KV=4 vs H_Q=32 节省 8x KV cache 内存。
推理时 batch 可以更大。

### 实现

只需要 map Q head → KV head：

```python
q_h_idx = bh_idx % N_Q_HEADS
kv_h_idx = q_h_idx * N_KV_HEADS // N_Q_HEADS  # 整数除法，GQA ratio 必须整除

q_base = Q_ptr + b_idx * stride_qb + q_h_idx * stride_qh
k_base = K_ptr + b_idx * stride_kb + kv_h_idx * stride_kh  # 注意用 kv_h_idx
v_base = V_ptr + b_idx * stride_vb + kv_h_idx * stride_vh
```

Grid 依然以 Q heads 为单位：`grid = (cdiv(N, BLOCK_Q), B * N_Q_HEADS)`

### 内存节省

PyTorch SDPA 在 GQA 时会 `repeat_interleave(K, ratio, dim=1)` 扩展 KV，导致 KV 内存是原来 ratio 倍。
我们的 Triton kernel 直接用 GQA map，不扩展，**内存节省 ~5x**（实测，见 baseline.md）。

---

## 6. Causal Masking 与 Early Termination

Causal attention：位置 i 只能 attend 到 j ≤ i（只看过去，不看未来）。

### 两个优化

**1. Early termination**：对第 `q_block_idx` 个 Q block，KV loop 最多到 `(q_block_idx+1)*BLOCK_Q`，
因为所有 KV 位置 j > q_i 都会被 mask。

```python
if IS_CAUSAL:
    kv_end = (q_block_idx + 1) * BLOCK_Q
else:
    kv_end = SEQ_LEN
```

这使得 causal attention 的 FLOPs 约减半（三角形 vs 正方形），speedup 随 N 增大。

**2. Causal mask**：在 scores 上加条件：

```python
if IS_CAUSAL:
    valid = (kv_offsets[None, :] <= q_offsets[:, None]) & kv_mask[None, :]
else:
    valid = kv_mask[None, :]
scores = tl.where(valid, scores, -float("inf"))
```

### 实测 speedup（vs full causal SDPA）

| N | Speedup |
|---|---------|
| 512  | 1.10x |
| 1024 | 1.38x |
| 2048 | 1.52x |
| 4096 | 1.64x |
| 128K | 2.81x |

---

## 7. Sliding Window Attention (SWA)

Gemma4 使用局部 attention：每个 token 只 attend 最近 `slide_size=1024` 个 token。

### 数学定义

Token i attend to token j if:
- **Causal**: j ≤ i
- **Window**: i - j < slide_size（即 j > i - slide_size）

Combined: `max(0, i - slide_size + 1) ≤ j ≤ i`

### 实现

**核心优化**：提前算出 KV loop 的起始位置，跳过窗口外的 KV 块。

```python
if IS_CAUSAL and SLIDE_SIZE > 0:
    # Q block 第一个位置 = q_block_idx * BLOCK_Q
    # 它能 attend 的最早 KV 位置 = q_block_idx * BLOCK_Q - SLIDE_SIZE + 1
    kv_min = tl.maximum(0, q_block_idx * BLOCK_Q - SLIDE_SIZE + 1)
    kv_loop_start = (kv_min // BLOCK_KV) * BLOCK_KV  # 向下取整到 block 边界
else:
    kv_loop_start = 0

for kv_start in range(kv_loop_start, kv_end, BLOCK_KV):
    ...
    if IS_CAUSAL and SLIDE_SIZE > 0:
        valid = (kv_offsets[None, :] <= q_offsets[:, None]) & \
                (q_offsets[:, None] - kv_offsets[None, :] < SLIDE_SIZE) & \
                kv_mask[None, :]
```

### ⚠️ NaN Bug 与修复

**问题**：kv_loop_start 是按 Q block 第一个位置保守估算的，
可能会包含对某些 Q row 全部 masked 的 KV 块。

如果此时 `m_i = -inf`（该 Q row 尚未见到任何有效 KV），则：
- `block_max = -inf`（全 mask）
- `new_max = max(-inf, -inf) = -inf`
- `alpha = exp(-inf - (-inf)) = exp(nan) = nan` ← 💥

**修复**：在计算 alpha 和 p 时，将 `-inf` clamp 到大负数 `-1e20`：

```python
alpha = tl.exp(tl.maximum(m_i, -1e20) - tl.maximum(new_max, -1e20))
p = tl.exp(scores - tl.maximum(new_max, -1e20)[:, None])
```

效果：当 new_max=-inf 时，alpha=exp(0)=1（不改变 acc），p=exp(-inf)=0（不贡献）。✓

### 性能

SWA 复杂度 O(N × S) vs Full Causal O(N²/2)。**断点在 N = 2×slide_size**。

**Gemma4 Sliding 层 config**（H_Q=32, H_KV=16, D=256, slide=1024）：

| N | SDPA Causal (ms) | Triton SWA (ms) | SWA vs SDPA |
|---|-----------------|-----------------|-------------|
| 1024 | 0.115 | 0.156 | 0.74x（N=S，无工作量节省） |
| 2048 | 0.250 | 0.269 | 0.93x（N=2S，刚到断点） |
| 4096 | 0.647 | 0.489 | 1.32x |
| 8192 | 2.074 | 0.944 | 2.20x |
| 16384 | 7.990 | 1.992 | 4.01x |

**为什么断点是 N=2S？**
- SWA work ≈ S²/2 + (N-S)·S = S(S + 2(N-S))/2 = S(2N-S)/2
- Full causal work = N²/2
- Ratio = SWA/Full = S(2N-S) / N²
- N=S: ratio = 1 (完全相同)
- N=2S: ratio = 3S·S / (4S²) = 0.75
- N=4S: ratio = 7S·S / (16S²) ≈ 0.44
- N=16S: ratio ≈ 0.12

约 N=2S 时工作量减少 25%，刚好抵消 kernel 常数开销；N≥4S 开始明显胜出。

### Backward 中的 SWA

dQ kernel：KV loop 同样加 kv_loop_start + sliding window mask。

dKV kernel：需要额外计算 **Q loop 上界**（窗口外的 Q 不会 attend 到这个 KV block）：

```python
if IS_CAUSAL and SLIDE_SIZE > 0:
    kv_last = kv_block_idx * BLOCK_KV + BLOCK_KV - 1
    q_max = kv_last + SLIDE_SIZE - 1  # 最晚能 attend 到此 KV block 的 Q 位置
    q_loop_end = ((q_max // BLOCK_Q) + 1) * BLOCK_Q
    q_loop_end = tl.minimum(SEQ_LEN, q_loop_end)
```

---

## 8. Backward Pass：梯度计算

Backward 的核心挑战：Attention weight `P = softmax(QK^T/sqrt(D))` 不会显式存储（太大）。
Flash Attention 的解法：**在 backward 中重新计算 P**，只需存储 logsumexp (LSE)。

### 保存 LSE

Forward 末尾存储 `lse = m_i + log(l_i)`（每个 Q position 一个标量），
Backward 用它重算 P：

```python
p = exp(Q @ K^T / sqrt(D) - lse[:, None])
```

### 梯度公式

- `dV = P^T @ dO`
- `dP = dO @ V^T`
- `delta = rowsum(dO * O)`  ← 预计算，避免 KV loop 内重复计算
- `dS = P * (dP - delta[:, None])`  ← softmax Jacobian 的简化形式
- `dQ += scale * dS @ K`
- `dK += scale * dS^T @ Q`

### 两个独立 kernel

1. **dQ kernel**：grid over Q blocks，iterate KV blocks，累加 dQ
2. **dKV kernel**：grid over KV blocks，iterate Q blocks，累加 dK 和 dV

GQA 的 dKV 特殊处理（split + reduce 设计）：
- 不按 KV head 分 program（只有 H_KV=4 个头，SM 利用率低）
- 改为按 Q head 分 program（H_Q=32），每个 program 计算该 Q head 对应的 dK, dV
- 输出 `dk_expanded(B, H_Q, N, D)`，再 `view+sum → dk(B, H_KV, N, D)`
- 代价：额外 `2 × B × H_Q × N × D` 内存；收益：32x 更多 programs，短序列提升显著

---

## 9. 性能调优：从 1x 到 2x

### Block Size 选择（HEAD_DIM=512, H100）

| 参数 | 最优值 | 原因 |
|------|--------|------|
| BLOCK_Q | 64 | 更大 → 更好 SM 利用率；128 → shared memory OOM |
| BLOCK_KV | 32 | 更小 → 更少 register pressure；64 → shared memory OOM |
| BLOCK_D | 512 (=HEAD_DIM) | 不做 D-tiling；tiling 反而慢（1.23x vs 0.91x） |
| num_warps | 8 | 至关重要！4 warps 慢 3.3x（HEAD_DIM=512 时） |
| num_stages | 2 | 3 stages → shared memory OOM |

### 为什么 num_warps=8 这么重要？

HEAD_DIM=512 时，每个 warp 处理更多数据，需要更多 warp 来隐藏 memory latency。
实测：4 warps → 1.469ms，8 warps → 0.441ms（同配置）。

### 为什么不做 D-tiling？

D-tiling 把 HEAD_DIM 分块，理论上可以减少 register pressure。
但 H100 的寄存器文件足够大，分块反而增加了 loop overhead 和 load 次数。
实测：BLOCK_D=512（无 tiling）比 BLOCK_D=128（4次 tiling）快 35%+。

### fp16 vs fp32

在 `tl.dot` 中保持 fp16 输入（走 tensor core），只在 softmax 计算时转 fp32：

```python
scores = tl.dot(q_block, tl.trans(k_block))  # fp16 input → tensor core
scores = scores.to(tl.float32) * scale        # fp32 for numerical stability
```

---

## 10. 坑与经验总结

### 坑 1：`inf - inf = nan`（SWA 的 NaN bug）

Online softmax 中，若某 Q row 所有有效 KV 都在后续 block 里，
当前 block 全 mask 时 m_i=-inf, block_max=-inf → alpha=nan。
**修复**：clamp 到 -1e20 再做减法。

### 坑 2：num_warps 对大 HEAD_DIM 至关重要

不是所有 kernel 都 num_warps=4。HEAD_DIM≥256 时强烈推荐 8。
实测差异 3.3x，差距超过其他任何单一调优。

### 坑 3：D-tiling 不一定有用

直觉上 HEAD_DIM=512 很大应该 tiling，但实测无 tiling 更快。
结论：先测量，不要凭直觉优化。

### 坑 4：Grid reorder 对 GQA 无效

期望 GQA group 内的 Q heads 连续调度可以共享 L2 中的 KV data。
实测：短 seq <2%，长 seq 甚至退步（因为 L2=50MB < KV size）。

### 坑 5：dKV BQ=32 引发 register spill

dKV kernel 需要在寄存器中同时保持 K(BKV×D), V(BKV×D), dK(BKV×D), dV(BKV×D)。
BQ 越大，分析中间量越多，寄存器用量超限则 spill 到 L2/HBM，性能灾难性下降（0.22x）。
BQ=16 是安全上限。

### 总结原则

1. **先测量，后优化**：不确定时写 sweep，用数据说话
2. **一次只改一个变量**：否则不知道是哪个改动有效
3. **警惕 nan 和 inf**：浮点特殊值传播很快，要在算法设计时就规避
4. **constexpr 分支很强大**：用 `if SLIDE_SIZE > 0`（constexpr 分支）零成本添加条件

---

> 最后更新：2026-04-15（完成 SWA 实现）
