# Triton Flash Attention: 大 HEAD_DIM 调优经验

> 适用场景：HEAD_DIM ≥ 256 的 flash attention Triton kernel 调优

---

## 核心经验

### 1. D-tiling 不一定比全量 dot 快

- **直觉**：HEAD_DIM=512 太大，应该分块加载以降低 register pressure
- **实测**：BLOCK_D=512（无 D-tiling）比 BLOCK_D=128（4 次 D-tile）快 35%+
- **原因**：D-tiling 每 KV iteration 多次加载 Q chunk，额外 HBM 流量 > 寄存器节省的收益
- **结论**：先试 BLOCK_D=HEAD_DIM；只在 OOM registers 时才退回到 D-tiling

### 2. num_warps 对大 block 至关重要

- BLOCK_Q=64 + HEAD_DIM=512 时，4 warps 比 8 warps 慢 3.3x
- 原因：大 block 意味着每个 program 有大量 memory 请求需要 overlap
- 8 warps (256 threads) 能更好地 hide memory latency
- 但 16 warps 反而严重退化（太多 warps 争抢 register file）

### 3. Shared memory 是硬限制

- H100 shared memory: 228KB/SM（配置依赖）
- (BLOCK_Q=64, BLOCK_KV=64, HEAD_DIM=512) 需要 ~320KB → OOM
- (BLOCK_Q=128, BLOCK_KV=32, HEAD_DIM=512) 需要 ~256KB → OOM
- 安全组合: BLOCK_Q=64 + BLOCK_KV=32（最优）或 BLOCK_Q=32 + BLOCK_KV=64

### 4. num_stages 与 shared memory 冲突

- num_stages=3 会增加 pipeline buffer 占用 → 容易 OOM shared memory
- 对于大 HEAD_DIM kernel，num_stages=2 通常是上限

### 5. fp16 输入直传 tl.dot

- 原 kernel 用 `.to(tl.float32)` 后做 dot → 走 CUDA core
- fp16/bf16 直传 → 走 tensor core，accumulate 自动 fp32
- 注意：`tl.dot(p.to(v.dtype), v)` 中 p 需要从 fp32 cast 回 fp16

---

## Sweep 方法论

1. 先确定 HEAD_DIM 是否需要 D-tiling（试 BLOCK_D=HEAD_DIM vs 小 BLOCK_D）
2. 固定 BLOCK_D 后 sweep (BLOCK_Q, BLOCK_KV)：从小到大，直到 OOM
3. 每个 (BLOCK_Q, BLOCK_KV) 试 num_warps={4,8}
4. 最后微调 num_stages（通常 2 就够）
5. 用多个 shape 验证最优配置的泛化性（短/长 seq, 不同 batch）

---

## Causal mask 实现要点

### KV loop early termination

```python
if IS_CAUSAL:
    kv_end = (q_block_idx + 1) * BLOCK_Q  # 不需要 min(SEQ_LEN, ...)，kv_mask 处理越界
else:
    kv_end = SEQ_LEN
```

- 第一个 Q block 只迭代 ~1 个 KV block，最后一个迭代 ~全部
- 平均减半迭代次数 → causal 约 2x 快于 non-causal

### Causal mask 与 kv_mask 合并

```python
if IS_CAUSAL:
    valid = (kv_offsets[None, :] <= q_offsets[:, None]) & kv_mask[None, :]
else:
    valid = kv_mask[None, :]
scores = tl.where(valid, scores, -float("inf"))
```

- 合并避免两次 tl.where 调用
- 对所有 KV block 都应用 causal mask（包括完全在对角线下方的块）— 简单且开销可忽略

### 注意：不能用 break

- Triton JIT 不支持 `break`，只能通过调整 `range` 上界来 early terminate

---

## Grid reorder 对 GQA L2 复用无效（失败经验）

**假设**：将 GQA group 内 Q heads 交叉排列在 grid 中可提升 K/V 的 L2 cache 复用

**结果**：短 seq <2% 提升，长 seq (N≥32K) 反而 -3.6%。

**原因分析**：
1. N≥32K 时 KV 数据 per head = 64MB > L2 50MB → 无法 cache
2. Grid reorder 改变了 GPU scheduler 的 thread block 调度模式，长 seq 下反而降低效率
3. 原始 grid `(cdiv(N,BQ), B*H_Q)` 已自然产生了不错的 L2 locality（同 KV head 的 Q heads 在 pid[1] 中相邻）

**结论**：当 BW utilization 已达 84-90% 时，Triton 层面的 grid reorder 收益极小。进一步优化需 CUDA 级控制（persistent kernel / TMA / warp specialization）。
