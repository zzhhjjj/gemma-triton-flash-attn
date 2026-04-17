# Q_SPLIT dKV：低 H_KV 短 N 场景的 SM-occupancy 救星

## 问题场景
GQA 比 ≥ 8:1 且 `H_KV = 1` 的模型（Gemma-4-E2B 实际部署场景）在短 N 时：

```
dKV kernel grid = cdiv(N, BLOCK_KV) × B × H_KV
           = cdiv(1024, 32) × 1 × 1 = 32 programs
H100 有 132 SMs → SM 占用率 24%
```

结果：dKV kernel 时间在 N=1K→2K→4K 几乎不变（0.273 / 0.270 / 0.293ms），
**fixed launch overhead dominate，不是 compute-bound**。Triton fwd+bwd 在 N=1K
Config B 下速度只有 SDPA (cuDNN backend) 的 0.48×。

## 方案：沿 Q 维度 split + atomic_add

pack-GQA dKV kernel 每个 program 原本处理 `(kv_block, h_kv)` tile，内层 loop 吞
`GQA_RATIO × N/BQ` 次 Q iter。把 Q loop 切成 `Q_SPLITS` 份：

```python
@triton.jit
def _flash_attn_gqa_bwd_dkv_packed_kernel(
    ..., Q_SPLITS: tl.constexpr,  # 新参数
):
    # Grid: (cdiv(N, BKV), B * H_KV, Q_SPLITS)
    kv_block_idx = tl.program_id(0)
    bkvh_idx = tl.program_id(1)
    split_idx = tl.program_id(2)
    ...
    # 计算 Q loop 范围
    if Q_SPLITS > 1:
        total_q_blocks = (q_end - q_start + BLOCK_Q - 1) // BLOCK_Q
        blocks_per_split = (total_q_blocks + Q_SPLITS - 1) // Q_SPLITS
        q_lo = q_start + split_idx * blocks_per_split * BLOCK_Q
        q_hi = tl.minimum(q_end, q_start + (split_idx+1) * blocks_per_split * BLOCK_Q)
    else:
        q_lo, q_hi = q_start, q_end
    ...
    # 输出：QS>1 用 atomic_add 累加（caller 预 zero）
    if Q_SPLITS > 1:
        tl.atomic_add(dk_ptrs, dk_acc.to(dtype), mask=...)
        tl.atomic_add(dv_ptrs, dv_acc.to(dtype), mask=...)
    else:
        tl.store(dk_ptrs, dk_acc.to(dtype), mask=...)
        tl.store(dv_ptrs, dv_acc.to(dtype), mask=...)
```

## 启发式（tuned on H100 132 SMs）

目标 ≈ 2 waves (256 programs)：

```python
raw_grid = triton.cdiv(N, BLOCK_KV) * B * H_KV
if raw_grid >= 256:
    Q_SPLITS = 1     # 正常 SM 饱和
elif raw_grid >= 128:
    Q_SPLITS = 2
elif raw_grid >= 64:
    Q_SPLITS = 4
else:
    Q_SPLITS = 8     # 例 Config B N=1K: 32 → 256
```

关键：**gated on raw_grid**。Healthy grid (Config A: raw=512) 用 QS=1 避 atomic；
starved grid (Config B: raw=32) 用 QS=8 救 occupancy。

## 为什么这次赢了，以前"atomic fuse"输

| 方案 | 问题 | 测出的结果 |
|------|------|------------|
| 2026-04-16 全局 atomic_add dKV fuse | Gemma4 Config A 下 grid 已健康，atomic 只添乱 | **慢 3-5%，拒绝** |
| 2026-04-17 Q_SPLIT gated | 只在 grid < 256 时开启，低 H_KV 场景 SM idle 严重过 atomic 代价 | **单 kernel -44%** |

**翻案要点**：旧结论不是错的——在 grid 饱和场景下 atomic 就是赔本。但
**grid 不饱和时 tradeoff 反转**：SM 空转的浪费 > atomic 竞争的浪费。

## 实测数据

**Config B (H_Q=8, H_KV=1, D=256, slide=512)**，H100：

| N | raw_grid | QS=1 | QS=2 | QS=4 | QS=8 | best |
|---|----------|------|------|------|------|------|
| 1K | 32 | 0.271 | 0.208 | 0.167 | **0.151** | QS=8 (-44%) |
| 2K | 64 | 0.273 | 0.213 | **0.192** | 0.282 | QS=4 (-30%) |
| 4K | 128 | 0.296 | **0.249** | 0.360 | 0.477 | QS=2 (-16%) |
| 8K | 256 | **0.360** | 0.485 | 0.670 | 0.930 | QS=1 (sat) |

**Config A (H_Q=32, H_KV=16, D=256)**：raw_grid ≥ 512 across all N → QS=1 永远最优。

## 端到端 fwd+bwd vs SDPA
Config B 累计提升（delta-fusion + Q_SPLIT）：
- N=1K: 0.48× → **0.63×** (+31%)
- N=2K: 0.71× → **0.87×** (+23%)
- N=4K: 1.44× → **1.65×** (+15%)

## 陷阱

1. **`dk, dv` 要 pre-zeroed 当 QS>1**：atomic_add 累加，不能从 uninit 开始。
   ```python
   if Q_SPLITS > 1:
       dk = torch.zeros_like(k); dv = torch.zeros_like(v)
   else:
       dk = torch.empty_like(k); dv = torch.empty_like(v)
   ```
2. **短 N 尾端 KV block 的 splits 可能全 no-op**：causal 下最后几个 kv block 的 Q 范围很小，
   高 QS 导致 `q_lo >= q_hi` 的程序，dk_acc/dv_acc 保持零，atomic_add 0 无害但浪费。
   cost 很小，不值得 early-return。

## 什么时候用
- GQA ratio ≥ 4 且 H_KV 小 (≤ 4) 且 短 N (≤ 4K) 时
- 单看 raw_grid: `< 256 = B × H_KV × cdiv(N, BLOCK_KV)` 是 gate 条件
- 不适用于 healthy grid（Config A）、长 N、B 大
