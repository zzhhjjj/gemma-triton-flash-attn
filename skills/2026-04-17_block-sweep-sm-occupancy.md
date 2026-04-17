# Block-size sweep 必须覆盖最小目标 N（SM occupancy 陷阱）

**适用场景**：调 Triton kernel 的 block size 时，sweep 只看一两个"代表性"的
N 得到的最优 config，在短 N 可能产生 8-15% 的回退。

## 问题

**H100 有 132 SMs**。一个 dKV-style kernel 的 grid 是
`(N / BKV, B × H_KV)`。在 H_KV=1, B=1 的场景下：

| N    | BKV=32 | BKV=64 | SM 利用率（BKV=64） |
|------|--------|--------|---------------------|
| 2K   | 64 prog| **32 prog** | **24%** ⚠ |
| 4K   | 128    | 64     | 48% |
| 8K   | 256    | 128    | 97% |
| 16K  | 512    | 256    | **saturated** |
| 32K  | 1024   | 512    | saturated |

当 grid < SM count 时，大 tile 的 compute-per-program 优势反而被 SM 闲置
抵消 —— **更大的块意味着同一块内做更多活，但块太少，部分 SM 吃不到饭。**

## 实例（2026-04-17 D=256 SWA dKV）

Sweep 只跑了 N=16K/32K → 找到 `(BKV=64, BQ=128, w=8, s=1)` 胜旧默认 10-14%。

E2E 应用后：
- N≥8K: -3.3% ~ -6.0% 提升（符合预期）
- **N=2K: +9.9% 回退**
- **N=2K slide=1024: +10.6% 回退**

根因：N=2K × BKV=64 = 32 programs = 132 SMs 的 24%。大 tile 瞎白设。

## 对策：N-gate（运行时分支）

```python
if D >= 512:
    # D=512 config
elif N >= 8192:
    # long-N optimal (big tiles)
else:
    # short-N optimal (smaller tiles, more programs)
```

开销：Python 层一个 `if` + Triton 编译不同 constexpr 组合 → 运行时纳秒级，
首次编译缓存后无增量开销。

## 预防：sweep 设计 checklist

sweep 之前，列出 **目标部署的 N 范围**，确保 sweep 至少覆盖：

1. **最小目标 N**（e.g., 短 context inference）
2. **最常见 N**（e.g., 训练典型 seq len）
3. **最大目标 N**（e.g., long context）

每个点单独拿胜者，然后：
- 如果同一 config 在所有 N 上都 win → 直接 ship
- 如果不同 N 需要不同 config → N-gate 路由

**只 sweep 长 N 的教训**：`BKV` 大（或任何会减少 grid 大小的参数）会在短 N
出现回退，sweep 查不到。

## 量化判据

当 `grid_size < 0.5 × N_SMs`（H100 即 grid < 66），可以警觉："大 tile 可能
吃亏"。如果 sweep 候选在此范围内，要么 N-gate 切回小 tile，要么接受回退。

## 关联

- `baseline.md` "D=256 SWA pack-GQA dKV N-gated Block Size (2026-04-17)"
- `attention.py:1509-1527` —— N-gate 实现
- `skills/2026-04-17_pack-gqa-register-model.md` —— pack-GQA BKV/BQ 模型
