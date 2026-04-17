# Block-config gate 应该基于 raw_grid，不是 N 单维

## 场景
挑 dKV kernel 的 `(BKV, BQ, w, s)` config 时，常见的反射是按 `N` 分段：
```python
if N >= 8192:
    BKV, BQ, w = 64, 128, 8  # big tile
else:
    BKV, BQ, w = 32, 64, 4   # small tile
```

这在 mixed-config 项目里埋雷：**N 不是唯一决定 SM occupancy 的维度**，
`B`、`H_KV` 同样重要。

## 翻车
2026-04-16 上面 `N<8K → 小 tile` 的判断基于 Gemma-4-E2B (H_Q=8, H_KV=1) 的
profiling：BKV=64 时 `raw_grid = N/64 × 1 = 16 @ N=1K` 严重 starved SMs。
结论：短 N 用大 BKV 会 starve，必须 BKV=32。

然后这个 config 被推广到 **所有** D<512 短 N 场景。但 Config A
(H_Q=32, H_KV=16) 下 `raw_grid = N/64 × 16 = 256 @ N=1K` **健康无比**，
用 small tile 浪费算力 — 多花了 23-35% 时间。

## 正解：用 raw_grid 做 gate

```python
grid_at_bkv64 = triton.cdiv(N, 64) * B * H_KV
if grid_at_bkv64 >= 128:  # one healthy wave on 132 SMs
    BKV, BQ, w, s = 64, 128, 8, 1   # big-tile win
else:
    BKV, BQ, w, s = 32, 64, 4, 2    # small tile, with Q_SPLITS compensating
```

判据直接反映目标（grid 是否健康），自动覆盖 H_KV 和 B 的影响。

## 教训系统性
- 任何 config 选择条件都应基于 **架构量**（grid、SM util、shmem 预算），
  不基于 **用户可变维度** (N/B/H) 单独。
- 推广到新 config 时，**重新推导判据**，不要 copy-paste 原有阈值。
- Triton 优化里，H_KV=1 (MQA-extreme) 和 H_KV=16 是两种不同的 kernel
  占用特征 — 别用一个数据点推广全场景。

## 实测收益（这次翻案）
Config A (H_Q=32, H_KV=16, D=256, slide=1024) dKV time:
| N | Before (误配 BKV=32) | After (正确 BKV=64) | Δ |
|---|---|---|---|
| 1K | 0.286 ms | 0.219 ms | -23% |
| 2K | 0.653 ms | 0.454 ms | -30% |
| 4K | 1.353 ms | 0.882 ms | -35% |

End-to-end fwd+bwd vs SDPA: N=1K 0.74×→0.83×, N=2K 0.86×→1.05× (crosses
SDPA), N=4K 1.30×→1.61×.

## 反向指引
遇到 "旧 config 被短 N 论文改动" 时，先问：
1. 旧结论的实验条件是哪些 (H, D, B, slide)？
2. 当前场景的 raw_grid 和旧场景的 raw_grid 在同一数量级吗？

如果否：旧结论不适用，直接当前场景跑新 sweep，别相信旧 carveout。
