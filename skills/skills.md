# skills/ 索引

> 遇到问题先查这里，有现成方案直接复用，避免重复踩坑。

| 文件 | 适用场景（一句话） |
|------|-------------------|
| `2026-04-16_triton-large-headdim.md` | Triton flash attention HEAD_DIM≥256 时的 block size / warp / stage 调优经验 |
| `2026-04-17_pack-gqa-register-model.md` | Pack-GQA backward kernel block size 调优：BKV 决定 accumulator, BQ 不进 accumulator |
| `2026-04-17_block-sweep-sm-occupancy.md` | Block-size sweep 必须覆盖最小目标 N，否则 SM occupancy 陷阱导致短 N 回退；解法是 N-gate |
| `2026-04-17_fp16-cossim-pitfall.md` | FP16 cos sim 会误报（同一数据 0.98 vs fp32 0.9999）；校验 Triton kernel 必须 `.float()` 后再 cos sim |
| `2026-04-17_delta-fusion-into-dQ.md` | Delta kernel 融入 dQ prologue 省 1 launch；短 N launch-bound kernel（N × B × H 个 tiny program）值得往大 kernel 里塞 |
| `2026-04-17_q-split-dkv.md` | dKV pack-GQA kernel Q_SPLIT + atomic_add 回收 SM occupancy；gate 在 raw_grid < 256 才开，目标 2-wave on 132 SMs |
| `2026-04-17_grid-gate-over-n-gate.md` | Block-config gate 应该基于 raw_grid 不是 N 单维；H_KV/B 同样影响 grid，误推广旧结论就会埋雷 (Config A 短 N BKV=32 碎片 4 月误用案例) |
| `2026-04-17_alloc-not-bottleneck.md` | training bwd 的 alloc 只占 3-5%（caching allocator 很快），别优化；autograd.Function 框架 80μs CPU-exclusive 才是 short-N 剩余瓶颈 |

---

> 新增 skill 时在此追加一行。文件命名：`YYYY-MM-DD_<topic>.md`