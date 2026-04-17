# skills/ 索引

> 遇到问题先查这里，有现成方案直接复用，避免重复踩坑。

| 文件 | 适用场景（一句话） |
|------|-------------------|
| `2026-04-16_triton-large-headdim.md` | Triton flash attention HEAD_DIM≥256 时的 block size / warp / stage 调优经验 |
| `2026-04-17_pack-gqa-register-model.md` | Pack-GQA backward kernel block size 调优：BKV 决定 accumulator, BQ 不进 accumulator |

---

> 新增 skill 时在此追加一行。文件命名：`YYYY-MM-DD_<topic>.md`