# context/ 索引

> 按需读取，不要全量加载。每个模块一句话说清楚"里面有什么"。

| 文件 | 内容摘要 | 最后更新 |
|------|----------|----------|
| `baseline.md` | 完整 benchmark: fwd/bwd/causal/long(128K)/BF16/MFU/BW/batch/grid + SWA + Gemma4 synthetic E2E + Gemma-4-E2B real model + 短 N profiling + delta-fusion | 2026-04-17 |
| `arch.md` | kernel 结构（fwd+bwd）+ GQA/SWA/causal 路径 + dQ/dKV 设计 + **HF integration layer** (register_triton_attention / adapter) | 2026-04-16 |
| `hotspots.md` | 热点分析 + memory bound 定量确认(84-90% BW) + GQA KV sharing 分析 + 已排除误报 | 2026-04-16 |

> 新增模块时在此追加一行，保持索引完整。