# 跨 H100/H200/B200 整理计划

更新时间：2026-08-05 18:28 PST  
基线提交：`4f0abb5c17b9401f166251f380ebda422b93e77d`（`refactor`，工作区 dirty）

## 目标

1. 保留 H100、H200、B200 的生产实现、配置与可复现实验证据；
2. B200 varlen D512 full F+B 覆盖2K–256K与packed分布，全部 >1.5× 同语义 SDPA；
3. 生产包与历史区分层组织；三代硬件相关代码和实验只移动、不删除；
4. 当前测试、benchmark 和文档入口一致。

## 保留边界

- `sm90` compile-safe base：H100/H200 共享，必须保留；
- H100 product override：保留，当前状态为历史 tuning、待复认证；
- H200：保留安全配置和历史证据；尚无独立 product-qualified tuned override；
- `sm100` compile-safe base 与 B200 tuned override：必须保留；
- Ulysses、HF integration、packing/reference/public API：按公开能力保留；
- grouped forward、fused backward、split dKV 等失败结构属于硬件研发历史；可以连同复现脚本移出 production 文件，但源码、配置和结论必须完整保留。
- 任何 H100/H200/B200 历史数值、调优表、负面结论和环境限制都必须保留；整理只改变位置与状态标签。
- 只有同时满足“与 H100/H200/B200 均无关、无调用、无独立复现价值、无唯一实验结论”的内容才允许删除。

## 当前基线

- B200 E2B D512 production：单序列2.42×–4.33×、packed 2.46×–4.02× SDPA；
- 峰值显存：长端18.0625KiB/token；256K为4.516GiB；
- CPU：150 passed、50 GPU skipped；
- B200 GPU：完整200/200，另7卡各varlen 8/8；
- 当前14个tracked文件有修改，另有5个新中文状态文档；其中3个为`MM`，提交前必须统一stage；
- H100/H200 本轮只保证路径未删除，尚未重新跑性能认证。

## 进度

| 阶段 | 状态 | 完成标准 |
| --- | --- | --- |
| M0–M5 B200 可运行与 D512 优化 | 完成 | 2K–256K/packed、吞吐/显存、NSYS/NCU详见 `exp/b200_speedup/results.md` |
| M6.1 安全解耦 | 完成 | e331：CPU150；B200 GPU200 + 7×8/8；三代selection无回退 |
| M6.2 历史资产归档 | 待开始 | 按 H100/H200/B200/failed 分类；相关脚本、结果和结论完整保留 |
| M6.3 实验 kernel 分层 | 待开始 | grouped/fused/split/dV/dK/delta 连同复现资产迁移，不删除源码 |
| M6.4 三代硬件复认证 | 待开始 | 每代 correctness + selection + canonical benchmark 独立留证 |
| M7 真实模型验收 | 待开始 | 单卡 Gemma-4 F+B 冒烟通过 |

## 本轮改动

- 此前已把 `attention.py` 末尾H100历史benchmark迁到 `benchmarks/history/h100/attention_embedded_benchmark.py`；
- 本轮新增B200-only dKV head-grid、relaxed/BF16x2策略和精确raw-grid门控；
- raw68–105 q2/w4+BF16x2相对q3吞吐+0.69%–1.7%，峰值约-14%；
- benchmark已记录大梯度、显存和selection；当前docs已同步e331；
- H100/H200代码、配置和实验结论均保留，未删除或重命名tracked文件。

## 清理顺序

1. 先固化当前 dirty/untracked B200 工作；
2. 先移动入口和复现资产，再移动实验 kernel；所有迁移保持可追溯；
3. 每批只改一个轴，先跑 CPU 门禁，再跑目标 GPU 门禁；
4. 任何 H100/H200/B200 production selection 变化都单独审查，不与整理混合。
5. 删除前必须给出四项证据：三代硬件无关、零调用、无复现价值、无唯一结论。

## 风险

- 当前工作区未形成提交，3个`MM`文件的index落后于working tree；
- H200 目前依赖 `sm90` 安全基线，不应误写为已调优；
- 历史脚本含旧环境、硬编码峰值和非同语义 baseline，只能标为历史证据；
- packed紧凑block table仅完成设计，恢复GPU配额后才可验证；
- 真实模型 E2E、H100/H200 实机复认证仍未完成。
