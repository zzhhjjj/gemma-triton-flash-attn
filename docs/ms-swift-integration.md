# 在 ms-swift 里使用本 kernel（含小 SMEM 工作站卡支持）

> 实测环境：ms-swift 4.3.2 · transformers 5.12 · torch 2.11+cu130 · triton 3.6 ·
> RTX PRO 6000 Blackwell Server (SM120, 99KB SMEM/block) · gemma-4-12B (`gemma4_unified`)

## TL;DR

```bash
swift rlhf \
    --rlhf_type dpo \
    --attn_impl sdpa \
    --custom_register_path /path/to/repo/integrations/ms_swift/triton_attn_patch.py \
    ...其余参数不变
```

一行 `--custom_register_path` 接入，`--attn_impl` 保持 `sdpa`。patch 在进程内把
transformers 的 `ALL_ATTENTION_FUNCTIONS["sdpa"]` 换成本仓库 kernel：Gemma-4 的
40 个 sliding (D=256, SWA) 层和 8 个 global (D=512) 层全部走 Triton flash；
dropout>0 / softcap / 生成期短序列自动回退原生 SDPA。

## 为什么需要它（SDPA 的两个问题）

1. **D=512 无 fused kernel**：SDPA 对 head_dim=512 走 math 路径，物化 **fp32 N×N**
   分数矩阵 —— N=8192、batch 2、16 头时是 **8.0 GiB 的瞬时尖峰**，在 95GB 卡上
   基线 88GB 的训练直接 OOM（我们就是这么炸的）。
2. **sliding 层无 SWA fast path**：滑窗只能靠 4D mask 表达，长序列下 kernel 级
   慢 9 倍以上。

本 kernel 两类层都是 flash 式 **O(N) 内存**：尖峰结构性消除，OOM 根治。

## 实测数字（RTX PRO 6000 Blackwell, bf16）

**kernel 级**（B=2, Hq=16, Hkv=8, N=4096, fwd+bwd）：

| 形状 | Triton | SDPA | 提速 | 正确性 |
|---|---|---|---|---|
| D=512 full causal | 30.9ms | 79.1ms | **2.6×** | fwd \|Δ\|≤0.008, grads \|Δ\|≤0.031 |
| D=256 SWA(1024) | 4.2ms | 39.1ms | **9.2×** | 同上 |

**E2E 训练**（gemma-4-12B DPO, LoRA r64, max_len 8192, grad ckpt, grad_accum 3；
同数据同 seed 逐 step 对齐 34 步）：

| | 到 step34 | 提速 |
|---|---|---|
| SDPA | 11m51s | — |
| Triton | **8m03s** | **1.47×**（分段稳定 1.41–1.51×） |

step-1 loss 与 SDPA 精确一致（0.8197 = 0.8197）；后续发散为 bf16 混沌，属正常。

## 小 SMEM GPU 支持（本 fork 的核心改动）

上游 tile 配置按 H100 (228KB SMEM/block) 调优；工作站 Blackwell 只有 **99KB**，
D=512 fwd 直接 `OutOfResources`（需 144KB）。`flash_attn/attention.py` 现在会在
import 时读 `shared_memory_per_block_optin`，< 140KB 自动切小 tile（`_SMALL_SMEM`）：

| launch 点 | H100 配置 | 小 SMEM 配置 |
|---|---|---|
| fwd ×2 | D≥512: (BQ64,BKV32,s2) | D≥512: (32,16,s1)；D<512: (64,32,s1) |
| bwd dq | D≥512: (32,64,w8) | D≥512: (16,16,w4)；D<512: (32,32,w4) |
| bwd dkv | D≥512: (BKV16,BQ64,s2) | D≥512: (BKV16,BQ16,s1)；D<512: (BKV16,BQ32,s1) |

H100/H200 等大 SMEM 卡不受影响（沿用上游配置）。

## 三个集成要点（踩过的坑）

1. **包名冲突**：本仓库包目录叫 `flash_attn`，与 pip 的 flash-attn wheel 撞名。
   ms-swift 启动时先 import 真 flash-attn 并缓存 → 直接 `import flash_attn` 会被
   遮蔽。`integrations/ms_swift/triton_attn_patch.py` 用 importlib 以私有名
   `_gtfa` 加载本仓库，与 wheel 共存。
2. **attention_mask 传 None**：kernel 自建 causal/滑窗 mask；HF 的 4D mask 会触发
   适配器的 multimodal 守卫。纯文本 + **右 padding** 时传 None 是安全的（pad 在
   真实 token 之后，causal 天然屏蔽；pad 位无 loss）。**左 padding 勿用**。
3. **别用短输入冒烟**：N<16 时 `tl.dot` 直接编译错误；验证用 ≥512 token 的输入。

## 正确性验证方法（换 kernel 必做）

- 单层级：真实激活上逐层对比 kernel vs SDPA，\|Δ\| 应在 bf16 噪声（~0.1）内。
- **不要**用全模型末层 logits 判断 —— 48 层 bf16 误差累积可到 \|Δ\|≈3，是正常现象。
- 训练级：同 seed 跑若干步，**step-1 loss 必须一致**（权重未更新前 loss 只由
  forward 决定）；之后的逐步差异是混沌放大，看轨迹量级即可。
