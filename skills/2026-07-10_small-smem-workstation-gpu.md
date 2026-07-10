# 小 Shared Memory GPU（工作站 Blackwell 99KB）适配：_SMALL_SMEM 缩块方案

> 适用场景：在 shared memory/block < 140KB 的 GPU（如 RTX PRO 6000 Blackwell Server / RTX 5090 工作站卡，SM120，99KB）上跑本 kernel。H100 调优的 tile 配置会直接 `OutOfResources: shared memory`。

---

## 问题

- H100 有 **228KB/SM** shared memory（optin），本仓库所有 block config 按它调优。
- 工作站 Blackwell（SM120）只有 **99KB/block optin**（`shared_memory_per_block_optin=101376`）。
- D=512 fwd 默认 (BQ=64, BKV=32, stages=2) 需要 **147456B (144KB)** → 编译期 OutOfResources。
- 只调 `num_stages` 没用：SMEM 大头是 Q/K/V tile 本体（(BQ+2·BKV)×D×2B），不是 pipeline buffer。

## 方案：`_SMALL_SMEM` 自动检测 + 四处 launch 缩块

模块加载时读 `torch.cuda.get_device_properties(0).shared_memory_per_block_optin`，< 140KB 即启用小块配置（数学不变，只缩 tile）：

| launch 点 | H100 配置 | 小 SMEM 配置 |
|---|---|---|
| fwd (`attention_flash_gqa` + `FlashAttnGQAFunction.forward`) | D≥512: (BQ64, BKV32, s2) / D<512: (128, 64, s2) | D≥512: **(32, 16, s1)** / D<512: **(64, 32, s1)** |
| bwd dq | D≥512: (32, 64, w8) / D<512: (64, 64, w4) | D≥512: **(16, 16, w4)** / D<512: **(32, 32, w4)** |
| bwd dkv (packed) | D≥512: (BKV16, BQ64, w4, s2) / D<512: grid 启发式 | D≥512: **(BKV16, BQ16, w4, s1)** / D<512: **(BKV16, BQ32, w4, s1)** |

SMEM 估算口径：fwd ≈ (BQ + 2·BKV)×D×2B + ~16KB 杂项；dkv 的 fp32 dk/dv accumulator 是大头（BKV×D×4B ×2），所以 dkv 优先压 BKV。tl.dot 各维 ≥16 是下限。

## 实测（RTX PRO 6000 Blackwell，bf16，B=2 Hq=16 Hkv=8 N=4096）

| 形状 | 正确性 (vs SDPA) | fwd+bwd 提速 |
|---|---|---|
| D=512 full causal | fwd \|Δ\|max=0.008，dq/dk/dv \|Δ\|max ≤0.031 | **2.6×** (30.9ms vs 79.1ms) |
| D=256 SWA slide=1024 | 同上 | **9.2×** (4.2ms vs 39.1ms) |

E2E（gemma-4-12B DPO，LoRA r64，max_len 8192，grad ckpt，同数据同 seed 逐 step 对齐 34 步）：**1.47×**（SDPA 11m51s vs Triton 8m03s，分段稳定 1.41-1.51×）；step-1 loss 与 SDPA 精确一致（0.8197）。与 NeMo Automodel 在 Gemma4-31B 报的 E2E 1.4-1.5× 相符。

## 额外收益：根治 SDPA math-path OOM

SDPA 对 D=512 无 fused kernel → math path 物化 **fp32 N×N**（N=8192 时 2×16×8192²×4B = **8.0GiB** 瞬时尖峰），在 95GB 卡上顶爆 88GB 基线。本 kernel flash 式 O(N)，尖峰结构性消除。

## 踩坑

1. **短输入测不出**：N < 16 时 `tl.dot` K 维不足直接 CompilationError（`Input shapes should have M>=1, N>=1 and K>=16`）——别用 10-token prompt 冒烟。
2. **包名冲突**：仓库包目录叫 `flash_attn`，与 pip 的 flash-attn 2.8.3 撞名。训练框架（ms-swift）会先 import 真包并缓存 → 本仓库被遮蔽。解法：`importlib.util.spec_from_file_location` 用私有模块名加载（见 `docs/ms-swift-integration.md`）。
3. **多层 bf16 累积 ≠ kernel 错误**：48 层全模型 forward 换 kernel 后末层 logits |Δ| 可到 ~3（误差逐层放大），单层对比 |Δ| ~0.1 才是正确的判据；训练判据用 step-1 loss 是否一致。
