# 架构与关键路径

> attention.py 中所有 kernel 的结构与调用关系。

---

## 文件结构：flash_attn/attention.py

### 1. PyTorch Reference

- `attention(q, k, v)` — 直接调用 `torch.nn.functional.scaled_dot_product_attention`，MHA only
- `attention_gqa_ref(q, k, v)` — GQA reference，expand KV heads 后调用 SDPA

### 2. Naive Triton（MHA only）

- `_attention_naive` kernel — 每个 program 处理一行 output `out[b,h,q_idx,:]`
- 全量加载 KV 到寄存器，O(N) per row，仅适用于短 seq
- Wrapper: `attention_triton(q, k, v)`

### 3. Flash Attention（MHA only）

- `_flash_attn_kernel` — 标准 flash attention，tiled over Q (BLOCK_Q) 和 KV (BLOCK_KV)
- Online softmax: running max + denominator，KV 分块流式处理
- 输入 cast 到 fp32 后做 tl.dot → **走 CUDA core 而非 tensor core**
- Wrapper: `attention_triton_opt(q, k, v)`

### 4. Flash Attention GQA（Gemma4 优化目标）

- `_flash_attn_gqa_kernel` — 支持 GQA + 大 HEAD_DIM
- **GQA 映射**: `kv_h_idx = q_h_idx * N_KV_HEADS // N_Q_HEADS`
- **Score 计算**: D-tiling loop `for d_start in range(0, HEAD_DIM, BLOCK_D)`
  - 当 BLOCK_D=HEAD_DIM 时 loop 退化为单次执行（无 D-tiling），实测最快
  - Q 和 K 以 (*, BLOCK_D) 分块加载，避免同时 materialise 全量 Q+K
- **V 累加**: 全 HEAD_DIM 宽度的 tl.dot，acc 为 (BLOCK_Q, HEAD_DIM) fp32
- **Tensor core**: fp16/bf16 输入直传 tl.dot，不 cast 到 fp32
- **Causal mask** (`IS_CAUSAL` constexpr):
  - KV loop early termination: `kv_end = (q_block_idx + 1) * BLOCK_Q` 而非 `SEQ_LEN`
  - Per-tile mask: `kv_offsets <= q_offsets`，与 `kv_mask` 合并为 `valid`
  - 总 FLOPs 约为 non-causal 一半，speedup 随 N 增大
- **LSE 输出** (`STORE_LSE` constexpr): 存储 `m_i + log(l_i)` 供 backward 使用
- Wrapper: `attention_flash_gqa(q, k, v, causal=False, ...)` (inference, STORE_LSE=False)

### 5. Flash Attention GQA — Backward

- **dQ kernel** (`_flash_attn_gqa_bwd_dq_kernel`):
  - Grid: `(cdiv(N, BQ), B * H_Q)` — 与 forward 相同
  - 每个 program 加载 Q block + dO block，遍历 KV blocks
  - 重算 P = exp(S - LSE)，计算 dS = P * (dO@V^T - D)，累加 dQ += scale * dS @ K
  - Block sizes: BQ=32, BKV=32
- **dKV kernel** (`_flash_attn_gqa_bwd_dkv_kernel`) — **split 设计**:
  - Grid: `(cdiv(N, BKV), B * H_Q)` — per Q head（不是 H_KV）
  - 每个 program 加载 K/V block，遍历该 Q head 的所有 Q blocks
  - 输出 per-Q-head dk/dv → caller 用 view+sum reduce 为 (B, H_KV, N, D)
  - 优势：8x 更多 programs（32 vs 4），短 seq SM 利用率大幅提升
  - Block sizes: BQ=16, BKV=32
- **Delta**: D_i = rowsum(dO_i * O_i)，PyTorch 计算
- **Autograd**: `FlashAttnGQAFunction` + `flash_attn_gqa_train(q, k, v, causal)`

---

## Grid 与 Parallelism

| Kernel | Grid | 每个 Program 处理 |
|--------|------|--------------------|
| naive  | (B*H*N,) | 一行 output |
| flash_opt | (cdiv(N,BQ), B*H) | BQ 行 output |
| flash_gqa | (cdiv(N,BQ), B*H_Q) | BQ 行 output，自动映射到 KV head |

---

## 关键路径（GQA kernel 内循环）

```
# Non-causal: kv_end = SEQ_LEN
# Causal:     kv_end = (q_block_idx + 1) * BLOCK_Q  (early termination)

foreach KV block (stride BLOCK_KV, up to kv_end):
  1. Load K chunk(s), compute scores via tl.dot    ← compute bound
  2. [Causal] Apply mask: kv_offsets <= q_offsets
  3. Online softmax: max, exp, rescale              ← element-wise
  4. Load V block, acc += tl.dot(p, V)              ← memory + compute
```

- Non-causal N=4096: KV loop 迭代 4096/32 = 128 次（all programs same）
- Causal N=4096: first Q block → 2 iters, last Q block → 128 iters（平均 ~64）
- 每次迭代主要成本: 2 个 tl.dot（score + V accum）+ 2 个 block load（K + V）
- Memory: V load dominates = (BLOCK_KV × HEAD_DIM × 2) = 32KB/iter

---

## 输入 Layout

所有 kernel 假设 `(B, H, N, D)` layout。GQA kernel 中 Q 和 K/V 的 H 维度可不同。
Stride 参数完全通用，支持非 contiguous tensor（但 contiguous 最快）。

---

## HuggingFace Integration Layer (`flash_attn/hf_integration.py`)

**目的**：把 Triton kernel 接入 transformers 的 attention interface 注册表，
让任何使用 `ALL_ATTENTION_FUNCTIONS` 的模型（Gemma4、Llama、Mistral 等）可以
通过一个 config flag 切换到我们的 kernel，无需改 modeling_*.py 文件。

### API

| 函数 | 说明 |
|------|------|
| `register_triton_attention(name="triton_gqa")` | 注册到 `ALL_ATTENTION_FUNCTIONS[name]` |
| `triton_gqa_attention(module, q, k, v, attention_mask, *, scaling, sliding_window, ...)` | adapter 本体 |
| `patch_transformers_5_5_4_flash_attn_key()` | transformers 5.5.4 `KeyError: 'flash_attn'` bug 的 workaround |

### Adapter 职责（`triton_gqa_attention`）

transformers 的 attention interface 契约：
```python
def interface(module, q, k, v, attention_mask, *,
              dropout, scaling, softcap=None, sliding_window=None, **kwargs
              ) -> tuple[out, attn_weights]:
    # q: (B, H_Q,  N, D), k/v: (B, H_KV, N, D)
    # Return out shape: (B, N, H_Q, D) — 因为模型外层会 .reshape(B, N, H_Q*D)
```

Adapter 做的 5 件事：

1. **Scaling 协调** — 我们的 kernel 内部已乘 `1/√D`。若 module 传入 `scaling`
   不等于 `1/√D`（Gemma4 传 1.0，因 scale 已 fold 进 q_norm），预乘 q 以消除：
   ```python
   q = q * (scaling / (1/√D))
   ```
2. **Sliding window** — `sliding_window=None` → `slide_size=0` (full causal)；
   `sliding_window=512` → `slide_size=512` (SWA)。每层 config 决定传哪个。
3. **attention_mask 忽略** — HF 传入的 additive mask 通常就是 causal + SWA
   的组合，kernel 内部已根据 `IS_CAUSAL` + `SLIDE_SIZE` 自行构造，无需外部 mask。
4. **输出 transpose** — kernel 输出 (B, H_Q, N, D)，HF 下游期望 (B, N, H_Q, D)，
   adapter 做 `transpose(1, 2).contiguous()`。
5. **不支持的 feature 报错** — `softcap is not None` 或 `dropout != 0.0` →
   `raise NotImplementedError`（宁可显式失败，不要静默错误）。

### 使用方式（用户视角）

```python
from gemma_triton_flash_attn import register_triton_attention, patch_transformers_5_5_4_flash_attn_key
import transformers
patch_transformers_5_5_4_flash_attn_key()         # (仅 transformers 5.5.4 需要)

from transformers import AutoModelForCausalLM
register_triton_attention()                        # 注册 "triton_gqa"
model = AutoModelForCausalLM.from_pretrained(...)
model.config._attn_implementation = "triton_gqa"   # opt in
if hasattr(model.config, "text_config"):
    model.config.text_config._attn_implementation = "triton_gqa"
```

之后模型每次 forward，每个 attention layer 都走我们的 Triton kernel。在 Gemma-4-E2B
上实测：35 层（7 full + 28 sliding）全部路由到 `triton_gqa_attention`。
