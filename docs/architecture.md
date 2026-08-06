# Architecture

## 仓库结构

```
flash_attn/
  attention.py       # production Triton kernels、autograd 与 public API
  registry.py        # implementation/config selection
  hf_integration.py  # HF、varlen 与 Ulysses adapter
  performance.py     # benchmark 结果 schema
  regression.py      # 性能回归规则
  telemetry.py       # selection/fallback 记录

tests/               # 当前 pytest 门禁
benchmarks/          # canonical benchmark 与待归档历史脚本
benchmarks/history/  # 按硬件保存的历史复现入口
exp/b200_speedup/    # B200 D512 原始证据与工作记录
```

## Production kernel

| Symbol | 作用 | 状态 |
| --- | --- | --- |
| `_flash_attn_gqa_kernel` | batched forward | production |
| `_flash_attn_gqa_varlen_fwd_kernel` | varlen forward | production |
| `_flash_attn_gqa_bwd_dq_kernel` | batched dQ，prologue 同时计算 delta | production |
| `_flash_attn_gqa_varlen_bwd_dq_kernel` | varlen dQ，prologue 同时计算 delta | production |
| `_flash_attn_gqa_bwd_dkv_packed_kernel` | batched packed dK/dV | production |
| `_flash_attn_gqa_varlen_bwd_dkv_packed_kernel` | varlen packed dK/dV | production |
| `_delta_kernel` | 旧的独立 delta launch | 历史实验；production 不调用 |
| grouped/fused/split/dV-only/dK-only kernels | 失败或被替代的结构 | 待连同复现脚本归档 |

## 训练数据流

```
public API
  → registry 选择 implementation 与 role config
  → forward 保存 q/k/v/o/lse
  → backward
      → dQ kernel prologue 计算并保存 delta
      → dQ
      → packed dK/dV
```

production backward 不再单独启动 `_delta_kernel`。

## 保留的设计结论

- dQ 每个 Q head 只有一个 owning KV head，不需要 pack-GQA；
- dK/dV 会累积同一GQA group的多个Q head；基础路径在program内展开，
  B200部分区间把Q head映射进grid并用relaxed atomic汇总；两者都不需要expand buffer；
- forward 通过 `STORE_LSE` 区分 inference/training，训练保存 LSE，
  inference 避免额外 HBM 写；
- `torch.autograd.Function` 负责精确保存 q/k/v/o/lse，并让每次调用
  明确经过 registry 选择。

H100 早期 wrapper tuning 结论也保留，但不等同于当前 compile-safe base：

| D | BQ (fwd) | BKV (fwd) | BQ (dQ) | BKV (dKV) |
| ---: | ---: | ---: | ---: | ---: |
| 64/96/128 | 128 | 64 | 64 | 64 |
| 256 | 128 | 64 | 64 | 64 |
| 512 | 64 | 32 | 32 | 32 |

H100 D512 production shape 的历史 sweep 结论为
`BQ=64, BKV=32, warps=8, stages=2`；对应脚本
`benchmarks/d512_prod_tune.py`。这些值保留用于 H100 复认证，不覆盖
H200 安全基线或 B200 override。

## 跨硬件配置层级

```
sm90 compile-safe base
├── H100 batched forward tuned override（历史 tuning，待复认证）
└── H200 使用安全基线（尚无独立 tuned override）

sm100 compile-safe base
└── B200 varlen D512 dKV tuned override（已验证）
```

- H100/H200/B200 的实现与证据都保留；
- 历史性能表、负面实验和软件栈限制也必须保留；只允许改变“当前/历史”标签；
- B200 结果不得覆盖 `sm90`；
- base 负责 compile-safe，product override 必须有对应实机与软件栈证据；
- 未识别产品使用 base，不猜测复用 tuned config。

当前B200 varlen D512 dKV是分层override：BQ16/BKV16/w4/s2为安全回退，
已验证网格使用BKV64/w4/s3；E2B BF16 batch1进一步按raw-grid选择
head-grid q3、q2/w4+BF16x2或q1+BF16x2。完整边界与证据见
`exp/b200_speedup/results.md`。H100性能表属于历史数据，H200尚待重新认证。

## Target model attention shapes

⚠️ Gemma-4 has **two head dims per model** — `head_dim` for sliding layers
and `global_head_dim` for full-attention layers. Reading `head_dim` alone
will mis-shape the full-attention path. Always query both:

```python
from transformers import AutoConfig
tc = AutoConfig.from_pretrained("google/gemma-4-E2B").text_config
# tc.head_dim         → sliding D  (256)
# tc.global_head_dim  → full-attn D (512)
# tc.num_attention_heads        → H_Q
# tc.num_key_value_heads        → H_KV (sliding); fallback for full
# tc.num_global_key_value_heads → H_KV (full only); None ⇒ same as above
# tc.layer_types                → ["sliding_attention" | "full_attention", ...]
# tc.sliding_window             → SWA window
```

Production shapes the kernel actually serves:

| Model | Layer type | H_Q | H_KV | D | slide | layers |
|---|---|----:|----:|----:|----:|----:|
| **Gemma-4-E2B** (35 layers) | sliding | 8 | 1 | 256 | 512 | 28 |
|                              | full    | 8 | 1 | **512** | — | 7 |
| **Gemma-4-E4B** (42 layers) | sliding | 8 | 2 | 256 | 512 | 35 |
|                              | full    | 8 | 2 | **512** | — | 7 |
| **Gemma-4-26B-A4B MoE** (30 layers) | sliding | 16 | 8 | 256 | 1024 | 25 |
|                                      | full    | 16 | 8 | **512** | — | 5 |
| **Gemma-3-12B** (48 layers, ref) | full | 16 | 8 | 256 | — | — |

**Per-shape forward MFU on H100** (single attention call, bf16, causal,
peak = 989 TFLOPS, measured `benchmarks/mfu_sweep.py`, N=8K):

| Shape | Triton | SDPA | speedup |
|---|---:|---:|---:|
| E2B sliding D=256 slide=512    | **25.4%** | 0.7%  | 38× |
| E2B full D=512                 | **17.6%** | 10.6% | 1.7× |
| E4B sliding D=256 slide=512    | **24.5%** | 0.7%  | 37× |
| E4B full D=512                 | **17.6%** | 10.5% | 1.7× |
| MoE sliding D=256 slide=1024   | **32.1%** | 1.4%  | 23× |
| MoE full D=512                 | **18.6%** | 10.8% | 1.7× |
| Gemma3-12B full D=256          | **47.5%** | 26.7% | 1.8× |

D=512 full attention is the lowest-MFU path but is only 5–7 layers per
model. Per-token attention time is still dominated by it (single full
layer is ≈24× a sliding layer at N=8K), so this is where the remaining
optimization headroom lives — see
[`optimization_notes.md`](optimization_notes.md) for what was tried.

## Multimodal routing (Gemma-4 is text + vision + audio)

`google/gemma-4-*` ships three attention implementations — only two route
through our kernel:

| Modality | Class | Pattern | Routes via `attention_interface`? | Our kernel? |
|---|---|---|---|---|
| **Text**   | `Gemma4TextAttention`   | causal (sliding or full), GQA, RMSNorm Q/K/V, scaling=1.0; **MoE 26B-A4B only**: sliding layers also OR-in image-group bidirectional mask (`q_group == kv_group & ≥ 0`) | ✅ yes | ✅ supported (OR-mask via ContextVar — see [Image-group OR-mask](#image-group-or-mask-26b-a4b-only)) |
| **Vision** | `Gemma4VisionAttention` | **bidirectional** (`is_causal=False`), MHA H=12 D=64, 16 layers, no SWA, no softcap | ✅ yes | ✅ supported (verified `tests/test_noncausal_vision_shape.py`) |
| **Audio**  | `Gemma4AudioAttention`  | chunked-local (chunk=12, left=12, right=0) + Shaw relative position bias + tanh softcap (50.0), 12 layers H=8 D=128 | ❌ no — computes attention inline | ❌ unsupported, stays on eager |

Implications for training:

- Setting `model.config._attn_implementation = "triton_gqa"` is a no-op for
  audio (it never queries the registry). No fallback needed — audio just
  uses its own custom path.
- Vision goes through our adapter exactly like text. The adapter sees
  `module.is_causal=False` and forwards `causal=False` to the kernel; no
  vision-specific code path. The non-causal kernel branch
  (`IS_CAUSAL: tl.constexpr = False` in fwd, dQ, and packed dKV bwd) is
  the only thing serving vision.
- Audio's softcap + relative position bias would need a separate kernel
  (or a softcap+rel-pos extension) to fuse — out of scope for now.

## Gemma-4 attention quirks

Beyond the dual `head_dim` / `global_head_dim` above, Gemma-4 layers carry
several non-standard contracts that the integration must respect. Source:
`transformers.models.gemma4.modeling_gemma4.Gemma4TextAttention`.

### Q/K/V RMSNorm before attention

Each layer applies `q_norm`, `k_norm`, and `v_norm` (RMSNorm, `with_scale=False`
for V) to the projection outputs **before** RoPE and the attention call.
Two consequences for kernel work:

- `module.scaling = 1.0` — the standard `1/sqrt(D)` is folded out. The
  adapter ([`flash_attn/hf_integration.py`](../flash_attn/hf_integration.py),
  `triton_gqa_attention`) reconciles this by pre-multiplying Q with
  `scaling / (1/sqrt(D))` since the kernel hard-codes `1/sqrt(D)` internally.
  **Don't change the kernel's internal scale** — the adapter is the only
  layer that sees the module's `scaling`.
- Any future fused-norm-into-attention work needs to fuse three RMSNorms,
  not just one (Q and K are the typical pair, but Gemma also norms V).

### `attention_k_eq_v` on full layers

When `config.attention_k_eq_v` is true, full-attention layers omit the V
projection entirely: `value_states = key_states` (post-`v_norm`). The K
and V tensors handed to the kernel are the **same buffer**. The kernel
already treats them as independent inputs, so this is a no-op for us, but
do not assume `id(k) != id(v)` anywhere downstream.

### Per-layer-type KV head count

`num_global_key_value_heads` overrides `num_key_value_heads` for full
layers when `attention_k_eq_v` is set. When it's `None` (default), full
layers reuse the sliding count. This matters when computing GQA ratios
per-shape: always read the layer-type-specific value, not just
`num_key_value_heads`.

### Cross-layer KV sharing

The last `num_kv_shared_layers` layers (20 in E2B, 18 in E4B, etc.) do
not project their own K/V. Instead they pull from
`shared_kv_states[kv_shared_layer_index]` — the K/V tensors stashed by
the last non-shared layer of the same type (`store_full_length_kv = True`).
The dict is passed as a kwarg through every decoder layer's `forward`.

Two integration consequences:

- **FSDP2 per-layer sharding breaks the dict identity.** FSDP2's pre-forward
  hook does `tree_flatten / tree_unflatten` on kwargs; `dict` is a registered
  pytree container, so unflatten rebuilds a fresh empty dict at each layer
  boundary. KV-shared layers then `KeyError`. The fix is
  [`patch_gemma4_shared_kv_states_for_fsdp2`](../flash_attn/hf_integration.py)
  which substitutes a pytree-opaque `_SharedKVStatesHolder` (object, not
  dict) so identity survives the round-trip. Required *before* any model
  load when training with per-layer FSDP2.
- The kernel itself is unaffected — it sees normal K/V tensors. The sharing
  happens entirely above the attention call.

### Different RoPE per layer type

| Layer type | `rope_theta` | `partial_rotary_factor` |
|---|---:|---:|
| sliding | 10,000 | 1.0 |
| full    | 1,000,000 | 0.25 |

Full layers use NTK-style proportional scaling (`rope_type=proportional`).
RoPE is applied above the kernel — Q and K are already rotated when handed
in — so this is informational, not actionable for kernel work, but anyone
adding RoPE fusion needs to handle both regimes.

### Image-group OR-mask (26B-A4B only)

`Gemma-4-26B-A4B`'s text config sets `use_bidirectional_attention = "vision"` (E2B/E4B leave it `None`). At mask-build time, `create_causal_mask_mapping` adds `or_mask_function = token_type_ids_mask_function(...)` to **`sliding_mask_kwargs` only** — full-attention layers stay pure causal. The OR clause is `(q_group == kv_group) & (q_group >= 0)`, where group ids are derived from `mm_token_type_ids` (1=image / 2=image-padding) by cumsumming new-image-starts.

By the time the mask reaches our adapter it's a `BlockMask` or 4D bool — group identity is baked in and not extractable. Instead we plumb the raw token-type ids through a ContextVar:

```
Gemma4Model.forward(mm_token_type_ids=...)            ── patched wrapper
    └── _compute_image_group_state(mm_token_type_ids)   group_ids, group_lo, group_hi_excl
    └── _image_group_state.set(...)                     ContextVar
            ↓
        triton_gqa_attention                             reads ContextVar
            └── flash_attn_gqa_train(q,k,v, group_ids, group_lo, group_hi_excl, ...)
                    └── _flash_attn_gqa_kernel (HAS_GROUP_IDS=True)
                            ├── widen kv_loop_start = min(swa_lo, image_lo)
                            ├── widen kv_end       = max(causal_end, image_hi)
                            └── valid |= (q_group == kv_group) & (q_group >= 0)
```

The same widen-bounds + OR-mask pattern is mirrored in `_flash_attn_gqa_bwd_dq_kernel` (Q-major) and `_flash_attn_gqa_bwd_dkv_packed_kernel` (KV-major).

`HAS_GROUP_IDS` is a `tl.constexpr` — when False (text-only or non-MoE Gemma), the kernel compiles to its original causal/SWA path with zero overhead. Enable path: `patch_gemma4_image_group_ids_for_kernel()` (no-op on E2B/E4B). Tests: `tests/test_image_group_mask.py`, `tests/gemma4_integration/test_adapter_multimodal.py`.

### Layer type interleaving

`config.layer_types` is the per-layer schedule. E2B/E4B/MoE all follow a
"5 sliding + 1 full" repeating pattern (with the exact ratio varying).
The kernel doesn't care about position — each call is independent — but
this means roughly 1 in 6 attention calls is the slow D=512 path.

## For more detail

- [`integration.md`](integration.md) — adapter internals and registry mechanism
- [`optimization_notes.md`](optimization_notes.md) — what was tried and why
- [`api.md`](api.md) — every public function and its signature
- [`../context/baseline.md`](../context/baseline.md) — full benchmark history
