import torch
import triton
import triton.language as tl
import math


# =====================================================================
# PyTorch reference: scaled dot-product attention
# =====================================================================

def attention(q, k, v):
    """
    Standard scaled dot-product attention.
    q, k, v: (batch, n_heads, seq_len, head_dim)
    Returns:  (batch, n_heads, seq_len, head_dim)
    """
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


# =====================================================================
# Naive Triton attention
#
# Each program computes one output row: out[b, h, q_idx, :].
# Materializes the full attention row in registers — O(N) per row.
# Only works when seq_len fits in BLOCK_SEQ.
# =====================================================================

@triton.jit
def _attention_naive(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    N_HEADS,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    scale,
    BLOCK_SEQ: tl.constexpr,
):
    # Each program handles one row: out[b, h, q_idx, :]
    pid = tl.program_id(0)
    q_idx = pid % SEQ_LEN
    tmp = pid // SEQ_LEN
    h_idx = tmp % N_HEADS
    b_idx = tmp // N_HEADS

    # Base pointers for this (b, h) slice
    q_base = Q_ptr + b_idx * stride_qb + h_idx * stride_qh
    k_base = K_ptr + b_idx * stride_kb + h_idx * stride_kh
    v_base = V_ptr + b_idx * stride_vb + h_idx * stride_vh
    o_base = O_ptr + b_idx * stride_ob + h_idx * stride_oh

    # Load q[q_idx, :] — shape (HEAD_DIM,)
    d_offsets = tl.arange(0, HEAD_DIM)
    q_row = tl.load(q_base + q_idx * stride_qn + d_offsets * stride_qd).to(tl.float32)

    # Load K block: (BLOCK_SEQ, HEAD_DIM)
    kv_offsets = tl.arange(0, BLOCK_SEQ)
    kv_mask = kv_offsets < SEQ_LEN

    k_ptrs = k_base + kv_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd
    k_block = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)

    # scores = K @ q — (BLOCK_SEQ,) via elementwise mul + reduce
    scores = tl.sum(k_block * q_row[None, :], axis=1) * scale

    # Softmax over scores
    row_max = tl.max(scores, axis=0)
    exp_scores = tl.where(kv_mask, tl.exp(scores - row_max), 0.0)
    denom = tl.sum(exp_scores, axis=0)
    attn_weights = exp_scores / denom  # (BLOCK_SEQ,)

    # Load V block: (BLOCK_SEQ, HEAD_DIM)
    v_ptrs = v_base + kv_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd
    v_block = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)

    # out = attn_weights @ V — (HEAD_DIM,)
    out_row = tl.sum(attn_weights[:, None] * v_block, axis=0)

    # Store
    tl.store(o_base + q_idx * stride_on + d_offsets * stride_od, out_row)


def attention_triton(q, k, v):
    B, H, N, D = q.shape
    output = torch.empty_like(q)

    BLOCK_SEQ = triton.next_power_of_2(N)

    grid = (B * H * N,)
    _attention_naive[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        N_HEADS=H,
        SEQ_LEN=N,
        HEAD_DIM=D,
        scale=1.0 / math.sqrt(D),
        BLOCK_SEQ=BLOCK_SEQ,
    )
    return output


# =====================================================================
# Optimized Triton: Flash Attention
#
# Based on Flash Attention (Dao et al., 2022).
# Key ideas:
#   - Each program handles a BLOCK of query rows (not just one)
#   - Tiles over KV in the inner loop — bounded memory, no full attn mat
#   - Online softmax: running max + denominator across KV tiles
#   - When max changes, rescale accumulator: acc *= exp(old_max - new_max)
#   - Fused: score → softmax → V accumulation in one KV pass
#
# Memory per program: O(BLOCK_Q * HEAD_DIM) — independent of seq_len.
# =====================================================================

@triton.jit
def _flash_attn_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    N_HEADS,
    SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    # Grid: (cdiv(SEQ_LEN, BLOCK_Q), B * H)
    q_block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    h_idx = bh_idx % N_HEADS
    b_idx = bh_idx // N_HEADS

    q_base = Q_ptr + b_idx * stride_qb + h_idx * stride_qh
    k_base = K_ptr + b_idx * stride_kb + h_idx * stride_kh
    v_base = V_ptr + b_idx * stride_vb + h_idx * stride_vh
    o_base = O_ptr + b_idx * stride_ob + h_idx * stride_oh

    # Query block offsets
    q_offsets = q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < SEQ_LEN
    d_offsets = tl.arange(0, HEAD_DIM)

    # Load Q block: (BLOCK_Q, HEAD_DIM)
    q_ptrs = q_base + q_offsets[:, None] * stride_qn + d_offsets[None, :] * stride_qd
    q_block = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    # Initialize accumulators
    m_i = tl.full([BLOCK_Q], value=-float("inf"), dtype=tl.float32)  # running max
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)                      # running sum
    acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)            # output accum

    # Iterate over KV blocks
    for kv_start in range(0, SEQ_LEN, BLOCK_KV):
        kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offsets < SEQ_LEN

        # Load K block: (BLOCK_KV, HEAD_DIM)
        k_ptrs = k_base + kv_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd
        k_block = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)

        # Compute scores: Q @ K^T → (BLOCK_Q, BLOCK_KV)
        scores = tl.dot(q_block, tl.trans(k_block)) * scale

        # Mask out invalid KV positions
        scores = tl.where(kv_mask[None, :], scores, -float("inf"))

        # Online softmax update
        block_max = tl.max(scores, axis=1)           # (BLOCK_Q,)
        new_max = tl.maximum(m_i, block_max)

        # Rescale factor for old accumulators
        alpha = tl.exp(m_i - new_max)
        # Attention weights for this block
        p = tl.exp(scores - new_max[:, None])        # (BLOCK_Q, BLOCK_KV)

        # Update running sum and rescale accumulator
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # Load V block: (BLOCK_KV, HEAD_DIM)
        v_ptrs = v_base + kv_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd
        v_block = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)

        # Accumulate: acc += P @ V
        acc += tl.dot(p, v_block)

        m_i = new_max

    # Final normalization
    acc = acc / l_i[:, None]

    # Store output block
    o_ptrs = o_base + q_offsets[:, None] * stride_on + d_offsets[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=q_mask[:, None])


def attention_triton_opt(q, k, v):
    B, H, N, D = q.shape
    output = torch.empty_like(q)

    BLOCK_Q = min(64, triton.next_power_of_2(N))
    BLOCK_KV = min(64, triton.next_power_of_2(N))

    grid = (triton.cdiv(N, BLOCK_Q), B * H)

    _flash_attn_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        N_HEADS=H,
        SEQ_LEN=N,
        HEAD_DIM=D,
        scale=1.0 / math.sqrt(D),
        BLOCK_Q=BLOCK_Q,
        BLOCK_KV=BLOCK_KV,
    )
    return output


# =====================================================================
# Flash Attention with GQA (Grouped Query Attention) support
#
# Extends Flash Attention for configs like Gemma4:
#   - HEAD_DIM=512 (large): Q preloaded once, K/V streamed per tile.
#     Score computation uses D-tiling (BLOCK_D) so full Q and K blocks
#     are never live simultaneously, keeping peak registers manageable.
#   - GQA: Q has N_Q_HEADS, K/V have N_KV_HEADS (ratio must be integer).
#   - Keeps fp16/bf16 inputs to tl.dot for tensor core acceleration.
#   - Sliding Window Attention (SWA): SLIDE_SIZE > 0 limits each query
#     to attend only to the most recent SLIDE_SIZE KV positions.
#     SLIDE_SIZE = 0 disables sliding window (full / causal attention).
# =====================================================================

@triton.jit
def _flash_attn_gqa_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    N_Q_HEADS,
    N_KV_HEADS,
    SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDE_SIZE: tl.constexpr,  # 0 = no sliding window; >0 = window width
    LSE_ptr,
    stride_lseb, stride_lseh, stride_lsen,
    STORE_LSE: tl.constexpr,
    # Image-bidirectional OR-mask (Gemma-4 multimodal sliding layers).
    # When HAS_GROUP_IDS=False the three pointers are unused and constant-folded
    # away by the compiler. Group_ids[b, n] = -1 for non-vision tokens, else
    # a per-batch group index. Group_lo/group_hi_excl are precomputed per-token
    # bounds: for vision tokens, the [first, last+1) span of their image; for
    # non-vision tokens, [n, n+1) (i.e., self-only — they never extend the
    # block's iteration range beyond the standard SWA window).
    GroupIds_ptr,
    GroupLo_ptr,
    GroupHi_ptr,
    stride_gb, stride_gn,
    HAS_GROUP_IDS: tl.constexpr,
):
    # Grid: (cdiv(SEQ_LEN, BLOCK_Q), B * N_Q_HEADS)
    q_block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    q_h_idx = bh_idx % N_Q_HEADS
    b_idx = bh_idx // N_Q_HEADS

    # GQA: map Q head to KV head group
    kv_h_idx = q_h_idx * N_KV_HEADS // N_Q_HEADS

    q_base = Q_ptr + b_idx * stride_qb + q_h_idx * stride_qh
    k_base = K_ptr + b_idx * stride_kb + kv_h_idx * stride_kh
    v_base = V_ptr + b_idx * stride_vb + kv_h_idx * stride_vh
    o_base = O_ptr + b_idx * stride_ob + q_h_idx * stride_oh

    q_offsets = q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < SEQ_LEN
    d_range = tl.arange(0, HEAD_DIM)

    # Image-group state for this Q block (loaded only when HAS_GROUP_IDS).
    if HAS_GROUP_IDS:
        g_base = GroupIds_ptr + b_idx * stride_gb
        glo_base = GroupLo_ptr + b_idx * stride_gb
        ghi_base = GroupHi_ptr + b_idx * stride_gb
        # group_id for each q in the block (-1 for non-vision)
        q_group_ids = tl.load(g_base + q_offsets * stride_gn, mask=q_mask, other=-1)
        # group bounds — for non-vision tokens these collapse to (n, n+1) and
        # don't extend beyond the SWA window.
        q_group_lo = tl.load(glo_base + q_offsets * stride_gn, mask=q_mask, other=SEQ_LEN)
        q_group_hi = tl.load(ghi_base + q_offsets * stride_gn, mask=q_mask, other=0)
        # Per-block bidirectional reach: union of all q's image spans.
        # Non-vision q's contribute (n, n+1) which sit inside the SWA window
        # (since SWA includes self), so they never expand the bounds.
        block_img_lo = tl.min(q_group_lo, axis=0)
        block_img_hi = tl.max(q_group_hi, axis=0)

    # Determine KV iteration range.
    if IS_CAUSAL:
        kv_end = (q_block_idx + 1) * BLOCK_Q
    else:
        kv_end = SEQ_LEN

    if IS_CAUSAL and SLIDE_SIZE > 0:
        kv_min = tl.maximum(0, q_block_idx * BLOCK_Q - SLIDE_SIZE + 1)
        kv_loop_start = (kv_min // BLOCK_KV) * BLOCK_KV
    else:
        kv_loop_start = 0

    # Extend bounds to cover the full image-group span when present.
    # Non-image blocks leave bounds unchanged (block_img_hi <= kv_end already,
    # block_img_lo >= kv_loop_start already). Round to BLOCK_KV.
    if HAS_GROUP_IDS:
        kv_loop_start = tl.minimum(kv_loop_start, (block_img_lo // BLOCK_KV) * BLOCK_KV)
        kv_end = tl.maximum(kv_end, block_img_hi)

    LOG2E: tl.constexpr = 1.4426950408889634
    scale_log2e = scale * LOG2E

    m_i = tl.full([BLOCK_Q], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    # For causal without SWA, split the KV loop into an unmasked off-diagonal
    # phase (queries all see all keys in block) and a masked diagonal phase.
    # FA2/FA3 use the same pattern. At HEAD_DIM=512 the register pressure from
    # the two-loop code makes this a net loss (matmul already dominates, mask
    # overhead is negligible), so skip the split there.
    # When HAS_GROUP_IDS, every tile may need an OR-mask check, so the unmasked
    # phase optimization is unsafe — disable it.
    USE_SPLIT: tl.constexpr = (HEAD_DIM < 512)
    if ((IS_CAUSAL and SLIDE_SIZE == 0) and USE_SPLIT) and (not HAS_GROUP_IDS):
        kv_end_unmasked = (q_block_idx * BLOCK_Q) // BLOCK_KV * BLOCK_KV
    else:
        kv_end_unmasked = kv_loop_start  # skip unmasked phase

    # Phase 1: unmasked KV iterations (off-diagonal, dense) — no mask op.
    for kv_start in range(kv_loop_start, kv_end_unmasked, BLOCK_KV):
        kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
        q_ptrs = q_base + q_offsets[:, None] * stride_qn + d_range[None, :] * stride_qd
        q_chunk = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
        k_ptrs = k_base + kv_offsets[:, None] * stride_kn + d_range[None, :] * stride_kd
        k_chunk = tl.load(k_ptrs)
        scores = tl.dot(q_chunk, tl.trans(k_chunk)) * scale_log2e

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(m_i, block_max)
        alpha = tl.math.exp2(m_i - new_max)
        p = tl.math.exp2(scores - new_max[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = v_base + kv_offsets[:, None] * stride_vn + d_range[None, :] * stride_vd
        v_block = tl.load(v_ptrs)
        acc += tl.dot(p.to(v_block.dtype), v_block)

        m_i = new_max

    # Phase 2: masked KV iterations (diagonal + seq boundary + SWA).
    for kv_start in range(kv_end_unmasked, kv_end, BLOCK_KV):
        kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offsets < SEQ_LEN

        q_ptrs = q_base + q_offsets[:, None] * stride_qn + d_range[None, :] * stride_qd
        q_chunk = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
        k_ptrs = k_base + kv_offsets[:, None] * stride_kn + d_range[None, :] * stride_kd
        k_chunk = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
        scores = tl.dot(q_chunk, tl.trans(k_chunk)) * scale_log2e

        if IS_CAUSAL:
            if SLIDE_SIZE > 0:
                valid = (kv_offsets[None, :] <= q_offsets[:, None]) & \
                        (q_offsets[:, None] - kv_offsets[None, :] < SLIDE_SIZE) & \
                        kv_mask[None, :]
            else:
                valid = (kv_offsets[None, :] <= q_offsets[:, None]) & kv_mask[None, :]
        else:
            valid = kv_mask[None, :]
        # Image-bidirectional OR-mask: queries within a vision span see all
        # keys in the same span (regardless of causal direction or SWA window).
        # Mirrors `token_type_ids_mask_function` in modeling_gemma4.py:1990.
        if HAS_GROUP_IDS:
            kv_group_ids = tl.load(g_base + kv_offsets * stride_gn,
                                   mask=kv_mask, other=-1)
            bidir = (q_group_ids[:, None] == kv_group_ids[None, :]) & \
                    (q_group_ids[:, None] >= 0) & kv_mask[None, :]
            valid = valid | bidir
        scores = tl.where(valid, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(m_i, block_max)
        # SWA may produce all-masked rows (no key visible); HAS_GROUP_IDS
        # implies SWA path and inherits the same NaN-clamp safeguard.
        if SLIDE_SIZE > 0:
            safe_new = tl.maximum(new_max, -1e20)
            alpha = tl.math.exp2(tl.maximum(m_i, -1e20) - safe_new)
            p = tl.math.exp2(scores - safe_new[:, None])
        else:
            alpha = tl.math.exp2(m_i - new_max)
            p = tl.math.exp2(scores - new_max[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = v_base + kv_offsets[:, None] * stride_vn + d_range[None, :] * stride_vd
        v_block = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)
        acc += tl.dot(p.to(v_block.dtype), v_block)

        m_i = new_max

    acc = acc / l_i[:, None]

    o_ptrs = o_base + q_offsets[:, None] * stride_on + d_range[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=q_mask[:, None])

    # Convert LSE back to natural-log domain for bwd consumption: m_i is log2,
    # so lse_natural = m_i * ln(2) + ln(l_i).
    if STORE_LSE:
        LN2: tl.constexpr = 0.6931471805599453
        lse = m_i * LN2 + tl.log(l_i)
        lse_ptrs = LSE_ptr + b_idx * stride_lseb + q_h_idx * stride_lseh + q_offsets * stride_lsen
        tl.store(lse_ptrs, lse, mask=q_mask)


# =====================================================================
# Flash Attention GQA — Varlen (packed / cu_seqlens) forward kernel
#
# Grid: (cdiv(max_seqlen_q, BLOCK_Q), H_Q, B).
#   program_id(0) -> q_block_idx WITHIN the sample
#   program_id(1) -> q_h_idx
#   program_id(2) -> b_idx (sample index; O(1) cu_seqlens lookup)
#
# Tensors are packed: q (total_q, H_Q, D), k/v (total_k, H_KV, D),
# LSE (total_q, H_Q) fp32. Attention is block-diagonal: each sample attends
# only to its own K/V range [cu_seqlens_k[b], cu_seqlens_k[b+1]).
# =====================================================================

@triton.jit
def _flash_attn_gqa_varlen_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    CuSeqlensQ_ptr,  # int32 (B+1,) contiguous
    CuSeqlensK_ptr,  # int32 (B+1,) contiguous
    N_Q_HEADS,
    N_KV_HEADS,
    HEAD_DIM: tl.constexpr,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDE_SIZE: tl.constexpr,  # 0 = no sliding window
    LSE_ptr,
    stride_lset, stride_lseh,
    STORE_LSE: tl.constexpr,
):
    q_block_idx = tl.program_id(0)
    q_h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)

    # Per-sample bounds — O(1) load, no binary search.
    seq_start_q = tl.load(CuSeqlensQ_ptr + b_idx)
    seq_end_q = tl.load(CuSeqlensQ_ptr + b_idx + 1)
    seq_len_q = seq_end_q - seq_start_q
    seq_start_k = tl.load(CuSeqlensK_ptr + b_idx)
    seq_end_k = tl.load(CuSeqlensK_ptr + b_idx + 1)
    seq_len_k = seq_end_k - seq_start_k

    # Early-return for tiles past this sample's tail. Entire-block OOR — plain
    # `q_mask < seq_len_q` only masks within a tile; varlen has whole tiles past
    # the sample end that must not contribute LSE stores or output writes.
    if q_block_idx * BLOCK_Q >= seq_len_q:
        return

    # GQA: map Q head to KV head group
    kv_h_idx = q_h_idx * N_KV_HEADS // N_Q_HEADS

    q_h_base = Q_ptr + q_h_idx * stride_qh
    k_h_base = K_ptr + kv_h_idx * stride_kh
    v_h_base = V_ptr + kv_h_idx * stride_vh
    o_h_base = O_ptr + q_h_idx * stride_oh

    # Seq-local indices for mask arithmetic, global indices for pointers.
    q_local = q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_local < seq_len_q
    q_global = seq_start_q + q_local
    d_range = tl.arange(0, HEAD_DIM)

    # KV iteration range (seq-local).
    if IS_CAUSAL:
        kv_end_local = tl.minimum(seq_len_k, (q_block_idx + 1) * BLOCK_Q)
    else:
        kv_end_local = seq_len_k

    if IS_CAUSAL and SLIDE_SIZE > 0:
        kv_min_local = tl.maximum(0, q_block_idx * BLOCK_Q - SLIDE_SIZE + 1)
        kv_loop_start_local = (kv_min_local // BLOCK_KV) * BLOCK_KV
    else:
        kv_loop_start_local = 0

    LOG2E: tl.constexpr = 1.4426950408889634
    scale_log2e = scale * LOG2E

    m_i = tl.full([BLOCK_Q], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    # Two-phase split optimisation (same as batched kernel; skip at D=512 where
    # the two-loop code's register pressure is a net loss). For varlen the
    # unmasked phase is still safe because the sample's seq boundary is handled
    # by kv_end_local ≤ seq_len_k (phase 1 iterates up to kv_end_unmasked_local
    # which is monotone-rounded below the diagonal anyway).
    USE_SPLIT: tl.constexpr = (HEAD_DIM < 512)
    if (IS_CAUSAL and SLIDE_SIZE == 0) and USE_SPLIT:
        kv_end_unmasked_local = (q_block_idx * BLOCK_Q) // BLOCK_KV * BLOCK_KV
    else:
        kv_end_unmasked_local = kv_loop_start_local

    # Phase 1: unmasked (off-diagonal, dense). All queries see all keys in the tile.
    for kv_start_local in range(kv_loop_start_local, kv_end_unmasked_local, BLOCK_KV):
        kv_local = kv_start_local + tl.arange(0, BLOCK_KV)
        kv_global = seq_start_k + kv_local

        q_ptrs = q_h_base + q_global[:, None] * stride_qt + d_range[None, :] * stride_qd
        q_chunk = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
        k_ptrs = k_h_base + kv_global[:, None] * stride_kt + d_range[None, :] * stride_kd
        k_chunk = tl.load(k_ptrs)
        scores = tl.dot(q_chunk, tl.trans(k_chunk)) * scale_log2e

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(m_i, block_max)
        alpha = tl.math.exp2(m_i - new_max)
        p = tl.math.exp2(scores - new_max[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = v_h_base + kv_global[:, None] * stride_vt + d_range[None, :] * stride_vd
        v_block = tl.load(v_ptrs)
        acc += tl.dot(p.to(v_block.dtype), v_block)

        m_i = new_max

    # Phase 2: masked (diagonal + sample boundary + SWA).
    for kv_start_local in range(kv_end_unmasked_local, kv_end_local, BLOCK_KV):
        kv_local = kv_start_local + tl.arange(0, BLOCK_KV)
        kv_mask = kv_local < seq_len_k
        kv_global = seq_start_k + kv_local

        q_ptrs = q_h_base + q_global[:, None] * stride_qt + d_range[None, :] * stride_qd
        q_chunk = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
        k_ptrs = k_h_base + kv_global[:, None] * stride_kt + d_range[None, :] * stride_kd
        k_chunk = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
        scores = tl.dot(q_chunk, tl.trans(k_chunk)) * scale_log2e

        if IS_CAUSAL:
            if SLIDE_SIZE > 0:
                valid = (kv_local[None, :] <= q_local[:, None]) & \
                        (q_local[:, None] - kv_local[None, :] < SLIDE_SIZE) & \
                        kv_mask[None, :]
            else:
                valid = (kv_local[None, :] <= q_local[:, None]) & kv_mask[None, :]
        else:
            valid = kv_mask[None, :]
        scores = tl.where(valid, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(m_i, block_max)
        if SLIDE_SIZE > 0:
            safe_new = tl.maximum(new_max, -1e20)
            alpha = tl.math.exp2(tl.maximum(m_i, -1e20) - safe_new)
            p = tl.math.exp2(scores - safe_new[:, None])
        else:
            alpha = tl.math.exp2(m_i - new_max)
            p = tl.math.exp2(scores - new_max[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = v_h_base + kv_global[:, None] * stride_vt + d_range[None, :] * stride_vd
        v_block = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)
        acc += tl.dot(p.to(v_block.dtype), v_block)

        m_i = new_max

    acc = acc / l_i[:, None]

    o_ptrs = o_h_base + q_global[:, None] * stride_ot + d_range[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=q_mask[:, None])

    if STORE_LSE:
        LN2: tl.constexpr = 0.6931471805599453
        lse = m_i * LN2 + tl.log(l_i)
        lse_ptrs = LSE_ptr + q_global * stride_lset + q_h_idx * stride_lseh
        tl.store(lse_ptrs, lse, mask=q_mask)


# =====================================================================
# Grouped Flash Attention GQA — multi-head fusion for K/V reuse
#
# Motivation: Gemma-4-E2B has GQA 8:1 (H_Q=8, H_KV=1). Every 8 Q heads read
# the exact same K/V data, but the per-Q-head kernel above launches 8 programs
# per (batch, Q-block) and each program independently streams K/V from HBM.
# L2 does some of the dedup but HBM util is still 85-90% of peak, meaning
# redundant reads are non-trivial.
#
# This kernel launches 1 program per GROUP_SIZE Q heads (all mapping to the
# same KV head). K/V is loaded ONCE, then fed to GROUP_SIZE separate (Q, scores,
# softmax, acc) computations. HBM K/V traffic is reduced by exactly GROUP_SIZE×.
#
# Constraint: GROUP_SIZE must divide N_Q_HEADS/N_KV_HEADS (the GQA ratio).
# =====================================================================

@triton.jit
def _flash_attn_gqa_grouped_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    N_Q_HEADS,
    N_KV_HEADS,
    SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDE_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,  # Q heads processed per program; all share one KV head
    LSE_ptr,
    stride_lseb, stride_lseh, stride_lsen,
    STORE_LSE: tl.constexpr,
):
    # Grid: (cdiv(SEQ_LEN, BLOCK_Q), B * N_Q_HEADS // GROUP_SIZE)
    q_block_idx = tl.program_id(0)
    bg_idx = tl.program_id(1)
    n_groups = N_Q_HEADS // GROUP_SIZE
    group_idx = bg_idx % n_groups
    b_idx = bg_idx // n_groups

    # First Q head in this group; all GROUP_SIZE heads [q_h_base, q_h_base+GROUP_SIZE)
    # share a single KV head (GROUP_SIZE <= GQA ratio enforced by caller).
    q_h_base = group_idx * GROUP_SIZE
    kv_h_idx = q_h_base * N_KV_HEADS // N_Q_HEADS

    k_base = K_ptr + b_idx * stride_kb + kv_h_idx * stride_kh
    v_base = V_ptr + b_idx * stride_vb + kv_h_idx * stride_vh

    q_offsets = q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < SEQ_LEN
    d_range = tl.arange(0, HEAD_DIM)

    # KV iteration bounds (same logic as single-head kernel)
    if IS_CAUSAL:
        kv_end = (q_block_idx + 1) * BLOCK_Q
    else:
        kv_end = SEQ_LEN

    if IS_CAUSAL and SLIDE_SIZE > 0:
        kv_min = tl.maximum(0, q_block_idx * BLOCK_Q - SLIDE_SIZE + 1)
        kv_loop_start = (kv_min // BLOCK_KV) * BLOCK_KV
    else:
        kv_loop_start = 0

    # Per-head state: GROUP_SIZE separate (Q, m, l, acc). Triton's `tl.static_range`
    # unrolls at compile time, so we use immutable tuple rebuilding (Triton doesn't
    # support list __setitem__). Each element becomes a distinct SSA value.
    q_blocks = ()
    m_is = ()
    l_is = ()
    accs = ()
    for g in tl.static_range(GROUP_SIZE):
        q_h_idx = q_h_base + g
        q_base_g = Q_ptr + b_idx * stride_qb + q_h_idx * stride_qh
        q_ptrs = q_base_g + q_offsets[:, None] * stride_qn + d_range[None, :] * stride_qd
        qb = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
        q_blocks = q_blocks + (qb,)
        m_is = m_is + (tl.full([BLOCK_Q], value=-float("inf"), dtype=tl.float32),)
        l_is = l_is + (tl.zeros([BLOCK_Q], dtype=tl.float32),)
        accs = accs + (tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32),)

    # KV loop — K/V loaded ONCE per iteration, used by all GROUP_SIZE heads.
    for kv_start in range(kv_loop_start, kv_end, BLOCK_KV):
        kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offsets < SEQ_LEN

        # Load K, V (shared across the whole group — this is the savings)
        k_ptrs = k_base + kv_offsets[:, None] * stride_kn + d_range[None, :] * stride_kd
        k_block = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
        v_ptrs = v_base + kv_offsets[:, None] * stride_vn + d_range[None, :] * stride_vd
        v_block = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

        # Mask: sequence boundary + optional causal + optional sliding window
        if IS_CAUSAL:
            if SLIDE_SIZE > 0:
                valid = (kv_offsets[None, :] <= q_offsets[:, None]) & \
                        (q_offsets[:, None] - kv_offsets[None, :] < SLIDE_SIZE) & \
                        kv_mask[None, :]
            else:
                valid = (kv_offsets[None, :] <= q_offsets[:, None]) & kv_mask[None, :]
        else:
            valid = kv_mask[None, :]

        # Per-head compute (unrolled). Rebuild state tuples because individual
        # elements can't be reassigned; we create new tuples per iteration.
        new_m_is = ()
        new_l_is = ()
        new_accs = ()
        for g in tl.static_range(GROUP_SIZE):
            scores = tl.dot(q_blocks[g], tl.trans(k_block)) * scale
            scores = tl.where(valid, scores, -float("inf"))

            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(m_is[g], block_max)
            if SLIDE_SIZE > 0:
                alpha = tl.exp(tl.maximum(m_is[g], -1e20) - tl.maximum(new_max, -1e20))
                p = tl.exp(scores - tl.maximum(new_max, -1e20)[:, None])
            else:
                alpha = tl.exp(m_is[g] - new_max)
                p = tl.exp(scores - new_max[:, None])

            new_l = l_is[g] * alpha + tl.sum(p, axis=1)
            new_acc = accs[g] * alpha[:, None] + tl.dot(p.to(v_block.dtype), v_block)

            new_m_is = new_m_is + (new_max,)
            new_l_is = new_l_is + (new_l,)
            new_accs = new_accs + (new_acc,)
        m_is = new_m_is
        l_is = new_l_is
        accs = new_accs

    # Normalize and store — one output per Q head
    for g in tl.static_range(GROUP_SIZE):
        q_h_idx = q_h_base + g
        o_base_g = O_ptr + b_idx * stride_ob + q_h_idx * stride_oh
        out = accs[g] / l_is[g][:, None]
        o_ptrs = o_base_g + q_offsets[:, None] * stride_on + d_range[None, :] * stride_od
        tl.store(o_ptrs, out, mask=q_mask[:, None])

        if STORE_LSE:
            lse = m_is[g] + tl.log(l_is[g])
            lse_ptrs = LSE_ptr + b_idx * stride_lseb + q_h_idx * stride_lseh + q_offsets * stride_lsen
            tl.store(lse_ptrs, lse, mask=q_mask)


def attention_flash_gqa(q, k, v, causal=False, slide_size=0,
                        group_ids=None, group_lo=None, group_hi_excl=None,
                        BLOCK_Q=None, BLOCK_KV=None,
                        BLOCK_D=None, num_warps=None, num_stages=None):
    """
    Flash Attention with GQA support and optional sliding window attention.
    q: (B, N_Q_HEADS, N, D)
    k: (B, N_KV_HEADS, N, D)
    v: (B, N_KV_HEADS, N, D)
    causal:     if True, apply causal mask (q_i attends only to k_j where j <= i)
    slide_size: sliding window size (0 = disabled). When > 0 and causal=True,
                q_i attends to k_j where max(0, i - slide_size + 1) <= j <= i.
    group_ids:  optional (B, N) int32 vision-group ids for the image-
                bidirectional OR-mask (Gemma-4 multimodal). -1 = non-vision.
                Must be passed together with `group_lo` and `group_hi_excl`.
    group_lo / group_hi_excl: (B, N) int32 — per-token bidirectional reach
                used to extend the kernel's KV iteration bounds when an image
                spans beyond the SWA window.
    """
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    output = torch.empty_like(q)

    # Normalize: when window covers the whole sequence, SWA degenerates to
    # full causal. Take the full-causal path (skips window mask + NaN clamp).
    if slide_size > 0 and slide_size >= N:
        slide_size = 0

    # Tuned defaults (sweeps in context/baseline.md). Block sizes scale inversely
    # with HEAD_DIM to keep shared memory usage ~constant at ~128KB.
    #   D=512 (Gemma4 full attn):     (BQ=64,  BKV=32) — shared memory constrained
    #   D=256 (Gemma4 sliding attn):  (BQ=128, BKV=64) — more headroom, larger blocks win
    #   D<256:                        (BQ=128, BKV=64) — same as D=256
    # num_warps=8 for D>=256 (large tiles need many warps to hide latency).
    if BLOCK_Q is None:
        BLOCK_Q = 64 if D >= 512 else 128
    if BLOCK_KV is None:
        BLOCK_KV = 32 if D >= 512 else 64
    if BLOCK_D is None:
        BLOCK_D = D  # no D-tiling: single tl.dot per KV tile
    if num_warps is None:
        num_warps = 8 if D >= 256 else 4
    if num_stages is None:
        num_stages = 2

    BLOCK_Q = min(BLOCK_Q, triton.next_power_of_2(N))
    BLOCK_KV = min(BLOCK_KV, triton.next_power_of_2(N))

    grid = (triton.cdiv(N, BLOCK_Q), B * H_Q)

    has_group_ids = group_ids is not None
    if has_group_ids:
        assert group_lo is not None and group_hi_excl is not None, \
            "group_ids requires both group_lo and group_hi_excl"
        g_strides = (group_ids.stride(0), group_ids.stride(1))
    else:
        g_strides = (0, 0)

    _flash_attn_gqa_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        N_Q_HEADS=H_Q,
        N_KV_HEADS=H_KV,
        SEQ_LEN=N,
        HEAD_DIM=D,
        scale=1.0 / math.sqrt(D),
        BLOCK_Q=BLOCK_Q,
        BLOCK_KV=BLOCK_KV,
        BLOCK_D=BLOCK_D,
        IS_CAUSAL=causal,
        SLIDE_SIZE=slide_size,
        LSE_ptr=None, stride_lseb=0, stride_lseh=0, stride_lsen=0,
        STORE_LSE=False,
        GroupIds_ptr=group_ids, GroupLo_ptr=group_lo, GroupHi_ptr=group_hi_excl,
        stride_gb=g_strides[0], stride_gn=g_strides[1],
        HAS_GROUP_IDS=has_group_ids,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


# =====================================================================
# Reference GQA attention (expands KV heads to match Q heads)
# =====================================================================

def attention_gqa_ref(q, k, v, causal=False):
    """PyTorch SDPA reference for GQA. Expands KV heads to match Q heads."""
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    if H_Q != H_KV:
        ratio = H_Q // H_KV
        k = k.repeat_interleave(ratio, dim=1)
        v = v.repeat_interleave(ratio, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)


def attention_swa_ref(q, k, v, slide_size):
    """
    PyTorch reference for causal sliding window attention (GQA).
    Builds an explicit (N, N) boolean mask and uses SDPA.
    Only suitable for small N (reference / correctness checks).
    """
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    if H_Q != H_KV:
        ratio = H_Q // H_KV
        k = k.repeat_interleave(ratio, dim=1)
        v = v.repeat_interleave(ratio, dim=1)
    idx = torch.arange(N, device=q.device)
    # q_i attends to k_j iff j <= i (causal) AND i - j < slide_size (window)
    causal_mask = idx[None, :] <= idx[:, None]          # (N, N)
    window_mask = (idx[:, None] - idx[None, :]) < slide_size  # (N, N)
    attend = causal_mask & window_mask                  # True = attend
    # SDPA float mask: 0.0 = attend, -inf = mask out
    float_mask = torch.where(attend, 0.0, float('-inf')).to(q.dtype)
    float_mask = float_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=float_mask)


# =====================================================================
# Varlen reference + pack helpers (used by tests and users)
#
# Packed layout:
#   q: (total_q, H_Q,  D)
#   k: (total_k, H_KV, D)
#   v: (total_k, H_KV, D)
#   cu_seqlens_q / cu_seqlens_k: int32, shape (B+1,) — cumulative sample boundaries
#
# Each sample is a contiguous slice of rows: sample b owns q[cu_seqlens_q[b]:cu_seqlens_q[b+1]].
# Attention is block-diagonal: tokens in sample b only attend to tokens in sample b.
# =====================================================================

def attention_gqa_varlen_ref(q, k, v, cu_seqlens_q, cu_seqlens_k,
                             max_seqlen_q, max_seqlen_k,
                             causal=False, window_size=0):
    """Per-sample SDPA reference for varlen attention.

    Reuses `attention_gqa_ref` (no SWA) and `attention_swa_ref` (SWA) per sample,
    so the numerical reference matches what the batched path already uses.

    Args:
        q: (total_q, H_Q, D)
        k/v: (total_k, H_KV, D)
        cu_seqlens_q / cu_seqlens_k: int32 (B+1,) on the same device as q
        max_seqlen_q / max_seqlen_k: ints (hints, unused here; kernel needs them)
        causal: bool
        window_size: 0 = no SWA; >0 = SWA window (equivalent to slide_size)

    Returns:
        (total_q, H_Q, D) packed output, same dtype as q.
    """
    total_q, H_Q, D = q.shape
    _, H_KV, _ = k.shape
    out = torch.empty_like(q)
    B = cu_seqlens_q.numel() - 1
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    for b in range(B):
        qs, qe = cu_q[b], cu_q[b + 1]
        ks, ke = cu_k[b], cu_k[b + 1]
        if qe == qs:
            continue
        # Reshape slice to (1, H, L, D) for the batched reference helpers.
        qi = q[qs:qe].transpose(0, 1).unsqueeze(0).contiguous()
        ki = k[ks:ke].transpose(0, 1).unsqueeze(0).contiguous()
        vi = v[ks:ke].transpose(0, 1).unsqueeze(0).contiguous()
        if window_size > 0 and causal and (ke - ks) > window_size:
            oi = attention_swa_ref(qi, ki, vi, slide_size=window_size)
        else:
            oi = attention_gqa_ref(qi, ki, vi, causal=causal)
        out[qs:qe] = oi.squeeze(0).transpose(0, 1).contiguous()
    return out


def pack_batched_to_varlen(q_bhnd, k_bhnd, v_bhnd, seqlens):
    """Pack a (B, H, N, D) batched tensor into (total, H, D) plus cu_seqlens.

    Only the first `seqlens[b]` rows of each batch element are kept.

    Args:
        q_bhnd: (B, H_Q,  N, D)
        k_bhnd: (B, H_KV, N, D)
        v_bhnd: (B, H_KV, N, D)
        seqlens: int tensor shape (B,), actual length of each sample, each ≤ N.

    Returns:
        q_packed (total_q, H_Q, D), k_packed / v_packed (total_k, H_KV, D),
        cu_seqlens (int32, B+1) on the same device as q_bhnd.

    cu_seqlens is shared between Q and K because the batched path has N_q == N_k
    per sample; callers that need different Q/K packing should build their own.
    """
    B, H_Q, N, D = q_bhnd.shape
    _, H_KV, _, _ = k_bhnd.shape
    seqlens = seqlens.to(dtype=torch.int64)
    total = int(seqlens.sum().item())
    device = q_bhnd.device
    q_packed = torch.empty(total, H_Q, D, dtype=q_bhnd.dtype, device=device)
    k_packed = torch.empty(total, H_KV, D, dtype=k_bhnd.dtype, device=device)
    v_packed = torch.empty(total, H_KV, D, dtype=v_bhnd.dtype, device=device)
    offset = 0
    for b in range(B):
        L = int(seqlens[b].item())
        # (H, L, D) -> (L, H, D)
        q_packed[offset:offset + L] = q_bhnd[b, :, :L, :].transpose(0, 1).contiguous()
        k_packed[offset:offset + L] = k_bhnd[b, :, :L, :].transpose(0, 1).contiguous()
        v_packed[offset:offset + L] = v_bhnd[b, :, :L, :].transpose(0, 1).contiguous()
        offset += L
    cu = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu[1:] = seqlens.to(dtype=torch.int32).cumsum(0)
    return q_packed, k_packed, v_packed, cu


def unpack_varlen_to_batched(x_packed, cu_seqlens, max_seqlen, n_heads):
    """Invert pack_batched_to_varlen for a single tensor.

    Args:
        x_packed: (total, H, D)
        cu_seqlens: int32 (B+1,)
        max_seqlen: int — pad each sample to this length along the seq axis
        n_heads: H

    Returns:
        (B, H, max_seqlen, D) with zeros past each sample's actual length.
    """
    total, H, D = x_packed.shape
    assert H == n_heads
    B = cu_seqlens.numel() - 1
    out = torch.zeros(B, H, max_seqlen, D, dtype=x_packed.dtype, device=x_packed.device)
    cu = cu_seqlens.tolist()
    for b in range(B):
        s, e = cu[b], cu[b + 1]
        L = e - s
        # (L, H, D) -> (H, L, D)
        out[b, :, :L, :] = x_packed[s:e].transpose(0, 1).contiguous()
    return out


# =====================================================================
# Flash Attention GQA — Backward Pass
#
# Two kernels:
#   1. dQ kernel: grid over Q blocks, iterate KV blocks, accumulate dQ
#   2. dKV kernel: grid over KV blocks, iterate Q blocks (all GQA heads),
#      accumulate dK and dV
#
# Both recompute attention weights P = exp(S - LSE) from Q, K, and
# the logsumexp (LSE) saved during forward.
# =====================================================================

@triton.jit
def _delta_kernel(
    dO_ptr, O_ptr, Delta_ptr,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_deltab, stride_deltah, stride_deltan,
    N_HEADS, SEQ_LEN,
    HEAD_DIM: tl.constexpr,
):
    # Grid: (SEQ_LEN, B * N_HEADS) — one program per (b, h, n)
    n_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    h_idx = bh_idx % N_HEADS
    b_idx = bh_idx // N_HEADS

    if n_idx >= SEQ_LEN:
        return

    d_offs = tl.arange(0, HEAD_DIM)
    do_base = dO_ptr + b_idx * stride_dob + h_idx * stride_doh + n_idx * stride_don
    o_base = O_ptr + b_idx * stride_ob + h_idx * stride_oh + n_idx * stride_on
    delta_ptr = Delta_ptr + b_idx * stride_deltab + h_idx * stride_deltah + n_idx * stride_deltan

    do_v = tl.load(do_base + d_offs * stride_dod).to(tl.float32)
    o_v = tl.load(o_base + d_offs * stride_od).to(tl.float32)
    delta = tl.sum(do_v * o_v)
    tl.store(delta_ptr, delta)


@triton.jit
def _flash_attn_gqa_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, dQ_ptr,
    LSE_ptr, Delta_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_dqb, stride_dqh, stride_dqn, stride_dqd,
    stride_lseb, stride_lseh, stride_lsen,
    stride_db, stride_dh, stride_dn,
    N_Q_HEADS, N_KV_HEADS, SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDE_SIZE: tl.constexpr,  # 0 = no sliding window
    STORE_DELTA: tl.constexpr,  # True: fuse delta = rowsum(dO * O) into dQ prologue
    # Image-bidirectional OR-mask (mirror of fwd kernel — see _flash_attn_gqa_kernel).
    GroupIds_ptr,
    GroupLo_ptr,
    GroupHi_ptr,
    stride_gb, stride_gn,
    HAS_GROUP_IDS: tl.constexpr,
):
    q_block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    q_h_idx = bh_idx % N_Q_HEADS
    b_idx = bh_idx // N_Q_HEADS
    kv_h_idx = q_h_idx * N_KV_HEADS // N_Q_HEADS

    q_base = Q_ptr + b_idx * stride_qb + q_h_idx * stride_qh
    k_base = K_ptr + b_idx * stride_kb + kv_h_idx * stride_kh
    v_base = V_ptr + b_idx * stride_vb + kv_h_idx * stride_vh
    do_base = dO_ptr + b_idx * stride_dob + q_h_idx * stride_doh
    dq_base = dQ_ptr + b_idx * stride_dqb + q_h_idx * stride_dqh

    q_offsets = q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < SEQ_LEN
    d_range = tl.arange(0, HEAD_DIM)

    # Load Q, dO blocks (persist across KV iterations)
    q_ptrs = q_base + q_offsets[:, None] * stride_qn + d_range[None, :] * stride_qd
    q_block = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
    do_ptrs = do_base + q_offsets[:, None] * stride_don + d_range[None, :] * stride_dod
    do_block = tl.load(do_ptrs, mask=q_mask[:, None], other=0.0)

    # Load LSE and Delta for this Q block
    lse_ptrs = LSE_ptr + b_idx * stride_lseb + q_h_idx * stride_lseh + q_offsets * stride_lsen
    lse = tl.load(lse_ptrs, mask=q_mask, other=0.0)
    delta_ptrs = Delta_ptr + b_idx * stride_db + q_h_idx * stride_dh + q_offsets * stride_dn
    if STORE_DELTA:
        # Fused delta: compute delta = rowsum(dO * O) in prologue, store for dKV kernel.
        # dO is already loaded; O adds ~BQ×D bytes per program (new load), but we save
        # an entire _delta_kernel launch which is otherwise launch-bound at short N
        # (32K programs of 512-byte work). Same do bandwidth, same o bandwidth, -1 launch.
        o_base = O_ptr + b_idx * stride_ob + q_h_idx * stride_oh
        o_ptrs = o_base + q_offsets[:, None] * stride_on + d_range[None, :] * stride_od
        o_block = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0)
        delta = tl.sum(do_block.to(tl.float32) * o_block.to(tl.float32), axis=1)
        tl.store(delta_ptrs, delta, mask=q_mask)
    else:
        delta = tl.load(delta_ptrs, mask=q_mask, other=0.0)

    # Determine KV iteration range (mirrors forward kernel logic)
    if IS_CAUSAL:
        kv_end = (q_block_idx + 1) * BLOCK_Q
    else:
        kv_end = SEQ_LEN

    if IS_CAUSAL and SLIDE_SIZE > 0:
        kv_min = tl.maximum(0, q_block_idx * BLOCK_Q - SLIDE_SIZE + 1)
        kv_loop_start = (kv_min // BLOCK_KV) * BLOCK_KV
    else:
        kv_loop_start = 0

    # Image-group state and bound expansion (mirrors fwd kernel).
    if HAS_GROUP_IDS:
        g_base = GroupIds_ptr + b_idx * stride_gb
        glo_base = GroupLo_ptr + b_idx * stride_gb
        ghi_base = GroupHi_ptr + b_idx * stride_gb
        q_group_ids = tl.load(g_base + q_offsets * stride_gn, mask=q_mask, other=-1)
        q_group_lo = tl.load(glo_base + q_offsets * stride_gn, mask=q_mask, other=SEQ_LEN)
        q_group_hi = tl.load(ghi_base + q_offsets * stride_gn, mask=q_mask, other=0)
        block_img_lo = tl.min(q_group_lo, axis=0)
        block_img_hi = tl.max(q_group_hi, axis=0)
        kv_loop_start = tl.minimum(kv_loop_start, (block_img_lo // BLOCK_KV) * BLOCK_KV)
        kv_end = tl.maximum(kv_end, block_img_hi)

    dq_acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    # Convert LSE to log2 domain once so we can use tl.math.exp2 inside the loop.
    LOG2E: tl.constexpr = 1.4426950408889634
    scale_log2e = scale * LOG2E
    lse_log2 = lse * LOG2E

    for kv_start in range(kv_loop_start, kv_end, BLOCK_KV):
        kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offsets < SEQ_LEN

        k_ptrs = k_base + kv_offsets[:, None] * stride_kn + d_range[None, :] * stride_kd
        k_block = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
        v_ptrs = v_base + kv_offsets[:, None] * stride_vn + d_range[None, :] * stride_vd
        v_block = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

        scores = tl.dot(q_block, tl.trans(k_block)).to(tl.float32) * scale_log2e
        if IS_CAUSAL:
            if SLIDE_SIZE > 0:
                valid = (kv_offsets[None, :] <= q_offsets[:, None]) & \
                        (q_offsets[:, None] - kv_offsets[None, :] < SLIDE_SIZE) & \
                        kv_mask[None, :]
            else:
                valid = (kv_offsets[None, :] <= q_offsets[:, None]) & kv_mask[None, :]
        else:
            valid = kv_mask[None, :]
        if HAS_GROUP_IDS:
            kv_group_ids = tl.load(g_base + kv_offsets * stride_gn,
                                   mask=kv_mask, other=-1)
            bidir = (q_group_ids[:, None] == kv_group_ids[None, :]) & \
                    (q_group_ids[:, None] >= 0) & kv_mask[None, :]
            valid = valid | bidir
        scores = tl.where(valid, scores, -float("inf"))
        p = tl.math.exp2(scores - lse_log2[:, None])

        dp = tl.dot(do_block, tl.trans(v_block)).to(tl.float32)
        ds = p * (dp - delta[:, None])
        ds = tl.where(valid, ds, 0.0)

        dq_acc += tl.dot(ds.to(k_block.dtype), k_block).to(tl.float32)

    dq_acc *= scale

    # Store dQ
    dq_ptrs = dq_base + q_offsets[:, None] * stride_dqn + d_range[None, :] * stride_dqd
    tl.store(dq_ptrs, dq_acc.to(q_block.dtype), mask=q_mask[:, None])


# =====================================================================
# Flash Attention GQA — Varlen backward dQ kernel
#
# Grid: (cdiv(max_seqlen_q, BQ), H_Q, B). Mirrors varlen fwd grid so the
# (q_block_idx, q_h_idx, b_idx) decomposition stays identical and the same
# per-sample early-return applies.
#
# Keeps STORE_DELTA fusion: prologue computes delta = rowsum(dO * O) into
# packed delta (total_q, H_Q), saving a separate delta kernel launch.
# =====================================================================

@triton.jit
def _flash_attn_gqa_varlen_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, dQ_ptr,
    LSE_ptr, Delta_ptr,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_dot, stride_doh, stride_dod,
    stride_ot, stride_oh, stride_od,
    stride_dqt, stride_dqh, stride_dqd,
    stride_lset, stride_lseh,
    stride_dt, stride_dh,
    CuSeqlensQ_ptr,  # int32 (B+1,)
    CuSeqlensK_ptr,  # int32 (B+1,)
    N_Q_HEADS, N_KV_HEADS,
    HEAD_DIM: tl.constexpr,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDE_SIZE: tl.constexpr,
    STORE_DELTA: tl.constexpr,
):
    q_block_idx = tl.program_id(0)
    q_h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)

    # Per-sample bounds.
    seq_start_q = tl.load(CuSeqlensQ_ptr + b_idx)
    seq_end_q = tl.load(CuSeqlensQ_ptr + b_idx + 1)
    seq_len_q = seq_end_q - seq_start_q
    seq_start_k = tl.load(CuSeqlensK_ptr + b_idx)
    seq_end_k = tl.load(CuSeqlensK_ptr + b_idx + 1)
    seq_len_k = seq_end_k - seq_start_k

    if q_block_idx * BLOCK_Q >= seq_len_q:
        return

    kv_h_idx = q_h_idx * N_KV_HEADS // N_Q_HEADS

    q_h_base = Q_ptr + q_h_idx * stride_qh
    k_h_base = K_ptr + kv_h_idx * stride_kh
    v_h_base = V_ptr + kv_h_idx * stride_vh
    do_h_base = dO_ptr + q_h_idx * stride_doh
    dq_h_base = dQ_ptr + q_h_idx * stride_dqh

    q_local = q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_local < seq_len_q
    q_global = seq_start_q + q_local
    d_range = tl.arange(0, HEAD_DIM)

    # Load Q, dO blocks (persist across KV loop).
    q_ptrs = q_h_base + q_global[:, None] * stride_qt + d_range[None, :] * stride_qd
    q_block = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
    do_ptrs = do_h_base + q_global[:, None] * stride_dot + d_range[None, :] * stride_dod
    do_block = tl.load(do_ptrs, mask=q_mask[:, None], other=0.0)

    # Load LSE (packed layout) and handle Delta (possibly fused).
    lse_ptrs = LSE_ptr + q_global * stride_lset + q_h_idx * stride_lseh
    lse = tl.load(lse_ptrs, mask=q_mask, other=0.0)
    delta_ptrs = Delta_ptr + q_global * stride_dt + q_h_idx * stride_dh
    if STORE_DELTA:
        # Prologue fusion: compute delta = rowsum(dO * O) and store for dKV.
        o_h_base = O_ptr + q_h_idx * stride_oh
        o_ptrs = o_h_base + q_global[:, None] * stride_ot + d_range[None, :] * stride_od
        o_block = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0)
        delta = tl.sum(do_block.to(tl.float32) * o_block.to(tl.float32), axis=1)
        tl.store(delta_ptrs, delta, mask=q_mask)
    else:
        delta = tl.load(delta_ptrs, mask=q_mask, other=0.0)

    # KV iteration bounds (seq-local; mirrors varlen fwd).
    if IS_CAUSAL:
        kv_end_local = tl.minimum(seq_len_k, (q_block_idx + 1) * BLOCK_Q)
    else:
        kv_end_local = seq_len_k
    if IS_CAUSAL and SLIDE_SIZE > 0:
        kv_min_local = tl.maximum(0, q_block_idx * BLOCK_Q - SLIDE_SIZE + 1)
        kv_loop_start_local = (kv_min_local // BLOCK_KV) * BLOCK_KV
    else:
        kv_loop_start_local = 0

    dq_acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    LOG2E: tl.constexpr = 1.4426950408889634
    scale_log2e = scale * LOG2E
    lse_log2 = lse * LOG2E

    for kv_start_local in range(kv_loop_start_local, kv_end_local, BLOCK_KV):
        kv_local = kv_start_local + tl.arange(0, BLOCK_KV)
        kv_mask = kv_local < seq_len_k
        kv_global = seq_start_k + kv_local

        k_ptrs = k_h_base + kv_global[:, None] * stride_kt + d_range[None, :] * stride_kd
        k_block = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
        v_ptrs = v_h_base + kv_global[:, None] * stride_vt + d_range[None, :] * stride_vd
        v_block = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

        scores = tl.dot(q_block, tl.trans(k_block)).to(tl.float32) * scale_log2e
        if IS_CAUSAL:
            if SLIDE_SIZE > 0:
                valid = (kv_local[None, :] <= q_local[:, None]) & \
                        (q_local[:, None] - kv_local[None, :] < SLIDE_SIZE) & \
                        kv_mask[None, :]
            else:
                valid = (kv_local[None, :] <= q_local[:, None]) & kv_mask[None, :]
        else:
            valid = kv_mask[None, :]
        scores = tl.where(valid, scores, -float("inf"))
        p = tl.math.exp2(scores - lse_log2[:, None])

        dp = tl.dot(do_block, tl.trans(v_block)).to(tl.float32)
        ds = p * (dp - delta[:, None])
        ds = tl.where(valid, ds, 0.0)

        dq_acc += tl.dot(ds.to(k_block.dtype), k_block).to(tl.float32)

    dq_acc *= scale

    dq_ptrs = dq_h_base + q_global[:, None] * stride_dqt + d_range[None, :] * stride_dqd
    tl.store(dq_ptrs, dq_acc.to(q_block.dtype), mask=q_mask[:, None])


# =====================================================================
# Fused backward: single kernel computes dQ, dK, dV together.
#
# Replaces (dQ_kernel + dKV_kernel + expand_reduce). Benefits:
#  - Eliminates redundant score recomputation (Q@K^T, dO@V^T done once)
#  - No expand buffer for dK/dV (saves GQA_RATIO× memory)
#  - No reduce step (saves ~50μs per bwd pass)
#  - 1 kernel launch instead of 2 (saves ~15μs)
#
# Cost: atomic_add contention on dK/dV. For GQA ratio R, R programs per KV
# head contend on the same dK/dV tile. Trade-off measured in benchmark.
#
# Requires: dK/dV pre-zeroed by caller (atomic_add accumulates into them).
# =====================================================================

@triton.jit
def _flash_attn_gqa_bwd_fused_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, dQ_ptr, dK_ptr, dV_ptr,
    LSE_ptr, Delta_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_dqb, stride_dqh, stride_dqn, stride_dqd,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_lseb, stride_lseh, stride_lsen,
    stride_db, stride_dh, stride_dn,
    N_Q_HEADS, N_KV_HEADS, SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDE_SIZE: tl.constexpr,
):
    q_block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    q_h_idx = bh_idx % N_Q_HEADS
    b_idx = bh_idx // N_Q_HEADS
    kv_h_idx = q_h_idx * N_KV_HEADS // N_Q_HEADS

    q_base = Q_ptr + b_idx * stride_qb + q_h_idx * stride_qh
    k_base = K_ptr + b_idx * stride_kb + kv_h_idx * stride_kh
    v_base = V_ptr + b_idx * stride_vb + kv_h_idx * stride_vh
    do_base = dO_ptr + b_idx * stride_dob + q_h_idx * stride_doh
    dq_base = dQ_ptr + b_idx * stride_dqb + q_h_idx * stride_dqh
    # dK/dV indexed by kv_h_idx — shared via atomic_add across GQA group programs
    dk_base = dK_ptr + b_idx * stride_dkb + kv_h_idx * stride_dkh
    dv_base = dV_ptr + b_idx * stride_dvb + kv_h_idx * stride_dvh

    q_offsets = q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < SEQ_LEN
    d_range = tl.arange(0, HEAD_DIM)

    # Load Q, dO, LSE, Delta — persist across KV loop
    q_ptrs = q_base + q_offsets[:, None] * stride_qn + d_range[None, :] * stride_qd
    q_block = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
    do_ptrs = do_base + q_offsets[:, None] * stride_don + d_range[None, :] * stride_dod
    do_block = tl.load(do_ptrs, mask=q_mask[:, None], other=0.0)
    lse_ptrs = LSE_ptr + b_idx * stride_lseb + q_h_idx * stride_lseh + q_offsets * stride_lsen
    lse = tl.load(lse_ptrs, mask=q_mask, other=0.0)
    delta_ptrs = Delta_ptr + b_idx * stride_db + q_h_idx * stride_dh + q_offsets * stride_dn
    delta = tl.load(delta_ptrs, mask=q_mask, other=0.0)

    # KV iteration bounds (same as dQ kernel)
    if IS_CAUSAL:
        kv_end = (q_block_idx + 1) * BLOCK_Q
    else:
        kv_end = SEQ_LEN
    if IS_CAUSAL and SLIDE_SIZE > 0:
        kv_min = tl.maximum(0, q_block_idx * BLOCK_Q - SLIDE_SIZE + 1)
        kv_loop_start = (kv_min // BLOCK_KV) * BLOCK_KV
    else:
        kv_loop_start = 0

    dq_acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    for kv_start in range(kv_loop_start, kv_end, BLOCK_KV):
        kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offsets < SEQ_LEN

        k_ptrs = k_base + kv_offsets[:, None] * stride_kn + d_range[None, :] * stride_kd
        k_block = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
        v_ptrs = v_base + kv_offsets[:, None] * stride_vn + d_range[None, :] * stride_vd
        v_block = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

        # Recompute scores, P (shared between all gradient computations)
        scores = tl.dot(q_block, tl.trans(k_block)).to(tl.float32) * scale
        if IS_CAUSAL:
            if SLIDE_SIZE > 0:
                valid = (kv_offsets[None, :] <= q_offsets[:, None]) & \
                        (q_offsets[:, None] - kv_offsets[None, :] < SLIDE_SIZE) & \
                        kv_mask[None, :]
            else:
                valid = (kv_offsets[None, :] <= q_offsets[:, None]) & kv_mask[None, :]
        else:
            valid = kv_mask[None, :]
        scores = tl.where(valid, scores, -float("inf"))
        p = tl.exp(scores - lse[:, None])

        # dV contribution = P^T @ dO (fp32 atomic to fp32 scratch dV_fp32)
        # (fp16 atomic_add loses precision with GQA_RATIO-way contention —
        #  each write rounds, errors accumulate. Cast to fp16 once at end.)
        dv_contrib = tl.dot(tl.trans(p.to(do_block.dtype)), do_block).to(tl.float32)
        dv_ptrs = dv_base + kv_offsets[:, None] * stride_dvn + d_range[None, :] * stride_dvd
        tl.atomic_add(dv_ptrs, dv_contrib, mask=kv_mask[:, None])

        # dP = dO @ V^T, then dS = P * (dP - delta)
        dp = tl.dot(do_block, tl.trans(v_block)).to(tl.float32)
        ds = p * (dp - delta[:, None])
        ds = tl.where(valid, ds, 0.0)

        # dQ += dS @ K (scale applied outside loop)
        dq_acc += tl.dot(ds.to(k_block.dtype), k_block).to(tl.float32)

        # dK contribution = scale * dS^T @ Q (fp32 atomic to dK_fp32)
        dk_contrib = (tl.dot(tl.trans(ds.to(q_block.dtype)), q_block).to(tl.float32)) * scale
        dk_ptrs = dk_base + kv_offsets[:, None] * stride_dkn + d_range[None, :] * stride_dkd
        tl.atomic_add(dk_ptrs, dk_contrib, mask=kv_mask[:, None])

    dq_acc *= scale
    dq_ptrs = dq_base + q_offsets[:, None] * stride_dqn + d_range[None, :] * stride_dqd
    tl.store(dq_ptrs, dq_acc.to(q_block.dtype), mask=q_mask[:, None])


@triton.jit
def _flash_attn_gqa_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, dK_ptr, dV_ptr,
    LSE_ptr, Delta_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_lseb, stride_lseh, stride_lsen,
    stride_db, stride_dh, stride_dn,
    N_Q_HEADS, N_KV_HEADS, SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDE_SIZE: tl.constexpr,  # 0 = no sliding window
    ATOMIC_REDUCE: tl.constexpr,  # True: atomic_add into shared (B, H_KV, N, D) dK/dV
):
    # Grid: (cdiv(SEQ_LEN, BLOCK_KV), B * N_Q_HEADS)
    # Split design: one program per Q head (not per KV head).
    # ATOMIC_REDUCE=False: output per-Q-head dK/dV (B, H_Q, N, D); caller reduces.
    # ATOMIC_REDUCE=True:  atomic_add into shared dK/dV (B, H_KV, N, D); caller
    #                      pre-zeros the buffers. Saves a reduce kernel + expand
    #                      buffer allocation at the cost of GQA_RATIO-way contention.
    kv_block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    q_h_idx = bh_idx % N_Q_HEADS
    b_idx = bh_idx // N_Q_HEADS
    kv_h_idx = q_h_idx * N_KV_HEADS // N_Q_HEADS

    k_base = K_ptr + b_idx * stride_kb + kv_h_idx * stride_kh
    v_base = V_ptr + b_idx * stride_vb + kv_h_idx * stride_vh
    q_base = Q_ptr + b_idx * stride_qb + q_h_idx * stride_qh
    do_base = dO_ptr + b_idx * stride_dob + q_h_idx * stride_doh
    if ATOMIC_REDUCE:
        # dK/dV indexed by kv_h_idx (shared across GQA group via atomic_add)
        dk_base = dK_ptr + b_idx * stride_dkb + kv_h_idx * stride_dkh
        dv_base = dV_ptr + b_idx * stride_dvb + kv_h_idx * stride_dvh
    else:
        # dK/dV indexed by q_h_idx (per-Q-head, reduced later)
        dk_base = dK_ptr + b_idx * stride_dkb + q_h_idx * stride_dkh
        dv_base = dV_ptr + b_idx * stride_dvb + q_h_idx * stride_dvh
    lse_base = LSE_ptr + b_idx * stride_lseb + q_h_idx * stride_lseh
    delta_base = Delta_ptr + b_idx * stride_db + q_h_idx * stride_dh

    kv_offsets = kv_block_idx * BLOCK_KV + tl.arange(0, BLOCK_KV)
    kv_mask = kv_offsets < SEQ_LEN
    d_range = tl.arange(0, HEAD_DIM)

    # Load K, V blocks (persist across Q iterations)
    k_ptrs = k_base + kv_offsets[:, None] * stride_kn + d_range[None, :] * stride_kd
    k_block = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
    v_ptrs = v_base + kv_offsets[:, None] * stride_vn + d_range[None, :] * stride_vd
    v_block = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

    dk_acc = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    # Determine Q iteration range.
    # Causal lower bound: only Q blocks that can attend to this KV block.
    # SWA upper bound: Q blocks too far ahead cannot attend to this KV block.
    if IS_CAUSAL:
        q_start = (kv_block_idx * BLOCK_KV // BLOCK_Q) * BLOCK_Q
    else:
        q_start = 0

    if IS_CAUSAL and SLIDE_SIZE > 0:
        # Latest Q position attending to earliest KV in this block:
        #   kv_first = kv_block_idx * BLOCK_KV
        #   q_max = kv_first + SLIDE_SIZE - 1  (beyond that, window excludes this KV)
        # But we need the last position of this KV block to be more conservative:
        #   kv_last = kv_block_idx * BLOCK_KV + BLOCK_KV - 1
        #   q_max = kv_last + SLIDE_SIZE - 1
        kv_last = kv_block_idx * BLOCK_KV + BLOCK_KV - 1
        q_max = kv_last + SLIDE_SIZE - 1
        # Round up to next Q block boundary (exclusive loop end)
        q_loop_end = ((q_max // BLOCK_Q) + 1) * BLOCK_Q
        q_loop_end = tl.minimum(SEQ_LEN, q_loop_end)
    else:
        q_loop_end = SEQ_LEN

    for q_start_pos in range(q_start, q_loop_end, BLOCK_Q):
        q_offsets = q_start_pos + tl.arange(0, BLOCK_Q)
        q_mask_local = q_offsets < SEQ_LEN

        # Load Q, dO
        q_ptrs = q_base + q_offsets[:, None] * stride_qn + d_range[None, :] * stride_qd
        q_block = tl.load(q_ptrs, mask=q_mask_local[:, None], other=0.0)
        do_ptrs = do_base + q_offsets[:, None] * stride_don + d_range[None, :] * stride_dod
        do_block = tl.load(do_ptrs, mask=q_mask_local[:, None], other=0.0)

        # Load LSE, Delta
        lse = tl.load(lse_base + q_offsets * stride_lsen, mask=q_mask_local, other=0.0)
        delta = tl.load(delta_base + q_offsets * stride_dn, mask=q_mask_local, other=0.0)

        # Recompute scores: (BLOCK_Q, BLOCK_KV)
        scores = tl.dot(q_block, tl.trans(k_block)).to(tl.float32) * scale
        if IS_CAUSAL:
            if SLIDE_SIZE > 0:
                valid = (kv_offsets[None, :] <= q_offsets[:, None]) & \
                        kv_mask[None, :] & q_mask_local[:, None] & \
                        (q_offsets[:, None] - kv_offsets[None, :] < SLIDE_SIZE)
            else:
                valid = (kv_offsets[None, :] <= q_offsets[:, None]) & kv_mask[None, :] & q_mask_local[:, None]
        else:
            valid = kv_mask[None, :] & q_mask_local[:, None]
        scores = tl.where(valid, scores, -float("inf"))
        p = tl.exp(scores - lse[:, None])

        # dV += P^T @ dO
        dv_acc += tl.dot(tl.trans(p.to(do_block.dtype)), do_block).to(tl.float32)

        # dP = dO @ V^T, dS = P * (dP - D)
        dp = tl.dot(do_block, tl.trans(v_block)).to(tl.float32)
        ds = p * (dp - delta[:, None])
        ds = tl.where(valid, ds, 0.0)

        # dK += scale * dS^T @ Q
        dk_acc += tl.dot(tl.trans(ds.to(q_block.dtype)), q_block).to(tl.float32)

    dk_acc *= scale

    dk_ptrs = dk_base + kv_offsets[:, None] * stride_dkn + d_range[None, :] * stride_dkd
    dv_ptrs = dv_base + kv_offsets[:, None] * stride_dvn + d_range[None, :] * stride_dvd
    if ATOMIC_REDUCE:
        # Triton 3.6: atomic_add doesn't support bf16. Caller provides fp32 buffers.
        tl.atomic_add(dk_ptrs, dk_acc, mask=kv_mask[:, None])
        tl.atomic_add(dv_ptrs, dv_acc, mask=kv_mask[:, None])
    else:
        tl.store(dk_ptrs, dk_acc.to(k_block.dtype), mask=kv_mask[:, None])
        tl.store(dv_ptrs, dv_acc.to(v_block.dtype), mask=kv_mask[:, None])


# =====================================================================
# Packed dKV kernel — pack-GQA style (inspired by flash-attn pack_gqa=True).
#
# Grid is (cdiv(N, BKV), B * N_KV_HEADS) — one program per KV block, NOT per
# Q head. Inner loop iterates over all GQA_RATIO Q heads mapping to this KV
# head and accumulates their contributions into a single dk_acc / dv_acc.
#
# Compared to _flash_attn_gqa_bwd_dkv_kernel (split per-Q-head + reduce):
#   + No expand buffer (saves GQA_RATIO× memory: no dk_expanded/dv_expanded)
#   + No reduce step (saves the view+sum kernel)
#   + K, V loaded ONCE per program (vs GQA_RATIO× across split programs)
#   - GQA_RATIO× longer inner loop (but same total work)
#   - GQA_RATIO× fewer programs in grid — risks SM under-utilisation at short N
#
# Register budget: identical to split dKV (same dk_acc/dv_acc/K/V live set).
# =====================================================================

@triton.jit
def _flash_attn_gqa_bwd_dkv_packed_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, dK_ptr, dV_ptr,
    LSE_ptr, Delta_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_lseb, stride_lseh, stride_lsen,
    stride_db, stride_dh, stride_dn,
    N_Q_HEADS, N_KV_HEADS, SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    GQA_RATIO: tl.constexpr,  # N_Q_HEADS // N_KV_HEADS
    IS_CAUSAL: tl.constexpr,
    SLIDE_SIZE: tl.constexpr,
    Q_SPLITS: tl.constexpr,  # 1 = direct store; >1 = atomic_add (caller pre-zeros dK/dV)
    # Image-bidirectional OR-mask (mirror of fwd). For KV-major iteration we
    # extend the Q-loop bounds based on the KV block's image-group reach.
    GroupIds_ptr,
    GroupLo_ptr,
    GroupHi_ptr,
    stride_gb, stride_gn,
    HAS_GROUP_IDS: tl.constexpr,
):
    # Grid: (cdiv(SEQ_LEN, BLOCK_KV), B * N_KV_HEADS, Q_SPLITS)
    kv_block_idx = tl.program_id(0)
    bkvh_idx = tl.program_id(1)
    split_idx = tl.program_id(2)
    kv_h_idx = bkvh_idx % N_KV_HEADS
    b_idx = bkvh_idx // N_KV_HEADS

    k_base = K_ptr + b_idx * stride_kb + kv_h_idx * stride_kh
    v_base = V_ptr + b_idx * stride_vb + kv_h_idx * stride_vh
    # dK/dV indexed by KV head — one program owns this tile entirely (Q_SPLITS=1)
    # or is one of Q_SPLITS programs sharing the tile via atomic_add
    dk_base = dK_ptr + b_idx * stride_dkb + kv_h_idx * stride_dkh
    dv_base = dV_ptr + b_idx * stride_dvb + kv_h_idx * stride_dvh

    kv_offsets = kv_block_idx * BLOCK_KV + tl.arange(0, BLOCK_KV)
    kv_mask = kv_offsets < SEQ_LEN
    d_range = tl.arange(0, HEAD_DIM)

    # Load K, V once — reused across all GQA_RATIO Q heads
    k_ptrs = k_base + kv_offsets[:, None] * stride_kn + d_range[None, :] * stride_kd
    k_block = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
    v_ptrs = v_base + kv_offsets[:, None] * stride_vn + d_range[None, :] * stride_vd
    v_block = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

    # Single accumulator sums contributions from all Q heads in the GQA group
    dk_acc = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    LOG2E: tl.constexpr = 1.4426950408889634
    scale_log2e = scale * LOG2E

    # Q iteration bounds (same as split kernel — derived from KV block position)
    if IS_CAUSAL:
        q_start = (kv_block_idx * BLOCK_KV // BLOCK_Q) * BLOCK_Q
    else:
        q_start = 0

    if IS_CAUSAL and SLIDE_SIZE > 0:
        kv_last = kv_block_idx * BLOCK_KV + BLOCK_KV - 1
        q_max = kv_last + SLIDE_SIZE - 1
        q_loop_end = ((q_max // BLOCK_Q) + 1) * BLOCK_Q
        q_loop_end = tl.minimum(SEQ_LEN, q_loop_end)
    else:
        q_loop_end = SEQ_LEN

    # Image-group state for this KV block. Extends Q-loop bounds symmetrically
    # to fwd/dQ: any image-KV in this block is bidirectionally visible to all
    # Q's in the same image span.
    if HAS_GROUP_IDS:
        g_base = GroupIds_ptr + b_idx * stride_gb
        glo_base = GroupLo_ptr + b_idx * stride_gb
        ghi_base = GroupHi_ptr + b_idx * stride_gb
        kv_group_ids = tl.load(g_base + kv_offsets * stride_gn,
                               mask=kv_mask, other=-1)
        kv_group_lo = tl.load(glo_base + kv_offsets * stride_gn,
                              mask=kv_mask, other=SEQ_LEN)
        kv_group_hi = tl.load(ghi_base + kv_offsets * stride_gn,
                              mask=kv_mask, other=0)
        block_img_lo = tl.min(kv_group_lo, axis=0)
        block_img_hi = tl.max(kv_group_hi, axis=0)
        q_start = tl.minimum(q_start, (block_img_lo // BLOCK_Q) * BLOCK_Q)
        q_loop_end = tl.maximum(q_loop_end, block_img_hi)
        # q_loop_end may exceed SEQ_LEN at the tail; mask handles it inside loop.

    # Q-split: each of Q_SPLITS programs handles a disjoint slice of Q blocks.
    # Activated when raw grid << 132 SMs (e.g. H_KV=1 short N). Atomic_add on
    # dK/dV (caller pre-zeroed) replaces direct store.
    if Q_SPLITS > 1:
        total_q_blocks = (q_loop_end - q_start + BLOCK_Q - 1) // BLOCK_Q
        blocks_per_split = (total_q_blocks + Q_SPLITS - 1) // Q_SPLITS
        q_lo = q_start + split_idx * blocks_per_split * BLOCK_Q
        q_hi = tl.minimum(q_loop_end, q_start + (split_idx + 1) * blocks_per_split * BLOCK_Q)
    else:
        q_lo = q_start
        q_hi = q_loop_end

    # Outer loop: Q heads (unrolled). Each Q head's contribution accumulates
    # into the SAME dk_acc/dv_acc — this replaces the expand+reduce pattern.
    for qh_offset in tl.static_range(GQA_RATIO):
        q_h_idx = kv_h_idx * GQA_RATIO + qh_offset
        q_base = Q_ptr + b_idx * stride_qb + q_h_idx * stride_qh
        do_base = dO_ptr + b_idx * stride_dob + q_h_idx * stride_doh
        lse_base = LSE_ptr + b_idx * stride_lseb + q_h_idx * stride_lseh
        delta_base = Delta_ptr + b_idx * stride_db + q_h_idx * stride_dh

        # Inner loop: Q blocks for this Q head (sliced by split_idx)
        for q_start_pos in range(q_lo, q_hi, BLOCK_Q):
            q_offsets = q_start_pos + tl.arange(0, BLOCK_Q)
            q_mask_local = q_offsets < SEQ_LEN

            q_ptrs = q_base + q_offsets[:, None] * stride_qn + d_range[None, :] * stride_qd
            q_block = tl.load(q_ptrs, mask=q_mask_local[:, None], other=0.0)
            do_ptrs = do_base + q_offsets[:, None] * stride_don + d_range[None, :] * stride_dod
            do_block = tl.load(do_ptrs, mask=q_mask_local[:, None], other=0.0)
            lse = tl.load(lse_base + q_offsets * stride_lsen, mask=q_mask_local, other=0.0)
            delta = tl.load(delta_base + q_offsets * stride_dn, mask=q_mask_local, other=0.0)
            lse_log2 = lse * LOG2E

            scores = tl.dot(q_block, tl.trans(k_block)).to(tl.float32) * scale_log2e
            if IS_CAUSAL:
                if SLIDE_SIZE > 0:
                    valid = (kv_offsets[None, :] <= q_offsets[:, None]) & \
                            kv_mask[None, :] & q_mask_local[:, None] & \
                            (q_offsets[:, None] - kv_offsets[None, :] < SLIDE_SIZE)
                else:
                    valid = (kv_offsets[None, :] <= q_offsets[:, None]) & \
                            kv_mask[None, :] & q_mask_local[:, None]
            else:
                valid = kv_mask[None, :] & q_mask_local[:, None]
            if HAS_GROUP_IDS:
                q_group_ids = tl.load(g_base + q_offsets * stride_gn,
                                      mask=q_mask_local, other=-1)
                bidir = (q_group_ids[:, None] == kv_group_ids[None, :]) & \
                        (q_group_ids[:, None] >= 0) & \
                        kv_mask[None, :] & q_mask_local[:, None]
                valid = valid | bidir
            scores = tl.where(valid, scores, -float("inf"))
            p = tl.math.exp2(scores - lse_log2[:, None])

            dv_acc += tl.dot(tl.trans(p.to(do_block.dtype)), do_block).to(tl.float32)

            dp = tl.dot(do_block, tl.trans(v_block)).to(tl.float32)
            ds = p * (dp - delta[:, None])
            ds = tl.where(valid, ds, 0.0)

            dk_acc += tl.dot(tl.trans(ds.to(q_block.dtype)), q_block).to(tl.float32)

    dk_acc *= scale

    dk_ptrs = dk_base + kv_offsets[:, None] * stride_dkn + d_range[None, :] * stride_dkd
    dv_ptrs = dv_base + kv_offsets[:, None] * stride_dvn + d_range[None, :] * stride_dvd
    if Q_SPLITS > 1:
        # Triton 3.6: atomic_add doesn't support bf16. Caller provides fp32 buffers.
        tl.atomic_add(dk_ptrs, dk_acc, mask=kv_mask[:, None])
        tl.atomic_add(dv_ptrs, dv_acc, mask=kv_mask[:, None])
    else:
        # Direct store — only this program writes this tile.
        tl.store(dk_ptrs, dk_acc.to(k_block.dtype), mask=kv_mask[:, None])
        tl.store(dv_ptrs, dv_acc.to(v_block.dtype), mask=kv_mask[:, None])


# =====================================================================
# Flash Attention GQA — Varlen backward pack-GQA dK/dV kernel
#
# Grid: (cdiv(max_seqlen_k, BLOCK_KV), B * N_KV_HEADS, Q_SPLITS).
# Packed dK/dV accumulate over all GQA Q heads via an inner static_range loop —
# single dk_acc/dv_acc per program → direct store (QS=1) or atomic_add (QS>1)
# to pre-zeroed packed dK/dV (total_k, H_KV, D).
#
# Per-sample bounds MUST be computed before the tl.static_range(GQA_RATIO)
# loop so the Triton unroller can fold them once instead of re-emitting GQA
# times.
#
# Atomic safety: kv_global = seq_start_k + kv_local is unique across b_idx
# (cu_seqlens strictly monotone), so programs from different samples never
# write to the same packed row of dK/dV. Only Q_SPLITS programs within a single
# sample's (kv_h, kv_block) cohort contend — identical to batched.
# =====================================================================

@triton.jit
def _flash_attn_gqa_varlen_bwd_dkv_packed_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, dK_ptr, dV_ptr,
    LSE_ptr, Delta_ptr,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_dot, stride_doh, stride_dod,
    stride_dkt, stride_dkh, stride_dkd,
    stride_dvt, stride_dvh, stride_dvd,
    stride_lset, stride_lseh,
    stride_dt, stride_dh,
    CuSeqlensQ_ptr,  # int32 (B+1,)
    CuSeqlensK_ptr,  # int32 (B+1,)
    N_Q_HEADS, N_KV_HEADS,
    HEAD_DIM: tl.constexpr,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    GQA_RATIO: tl.constexpr,  # N_Q_HEADS // N_KV_HEADS
    IS_CAUSAL: tl.constexpr,
    SLIDE_SIZE: tl.constexpr,
    Q_SPLITS: tl.constexpr,
):
    kv_block_idx = tl.program_id(0)
    bkvh_idx = tl.program_id(1)
    split_idx = tl.program_id(2)
    kv_h_idx = bkvh_idx % N_KV_HEADS
    b_idx = bkvh_idx // N_KV_HEADS

    # Per-sample bounds (hoisted outside the GQA static_range).
    seq_start_q = tl.load(CuSeqlensQ_ptr + b_idx)
    seq_end_q = tl.load(CuSeqlensQ_ptr + b_idx + 1)
    seq_len_q = seq_end_q - seq_start_q
    seq_start_k = tl.load(CuSeqlensK_ptr + b_idx)
    seq_end_k = tl.load(CuSeqlensK_ptr + b_idx + 1)
    seq_len_k = seq_end_k - seq_start_k

    if kv_block_idx * BLOCK_KV >= seq_len_k:
        # OOR kv_block for this sample — no atomic writes, no stores.
        return

    k_h_base = K_ptr + kv_h_idx * stride_kh
    v_h_base = V_ptr + kv_h_idx * stride_vh
    dk_h_base = dK_ptr + kv_h_idx * stride_dkh
    dv_h_base = dV_ptr + kv_h_idx * stride_dvh

    kv_local = kv_block_idx * BLOCK_KV + tl.arange(0, BLOCK_KV)
    kv_mask = kv_local < seq_len_k
    kv_global = seq_start_k + kv_local
    d_range = tl.arange(0, HEAD_DIM)

    # Load K, V once — reused across all GQA_RATIO Q heads.
    k_ptrs = k_h_base + kv_global[:, None] * stride_kt + d_range[None, :] * stride_kd
    k_block = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
    v_ptrs = v_h_base + kv_global[:, None] * stride_vt + d_range[None, :] * stride_vd
    v_block = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

    dk_acc = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    LOG2E: tl.constexpr = 1.4426950408889634
    scale_log2e = scale * LOG2E

    # Q iteration bounds (seq-local). Symmetric to varlen fwd/dQ.
    if IS_CAUSAL:
        q_start_local = (kv_block_idx * BLOCK_KV // BLOCK_Q) * BLOCK_Q
    else:
        q_start_local = 0

    if IS_CAUSAL and SLIDE_SIZE > 0:
        kv_last_local = kv_block_idx * BLOCK_KV + BLOCK_KV - 1
        q_max_local = kv_last_local + SLIDE_SIZE - 1
        q_loop_end_local = ((q_max_local // BLOCK_Q) + 1) * BLOCK_Q
        q_loop_end_local = tl.minimum(seq_len_q, q_loop_end_local)
    else:
        q_loop_end_local = seq_len_q

    # Q-split slicing (local bounds).
    if Q_SPLITS > 1:
        total_q_blocks = (q_loop_end_local - q_start_local + BLOCK_Q - 1) // BLOCK_Q
        blocks_per_split = (total_q_blocks + Q_SPLITS - 1) // Q_SPLITS
        q_lo_local = q_start_local + split_idx * blocks_per_split * BLOCK_Q
        q_hi_local = tl.minimum(
            q_loop_end_local,
            q_start_local + (split_idx + 1) * blocks_per_split * BLOCK_Q,
        )
    else:
        q_lo_local = q_start_local
        q_hi_local = q_loop_end_local

    # Outer loop: Q heads (unrolled). Each contributes into dk_acc/dv_acc.
    for qh_offset in tl.static_range(GQA_RATIO):
        q_h_idx = kv_h_idx * GQA_RATIO + qh_offset
        q_h_base = Q_ptr + q_h_idx * stride_qh
        do_h_base = dO_ptr + q_h_idx * stride_doh

        for q_start_pos_local in range(q_lo_local, q_hi_local, BLOCK_Q):
            q_local = q_start_pos_local + tl.arange(0, BLOCK_Q)
            q_mask_local = q_local < seq_len_q
            q_global = seq_start_q + q_local

            q_ptrs = q_h_base + q_global[:, None] * stride_qt + d_range[None, :] * stride_qd
            q_block = tl.load(q_ptrs, mask=q_mask_local[:, None], other=0.0)
            do_ptrs = do_h_base + q_global[:, None] * stride_dot + d_range[None, :] * stride_dod
            do_block = tl.load(do_ptrs, mask=q_mask_local[:, None], other=0.0)

            lse_ptrs = LSE_ptr + q_global * stride_lset + q_h_idx * stride_lseh
            lse = tl.load(lse_ptrs, mask=q_mask_local, other=0.0)
            delta_ptrs = Delta_ptr + q_global * stride_dt + q_h_idx * stride_dh
            delta = tl.load(delta_ptrs, mask=q_mask_local, other=0.0)
            lse_log2 = lse * LOG2E

            scores = tl.dot(q_block, tl.trans(k_block)).to(tl.float32) * scale_log2e
            if IS_CAUSAL:
                if SLIDE_SIZE > 0:
                    valid = (kv_local[None, :] <= q_local[:, None]) & \
                            kv_mask[None, :] & q_mask_local[:, None] & \
                            (q_local[:, None] - kv_local[None, :] < SLIDE_SIZE)
                else:
                    valid = (kv_local[None, :] <= q_local[:, None]) & \
                            kv_mask[None, :] & q_mask_local[:, None]
            else:
                valid = kv_mask[None, :] & q_mask_local[:, None]
            scores = tl.where(valid, scores, -float("inf"))
            p = tl.math.exp2(scores - lse_log2[:, None])

            dv_acc += tl.dot(tl.trans(p.to(do_block.dtype)), do_block).to(tl.float32)

            dp = tl.dot(do_block, tl.trans(v_block)).to(tl.float32)
            ds = p * (dp - delta[:, None])
            ds = tl.where(valid, ds, 0.0)

            dk_acc += tl.dot(tl.trans(ds.to(q_block.dtype)), q_block).to(tl.float32)

    dk_acc *= scale

    dk_ptrs = dk_h_base + kv_global[:, None] * stride_dkt + d_range[None, :] * stride_dkd
    dv_ptrs = dv_h_base + kv_global[:, None] * stride_dvt + d_range[None, :] * stride_dvd
    if Q_SPLITS > 1:
        # Triton 3.6: atomic_add does not support bf16. dk_acc/dv_acc are fp32;
        # when Q_SPLITS > 1 the caller allocates dK/dV as fp32 buffers.
        tl.atomic_add(dk_ptrs, dk_acc, mask=kv_mask[:, None])
        tl.atomic_add(dv_ptrs, dv_acc, mask=kv_mask[:, None])
    else:
        tl.store(dk_ptrs, dk_acc.to(k_block.dtype), mask=kv_mask[:, None])
        tl.store(dv_ptrs, dv_acc.to(v_block.dtype), mask=kv_mask[:, None])


# =====================================================================
# Split-accumulator pack-GQA backward — experiment 4 (KEPT FOR REFERENCE,
# NOT ON HOT PATH)
#
# Motivation: packed kernel hits 302 register spills at HEAD_DIM=512 because
# it holds BOTH dk_acc and dv_acc plus 4 matmuls per inner iter (scores,
# P^T@dO, dO@V^T, dS^T@Q) in a single live state.
#
# Split: dV-only needs only Q, K, dO, LSE (no V, no delta) — 2 matmuls and
# 1 accumulator. dK-only needs Q, K, V, dO, LSE, delta — 3 matmuls and
# 1 accumulator.
#
# Result @ D=512 Gemma4 full causal:
#   - spills drop 302 → 46 (dV) / 92 (dK)
#   - BUT total time: +35-36% vs packed (N=4K 9.27 → 12.49ms; N=8K 35.48 → 48.39ms)
#   - Why: both kernels recompute scores + P. The compute sharing in packed
#     is worth more than the ~19% spill overhead.
#
# Conclusion: kept in source since the analysis is instructive, but
# `_flash_attn_gqa_bwd_dkv_packed_kernel` remains the hot path.
# Benchmark: `benchmarks/dkv_split_bench.py`.
# =====================================================================

@triton.jit
def _flash_attn_gqa_bwd_dv_only_kernel(
    Q_ptr, K_ptr, dO_ptr, dV_ptr,
    LSE_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_lseb, stride_lseh, stride_lsen,
    N_Q_HEADS, N_KV_HEADS, SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDE_SIZE: tl.constexpr,
):
    # Grid: (cdiv(SEQ_LEN, BLOCK_KV), B * N_KV_HEADS)
    kv_block_idx = tl.program_id(0)
    bkvh_idx = tl.program_id(1)
    kv_h_idx = bkvh_idx % N_KV_HEADS
    b_idx = bkvh_idx // N_KV_HEADS

    k_base = K_ptr + b_idx * stride_kb + kv_h_idx * stride_kh
    dv_base = dV_ptr + b_idx * stride_dvb + kv_h_idx * stride_dvh

    kv_offsets = kv_block_idx * BLOCK_KV + tl.arange(0, BLOCK_KV)
    kv_mask = kv_offsets < SEQ_LEN
    d_range = tl.arange(0, HEAD_DIM)

    k_ptrs = k_base + kv_offsets[:, None] * stride_kn + d_range[None, :] * stride_kd
    k_block = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)

    dv_acc = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    LOG2E: tl.constexpr = 1.4426950408889634
    scale_log2e = scale * LOG2E

    if IS_CAUSAL:
        q_start = (kv_block_idx * BLOCK_KV // BLOCK_Q) * BLOCK_Q
    else:
        q_start = 0

    if IS_CAUSAL and SLIDE_SIZE > 0:
        kv_last = kv_block_idx * BLOCK_KV + BLOCK_KV - 1
        q_max = kv_last + SLIDE_SIZE - 1
        q_loop_end = ((q_max // BLOCK_Q) + 1) * BLOCK_Q
        q_loop_end = tl.minimum(SEQ_LEN, q_loop_end)
    else:
        q_loop_end = SEQ_LEN

    for qh_offset in tl.static_range(GQA_RATIO):
        q_h_idx = kv_h_idx * GQA_RATIO + qh_offset
        q_base = Q_ptr + b_idx * stride_qb + q_h_idx * stride_qh
        do_base = dO_ptr + b_idx * stride_dob + q_h_idx * stride_doh
        lse_base = LSE_ptr + b_idx * stride_lseb + q_h_idx * stride_lseh

        for q_start_pos in range(q_start, q_loop_end, BLOCK_Q):
            q_offsets = q_start_pos + tl.arange(0, BLOCK_Q)
            q_mask_local = q_offsets < SEQ_LEN

            q_ptrs = q_base + q_offsets[:, None] * stride_qn + d_range[None, :] * stride_qd
            q_block = tl.load(q_ptrs, mask=q_mask_local[:, None], other=0.0)
            do_ptrs = do_base + q_offsets[:, None] * stride_don + d_range[None, :] * stride_dod
            do_block = tl.load(do_ptrs, mask=q_mask_local[:, None], other=0.0)
            lse = tl.load(lse_base + q_offsets * stride_lsen, mask=q_mask_local, other=0.0)
            lse_log2 = lse * LOG2E

            scores = tl.dot(q_block, tl.trans(k_block)).to(tl.float32) * scale_log2e
            if IS_CAUSAL:
                if SLIDE_SIZE > 0:
                    valid = (kv_offsets[None, :] <= q_offsets[:, None]) & \
                            kv_mask[None, :] & q_mask_local[:, None] & \
                            (q_offsets[:, None] - kv_offsets[None, :] < SLIDE_SIZE)
                else:
                    valid = (kv_offsets[None, :] <= q_offsets[:, None]) & \
                            kv_mask[None, :] & q_mask_local[:, None]
            else:
                valid = kv_mask[None, :] & q_mask_local[:, None]
            scores = tl.where(valid, scores, -float("inf"))
            p = tl.math.exp2(scores - lse_log2[:, None])

            dv_acc += tl.dot(tl.trans(p.to(do_block.dtype)), do_block).to(tl.float32)

    dv_ptrs = dv_base + kv_offsets[:, None] * stride_dvn + d_range[None, :] * stride_dvd
    tl.store(dv_ptrs, dv_acc.to(k_block.dtype), mask=kv_mask[:, None])


@triton.jit
def _flash_attn_gqa_bwd_dk_only_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, dK_ptr,
    LSE_ptr, Delta_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_lseb, stride_lseh, stride_lsen,
    stride_db, stride_dh, stride_dn,
    N_Q_HEADS, N_KV_HEADS, SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDE_SIZE: tl.constexpr,
):
    kv_block_idx = tl.program_id(0)
    bkvh_idx = tl.program_id(1)
    kv_h_idx = bkvh_idx % N_KV_HEADS
    b_idx = bkvh_idx // N_KV_HEADS

    k_base = K_ptr + b_idx * stride_kb + kv_h_idx * stride_kh
    v_base = V_ptr + b_idx * stride_vb + kv_h_idx * stride_vh
    dk_base = dK_ptr + b_idx * stride_dkb + kv_h_idx * stride_dkh

    kv_offsets = kv_block_idx * BLOCK_KV + tl.arange(0, BLOCK_KV)
    kv_mask = kv_offsets < SEQ_LEN
    d_range = tl.arange(0, HEAD_DIM)

    k_ptrs = k_base + kv_offsets[:, None] * stride_kn + d_range[None, :] * stride_kd
    k_block = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
    v_ptrs = v_base + kv_offsets[:, None] * stride_vn + d_range[None, :] * stride_vd
    v_block = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

    dk_acc = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    LOG2E: tl.constexpr = 1.4426950408889634
    scale_log2e = scale * LOG2E

    if IS_CAUSAL:
        q_start = (kv_block_idx * BLOCK_KV // BLOCK_Q) * BLOCK_Q
    else:
        q_start = 0

    if IS_CAUSAL and SLIDE_SIZE > 0:
        kv_last = kv_block_idx * BLOCK_KV + BLOCK_KV - 1
        q_max = kv_last + SLIDE_SIZE - 1
        q_loop_end = ((q_max // BLOCK_Q) + 1) * BLOCK_Q
        q_loop_end = tl.minimum(SEQ_LEN, q_loop_end)
    else:
        q_loop_end = SEQ_LEN

    for qh_offset in tl.static_range(GQA_RATIO):
        q_h_idx = kv_h_idx * GQA_RATIO + qh_offset
        q_base = Q_ptr + b_idx * stride_qb + q_h_idx * stride_qh
        do_base = dO_ptr + b_idx * stride_dob + q_h_idx * stride_doh
        lse_base = LSE_ptr + b_idx * stride_lseb + q_h_idx * stride_lseh
        delta_base = Delta_ptr + b_idx * stride_db + q_h_idx * stride_dh

        for q_start_pos in range(q_start, q_loop_end, BLOCK_Q):
            q_offsets = q_start_pos + tl.arange(0, BLOCK_Q)
            q_mask_local = q_offsets < SEQ_LEN

            q_ptrs = q_base + q_offsets[:, None] * stride_qn + d_range[None, :] * stride_qd
            q_block = tl.load(q_ptrs, mask=q_mask_local[:, None], other=0.0)
            do_ptrs = do_base + q_offsets[:, None] * stride_don + d_range[None, :] * stride_dod
            do_block = tl.load(do_ptrs, mask=q_mask_local[:, None], other=0.0)
            lse = tl.load(lse_base + q_offsets * stride_lsen, mask=q_mask_local, other=0.0)
            delta = tl.load(delta_base + q_offsets * stride_dn, mask=q_mask_local, other=0.0)
            lse_log2 = lse * LOG2E

            scores = tl.dot(q_block, tl.trans(k_block)).to(tl.float32) * scale_log2e
            if IS_CAUSAL:
                if SLIDE_SIZE > 0:
                    valid = (kv_offsets[None, :] <= q_offsets[:, None]) & \
                            kv_mask[None, :] & q_mask_local[:, None] & \
                            (q_offsets[:, None] - kv_offsets[None, :] < SLIDE_SIZE)
                else:
                    valid = (kv_offsets[None, :] <= q_offsets[:, None]) & \
                            kv_mask[None, :] & q_mask_local[:, None]
            else:
                valid = kv_mask[None, :] & q_mask_local[:, None]
            scores = tl.where(valid, scores, -float("inf"))
            p = tl.math.exp2(scores - lse_log2[:, None])

            dp = tl.dot(do_block, tl.trans(v_block)).to(tl.float32)
            ds = p * (dp - delta[:, None])
            ds = tl.where(valid, ds, 0.0)

            dk_acc += tl.dot(tl.trans(ds.to(q_block.dtype)), q_block).to(tl.float32)

    dk_acc *= scale

    dk_ptrs = dk_base + kv_offsets[:, None] * stride_dkn + d_range[None, :] * stride_dkd
    tl.store(dk_ptrs, dk_acc.to(k_block.dtype), mask=kv_mask[:, None])


# =====================================================================
# Autograd Function wrapping forward + backward
# =====================================================================

class FlashAttnGQAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, slide_size, group_ids, group_lo, group_hi_excl):
        B, H_Q, N, D = q.shape
        _, H_KV, _, _ = k.shape
        output = torch.empty_like(q)
        lse = torch.empty(B, H_Q, N, dtype=torch.float32, device=q.device)

        # Normalize: window covering the whole sequence ≡ full causal.
        if slide_size > 0 and slide_size >= N:
            slide_size = 0

        # H200 + Triton 3.2 shmem budget is 228 KB. Older table
        # (BQ=64/BKV=32 @ D=512, BQ=128/BKV=64 @ D=256, w=8) compiled on
        # earlier Triton but now over-budget by ~10-25 KB. Shrink to the
        # same configs the varlen Function uses — mathematically identical,
        # ~10-20% slower at D=256/D=512 on H100 but compiles on both envs.
        # D<256 unchanged (was already fitting).
        if D >= 512:
            BLOCK_Q = 32
            BLOCK_KV = 32
            num_warps = 4
        elif D >= 256:
            BLOCK_Q = 64
            BLOCK_KV = 64
            num_warps = 4
        else:
            BLOCK_Q = 128
            BLOCK_KV = 64
            num_warps = 4
        BLOCK_D = D
        num_stages = 2
        BLOCK_Q = min(BLOCK_Q, triton.next_power_of_2(N))
        BLOCK_KV = min(BLOCK_KV, triton.next_power_of_2(N))
        grid = (triton.cdiv(N, BLOCK_Q), B * H_Q)

        has_group_ids = group_ids is not None
        if has_group_ids:
            assert group_lo is not None and group_hi_excl is not None
            g_strides = (group_ids.stride(0), group_ids.stride(1))
        else:
            g_strides = (0, 0)

        _flash_attn_gqa_kernel[grid](
            q, k, v, output,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D,
            scale=1.0 / math.sqrt(D),
            BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, BLOCK_D=BLOCK_D,
            IS_CAUSAL=causal,
            SLIDE_SIZE=slide_size,
            LSE_ptr=lse, stride_lseb=lse.stride(0), stride_lseh=lse.stride(1),
            stride_lsen=lse.stride(2), STORE_LSE=True,
            GroupIds_ptr=group_ids, GroupLo_ptr=group_lo, GroupHi_ptr=group_hi_excl,
            stride_gb=g_strides[0], stride_gn=g_strides[1],
            HAS_GROUP_IDS=has_group_ids,
            num_warps=num_warps, num_stages=num_stages,
        )

        # save_for_backward supports None entries (kept as None in saved_tensors).
        ctx.save_for_backward(q, k, v, output, lse, group_ids, group_lo, group_hi_excl)
        ctx.causal = causal
        ctx.slide_size = slide_size
        ctx.has_group_ids = has_group_ids
        return output

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, group_ids, group_lo, group_hi_excl = ctx.saved_tensors
        causal = ctx.causal
        slide_size = ctx.slide_size
        has_group_ids = ctx.has_group_ids
        g_strides_dq = (
            (group_ids.stride(0), group_ids.stride(1)) if has_group_ids else (0, 0)
        )
        B, H_Q, N, D = q.shape
        _, H_KV, _, _ = k.shape
        scale = 1.0 / math.sqrt(D)

        # delta = rowsum(dO * O) fused into dQ kernel prologue (STORE_DELTA=True).
        # Same bandwidth (dO already loaded by dQ; O is the only new read) but saves
        # an entire _delta_kernel launch. At short N, the standalone delta kernel was
        # launch-bound (grid = N × B × H_Q tiny programs of 512-byte work).
        delta = torch.empty(B, H_Q, N, dtype=torch.float32, device=q.device)

        dq = torch.empty_like(q)
        GQA_RATIO = H_Q // H_KV
        # dk, dv allocated after dKV kernel (expanded buffer + reduce)

        # --- dQ kernel ---
        # D-specific tuning (sweeps in context/baseline.md):
        #   D=512: (BQ=32, BKV=64, w=8)  — register-constrained, need 8 warps
        #   D=256: (BQ=64, BKV=64, w=4)  — more headroom, larger BQ + fewer warps win
        if D >= 512:
            BLOCK_Q_BW, BLOCK_KV_BW, num_warps_bw = 32, 64, 8
        else:
            BLOCK_Q_BW, BLOCK_KV_BW, num_warps_bw = 64, 64, 4
        BLOCK_Q_BW = min(BLOCK_Q_BW, triton.next_power_of_2(N))
        BLOCK_KV_BW = min(BLOCK_KV_BW, triton.next_power_of_2(N))
        grid_dq = (triton.cdiv(N, BLOCK_Q_BW), B * H_Q)

        _flash_attn_gqa_bwd_dq_kernel[grid_dq](
            q, k, v, do, o, dq, lse, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N,
            HEAD_DIM=D, scale=scale,
            BLOCK_Q=BLOCK_Q_BW, BLOCK_KV=BLOCK_KV_BW,
            IS_CAUSAL=causal,
            SLIDE_SIZE=slide_size,
            STORE_DELTA=True,
            GroupIds_ptr=group_ids, GroupLo_ptr=group_lo, GroupHi_ptr=group_hi_excl,
            stride_gb=g_strides_dq[0], stride_gn=g_strides_dq[1],
            HAS_GROUP_IDS=has_group_ids,
            num_warps=num_warps_bw, num_stages=2,
        )

        # --- dK, dV kernel (packed: KV-major, inline GQA Q-head loop) ---
        # Inspired by flash-attention's pack_gqa=True mode:
        #   Grid = (cdiv(N, BKV), B × H_KV) — one program per KV block.
        #   Inner loop iterates all GQA_RATIO Q heads, accumulating into a
        #   single dk_acc/dv_acc per program → direct write, no expand buffer,
        #   no atomic, no reduce kernel.
        # Saves GQA_RATIO × (B × H_KV × N × D × 2 × 2) bytes vs the old
        # split+reduce design (≈1 GB at Gemma-4-E2B N=32K).
        #
        # The old split path (_flash_attn_gqa_bwd_dkv_kernel with expand+reduce)
        # is kept in the source for reference but no longer on the hot path.
        if D >= 512:
            # Pack-GQA sweep @ N=4K Gemma4 (H_Q=32,H_KV=4): (BKV=16,BQ=64,w=4)
            # wins 9.36ms vs old (32,16,8)=13.13ms (-29%); BKV=16 halves
            # dk_acc/dv_acc shmem (32KB×2 vs 64KB×2), freeing budget for BQ=64
            # which cuts the inner Q loop 4× and reuses each Q/dO tile better.
            BLOCK_KV_DKV, BLOCK_Q_DKV, num_warps_dkv = 16, 64, 4
            num_stages_dkv = 2
        else:
            # D<512: (BKV=64, BQ=128, w=8, s=1) is the big-tile config that
            # wins when grid is healthy. We use it in TWO regimes:
            #   (a) grid_at_bkv64 >= 128: plain one-wave+ (Config A any N,
            #       Config B long N) — QS=1 below.
            #   (b) grid_at_bkv64 <= 16: extreme starve (Config B short N,
            #       H_KV=1, N≤1K) — rely on QS=8 to hit ~128 programs; big
            #       tile amortizes atomic cost. Sweep (2026-04-17,
            #       `dkv_config_b_bkv64.py`) @ Config B N=1K:
            #         (BKV=32,BQ=64,w=4,s=2,QS=8) = 0.153 ms
            #         (BKV=64,BQ=128,w=8,s=1,QS=8) = 0.144 ms (-6%)
            # Middle regime (grid_at_bkv64 in 32-64, Config B N=2K/4K): BKV=32
            # with tighter QS to hit ~256 programs wins slightly.
            grid_at_bkv64 = triton.cdiv(N, 64) * B * H_KV
            if grid_at_bkv64 >= 128 or grid_at_bkv64 <= 16:
                BLOCK_KV_DKV, BLOCK_Q_DKV, num_warps_dkv = 64, 128, 8
                num_stages_dkv = 1
            else:
                BLOCK_KV_DKV, BLOCK_Q_DKV, num_warps_dkv = 32, 64, 4
                num_stages_dkv = 2
        BLOCK_KV_DKV = min(BLOCK_KV_DKV, triton.next_power_of_2(N))
        BLOCK_Q_DKV = min(BLOCK_Q_DKV, triton.next_power_of_2(N))

        # Q_SPLITS: split each KV block's Q loop across multiple programs via
        # atomic_add, activated when the raw grid < target. Target differs by
        # BLOCK_KV (tuned 2026-04-17 in `dkv_config_b_bkv64.py`):
        #   BKV=64 (big tile): target ≈ 128 programs (one wave), each program
        #     has enough work to amortize atomic cost without needing 2 waves.
        #   BKV=32 (small tile): target ≈ 256 programs (two waves), since
        #     per-program work is halved and needs more parallelism.
        # Atomic cost (GQA_RATIO × QS programs per dK/dV tile contending) is
        # offset by better SM occupancy in low-grid regime. For healthy grids
        # QS>1 is net-negative by ~5-15%.
        raw_grid_dkv = triton.cdiv(N, BLOCK_KV_DKV) * B * H_KV
        target_grid = 128 if BLOCK_KV_DKV == 64 else 256
        if raw_grid_dkv >= target_grid:
            Q_SPLITS_DKV = 1
        elif raw_grid_dkv * 2 >= target_grid:
            Q_SPLITS_DKV = 2
        elif raw_grid_dkv * 4 >= target_grid:
            Q_SPLITS_DKV = 4
        else:
            Q_SPLITS_DKV = 8

        grid_dkv = (triton.cdiv(N, BLOCK_KV_DKV), B * H_KV, Q_SPLITS_DKV)

        # atomic_add path requires pre-zeroed buffers (multiple programs contribute).
        # Alloc micro-optimization (2026-04-17, `benchmarks/alloc_overhead.py`):
        # combine dk+dv into one 5D empty, zero once, then slice. Saves 1
        # allocator call + 1 cudaMemsetAsync for the QS>1 path. Tiny (~5μs) but
        # free since semantically identical.
        if Q_SPLITS_DKV > 1:
            # Triton 3.6 atomic_add doesn't support bf16 — accumulate in fp32
            dkv = torch.zeros((2,) + k.shape, dtype=torch.float32, device=k.device)
            dk, dv = dkv[0], dkv[1]
        else:
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)

        _flash_attn_gqa_bwd_dkv_packed_kernel[grid_dkv](
            q, k, v, do, dk, dv, lse, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N,
            HEAD_DIM=D, scale=scale,
            BLOCK_Q=BLOCK_Q_DKV, BLOCK_KV=BLOCK_KV_DKV,
            GQA_RATIO=GQA_RATIO,
            IS_CAUSAL=causal,
            SLIDE_SIZE=slide_size,
            Q_SPLITS=Q_SPLITS_DKV,
            GroupIds_ptr=group_ids, GroupLo_ptr=group_lo, GroupHi_ptr=group_hi_excl,
            stride_gb=g_strides_dq[0], stride_gn=g_strides_dq[1],
            HAS_GROUP_IDS=has_group_ids,
            num_warps=num_warps_dkv, num_stages=num_stages_dkv,
        )

        if Q_SPLITS_DKV > 1:
            dk = dk.to(k.dtype)
            dv = dv.to(v.dtype)

        return dq, dk, dv, None, None, None, None, None


def flash_attn_gqa_train(q, k, v, causal=False, slide_size=0,
                         group_ids=None, group_lo=None, group_hi_excl=None):
    """Flash Attention GQA with backward pass support for training.

    Args:
        q: (B, N_Q_HEADS, N, D)
        k: (B, N_KV_HEADS, N, D)
        v: (B, N_KV_HEADS, N, D)
        causal:     causal masking
        slide_size: sliding window size (0 = disabled, must be causal=True to have effect)
        group_ids / group_lo / group_hi_excl: optional (B, N) int32 tensors
            enabling the image-bidirectional OR-mask path for Gemma-4
            multimodal training. See `attention_flash_gqa` for semantics.
    """
    return FlashAttnGQAFunction.apply(
        q, k, v, causal, slide_size, group_ids, group_lo, group_hi_excl,
    )


# =====================================================================
# Flash Attention GQA — Varlen (packed / cu_seqlens) autograd Function
#
# Packed / FA2-compatible API: tensors are (total_tokens, H, D); cu_seqlens
# marks per-sample boundaries. See docstring of `flash_attn_gqa_varlen` for
# semantics.
#
# v1 assumes cu_seqlens_q == cu_seqlens_k (same packing for Q and K) — the
# standard training case. Distinct Q/K packing is a v2 extension.
# =====================================================================

class FlashAttnGQAVarlenFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, causal, window_size):
        total_q, H_Q, D = q.shape
        _, H_KV, _ = k.shape
        B = cu_seqlens_q.numel() - 1

        # Normalize: window covering max_seqlen_k means degenerate full-causal.
        slide_size = int(window_size)
        if slide_size > 0 and slide_size >= int(max_seqlen_k):
            slide_size = 0

        output = torch.empty_like(q)
        lse = torch.empty(total_q, H_Q, dtype=torch.float32, device=q.device)

        # Forward block sizes.
        # H200 + Triton 3.2 shmem budget is 228 KB. Each (BQ, BKV) block
        # footprint is roughly (BQ + 2 * num_stages * BKV) * HEAD_DIM * 2 bytes
        # for the Q/K/V tiles plus the fp32 accumulator (BQ * D * 4). The
        # batched kernel's original table (BQ=64/BKV=32 @ D=512, BQ=128/BKV=64
        # @ D=256) was tuned against older Triton's looser accounting; Triton
        # 3.2 reports 256-257 KB for those, just over the cap. Shrink:
        #   D=512: BQ=32, BKV=32 (was BQ=64)
        #   D=256: BQ=64, BKV=64 (was BQ=128) — keep larger BKV for HBM reuse
        #   D<256: BQ=128, BKV=64 (fits easily)
        # tl.dot requires tile dims ≥ 16, so clamp the floor to 16.
        if D >= 512:
            BLOCK_Q = 32
            BLOCK_KV = 32
            num_warps = 4
        elif D >= 256:
            BLOCK_Q = 64
            BLOCK_KV = 64
            num_warps = 4
        else:
            BLOCK_Q = 128
            BLOCK_KV = 64
            num_warps = 4
        num_stages = 2
        BLOCK_Q = max(16, min(BLOCK_Q, triton.next_power_of_2(int(max_seqlen_q))))
        BLOCK_KV = max(16, min(BLOCK_KV, triton.next_power_of_2(int(max_seqlen_k))))

        grid = (triton.cdiv(int(max_seqlen_q), BLOCK_Q), H_Q, B)
        _flash_attn_gqa_varlen_fwd_kernel[grid](
            q, k, v, output,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            cu_seqlens_q, cu_seqlens_k,
            N_Q_HEADS=H_Q, N_KV_HEADS=H_KV,
            HEAD_DIM=D,
            scale=1.0 / math.sqrt(D),
            BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, BLOCK_D=D,
            IS_CAUSAL=causal, SLIDE_SIZE=slide_size,
            LSE_ptr=lse, stride_lset=lse.stride(0), stride_lseh=lse.stride(1),
            STORE_LSE=True,
            num_warps=num_warps, num_stages=num_stages,
        )

        ctx.save_for_backward(q, k, v, output, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.max_seqlen_q = int(max_seqlen_q)
        ctx.max_seqlen_k = int(max_seqlen_k)
        ctx.causal = causal
        ctx.slide_size = slide_size
        ctx.H_KV = H_KV
        return output

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        causal = ctx.causal
        slide_size = ctx.slide_size
        H_KV = ctx.H_KV
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k

        total_q, H_Q, D = q.shape
        total_k, _, _ = k.shape
        B = cu_seqlens_q.numel() - 1
        scale = 1.0 / math.sqrt(D)
        GQA_RATIO = H_Q // H_KV

        # Packed delta (filled by dQ prologue via STORE_DELTA=True).
        delta = torch.empty(total_q, H_Q, dtype=torch.float32, device=q.device)
        dq = torch.empty_like(q)

        # --- dQ kernel ---
        if D >= 512:
            BLOCK_Q_BW, BLOCK_KV_BW, num_warps_bw = 32, 64, 8
        else:
            BLOCK_Q_BW, BLOCK_KV_BW, num_warps_bw = 64, 64, 4
        BLOCK_Q_BW = max(16, min(BLOCK_Q_BW, triton.next_power_of_2(max_seqlen_q)))
        BLOCK_KV_BW = max(16, min(BLOCK_KV_BW, triton.next_power_of_2(max_seqlen_k)))
        grid_dq = (triton.cdiv(max_seqlen_q, BLOCK_Q_BW), H_Q, B)

        _flash_attn_gqa_varlen_bwd_dq_kernel[grid_dq](
            q, k, v, do, o, dq, lse, delta,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            lse.stride(0), lse.stride(1),
            delta.stride(0), delta.stride(1),
            cu_seqlens_q, cu_seqlens_k,
            N_Q_HEADS=H_Q, N_KV_HEADS=H_KV,
            HEAD_DIM=D, scale=scale,
            BLOCK_Q=BLOCK_Q_BW, BLOCK_KV=BLOCK_KV_BW,
            IS_CAUSAL=causal, SLIDE_SIZE=slide_size,
            STORE_DELTA=True,
            num_warps=num_warps_bw, num_stages=2,
        )

        # --- dK/dV kernel (pack-GQA) ---
        if D >= 512:
            BLOCK_KV_DKV, BLOCK_Q_DKV, num_warps_dkv = 16, 64, 4
            num_stages_dkv = 2
        else:
            grid_at_bkv64 = triton.cdiv(max_seqlen_k, 64) * B * H_KV
            if grid_at_bkv64 >= 128 or grid_at_bkv64 <= 16:
                BLOCK_KV_DKV, BLOCK_Q_DKV, num_warps_dkv = 64, 128, 8
                num_stages_dkv = 1
            else:
                BLOCK_KV_DKV, BLOCK_Q_DKV, num_warps_dkv = 32, 64, 4
                num_stages_dkv = 2
        BLOCK_KV_DKV = max(16, min(BLOCK_KV_DKV, triton.next_power_of_2(max_seqlen_k)))
        BLOCK_Q_DKV = max(16, min(BLOCK_Q_DKV, triton.next_power_of_2(max_seqlen_q)))

        raw_grid_dkv = triton.cdiv(max_seqlen_k, BLOCK_KV_DKV) * B * H_KV
        target_grid = 128 if BLOCK_KV_DKV == 64 else 256
        if raw_grid_dkv >= target_grid:
            Q_SPLITS_DKV = 1
        elif raw_grid_dkv * 2 >= target_grid:
            Q_SPLITS_DKV = 2
        elif raw_grid_dkv * 4 >= target_grid:
            Q_SPLITS_DKV = 4
        else:
            Q_SPLITS_DKV = 8

        grid_dkv = (triton.cdiv(max_seqlen_k, BLOCK_KV_DKV), B * H_KV, Q_SPLITS_DKV)

        if Q_SPLITS_DKV > 1:
            # Triton 3.6 atomic_add doesn't support bf16 — accumulate in fp32
            dkv = torch.zeros((2,) + k.shape, dtype=torch.float32, device=k.device)
            dk, dv = dkv[0], dkv[1]
        else:
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)

        _flash_attn_gqa_varlen_bwd_dkv_packed_kernel[grid_dkv](
            q, k, v, do, dk, dv, lse, delta,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            lse.stride(0), lse.stride(1),
            delta.stride(0), delta.stride(1),
            cu_seqlens_q, cu_seqlens_k,
            N_Q_HEADS=H_Q, N_KV_HEADS=H_KV,
            HEAD_DIM=D, scale=scale,
            BLOCK_Q=BLOCK_Q_DKV, BLOCK_KV=BLOCK_KV_DKV,
            GQA_RATIO=GQA_RATIO,
            IS_CAUSAL=causal, SLIDE_SIZE=slide_size,
            Q_SPLITS=Q_SPLITS_DKV,
            num_warps=num_warps_dkv, num_stages=num_stages_dkv,
        )

        if Q_SPLITS_DKV > 1:
            dk = dk.to(k.dtype)
            dv = dv.to(v.dtype)

        return dq, dk, dv, None, None, None, None, None, None


def flash_attn_gqa_varlen(q, k, v,
                          cu_seqlens_q, cu_seqlens_k,
                          max_seqlen_q, max_seqlen_k,
                          causal=False, window_size=0):
    """Variable-length (packed-sequence) Flash Attention with GQA and SWA.

    FA2-compatible API. Tensors are packed across samples along a single token
    axis; per-sample boundaries come from cu_seqlens.

    Args:
        q: (total_q, N_Q_HEADS, D)
        k: (total_k, N_KV_HEADS, D)
        v: (total_k, N_KV_HEADS, D)
        cu_seqlens_q: int32, shape (B+1,), cumulative sample-start offsets for q.
        cu_seqlens_k: int32, shape (B+1,), cumulative sample-start offsets for k/v.
        max_seqlen_q: int — max over (cu_seqlens_q[i+1] - cu_seqlens_q[i]).
        max_seqlen_k: int — max over (cu_seqlens_k[i+1] - cu_seqlens_k[i]).
        causal: bool. When True and window_size=0, standard causal block-diagonal.
        window_size: 0 disables SWA. >0 applies a per-sample left window of size
            `window_size` (equivalent to slide_size in the batched API).

    Returns:
        out: (total_q, N_Q_HEADS, D), same dtype as q.

    v1 limitations:
        - Text-only (no image-group OR-mask).
        - Assumes cu_seqlens_q == cu_seqlens_k (same packing for Q and K/V).
          Distinct Q/K packing is a v2 extension.
    """
    # --- Type + shape validation ---
    assert q.is_cuda and k.is_cuda and v.is_cuda, "q/k/v must be on CUDA"
    assert q.dtype == k.dtype == v.dtype, "q/k/v dtype mismatch"
    assert q.dtype in (torch.float16, torch.bfloat16), \
        f"dtype must be fp16/bf16, got {q.dtype}"
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3, \
        f"q/k/v must be 3-D packed (total, H, D), got {q.shape}, {k.shape}, {v.shape}"
    assert k.shape == v.shape, f"k/v shape mismatch: {k.shape} vs {v.shape}"
    total_q, H_Q, D = q.shape
    total_k, H_KV, D_kv = k.shape
    assert D == D_kv, f"q/k head_dim mismatch: {D} vs {D_kv}"
    assert H_Q % H_KV == 0, f"GQA ratio must be integer: H_Q={H_Q}, H_KV={H_KV}"
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32, \
        "cu_seqlens must be int32"
    assert cu_seqlens_q.is_cuda and cu_seqlens_k.is_cuda, "cu_seqlens must be on CUDA"
    assert cu_seqlens_q.device == q.device and cu_seqlens_k.device == q.device, \
        "cu_seqlens device mismatch"
    assert cu_seqlens_q.numel() == cu_seqlens_k.numel(), \
        "cu_seqlens_q and cu_seqlens_k must have equal length"
    # v1: require matching Q/K packing. Helpful error message for v2 upgrade path.
    assert torch.equal(cu_seqlens_q, cu_seqlens_k), (
        "v1 requires cu_seqlens_q == cu_seqlens_k (same packing for Q and KV). "
        "Distinct Q/K packing is a v2 extension."
    )
    assert int(max_seqlen_q) > 0 and int(max_seqlen_k) > 0, "max_seqlen must be positive"

    # Token-axis contiguity: kernel assumes D-axis is last-contiguous.
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, \
        "D axis must be contiguous (stride(-1) == 1)"

    return FlashAttnGQAVarlenFunction.apply(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        int(max_seqlen_q), int(max_seqlen_k),
        bool(causal), int(window_size),
    )


# --- Benchmark ---
if __name__ == "__main__":
    from utils import benchmark, benchmark_fn
    import sys
    from functools import partial

    def make_qkv(shape, dtype, device):
        B, H, N, D = shape
        q = torch.randn(B, H, N, D, dtype=dtype, device=device)
        k = torch.randn(B, H, N, D, dtype=dtype, device=device)
        v = torch.randn(B, H, N, D, dtype=dtype, device=device)
        return (q, k, v)

    def make_qkv_gqa(shape, dtype, device):
        """shape = (B, H_Q, N, D, H_KV)"""
        B, H_Q, N, D, H_KV = shape
        q = torch.randn(B, H_Q, N, D, dtype=dtype, device=device)
        k = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)
        v = torch.randn(B, H_KV, N, D, dtype=dtype, device=device)
        return (q, k, v)

    mode = sys.argv[1] if len(sys.argv) > 1 else "mha"

    if mode == "gemma4":
        # Gemma4 config: H_Q=32, H_KV=4, D=512
        print("=== Gemma4 GQA Benchmark (H_Q=32, H_KV=4, D=512) ===")
        benchmark(
            implementations={
                "pytorch_sdpa": attention_gqa_ref,
                "flash_gqa": attention_flash_gqa,
            },
            input_shapes=[
                # (B, H_Q, N, D, H_KV)
                (1, 32, 128, 512, 4),
                (1, 32, 256, 512, 4),
                (1, 32, 512, 512, 4),
                (1, 32, 1024, 512, 4),
                (1, 32, 2048, 512, 4),
                (1, 32, 4096, 512, 4),
                (2, 32, 1024, 512, 4),
                (2, 32, 2048, 512, 4),
            ],
            input_fn=make_qkv_gqa,
            dtype=torch.float16,
            device="cuda",
            warmup=10,
            rep=100,
            verify=True,
            atol=5e-2,
            rtol=5e-2,
        )

    elif mode == "causal":
        # Gemma4 causal attention benchmark
        print("=== Gemma4 GQA Causal Benchmark (H_Q=32, H_KV=4, D=512) ===")
        ref_causal = partial(attention_gqa_ref, causal=True)
        triton_causal = partial(attention_flash_gqa, causal=True)
        benchmark(
            implementations={
                "pytorch_sdpa_causal": ref_causal,
                "flash_gqa_causal": triton_causal,
            },
            input_shapes=[
                (1, 32, 128, 512, 4),
                (1, 32, 256, 512, 4),
                (1, 32, 512, 512, 4),
                (1, 32, 1024, 512, 4),
                (1, 32, 2048, 512, 4),
                (1, 32, 4096, 512, 4),
                (2, 32, 2048, 512, 4),
            ],
            input_fn=make_qkv_gqa,
            dtype=torch.float16,
            device="cuda",
            warmup=10,
            rep=100,
            verify=True,
            atol=5e-2,
            rtol=5e-2,
        )

    elif mode == "long":
        # Long sequence benchmark (Gemma4 non-causal + causal)
        print("=== Gemma4 Long Sequence Benchmark (H_Q=32, H_KV=4, D=512) ===")
        ref_causal = partial(attention_gqa_ref, causal=True)
        triton_causal = partial(attention_flash_gqa, causal=True)
        benchmark(
            implementations={
                "sdpa_causal": ref_causal,
                "triton_causal": triton_causal,
            },
            input_shapes=[
                (1, 32, 4096, 512, 4),
                (1, 32, 8192, 512, 4),
                (1, 32, 16384, 512, 4),
            ],
            input_fn=make_qkv_gqa,
            dtype=torch.float16,
            device="cuda",
            warmup=5,
            rep=20,
            verify=True,
            atol=5e-2,
            rtol=5e-2,
        )

    elif mode == "bf16":
        # BF16 benchmark
        print("=== Gemma4 GQA BF16 Benchmark (H_Q=32, H_KV=4, D=512) ===")
        benchmark(
            implementations={
                "pytorch_sdpa": attention_gqa_ref,
                "flash_gqa": attention_flash_gqa,
            },
            input_shapes=[
                (1, 32, 512, 512, 4),
                (1, 32, 1024, 512, 4),
                (1, 32, 2048, 512, 4),
                (1, 32, 4096, 512, 4),
            ],
            input_fn=make_qkv_gqa,
            dtype=torch.bfloat16,
            device="cuda",
            warmup=10,
            rep=100,
            verify=True,
            atol=5e-2,
            rtol=5e-2,
        )

    elif mode == "sweep":
        # Sweep block sizes for Gemma4 to find best config
        print("=== Block Size Sweep for Gemma4 (B=1, H_Q=32, H_KV=4, N=1024, D=512) ===")
        shape = (1, 32, 1024, 512, 4)
        args = make_qkv_gqa(shape, torch.float16, "cuda")
        ref_out = attention_gqa_ref(*args)
        ref_time = benchmark_fn(attention_gqa_ref, *args, warmup=10, rep=50)
        print(f"PyTorch SDPA: {ref_time:.4f} ms\n")

        configs = [
            # (BLOCK_Q, BLOCK_KV, BLOCK_D, num_warps, num_stages)
            (16, 32, 128, 4, 2),
            (16, 32, 128, 4, 3),
            (16, 32, 128, 8, 2),
            (16, 64, 128, 4, 2),
            (16, 64, 128, 4, 3),
            (16, 64, 128, 8, 2),
            (16, 64, 64, 4, 3),
            (16, 64, 256, 4, 3),
            (32, 32, 128, 4, 2),
            (32, 32, 128, 8, 2),
            (32, 64, 128, 4, 2),
            (32, 64, 128, 8, 2),
            (64, 32, 128, 4, 2),
            (64, 64, 128, 4, 2),
            (16, 128, 128, 4, 2),
            (16, 128, 128, 8, 2),
        ]
        print(f"{'Config (BQ,BKV,BD,W,S)':<30} {'Time (ms)':>10} {'vs SDPA':>8} {'Correct':>8}")
        print("-" * 60)
        for bq, bkv, bd, nw, ns in configs:
            try:
                fn = partial(attention_flash_gqa,
                             BLOCK_Q=bq, BLOCK_KV=bkv, BLOCK_D=bd,
                             num_warps=nw, num_stages=ns)
                out = fn(*args)
                correct = torch.allclose(ref_out, out, atol=5e-2, rtol=5e-2)
                t = benchmark_fn(fn, *args, warmup=10, rep=50)
                ratio = ref_time / t
                print(f"({bq:>2},{bkv:>3},{bd:>3},{nw},{ns})"
                      f"{'':>16} {t:>10.4f} {ratio:>7.2f}x {'OK' if correct else 'FAIL':>8}")
            except Exception as e:
                print(f"({bq:>2},{bkv:>3},{bd:>3},{nw},{ns})"
                      f"{'':>16} {'ERROR':>10} {str(e)[:40]}")

    elif mode == "swa":
        # Sliding Window Attention benchmark and correctness test.
        # Gemma4 Sliding layer config: H_Q=32, H_KV=16, D=256, slide_size=1024
        # (distinct from the full-attention layer config H_KV=4, D=512).
        SLIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
        print(f"=== Gemma4 Sliding Attention (slide_size={SLIDE}, H_Q=32, H_KV=16, D=256) ===")
        print()

        # --- Correctness check ---
        print("Correctness check:")
        for N in [128, 256, 512, 1024, 2048, 4096]:
            shape = (1, 32, N, 256, 16)
            args = make_qkv_gqa(shape, torch.float16, "cuda")
            ref_out = attention_swa_ref(*args, slide_size=SLIDE)
            triton_out = attention_flash_gqa(*args, causal=True, slide_size=SLIDE)
            ok = torch.allclose(ref_out, triton_out, atol=5e-2, rtol=5e-2)
            max_err = (ref_out - triton_out).abs().max().item()
            print(f"  N={N:>5}: {'OK' if ok else 'FAIL'} (max_err={max_err:.4f})")

        print()

        # --- Forward benchmark: SWA vs full causal vs SDPA ---
        print("Forward benchmark:")
        triton_causal = partial(attention_flash_gqa, causal=True)
        triton_swa = partial(attention_flash_gqa, causal=True, slide_size=SLIDE)
        sdpa_causal = partial(attention_gqa_ref, causal=True)
        benchmark(
            implementations={
                "sdpa_causal":   sdpa_causal,
                "triton_causal": triton_causal,
                f"triton_swa_{SLIDE}": triton_swa,
            },
            input_shapes=[
                (1, 32, 512,  256, 16),
                (1, 32, 1024, 256, 16),
                (1, 32, 2048, 256, 16),
                (1, 32, 4096, 256, 16),
                (1, 32, 8192, 256, 16),
                (1, 32, 16384, 256, 16),
            ],
            input_fn=make_qkv_gqa,
            dtype=torch.float16,
            device="cuda",
            warmup=10,
            rep=30,
            verify=False,
        )

        print()

        # --- Backward correctness check ---
        print("Backward correctness check (train mode):")
        for N in [128, 256, 512, 1024, 2048]:
            shape = (1, 32, N, 256, 16)
            args_ref  = make_qkv_gqa(shape, torch.float16, "cuda")
            args_tri  = [t.clone().requires_grad_(True) for t in args_ref]
            args_ref  = [t.clone().requires_grad_(True) for t in args_ref]

            ref_out = attention_swa_ref(*args_ref, slide_size=SLIDE)
            ref_out.sum().backward()

            tri_out = flash_attn_gqa_train(*args_tri, causal=True, slide_size=SLIDE)
            tri_out.sum().backward()

            dq_ok = torch.allclose(args_ref[0].grad, args_tri[0].grad, atol=1e-1, rtol=1e-1)
            dk_ok = torch.allclose(args_ref[1].grad, args_tri[1].grad, atol=1e-1, rtol=1e-1)
            dv_ok = torch.allclose(args_ref[2].grad, args_tri[2].grad, atol=1e-1, rtol=1e-1)
            print(f"  N={N:>5}: dQ={'OK' if dq_ok else 'FAIL'} dK={'OK' if dk_ok else 'FAIL'} dV={'OK' if dv_ok else 'FAIL'}")

    elif mode == "swa_bwd":
        SLIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
        print("=== SWA Fwd+Bwd Benchmark (slide=%d, H_Q=32, H_KV=16, D=256) ===" % SLIDE)
        print()

        def sdpa_causal_fwd_bwd(q, k, v):
            ratio = q.shape[1] // k.shape[1]
            k_exp = k.repeat_interleave(ratio, dim=1)
            v_exp = v.repeat_interleave(ratio, dim=1)
            out = torch.nn.functional.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
            out.sum().backward()
            q.grad = k.grad = v.grad = None

        def triton_swa_fwd_bwd(q, k, v):
            out = flash_attn_gqa_train(q, k, v, causal=True, slide_size=SLIDE)
            out.sum().backward()
            q.grad = k.grad = v.grad = None

        print("%6s | %16s | %15s | %8s" % ("N", "SDPA-causal (ms)", "Triton-SWA (ms)", "Speedup"))
        print("-" * 56)
        for N in [512, 1024, 2048, 4096, 8192]:
            q = torch.randn(1, 32, N, 256, dtype=torch.float16, device="cuda").requires_grad_(True)
            k = torch.randn(1, 16, N, 256, dtype=torch.float16, device="cuda").requires_grad_(True)
            v = torch.randn(1, 16, N, 256, dtype=torch.float16, device="cuda").requires_grad_(True)
            t_sdpa   = benchmark_fn(sdpa_causal_fwd_bwd, q, k, v, warmup=5, rep=20)
            t_triton = benchmark_fn(triton_swa_fwd_bwd,  q, k, v, warmup=5, rep=20)
            print("%6d | %16.3f | %15.3f | %7.2fx" % (N, t_sdpa, t_triton, t_sdpa/t_triton))

    else:
        # Original MHA benchmark
        benchmark(
            implementations={
                "pytorch": attention,
                "triton": attention_triton,
                "triton_opt": attention_triton_opt,
            },
            input_shapes=[
                # (B, H, N, D)
                (1, 8, 128, 64),
                (1, 8, 256, 64),
                (1, 8, 512, 64),
                (1, 8, 1024, 64),
                (2, 8, 1024, 64),
                (1, 8, 2048, 64),
                (1, 8, 4096, 64),
            ],
            input_fn=make_qkv,
            dtype=torch.float16,
            device="cuda",
            warmup=10,
            rep=100,
            verify=True,
            atol=5e-2,
            rtol=5e-2,
        )
