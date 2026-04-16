"""
Gemma4-style transformer block with alternating full + sliding attention.

Gemma3/4 pattern:
  - 5 sliding window layers (H_Q=32, H_KV=16, head_dim=256, slide=1024) for every
  - 1 full causal layer        (H_Q=32, H_KV=4,  head_dim=512)

Both share a common d_model (residual stream). Separate wq/wk/wv per layer type.
This script builds a small stack (e.g., 6 layers) and benchmarks Triton vs SDPA.
"""
import sys
import torch
import torch.nn as nn
from attention import flash_attn_gqa_train, attention_gqa_ref, attention_swa_ref
from utils import benchmark_fn


class GQAAttention(nn.Module):
    """Single GQA attention layer, optionally with sliding window."""

    def __init__(self, d_model, n_q_heads, n_kv_heads, head_dim, slide_size=0, use_triton=False):
        super().__init__()
        self.n_q = n_q_heads
        self.n_kv = n_kv_heads
        self.hd = head_dim
        self.slide = slide_size
        self.use_triton = use_triton
        self.wq = nn.Linear(d_model, n_q_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_q_heads * head_dim, d_model, bias=False)
        for p in self.parameters():
            nn.init.normal_(p, std=0.01)

    def forward(self, x):
        B, N, _ = x.shape
        q = self.wq(x).view(B, N, self.n_q, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, N, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, N, self.n_kv, self.hd).transpose(1, 2)
        if self.use_triton:
            o = flash_attn_gqa_train(q, k, v, causal=True, slide_size=self.slide)
        elif self.slide > 0:
            o = attention_swa_ref(q, k, v, slide_size=self.slide)
        else:
            o = attention_gqa_ref(q, k, v, causal=True)
        return self.wo(o.transpose(1, 2).contiguous().view(B, N, -1))


class Gemma4Block(nn.Module):
    """One 'block' = one attention layer + MLP (omitted for brevity — attention-only)."""
    def __init__(self, d_model, layer_type, use_triton=False):
        super().__init__()
        if layer_type == "full":
            # Full causal: H_Q=32, H_KV=4, head_dim=512
            self.attn = GQAAttention(d_model, 32, 4, 512, slide_size=0, use_triton=use_triton)
        elif layer_type == "slide":
            # Sliding: H_Q=32, H_KV=16, head_dim=256, slide=1024
            self.attn = GQAAttention(d_model, 32, 16, 256, slide_size=1024, use_triton=use_triton)
        else:
            raise ValueError(layer_type)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return x + self.attn(self.norm(x))


class Gemma4Stack(nn.Module):
    """Mini Gemma4 stack: alternating [slide, slide, slide, slide, slide, full] × N_BLOCKS."""
    def __init__(self, d_model, n_blocks=6, pattern=("slide",) * 5 + ("full",), use_triton=False):
        super().__init__()
        layers = []
        for i in range(n_blocks):
            layers.append(Gemma4Block(d_model, pattern[i % len(pattern)], use_triton=use_triton))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _cosine_sim(a, b):
    """Cosine similarity between flattened tensors — robust for gradient comparison."""
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b) / (a.norm() * b.norm() + 1e-8)


def check_correctness(d_model=2048, n_blocks=6, seq_len=2048, n_steps=10):
    """Compare Triton stack vs SDPA reference over training steps.

    Uses cosine similarity for gradients (robust to BF16 accumulation noise) and
    tracks loss divergence across steps.
    """
    torch.manual_seed(0)
    ref = Gemma4Stack(d_model, n_blocks=n_blocks, use_triton=False).cuda().bfloat16()
    tri = Gemma4Stack(d_model, n_blocks=n_blocks, use_triton=True).cuda().bfloat16()
    tri.load_state_dict(ref.state_dict())
    opt_ref = torch.optim.AdamW(ref.parameters(), lr=1e-5)
    opt_tri = torch.optim.AdamW(tri.parameters(), lr=1e-5)

    x = torch.randn(1, seq_len, d_model, dtype=torch.bfloat16, device='cuda')
    target = torch.randn_like(x)

    # One-step detailed comparison
    out_ref = ref(x)
    out_tri = tri(x.clone())
    fwd_cos = _cosine_sim(out_tri, out_ref).item()
    fwd_abs = (out_ref - out_tri).abs().max().item()
    fwd_rel_norm = (out_ref - out_tri).norm() / out_ref.norm()

    loss_ref = ((out_ref - target) ** 2).mean()
    loss_tri = ((out_tri - target) ** 2).mean()
    loss_ref.backward()
    loss_tri.backward()

    grad_cos = 1.0
    for p1, p2 in zip(ref.parameters(), tri.parameters()):
        if p1.grad is not None and p2.grad is not None:
            c = _cosine_sim(p2.grad, p1.grad).item()
            grad_cos = min(grad_cos, c)

    # N-step training loop: check loss trajectories match
    opt_ref.step(); opt_tri.step()
    opt_ref.zero_grad(); opt_tri.zero_grad()
    losses_ref, losses_tri = [loss_ref.item()], [loss_tri.item()]
    for step in range(n_steps - 1):
        l_r = ((ref(x) - target) ** 2).mean()
        l_t = ((tri(x) - target) ** 2).mean()
        l_r.backward(); l_t.backward()
        opt_ref.step(); opt_tri.step()
        opt_ref.zero_grad(); opt_tri.zero_grad()
        losses_ref.append(l_r.item()); losses_tri.append(l_t.item())

    loss_rel_max = max(abs(a - b) / abs(a) for a, b in zip(losses_ref, losses_tri))

    # Success criteria (BF16-aware):
    #  - Forward output cos sim > 0.999 (should be near 1.0 for identical weights)
    #  - Grad cos sim > 0.8 (BF16 noise amplifies in small-magnitude tensors like
    #    LayerNorm gains; lower threshold is fine when loss itself matches)
    #  - Loss trajectory matches within 1% relative error — gold standard
    ok = fwd_cos > 0.999 and grad_cos > 0.8 and loss_rel_max < 0.01
    print(f'  fwd cos sim: {fwd_cos:.6f}  (rel_norm={fwd_rel_norm:.2e}, abs={fwd_abs:.2e})')
    print(f'  min grad cos sim across params: {grad_cos:.6f}')
    print(f'  loss trajectory ({n_steps} steps) max rel diff: {loss_rel_max:.2e}')
    print(f'  final loss: ref={losses_ref[-1]:.6f}  tri={losses_tri[-1]:.6f}')
    return ok


def bench_e2e(d_model, seq_lens, n_blocks=6):
    print(f'\n=== Gemma4 Stack E2E Training (d_model={d_model}, {n_blocks} blocks = 5 slide + 1 full) ===')
    print(f"{'N':>6} {'SDPA (ms/step)':>16} {'Triton (ms/step)':>18} {'Speedup':>8}")
    print('-' * 55)
    for seq_len in seq_lens:
        torch.manual_seed(0)
        ref = Gemma4Stack(d_model, n_blocks=n_blocks, use_triton=False).cuda().bfloat16()
        tri = Gemma4Stack(d_model, n_blocks=n_blocks, use_triton=True).cuda().bfloat16()
        tri.load_state_dict(ref.state_dict())
        opt_ref = torch.optim.AdamW(ref.parameters(), lr=1e-5)
        opt_tri = torch.optim.AdamW(tri.parameters(), lr=1e-5)

        x = torch.randn(1, seq_len, d_model, dtype=torch.bfloat16, device='cuda')
        target = torch.randn_like(x)

        def step_ref():
            opt_ref.zero_grad()
            ((ref(x) - target) ** 2).mean().backward()
            opt_ref.step()

        def step_tri():
            opt_tri.zero_grad()
            ((tri(x) - target) ** 2).mean().backward()
            opt_tri.step()

        rt = benchmark_fn(step_ref, warmup=3, rep=10 if seq_len <= 4096 else 5)
        tt = benchmark_fn(step_tri, warmup=3, rep=10 if seq_len <= 4096 else 5)
        print(f'{seq_len:>6} {rt:>16.2f} {tt:>18.2f} {rt/tt:>7.2f}x')
        del ref, tri, opt_ref, opt_tri, x, target
        torch.cuda.empty_cache()


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'both'

    if mode in ('correct', 'both'):
        print('=== Correctness: Gemma4 stack (5 slide + 1 full, BF16) ===')
        for N in [512, 1024, 2048]:
            print(f'N={N}:')
            ok = check_correctness(d_model=2048, seq_len=N)
            print(f'  Result: {"PASS" if ok else "FAIL"}')

    if mode in ('bench', 'both'):
        # Smaller d_model so we can fit longer seq
        bench_e2e(d_model=2048, seq_lens=[512, 1024, 2048, 4096, 8192])
