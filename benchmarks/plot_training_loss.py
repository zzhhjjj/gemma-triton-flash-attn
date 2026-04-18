"""Plot 100-step FSDP2 training loss — SDPA vs Triton, Gemma-4-E2B, 8× H100."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent
data = json.loads((OUT / "training_loss_fsdp2.json").read_text())


def _ema(xs, alpha=0.2):
    out, s = [], xs[0]
    for x in xs:
        s = alpha * x + (1 - alpha) * s
        out.append(s)
    return out


sdpa = [d["loss"] for d in data["losses_sdpa"]]
tri = [d["loss"] for d in data["losses_triton"]]
steps = list(range(len(sdpa)))

fig, (ax, ax_diff) = plt.subplots(
    2, 1, figsize=(10, 7.5),
    gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
)

# Top: loss curves
ax.plot(steps, sdpa, marker="o", linestyle="-", color="#d62728",
        alpha=0.35, markersize=3, linewidth=0.8)
ax.plot(steps, tri, marker="s", linestyle="-", color="#2ca02c",
        alpha=0.35, markersize=3, linewidth=0.8)
ax.plot(steps, _ema(sdpa), color="#d62728", linewidth=2.2,
        label=f"SDPA   (mean last 50 = {np.mean(sdpa[-50:]):.3f})")
ax.plot(steps, _ema(tri), color="#2ca02c", linewidth=2.2,
        label=f"Triton (mean last 50 = {np.mean(tri[-50:]):.3f})")

ax.set_ylabel("Cross-entropy loss (nats)", fontsize=12)
ax.set_title(
    "Gemma-4-E2B FSDP2 training — SDPA vs Triton attention (8× H100)\n"
    f"fp32 master + bf16 matmul + fp32 grad reduce, "
    f"WikiText-2, seq_len={data['seq_len']}, "
    f"effective batch={data['effective_batch']}, AdamW lr={data['lr']}",
    fontsize=11,
)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=10)
ax.set_ylim(bottom=0)

d0 = data["step0_fwd_correctness"]
ax.text(
    0.02, 0.02,
    f"Step-0 forward correctness (same weights, same batch)\n"
    f"  SDPA    = {d0['sdpa_loss']:.6f}\n"
    f"  Triton  = {d0['triton_loss']:.6f}\n"
    f"  |Δ|     = {d0['abs_diff']:.2e} nats",
    transform=ax.transAxes, fontsize=9, family="monospace",
    verticalalignment="bottom",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc"),
)

# Bottom: per-step |loss diff|
abs_diff = [abs(a - b) for a, b in zip(sdpa, tri)]
ax_diff.plot(steps, abs_diff, color="#1f77b4", linewidth=1.5,
             label=f"|SDPA − Triton| per step  (mean = {np.mean(abs_diff):.3f}, "
                   f"max = {max(abs_diff):.3f})")
ax_diff.set_xlabel("Training step", fontsize=12)
ax_diff.set_ylabel("|loss diff|", fontsize=11)
ax_diff.grid(True, alpha=0.3)
ax_diff.legend(loc="upper right", fontsize=9)
ax_diff.set_ylim(bottom=0)

plt.tight_layout()
out = OUT / "training_loss_fsdp2.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
print(f"SDPA   mean last 50 = {np.mean(sdpa[-50:]):.4f}")
print(f"Triton mean last 50 = {np.mean(tri[-50:]):.4f}")
print(f"|loss diff| max = {max(abs_diff):.4f}, mean = {np.mean(abs_diff):.4f}")
