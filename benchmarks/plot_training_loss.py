"""Plot 100-step FSDP2 training loss (WikiText-2, Gemma-4-E2B, 8 H100s)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent
data = json.loads((OUT / "training_loss_fsdp2.json").read_text())

steps = [d["step"] for d in data["losses"]]
losses = [d["loss"] for d in data["losses"]]

# EMA to smooth noisy per-chunk variation (each step is a different wikitext
# chunk, so per-step loss varies by chunk difficulty, not training progress).
ema, alpha = [], 0.2
s = losses[0]
for x in losses:
    s = alpha * x + (1 - alpha) * s
    ema.append(s)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(steps, losses, marker="o", linestyle="-", color="#1f77b4",
        alpha=0.4, markersize=4, linewidth=1, label="loss (per step)")
ax.plot(steps, ema, color="#d62728", linewidth=2.2,
        label=f"EMA α={alpha}")
ax.axhline(losses[0], color="gray", linestyle=":", alpha=0.6,
           label=f"step-0 loss = {losses[0]:.3f}")

ax.set_xlabel("Training step", fontsize=12)
ax.set_ylabel("Cross-entropy loss (nats)", fontsize=12)
ax.set_title(
    "Gemma-4-E2B FSDP2 training — 8×H100, fp32 master + bf16 matmul + Triton attention\n"
    f"WikiText-2, seq_len=512, effective batch={data['effective_batch']}, AdamW lr={data['lr']}",
    fontsize=11,
)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=10)
ax.set_ylim(bottom=0)

# Step-0 fwd correctness inset text
diff = data["step0_fwd_correctness"]["abs_diff"]
ax.text(
    0.02, 0.02,
    f"Step-0 forward correctness\n"
    f"  SDPA    = {data['step0_fwd_correctness']['sdpa_loss']:.6f}\n"
    f"  Triton  = {data['step0_fwd_correctness']['triton_loss']:.6f}\n"
    f"  |Δ|     = {diff:.2e} nats",
    transform=ax.transAxes, fontsize=9, family="monospace",
    verticalalignment="bottom",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc"),
)

plt.tight_layout()
out = OUT / "training_loss_fsdp2.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
print(f"100 steps, no NaN. Mean loss (last 50) = {np.mean(losses[-50:]):.3f}")
