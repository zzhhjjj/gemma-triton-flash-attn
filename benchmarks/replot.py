"""Regenerate plots from benchmarks/results.json without re-running benchmarks."""
import json
from pathlib import Path

from run_final_benchmark import plot_speedup, plot_e2e, plot_memory

OUT = Path(__file__).parent
data = json.loads((OUT / "results.json").read_text())

plot_speedup(data["kernel_fwd"], data["kernel_fwdbwd"], data["e2e_fwd"],
             OUT / "flops_vs_sdpa.png")
plot_e2e(data["e2e_fwd"], OUT / "e2e_latency_vs_sdpa.png")
plot_memory(data["memory"], OUT / "memory_vs_sdpa.png")
