"""Scientific comparison chart from measured JSON; no GPU work."""

from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
rows = json.loads(
    (ROOT / "experiments/transolver_lab/evidence/publication/summary.json").read_text()
)
cases = sorted({r["case"] for r in rows})
modes = ["fp32", "tf32", "v3"]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
colors = ["#607d8b", "#ed9e3d", "#168e8d"]
for ax, key, label, scale in zip(
    axes,
    ["median_ms", "peak_allocated_mb", "pressure_l2_percent"],
    ["Median inference (s)", "Max allocated memory (GB)", "Pressure relative L2 (%)"],
    [1000, 1000, 1],
):
    for i, (mode, color) in enumerate(zip(modes, colors)):
        values = [
            next(r[key] for r in rows if r["case"] == case and r["mode"] == mode)
            / scale
            for case in cases
        ]
        ax.bar(
            np.arange(len(cases)) + (i - 1) * 0.24,
            values,
            width=0.23,
            color=color,
            label=mode.upper(),
        )
    ax.set_xticks(range(len(cases)), [f"Run {c}" for c in cases])
    ax.set_title(label, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle(
    "Three-case H100 validation • five warmed repetitions per method", fontsize=12
)
fig.savefig(ROOT / "docs/papers/arxiv/benchmark.png", dpi=180)
plt.close(fig)
