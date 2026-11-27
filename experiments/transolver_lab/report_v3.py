"""Format saved GPU measurements. This module performs no GPU work."""

import json
from pathlib import Path


def format_table(report):
    lines = [
        "Scope | Latency ms | Speedup vs 20,792 ms | Peak allocated MB | Pressure L2 %",
        "--- | ---: | ---: | ---: | ---:",
    ]
    layer = report["layer0"]
    lines.append(
        f"Layer 0, v3 | {layer['latency_ms']:.2f} | N/A | {layer['peak_allocated_vram_mb']:.2f} | N/A"
    )
    for key, label in [
        ("baseline", "Full20 PyTorch FP32"),
        ("full_model", "Full20 v3"),
    ]:
        row = report.get(key)
        if row:
            lines.append(
                f"{label} | {row['latency_ms']:.2f} | {row['speedup_vs_20792']:.4f}x | {row['peak_allocated_vram_mb']:.2f} | {row['pressure_relative_l2_percent']:.6f}"
            )
    lines.append(
        "Layer0 speedup uses its own reference (893.47 ms), not the full-model 20,792 ms denominator."
    )
    if report.get("full_model"):
        lines.append(
            f"Full v3 peak reserved memory: {report['full_model']['peak_reserved_vram_mb']:.2f} MB. Graph pools retain scratch storage; allocated memory alone understates the resident allocator footprint."
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(
        format_table(
            json.loads(
                Path("experiments/transolver_lab/evidence/v3-results.json").read_text()
            )
        )
    )
