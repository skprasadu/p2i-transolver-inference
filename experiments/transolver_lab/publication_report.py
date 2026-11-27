"""Generate a publication table solely from saved Modal measurements (CPU only)."""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/transolver_lab/evidence/publication"


def render():
    data = json.loads((EVIDENCE / "results.json").read_text())
    rows = []
    summaries = []
    for report in data["reports"]:
        samples = report["samples"]
        assert len(samples) == 5
        times = [s["latency_ms"] for s in samples]
        summary = dict(
            case=report["case"],
            mode=report["mode"],
            cells=report["cells"],
            vertices=report["vertices"],
            median_ms=statistics.median(times),
            min_ms=min(times),
            max_ms=max(times),
            stdev_ms=statistics.stdev(times),
            peak_allocated_mb=max(s["peak_allocated_mb"] for s in samples),
            peak_reserved_mb=max(s["peak_reserved_mb"] for s in samples),
            pressure_l2_percent=max(s["pressure_l2_percent"] for s in samples),
            shear_l2_percent=max(s["shear_l2_percent"] for s in samples),
        )
        summaries.append(summary)
    assert {(r["case"], r["mode"]) for r in summaries} == {
        (c, m) for c in (4, 17, 18) for m in ("fp32", "tf32", "v3")
    }, "Incomplete publication matrix"
    summaries.sort(key=lambda r: (r["case"], ["fp32", "tf32", "v3"].index(r["mode"])))
    for r in summaries:
        base = next(
            s for s in summaries if s["case"] == r["case"] and s["mode"] == "fp32"
        )
        delta = abs(r["pressure_l2_percent"] - base["pressure_l2_percent"])
        r["pressure_delta_pp"] = delta
        r["pressure_gate_005pp"] = delta <= 0.05
        r["pressure_gate_005relative"] = delta / base["pressure_l2_percent"] <= 0.0005
        rows.append(
            f"| {r['case']} | {r['mode']} | {r['median_ms']:.1f} [{r['min_ms']:.1f}, {r['max_ms']:.1f}] | {r['peak_allocated_mb']:.1f} / {r['peak_reserved_mb']:.1f} | {r['pressure_l2_percent']:.6f} | {'Pass' if r['pressure_gate_005pp'] else 'Fail'} |"
        )
    text = (
        """## Publication validation: three cases and repeated measurements

Cases 4, 17 and 18 were selected from the pinned publisher validation split before
examining their predictions. Each method ran in a fresh process, five sequential
complete-case repetitions; execution order was rotated between cases. All used
the same checkpoint, seed 42 and 200,000-cell partition convention. Original V3
CUDA code was unchanged. This follow-up reuses the earlier scoped sanitizer evidence;
it does not claim additional sanitizer coverage on cases 17 and 18.

| Case | Method | Median ms [min, max], n=5 | Peak allocated / reserved MB | Pressure L2 % | 0.05 pp gate |
|---|---|---:|---:|---:|---|
"""
        + "\n".join(rows)
        + """

Memory columns use the maximum over all five repetitions, including graph-pool
behavior across repeated runs. They must not be replaced by the smaller first-pass
numbers. These runs test repeatability within each method process. A disconnected job was
resumed in a second H100 allocation, retaining completed configurations; hardware
identity across that boundary was not recorded. This is not a systematic
cross-host variability study. Per-repetition measurements and standard deviations
are available in the accompanying JSON files.

Globally enabling TF32 is a distinct accuracy/performance trade-off, not automatically
a valid replacement under the declared pressure gate. A same-selective-precision,
non-tiled implementation and separate graph-only control remain untested; the
present evidence cannot isolate the speed contribution of each optimization.
"""
    )
    (EVIDENCE / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (EVIDENCE / "table.md").write_text(text)
    print(text)


if __name__ == "__main__":
    render()
