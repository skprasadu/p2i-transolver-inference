"""Milestone 3B-v2: fail-closed Layer0 gate before full-model benchmarking."""

import json
from pathlib import Path
import modal
from experiments.transolver_lab.benchmark_drivaer_cuda import image, volume

app = modal.App("p2i-transolver-v2-gate1")


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="H100!",
    cpu=8,
    memory=65536,
    timeout=2400,
    retries=0,
)
def execute(memcheck_only: bool = False):
    import subprocess, sys

    root = "/app/experiments/transolver_lab"
    subprocess.run(
        [sys.executable, "cuda/setup_v2.py", "build_ext", "--inplace"],
        cwd=root,
        check=True,
        timeout=600,
    )
    cmd = [sys.executable, "-m", "experiments.transolver_lab.drivaer_v2_worker"]
    if not memcheck_only:
        subprocess.run(cmd + ["probe"], check=True, timeout=300)
    volume.commit()
    probe = json.loads(Path("/data/v2-probe.json").read_text())
    if not probe["selected"]:
        return {"probe": probe, "passed": False}, ""
    if not memcheck_only:
        subprocess.run(cmd + ["gate"], check=True, timeout=600)
    volume.commit()
    logs = []
    safe = True
    for begin, stop in [
        (0, 2000000),
        (2000000, 4000000),
        (4000000, 6000000),
        (6000000, 8140177),
    ]:
        done = subprocess.run(
            [
                "compute-sanitizer",
                "--tool",
                "memcheck",
                "--error-exitcode",
                "99",
                *cmd,
                "sanitize",
                str(begin),
                str(stop),
            ],
            capture_output=True,
            text=True,
            timeout=400,
        )
        group = done.stdout + done.stderr
        print(group, flush=True)
        logs.append(group)
        safe = safe and done.returncode == 0 and "ERROR SUMMARY: 0 errors" in group
        if not safe:
            break
    log = "\n".join(logs)
    Path("/data/v2-memcheck.log").write_text(log)
    gate = json.loads(Path("/data/v2-gate.json").read_text())
    gate["memcheck_passed"] = safe
    gate["passed"] = gate["passed"] and gate["memcheck_passed"]
    gate["probe"] = probe
    Path("/data/v2-gate.json").write_text(json.dumps(gate, indent=2))
    volume.commit()
    return gate, log


@app.local_entrypoint()
def main(memcheck_only: bool = False):
    report, log = execute.remote(memcheck_only)
    out = Path("experiments/transolver_lab/evidence")
    (out / "v2-gate.json").write_text(json.dumps(report, indent=2))
    (out / "v2-memcheck.log").write_text(log)
    print(json.dumps(report, indent=2))
    assert report["passed"], "Gate1 failed: full20-layer run prohibited"
