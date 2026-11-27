"""Milestone 3B-v3: one graph-captured TF32 candidate, actual CFD accuracy gate."""

import json
from pathlib import Path
import modal
from experiments.transolver_lab.benchmark_drivaer_cuda import image, volume

app = modal.App("p2i-transolver-v3")


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="H100!",
    cpu=8,
    memory=65536,
    timeout=2400,
    retries=0,
)
def execute():
    import subprocess, sys

    root = "/app/experiments/transolver_lab"
    subprocess.run(
        [sys.executable, "cuda/setup_v2.py", "build_ext", "--inplace"],
        cwd=root,
        check=True,
        timeout=600,
    )
    cmd = [sys.executable, "-m", "experiments.transolver_lab.drivaer_v3_worker"]

    def run(stage):
        subprocess.run(cmd + [stage], check=True, timeout=600)
        volume.commit()
        return json.loads(Path("/data/v3-" + stage + ".json").read_text())

    layer = run("layer")
    logs = []
    safe = True

    def check(stage, *bounds):
        nonlocal safe
        done = subprocess.run(
            [
                "compute-sanitizer",
                "--tool",
                "memcheck",
                "--error-exitcode",
                "99",
                *cmd,
                stage,
                *map(str, bounds),
            ],
            capture_output=True,
            text=True,
            timeout=500,
        )
        log = done.stdout + done.stderr
        logs.append(log)
        print(log, flush=True)
        safe = safe and done.returncode == 0 and "ERROR SUMMARY: 0 errors" in log
        Path("/data/v3-memcheck.log").write_text("\n".join(logs))
        volume.commit()

    for begin, stop in [
        (0, 2000000),
        (2000000, 4000000),
        (4000000, 6000000),
        (6000000, 8140177),
    ]:
        check("sanitize-layer", begin, stop)
        if not safe:
            break
    result = {
        "layer0": layer,
        "memcheck_passed": safe,
        "full_model": None,
        "passed": False,
    }
    if layer["accuracy_passed"] and safe:
        result["baseline"] = run("baseline")
        result["full_model"] = run("full")
        check("sanitize-model")
        result["memcheck_passed"] = safe
        result["passed"] = (
            safe
            and result["full_model"]["pressure_gate_005_percentage_points"]
            and layer["latency_ms"] < 893
        )
    Path("/data/v3-results.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result, "\n".join(logs)


@app.local_entrypoint()
def main():
    report, log = execute.remote()
    out = Path("experiments/transolver_lab/evidence")
    (out / "v3-results.json").write_text(json.dumps(report, indent=2))
    (out / "v3-memcheck.log").write_text(log)
    print(json.dumps(report, indent=2))
    from experiments.transolver_lab.report_v3 import format_table

    print(format_table(report))
    assert report["passed"], "V3 failed its explicit release gates; evidence saved"
