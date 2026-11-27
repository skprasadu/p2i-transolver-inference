"""Milestone 3B: trained, full-case FP32 CUDA equivalence and paired timing."""

from pathlib import Path
import json
import modal
from experiments.transolver_lab.validate_drivaerml import image, volume

app = modal.App("p2i-transolver-drivaer-cuda")
image = (
    image.apt_install("build-essential")
    .pip_install("ninja", "setuptools")
    .env(
        {
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "MAX_JOBS": "2",
            "CC": "gcc",
            "CXX": "g++",
            "PYTHONPATH": "/app/experiments/transolver_lab:/app",
        }
    )
)


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="H100!",
    cpu=8,
    memory=65536,
    timeout=3600,
    retries=0,
)
def execute():
    import subprocess, sys, os

    root = "/app/experiments/transolver_lab"
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=root,
        check=True,
        timeout=600,
    )
    env = dict(os.environ, PYTHONPATH=root + ":/app")
    check = subprocess.run(
        [
            "compute-sanitizer",
            "--tool",
            "memcheck",
            "--error-exitcode",
            "99",
            sys.executable,
            "-m",
            "experiments.transolver_lab.drivaer_cuda_worker",
            "--sanitize",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=1500,
    )
    print(check.stdout, flush=True)
    print(check.stderr, flush=True)
    Path("/data/cuda-memcheck.log").write_text(check.stdout + check.stderr)
    volume.commit()
    assert (
        check.returncode == 0
        and "ERROR SUMMARY: 0 errors" in check.stdout + check.stderr
    ), "Memcheck/correctness failed"
    Path("/data/cuda-benchmark.json").unlink(missing_ok=True)
    outcome = subprocess.run(
        [sys.executable, "-m", "experiments.transolver_lab.drivaer_cuda_worker"],
        env=env,
        timeout=1800,
    )
    volume.commit()
    result = json.loads(Path("/data/cuda-benchmark.json").read_text())
    assert outcome.returncode == 0 or not result["passed"], "Unexpected worker failure"
    return result, Path("/data/cuda-memcheck.log").read_text()


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="H100!",
    cpu=8,
    memory=65536,
    timeout=3600,
    retries=0,
)
def profile():
    import subprocess, sys, csv, io

    root = "/app/experiments/transolver_lab"
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=root,
        check=True,
        timeout=600,
    )
    # Preserve the original upstream golden-reference regression as well.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from experiments.transolver_lab.test_cuda_kernel import worker; worker(sanitize=True)",
        ],
        check=True,
        timeout=300,
    )
    result = json.loads(Path("/data/cuda-benchmark.json").read_text())
    for name, flag in [("PyTorch", []), ("Fused CUDA", ["--fused"])]:
        command = [
            "ncu",
            "--clock-control",
            "none",
            "--cache-control",
            "none",
            "--profile-from-start",
            "off",
            "--metrics",
            "dram__bytes.sum",
            "--csv",
            sys.executable,
            "-m",
            "experiments.transolver_lab.profile_drivaer_cuda",
            *flag,
        ]
        done = subprocess.run(command, capture_output=True, text=True, timeout=1500)
        Path("/data/hbm-" + name.replace(" ", "-") + ".log").write_text(
            done.stdout + done.stderr
        )
        assert done.returncode == 0, done.stderr[-2000:] + done.stdout[-2000:]
        lines = done.stdout.splitlines()
        first = next(i for i, line in enumerate(lines) if line.startswith('"ID",'))
        rows = list(csv.DictReader(io.StringIO("\n".join(lines[first:]))))
        values = [
            float(r["Metric Value"].replace(",", ""))
            for r in rows
            if r.get("Metric Name") == "dram__bytes.sum"
        ]
        assert values, "No HBM counters collected"
        row = result["benchmarks"][name]
        row["profiled_hbm_bytes"] = sum(values)
        row["profiled_kernel_count"] = len(values)
        row["effective_hbm_bandwidth_gb_s"] = sum(values) / (row["latency_ms"] * 1e6)
        print(name, row, flush=True)
        Path("/data/cuda-benchmark.json").write_text(json.dumps(result, indent=2))
        volume.commit()
    result["bandwidth_note"] = (
        "Nsight Compute dram__bytes.sum across full-case kernels / uninstrumented wall latency. Separate profiling pass; clock and cache control disabled."
    )
    Path("/data/cuda-benchmark.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result, Path("/data/cuda-memcheck.log").read_text()


@app.local_entrypoint()
def main(profile_only: bool = False):
    result, log = profile.remote() if profile_only else execute.remote()
    out = Path("experiments/transolver_lab/evidence")
    out.mkdir(exist_ok=True)
    (out / "drivaer-cuda-benchmark.json").write_text(json.dumps(result, indent=2))
    (out / "drivaer-cuda-memcheck.log").write_text(log)
    print(json.dumps(result, indent=2))
    print("\nImplementation | Total ms | Speedup | Peak MB | Effective HBM GB/s")
    for name, row in result["benchmarks"].items():
        print(
            f"{name} | {row['latency_ms']:.3f} | {row['speedup']:.3f} | {row['peak_allocated_vram_mb']:.3f} | {row['effective_hbm_bandwidth_gb_s']}"
        )
    print(
        "FP32 equivalence gate:",
        "PASS" if result["passed"] else "FAIL — diagnostic only",
    )
    assert result[
        "passed"
    ], "Milestone 3B numerical gate failed; evidence saved, candidate not approved"
