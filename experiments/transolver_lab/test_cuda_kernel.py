"""Modal-only compile, full-size correctness/memcheck and paired benchmarks.

Run: modal run experiments/transolver_lab/test_cuda_kernel.py
"""

from pathlib import Path
import json
import modal

app = modal.App("p2i-transolver-cuda-test")
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential")
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("einops==0.8.1", "ninja", "setuptools", "modal")
    .env(
        {
            "PYTHONPATH": "/app",
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "MAX_JOBS": "2",
            "CC": "gcc",
            "CXX": "g++",
        }
    )
    .add_local_dir(
        "experiments/transolver_lab",
        "/app/experiments/transolver_lab",
        copy=True,
        ignore=["**/__pycache__/**", "evidence", "build", "*.so"],
    )
)


def worker(sanitize=False):
    import statistics
    import torch
    import p2i_physics_attention_cuda as ext
    from experiments.transolver_lab.workload import setup

    torch.manual_seed(42)
    model = setup()

    def project(x):
        b, n, _ = x.shape
        return tuple(
            layer(x).reshape(b, n, 8, 64).permute(0, 2, 1, 3).contiguous()
            for layer in (model.in_project_x, model.in_project_fx)
        )

    w = model.in_project_slice.weight
    b = model.in_project_slice.bias
    t = model.temperature

    def reference_slice(x, f):
        weights = model.softmax(model.in_project_slice(x) / t)
        return torch.einsum("bhnc,bhng->bhgc", f, weights) / (
            weights.sum(2)[..., None] + 1e-5
        )

    def fused(x):
        xm, fm = project(x)
        tokens = ext.slice(xm, fm, w, b, t)
        q, k, v = model.to_q(tokens), model.to_k(tokens), model.to_v(tokens)
        attn = model.softmax(q @ k.transpose(-1, -2) * model.scale)
        out = ext.unslice(xm, (attn @ v).contiguous(), w, b, t)
        return model.to_out(out.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1))

    errors = []
    with torch.inference_mode():
        # Tail, multi-batch and full requested size; non-default stream checks launch semantics.
        for batch, n in [(2, 37), (1, 1025), (1, 500000)]:
            torch.manual_seed(42)
            x = torch.randn(batch, n, 3, device="cuda")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                xm, fm = project(x)
                actual = ext.slice(xm, fm, w, b, t)
                expected = reference_slice(xm, fm)
                slice_error = (actual - expected).abs().max().item()
                output_error = (fused(x) - model(x)).abs().max().item()
            diagnostic = {}
            if n == 500000:
                with torch.cuda.stream(stream):
                    weights = model.softmax(model.in_project_slice(xm) / t)
                    high = (
                        torch.einsum("bhnc,bhng->bhgc", fm.double(), weights.double())
                        / (weights.double().sum(2)[..., None] + 1e-5)
                    ).float()
                    diagnostic = {
                        "custom_vs_fp64_aggregation": (actual - high)
                        .abs()
                        .max()
                        .item(),
                        "reference_vs_fp64_aggregation": (expected - high)
                        .abs()
                        .max()
                        .item(),
                    }
                    assert diagnostic["custom_vs_fp64_aggregation"] < 1e-4, diagnostic
                del weights, high
            else:
                assert slice_error < 1e-4, (n, "slice", slice_error)
            assert output_error < 1e-4, (n, "full", output_error)
            errors.append(
                {
                    "batch": batch,
                    "points": n,
                    "slice_max_abs_error": slice_error,
                    "full_max_abs_error": output_error,
                    **diagnostic,
                }
            )
            del xm, fm, actual, expected, x
        if sanitize:
            print(json.dumps({"correctness": errors}))
            return
        torch.manual_seed(42)
        x = torch.randn(1, 500000, 3, device="cuda")
        xm, fm = project(x)

        def measure(fn):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            samples = []
            for _ in range(10):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = fn()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
                del out
            return {
                "median_ms": statistics.median(samples),
                "samples_ms": samples,
                "incremental_peak_MB": (torch.cuda.max_memory_allocated() - base) / 1e6,
            }

        rows = {}
        # Identical logical I/O bytes for both slice-stage implementations. NOT measured DRAM traffic.
        logical_bytes = (
            xm.numel() + fm.numel() + w.numel() + b.numel() + t.numel() + 8 * 64 * 64
        ) * 4
        for name, fn in [
            ("PyTorch slice", lambda: reference_slice(xm, fm)),
            ("Custom CUDA slice", lambda: ext.slice(xm, fm, w, b, t)),
        ]:
            rows[name] = measure(fn)
            rows[name]["effective_logical_GB_s"] = logical_bytes / (
                rows[name]["median_ms"] * 1e6
            )
        del xm, fm
        for name, fn in [
            ("PyTorch full block", lambda: model(x)),
            ("Custom CUDA full block", lambda: fused(x)),
        ]:
            rows[name] = measure(fn)
    return {
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "correctness": errors,
        "benchmarks": rows,
        "logical_slice_bytes": logical_bytes,
        "bandwidth_definition": "logical input+output bytes / median time; NOT profiler-measured DRAM bandwidth",
        "scope": "FP32 inference only; D=M=64, heads=8, input channels=3. Recompute weights for deslicing; no global BHNM tensor.",
    }


@app.function(image=image, gpu="H100!", cpu=8, memory=32768, timeout=2400, retries=0)
def run_remote():
    import subprocess, sys

    directory = "/app/experiments/transolver_lab"
    build = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=600,
    )
    print(build.stdout[-6000:])
    print(build.stderr[-3000:])
    assert build.returncode == 0, "Extension build failed"
    import os

    os.environ["PYTHONPATH"] = directory + ":/app"
    check = subprocess.run(
        [
            "compute-sanitizer",
            "--tool",
            "memcheck",
            "--error-exitcode",
            "99",
            sys.executable,
            "-m",
            "experiments.transolver_lab.test_cuda_kernel",
            "--sanitize-worker",
        ],
        capture_output=True,
        text=True,
        timeout=1500,
    )
    print(check.stdout)
    print(check.stderr)
    assert check.returncode == 0, "Memcheck/correctness subprocess failed"
    assert "ERROR SUMMARY: 0 errors" in check.stdout + check.stderr
    sys.path.insert(0, directory)
    result = worker()
    return result, build.stdout + build.stderr, check.stdout + check.stderr


@app.local_entrypoint()
def main():
    result, build, memcheck = run_remote.remote()
    out = Path("experiments/transolver_lab/evidence")
    out.mkdir(exist_ok=True)
    (out / "cuda-benchmark.json").write_text(json.dumps(result, indent=2))
    (out / "cuda-build.txt").write_text(build)
    (out / "cuda-memcheck.txt").write_text(memcheck)
    print("Implementation | median ms | effective logical GB/s | incremental peak MB")
    for name, row in result["benchmarks"].items():
        bw = row.get("effective_logical_GB_s")
        print(
            f"{name} | {row['median_ms']:.4f} | {bw if bw is not None else 'N/A'} | {row['incremental_peak_MB']:.2f}"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import sys

    if "--sanitize-worker" in sys.argv:
        worker(sanitize=True)
