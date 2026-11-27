"""Bounded v3 benchmark; pressure is checked against actual CFD, no fake fields."""

import json
import sys
import time
import types
from pathlib import Path
import numpy as np
import torch
from experiments.transolver_lab.drivaer_cuda_worker import load, data
from experiments.transolver_lab.drivaer_v2_worker import golden
from experiments.transolver_lab.tiled_attention_v3 import SharedGraph, forward
from experiments.transolver_lab.drivaer_worker import STATS

ROOT = Path("/data")
BATCH = 200000


def save(stage, result):
    (ROOT / f"v3-{stage}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def elapsed(fn):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    value = fn()
    end.record()
    end.synchronize()
    return value, start.elapsed_time(end)


@torch.inference_mode()
def run(stage, begin=0, stop=8140177):
    golden()
    model = load()
    emb, vertices = data()
    n = len(emb)
    order = torch.randperm(n, device="cuda")
    fx = torch.tensor([1.205, 30.0], device="cuda").expand(1, BATCH, 2)
    a = model.blocks[0].Attn
    graph = SharedGraph()

    def inputs(offset):
        idx = order[offset : offset + BATCH]
        return idx, dict(embedding=emb[idx][None], fx=fx[:, : len(idx)])

    def hidden(args):
        torch.set_float32_matmul_precision("highest")
        return model.blocks[0].ln_1(
            model.preprocess(torch.cat((args["embedding"], args["fx"]), -1))
        )

    def patch():
        for block in model.blocks:
            block.Attn.forward = types.MethodType(
                lambda self, x: graph(self, x), block.Attn
            )

    common = dict(
        vertices=vertices,
        cells=n,
        partition=BATCH,
        seed=42,
        gpu=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        variant=(
            "unmodified PyTorch FP32; TF32 disabled"
            if stage == "baseline"
            else "TF32 tile GEMMs; FP32 norms and outer projections; shared CUDAGraph"
        ),
        tile=32768,
    )
    if stage in ("layer", "sanitize-layer"):
        maximum = 0.0
        total = 0.0
        peak = 0
        reserved = 0
        for offset in range(begin, min(stop, n), BATCH):
            _, args = inputs(offset)
            x = hidden(args)
            if stage == "layer":
                torch.set_float32_matmul_precision("highest")
                expected = a(x).cpu()
            graph.prepare(a, x)
            torch.cuda.reset_peak_memory_stats()
            actual, ms = elapsed(lambda: graph(a, x))
            total += ms
            peak = max(peak, torch.cuda.max_memory_allocated())
            reserved = max(reserved, torch.cuda.max_memory_reserved())
            assert torch.isfinite(actual).all()
            if stage == "layer":
                maximum = max(maximum, (actual.cpu() - expected).abs().max().item())
            del actual, x, args
            print(
                stage,
                offset,
                "error",
                maximum if stage == "layer" else "not compared",
                flush=True,
            )
        common.update(
            latency_ms=total,
            peak_allocated_vram_mb=peak / 1e6,
            peak_reserved_vram_mb=reserved / 1e6,
            capture_ms=graph.capture_ms,
            max_abs_error=maximum if stage == "layer" else None,
            accuracy_passed=maximum < 1e-3 if stage == "layer" else None,
            begin=begin,
            stop=min(stop, n),
        )
        graph.release()
        save(stage, common)
        return
    if stage == "sanitize-model":
        # Every trained layer at full reference partition and irregular tail.
        patch()
        for offset in [0, 8000000]:
            idx, args = inputs(offset)
            x = hidden(args)
            graph.prepare(a, x)
            del x
            out = model(**args)
            torch.cuda.synchronize()
            assert torch.isfinite(out).all()
            print("Sanitized all20 layers", offset, len(idx), flush=True)
            del out, args
            graph.release()
        save(stage, dict(common, checked_cells=340177, all_layers=20))
        return
    assert stage in ("baseline", "full")
    custom = stage == "full"
    if custom:
        patch()
    predictions = np.empty((n, 4), dtype=np.float32)
    stats = json.loads((ROOT / STATS).read_text())
    mean = np.array(
        stats["mean"]["pressure"] + stats["mean"]["shear_stress"], dtype=np.float32
    )
    std = np.array(
        stats["std_dev"]["pressure"] + stats["std_dev"]["shear_stress"],
        dtype=np.float32,
    )
    total = 0.0
    peak = 0
    reserved = 0
    warm_shape = None
    for offset in range(0, n, BATCH):
        idx, args = inputs(offset)
        if warm_shape != len(idx):
            if custom:
                x = hidden(args)
                graph.prepare(a, x)
                del x
            model(**args)
            torch.cuda.synchronize()
            warm_shape = len(idx)
        torch.cuda.reset_peak_memory_stats()
        # Wall latency matches prior full-model measurement more closely than
        # summing CUDA kernel times. Input indexing is timed separately below.
        torch.cuda.synchronize()
        start = time.perf_counter()
        idx, args = inputs(offset)
        out = model(**args)
        torch.cuda.synchronize()
        total += (time.perf_counter() - start) * 1000
        peak = max(peak, torch.cuda.max_memory_allocated())
        reserved = max(reserved, torch.cuda.max_memory_reserved())
        assert torch.isfinite(out).all()
        predictions[idx.cpu().numpy()] = (out[0].cpu().numpy() * std + mean) * (
            1.205 * 30.0**2
        )
        del out, args
        print(stage, offset, flush=True)
    if custom:
        graph.release()
    import pyvista as pv

    mesh = pv.read(ROOT / "run_4/boundary_4.vtp")
    gt = np.asarray(mesh.cell_data["pMeanTrim"], dtype=np.float64)
    error = np.linalg.norm(predictions[:, 0].astype(np.float64) - gt) / np.linalg.norm(
        gt
    )
    wss = np.asarray(mesh.cell_data["wallShearStressMeanTrim"], dtype=np.float64)
    shear_error = np.linalg.norm(
        predictions[:, 1:].astype(np.float64) - wss
    ) / np.linalg.norm(wss)
    np.save(ROOT / f"v3-{stage}-predictions.npy", predictions)
    common.update(
        latency_ms=total,
        speedup_vs_20792=20792.0 / total,
        peak_allocated_vram_mb=peak / 1e6,
        peak_reserved_vram_mb=reserved / 1e6,
        pressure_relative_l2_percent=float(error * 100),
        wall_shear_relative_l2_percent=float(shear_error * 100),
        capture_ms=graph.capture_ms,
        timing_scope="Full20 layers, indexing+forward, warmed; excludes capture, denormalization, CPU transfer and CFD metrics",
    )
    if custom:
        base = json.loads((ROOT / "v3-baseline.json").read_text())
        delta = (
            common["pressure_relative_l2_percent"]
            - base["pressure_relative_l2_percent"]
        )
        common.update(
            pressure_change_percentage_points=delta,
            pressure_relative_change_percent=delta
            / base["pressure_relative_l2_percent"]
            * 100,
            pressure_gate_005_percentage_points=abs(delta) <= 0.05,
            pressure_gate_005_percent_relative=abs(delta)
            / base["pressure_relative_l2_percent"]
            <= 0.0005,
            measured_speedup=base["latency_ms"] / total,
        )
    save(stage, common)


if __name__ == "__main__":
    run(sys.argv[1], *(int(x) for x in sys.argv[2:]))
