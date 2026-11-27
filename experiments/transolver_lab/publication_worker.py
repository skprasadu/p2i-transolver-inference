"""Remote-only, fresh-process publication benchmark; original V3 kernel unchanged."""

import json
import sys
import time
import types
from pathlib import Path
import numpy as np
import torch
import pyvista as pv
from experiments.transolver_lab.drivaer_cuda_worker import load
from experiments.transolver_lab.drivaer_worker import STATS
from experiments.transolver_lab.drivaer_v2_worker import golden
from experiments.transolver_lab.tiled_attention_v3 import SharedGraph

ROOT = Path("/data")
BATCH = 200000


@torch.inference_mode()
def run(case, mode, repeats=5):
    golden()
    model = load()  # Resets deterministic seed and disables TF32.
    mesh = pv.read(ROOT / f"run_{case}/boundary_{case}.vtp")
    stl = pv.read(ROOT / f"run_{case}/drivaer_{case}.stl")
    origin = np.asarray(stl.cell_centers().points).mean(0)
    del stl
    centers = np.asarray(mesh.cell_centers().points, dtype=np.float32)
    normals = np.asarray(mesh.cell_normals, dtype=np.float32).copy()
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    emb = torch.from_numpy(
        np.concatenate(((centers - origin) / [12.0, 4.5, 3.25], normals), 1).astype(
            np.float32
        )
    ).cuda()
    n = len(emb)
    order = torch.randperm(n, device="cuda")
    fx = torch.tensor([1.205, 30.0], device="cuda").expand(1, BATCH, 2)
    gt = np.asarray(mesh.cell_data["pMeanTrim"], dtype=np.float64)
    shear = np.asarray(mesh.cell_data["wallShearStressMeanTrim"], dtype=np.float64)
    vertices = mesh.n_points
    del mesh, centers, normals
    stats = json.loads((ROOT / STATS).read_text())
    mean = np.array(
        stats["mean"]["pressure"] + stats["mean"]["shear_stress"], dtype=np.float32
    )
    std = np.array(
        stats["std_dev"]["pressure"] + stats["std_dev"]["shear_stress"],
        dtype=np.float32,
    )
    graph = SharedGraph()
    if mode == "v3":
        for block in model.blocks:
            block.Attn.forward = types.MethodType(
                lambda self, x: graph(self, x), block.Attn
            )
    samples = []
    for repeat in range(repeats):
        predictions = np.empty((n, 4), dtype=np.float32)
        total = 0.0
        peak = 0
        reserved = 0
        warm_shape = None
        for offset in range(0, n, BATCH):
            idx = order[offset : offset + BATCH]
            args = dict(embedding=emb[idx][None], fx=fx[:, : len(idx)])
            if warm_shape != len(idx):
                if mode == "v3":
                    torch.set_float32_matmul_precision("highest")
                    hidden = model.blocks[0].ln_1(
                        model.preprocess(torch.cat((args["embedding"], args["fx"]), -1))
                    )
                    graph.prepare(model.blocks[0].Attn, hidden)
                    del hidden
                else:
                    torch.set_float32_matmul_precision(
                        "high" if mode == "tf32" else "highest"
                    )
                model(**args)
                torch.cuda.synchronize()
                warm_shape = len(idx)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            idx = order[offset : offset + BATCH]
            args = dict(embedding=emb[idx][None], fx=fx[:, : len(idx)])
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
        sample = dict(
            latency_ms=total,
            peak_allocated_mb=peak / 1e6,
            peak_reserved_mb=reserved / 1e6,
            pressure_l2_percent=float(
                100
                * np.linalg.norm(predictions[:, 0].astype(np.float64) - gt)
                / np.linalg.norm(gt)
            ),
            shear_l2_percent=float(
                100
                * np.linalg.norm(predictions[:, 1:].astype(np.float64) - shear)
                / np.linalg.norm(shear)
            ),
        )
        samples.append(sample)
        result = dict(
            case=case,
            mode=mode,
            vertices=vertices,
            cells=n,
            partition=BATCH,
            seed=42,
            gpu=torch.cuda.get_device_name(),
            torch=str(torch.__version__),
            samples=samples,
            scope="Indexing+forward; per-shape warmup excluded; five sequential repetitions; fresh process per case/mode",
            precision="TF32 allowed globally" if mode == "tf32" else mode,
        )
        path = ROOT / "publication" / f"run-{case}-{mode}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        print(
            json.dumps(dict(case=case, mode=mode, repeat=repeat, **sample)), flush=True
        )
    graph.release()


if __name__ == "__main__":
    run(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
