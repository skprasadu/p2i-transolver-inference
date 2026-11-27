"""Control for reference repeatability on an identical production-size block."""

import json
from pathlib import Path
import modal
from experiments.transolver_lab.validate_drivaerml import image, volume

app = modal.App("p2i-transolver-reference-repeat")


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="H100!",
    cpu=8,
    memory=65536,
    timeout=300,
)
def check():
    import torch, numpy as np, pyvista as pv, zipfile
    from physicsnemo.models.transolver.transolver import Transolver
    from experiments.transolver_lab.drivaer_worker import ROOT, CHECKPOINT

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with zipfile.ZipFile(ROOT / CHECKPOINT) as z:
        model = Transolver(**json.loads(z.read("args.json"))["__args__"])
        model.load_state_dict(
            torch.load(z.open("model.pt"), map_location="cpu", weights_only=True),
            strict=True,
        )
    model.cuda().eval()
    mesh = pv.read(ROOT / "run_4/boundary_4.vtp")
    stl = pv.read(ROOT / "run_4/drivaer_4.stl")
    origin = np.asarray(stl.cell_centers().points).mean(axis=0)
    del stl
    centers = np.asarray(mesh.cell_centers().points, dtype=np.float32)
    normals = np.asarray(mesh.cell_normals, dtype=np.float32).copy()
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    emb = torch.from_numpy(
        np.concatenate(
            ((centers - origin) / [12.0, 4.5, 3.25], normals), axis=1
        ).astype(np.float32)
    ).cuda()
    order = torch.randperm(len(emb), device="cuda")
    x = emb[order[:200000]][None]
    fx = torch.tensor([1.205, 30.0], device="cuda").expand(1, 200000, 2)
    with torch.inference_mode():
        a = model(embedding=x, fx=fx)
        b = model(embedding=x, fx=fx)
        error = (a - b).abs().max().item()
    return {
        "same_input_repeat_max_abs_error": error,
        "cells": 200000,
        "seed": 42,
        "gpu": torch.cuda.get_device_name(),
    }


@app.local_entrypoint()
def main():
    result = check.remote()
    Path("experiments/transolver_lab/evidence/reference-repeat.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result, indent=2))
