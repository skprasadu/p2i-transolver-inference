"""Remote-only CFD validation. No custom kernels, fitted offsets, or synthetic fields."""

from pathlib import Path
import hashlib
import json
import zipfile
import time

ROOT = Path("/data")
MODEL_REV = "96477aeb86d24c26ccf0797bca1b3851268017d0"
DATA_REV = "5d448b209bf654503c64ce7261c34fa125f46392"
CASE = 4
CHECKPOINT = "transolver_drivaerml_surface_checkpoint/Transolver.0.501.mdlus"
STATS = "transolver_drivaerml_surface_checkpoint/global_stats.json"


def download():
    from huggingface_hub import hf_hub_download
    import csv
    import urllib.request

    split_url = (
        "https://raw.githubusercontent.com/NVIDIA/physicsnemo-cfd/"
        "06846c9fbc08b8e82551841897b5294f7e5ef82d/"
        "workflows/benchmarking/drivaer_ml_files/validation.csv"
    )
    ROOT.mkdir(exist_ok=True)
    split_text = urllib.request.urlopen(split_url, timeout=30).read().decode()
    assert CASE in {
        int(row["run_idx"]) for row in csv.DictReader(split_text.splitlines())
    }
    (ROOT / "nvidia-validation-split.csv").write_text(split_text)
    files = []
    for repo, revision, kind, names in [
        ("nvidia/transolver_drivaerml", MODEL_REV, "model", [CHECKPOINT, STATS]),
        (
            "neashton/drivaerml",
            DATA_REV,
            "dataset",
            [
                f"run_{CASE}/boundary_{CASE}.vtp",
                f"run_{CASE}/drivaer_{CASE}.stl",
                f"run_{CASE}/boundary_cell_area_{CASE}.npy",
                f"run_{CASE}/geo_ref_{CASE}.csv",
            ],
        ),
    ]:
        for name in names:
            path = Path(
                hf_hub_download(
                    repo, name, revision=revision, repo_type=kind, local_dir=str(ROOT)
                )
            )
            digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
            files.append({"file": name, "bytes": path.stat().st_size, "sha256": digest})
            print("Cached", name, path.stat().st_size, flush=True)
    manifest = {
        "model_revision": MODEL_REV,
        "dataset_revision": DATA_REV,
        "case": CASE,
        "validation_split_source": split_url,
        "files": files,
    }
    (ROOT / "provenance.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def solve(batch_size):
    import numpy as np
    import pyvista as pv
    import torch
    from physicsnemo.models.transolver.transolver import Transolver

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    mesh = pv.read(ROOT / f"run_{CASE}/boundary_{CASE}.vtp")
    print("Native mesh", mesh.n_points, mesh.n_cells, list(mesh.cell_data), flush=True)
    assert "pMeanTrim" in mesh.cell_data and "wallShearStressMeanTrim" in mesh.cell_data
    stl = pv.read(ROOT / f"run_{CASE}/drivaer_{CASE}.stl")
    origin = np.asarray(stl.cell_centers().points).mean(axis=0)
    del stl
    centers = np.asarray(mesh.cell_centers().points, dtype=np.float32)
    normals = np.asarray(mesh.cell_normals, dtype=np.float32).copy()
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    assert np.isfinite(normals).all()
    # Exact official TransolverDataPipe geometry convention (not global coordinate z-scores).
    embedding = np.concatenate(
        ((centers - origin) / [12.0, 4.5, 3.25], normals), axis=1
    ).astype(np.float32)
    del centers, normals
    with zipfile.ZipFile(ROOT / CHECKPOINT) as archive:
        args = json.loads(archive.read("args.json"))["__args__"]
        weights = torch.load(
            archive.open("model.pt"), map_location="cpu", weights_only=True
        )
    model = Transolver(**args)
    model.load_state_dict(weights, strict=True)
    model = model.cuda().eval()
    del weights
    stats = json.loads((ROOT / STATS).read_text())
    mean = torch.tensor(
        stats["mean"]["pressure"] + stats["mean"]["shear_stress"], device="cuda"
    )
    std = torch.tensor(
        stats["std_dev"]["pressure"] + stats["std_dev"]["shear_stress"], device="cuda"
    )
    # Checkpoint convention used by NVIDIA inference; NOT a claim of a new 30 m/s CFD solve.
    rho, model_speed = 1.205, 30.0
    scale = rho * model_speed**2
    emb = torch.from_numpy(embedding).cuda()
    fx = torch.tensor([rho, model_speed], device="cuda").expand(1, batch_size, 2)
    count = mesh.n_cells
    order = torch.randperm(count, device="cuda")
    pred = torch.empty((count, 4), device="cuda")
    slice_ids = torch.empty(count, dtype=torch.int16, device="cuda")
    capture = {}

    def hook(_module, _inputs, output):
        # Positive per-head temperature and softmax preserve argmax. No B,H,N,M copy.
        capture["ids"] = output[0, :, 0, :].argmax(-1).to(torch.int16)

    handle = model.blocks[-1].Attn.in_project_slice.register_forward_hook(hook)
    with torch.inference_mode():
        model(embedding=emb[order[:batch_size]][None], fx=fx)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for offset in range(0, count, batch_size):
            idx = order[offset : offset + batch_size]
            y = model(embedding=emb[idx][None], fx=fx[:, : len(idx)])[0]
            pred[idx] = (y * std + mean) * scale
            slice_ids[idx] = capture["ids"]
            if offset % (batch_size * 10) == 0:
                print(
                    f"Inferred {min(offset + batch_size, count):,}/{count:,} cells",
                    flush=True,
                )
        torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000
        peak = torch.cuda.max_memory_allocated() / 1e6
        reserved = torch.cuda.max_memory_reserved() / 1e6
    handle.remove()
    assert torch.isfinite(pred).all()
    gt = torch.from_numpy(
        np.column_stack(
            (mesh.cell_data["pMeanTrim"], mesh.cell_data["wallShearStressMeanTrim"])
        ).astype(np.float32)
    ).cuda()
    area_np = np.load(ROOT / f"run_{CASE}/boundary_cell_area_{CASE}.npy")
    assert len(area_np) == count
    areas = torch.from_numpy(area_np).cuda().double()

    def metrics(p, g):
        p, g = p.double(), g.double()
        e = p - g
        w = areas if e.ndim == 1 else areas[:, None]
        return {
            "relative_l2": (e.norm() / g.norm()).item(),
            "mae_native_units": e.abs().mean().item(),
            "area_weighted_relative_l2": (
                (w * e.square()).sum() / (w * g.square()).sum()
            )
            .sqrt()
            .item(),
            "prediction_range": [p.min().item(), p.max().item()],
            "ground_truth_range": [g.min().item(), g.max().item()],
        }

    result = {
        "case": f"run_{CASE}",
        "vertices": mesh.n_points,
        "surface_cells": count,
        "inference_latency_ms": latency,
        "peak_allocated_vram_mb": peak,
        "peak_reserved_vram_mb": reserved,
        "gpu": torch.cuda.get_device_name(),
        "batch_size": batch_size,
        "seed": 42,
        "precision": "FP32, TF32 disabled",
        "model_args": args,
        "pressure": metrics(pred[:, 0], gt[:, 0]),
        "wall_shear_vector": metrics(pred[:, 1:], gt[:, 1:]),
        "units": "Native DrivAerML kinematic pressure / wall shear (m^2/s^2); no Pa relabeling",
        "model_input": {"air_density": rho, "stream_velocity": model_speed},
        "cfd_reference_speed_m_s": 38.889,
        "validation_scope": "run_4 in NVIDIA demo validation list; not independent proof of original training membership",
        "slice_scope": "final attention block (19), head 0; argmax of 512 trained slice assignments",
        "occupied_slice_ids": torch.unique(slice_ids).cpu().tolist(),
        "timing_scope": "all cell blocks, model forward + slice extraction + denormalization + scatter; excludes loading/VTK writing",
        "model_revision": MODEL_REV,
        "dataset_revision": DATA_REV,
    }
    p = pred.cpu().numpy()
    g = gt.cpu().numpy()
    mesh.clear_data()
    for name, values in {
        "Pred_Pressure": p[:, 0],
        "GT_Pressure": g[:, 0],
        "Abs_Error": np.abs(p[:, 0] - g[:, 0]),
        "Wall_Shear_Stress": np.linalg.norm(p[:, 1:], axis=1),
        "Pred_WSS_Vector": p[:, 1:],
        "GT_WSS_Vector": g[:, 1:],
        "GT_WSS_Magnitude": np.linalg.norm(g[:, 1:], axis=1),
        "Physical_Slice_ID": slice_ids.cpu().numpy(),
    }.items():
        mesh.cell_data[name] = values
    mesh.save(ROOT / "drivaer_validated.vtp")
    (ROOT / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result
