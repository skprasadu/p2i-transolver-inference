"""Remote-only trained model validation. No changes to checkpoint or input partition."""

import json
import time
import types
import sys
import zipfile
from pathlib import Path
import torch
from experiments.transolver_lab.drivaer_worker import ROOT, CHECKPOINT


def fused(self, x):
    import p2i_physics_attention_cuda as ext

    xm, fm = self.project_input_onto_slices(x)
    xm = xm.permute(0, 2, 1, 3).contiguous()
    fm = fm.permute(0, 2, 1, 3).contiguous()
    w, b, t = (
        self.in_project_slice.weight,
        self.in_project_slice.bias,
        self.temperature.contiguous(),
    )
    tokens = ext.nvidia_slice(xm, fm, w, b, t)
    attended = self._compute_slice_attention_sdpa(tokens).contiguous()
    out = ext.nvidia_unslice(xm, attended, w, b, t).permute(0, 2, 1, 3).reshape(x.shape)
    return self.out_dropout(self.out_linear(out))


def load():
    from physicsnemo.models.transolver.transolver import Transolver

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with zipfile.ZipFile(ROOT / CHECKPOINT) as z:
        args = json.loads(z.read("args.json"))["__args__"]
        weights = torch.load(z.open("model.pt"), map_location="cpu", weights_only=True)
    model = Transolver(**args)
    model.load_state_dict(weights, strict=True)
    assert (
        len(model.blocks) == 20
        and args["slice_num"] == 512
        and args["n_hidden"] // args["n_head"] == 32
    )
    return model.cuda().eval()


def switch(model, custom):
    for block in model.blocks:
        attn = block.Attn
        if not hasattr(attn, "original_forward"):
            attn.original_forward = attn.forward
        attn.forward = (
            types.MethodType(fused, attn) if custom else attn.original_forward
        )


def data():
    import numpy as np
    import pyvista as pv

    mesh = pv.read(ROOT / "run_4/boundary_4.vtp")
    stl = pv.read(ROOT / "run_4/drivaer_4.stl")
    origin = np.asarray(stl.cell_centers().points).mean(axis=0)
    del stl
    centers = np.asarray(mesh.cell_centers().points, dtype=np.float32)
    normals = np.asarray(mesh.cell_normals, dtype=np.float32).copy()
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    emb = np.concatenate(
        ((centers - origin) / [12.0, 4.5, 3.25], normals), axis=1
    ).astype(np.float32)
    return torch.from_numpy(emb).cuda(), mesh.n_points


@torch.inference_mode()
def run(sanitize=False):
    model = load()
    emb, vertices = data()
    n = len(emb)
    batch = 200000
    order = torch.randperm(n, device="cuda")
    fx = torch.tensor([1.205, 30.0], device="cuda").expand(1, batch, 2)
    errors = [0.0] * 20
    full_error = 0.0
    # All layers use actual trained hidden states from real geometry.
    offsets = (
        [(0, 37), (37, 1025)]
        if sanitize
        else [(i, min(batch, n - i)) for i in range(0, n, batch)]
    )
    for offset, size in offsets:
        inputs = {
            "embedding": emb[order[offset : offset + size]][None],
            "fx": fx[:, :size],
        }
        saved = []
        handles = []
        switch(model, False)
        if sanitize:
            for j, block in enumerate(model.blocks):

                def isolated(m, i, o, j=j):
                    error = (fused(m, i[0]) - o).abs().max().item()
                    errors[j] = max(errors[j], error)
                    assert error < 1e-4, (j, size, error)

                handles.append(block.Attn.register_forward_hook(isolated))
            model(**inputs)
            for handle in handles:
                handle.remove()
            print("Isolated trained layers", size, max(errors), flush=True)
            continue
        for block in model.blocks:
            handles.append(
                block.Attn.register_forward_hook(
                    lambda m, i, o: saved.append(o.clone())
                )
            )
        expected = model(**inputs)
        for handle in handles:
            handle.remove()
        switch(model, True)
        for j, block in enumerate(model.blocks):

            def compare(m, i, o, j=j):
                errors[j] = max(errors[j], (o - saved[j]).abs().max().item())

            handles.append(block.Attn.register_forward_hook(compare))
        actual = model(**inputs)
        for handle in handles:
            handle.remove()
        error = (actual - expected).abs().max().item()
        full_error = max(full_error, error)
        print(
            "Compared",
            offset,
            size,
            "output",
            error,
            "worst layer",
            max(errors),
            flush=True,
        )
        report = {
            "vertices": vertices,
            "surface_cells": n,
            "layer_max_abs_errors": errors,
            "output_max_abs_error": full_error,
            "passed": max(errors) < 1e-4 and full_error < 1e-4,
        }
        (ROOT / "cuda-equivalence-progress.json").write_text(
            json.dumps(report, indent=2)
        )
        # Complete the diagnostic even if the release gate fails; never deploy it.
        del saved, expected, actual, inputs
    if sanitize:
        # Representative full-sized learned first layer, plus all-layer tail cases above.
        size = 16385
        switch(model, False)
        x = emb[order[:size]][None]
        captured = []
        handle = model.blocks[0].Attn.register_forward_pre_hook(
            lambda m, i: captured.append(i[0].clone())
        )
        model(embedding=x, fx=fx[:, :size])
        handle.remove()
        a = model.blocks[0].Attn
        error = (a(captured[0]) - fused(a, captured[0])).abs().max().item()
        assert error < 1e-4, error
        print(
            "Sanitizer scope: all20 layers at37/1025 real cells; first trained layer at16385 cells. Error",
            error,
            flush=True,
        )
        return
    rows = {}
    for name, custom in [("PyTorch", False), ("Fused CUDA", True)]:
        switch(model, custom)
        model(embedding=emb[order[:batch]][None], fx=fx)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for offset in range(0, n, batch):
            idx = order[offset : offset + batch]
            output = model(embedding=emb[idx][None], fx=fx[:, : len(idx)])
            del output
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000
        rows[name] = {
            "latency_ms": ms,
            "peak_allocated_vram_mb": torch.cuda.max_memory_allocated() / 1e6,
            "effective_hbm_bandwidth_gb_s": None,
        }
        print(name, rows[name], flush=True)
    for row in rows.values():
        row["speedup"] = rows["PyTorch"]["latency_ms"] / row["latency_ms"]
    report.update(
        benchmarks=rows,
        gpu=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        seed=42,
        batch_size=batch,
        timing_scope="All20 layers across all8140177 cells; identical partition. Model forward+indexing only. One warmed full pass each.",
        bandwidth_note="Not measured: wall time and logical bytes do not establish HBM bandwidth. Requires profiler DRAM counters.",
        normalization_note="Fused unnormalized partial sums divided by total norm+0.01; algebraically equivalent, different floating-point operation order; gated at1e-4.",
    )
    (ROOT / "cuda-benchmark.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    assert report[
        "passed"
    ], "FP32 equivalence release gate failed; diagnostic timings are not an approved replacement"


if __name__ == "__main__":
    run("--sanitize" in sys.argv)
