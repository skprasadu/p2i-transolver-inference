"""Remote-only v2 gates. Full car means all native cells in reference partitions."""

import json, time, sys
from pathlib import Path
import torch
from experiments.transolver_lab.drivaer_cuda_worker import load, data
from experiments.transolver_lab.tiled_attention_v2 import forward

ROOT = Path("/data")


def precision(mode):
    from experiments.transolver_lab import tiled_attention_v2

    tiled_attention_v2.use_corrected_tensor_cores = mode == "bf16x3"
    torch.set_float32_matmul_precision("high" if mode == "tf32" else "highest")
    torch.backends.cudnn.allow_tf32 = False


def timed(fn, repeats=3):
    fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        v = fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        del v
    return sorted(samples)[len(samples) // 2]


def golden():
    from experiments.transolver_lab.ref_physics_attention import Physics_Attention
    import p2i_tiled_softmax as ext

    torch.manual_seed(42)
    precision("fp32")
    a = Physics_Attention(3).cuda().eval()
    x = torch.randn(1, 1031, 3, device="cuda")
    xm = a.in_project_x(x).reshape(1, 1031, 8, 64)
    logits = a.in_project_slice(xm)
    actual = ext.softmax(logits, a.temperature.reshape(1, 1, 8, 1).contiguous(), None)
    expected = a.softmax(logits / a.temperature.permute(0, 2, 1, 3))
    error = (actual - expected).abs().max().item()
    assert error < 1e-4, error

    # Complete original reference module, substituting only the new kernel.
    class Softmax(torch.nn.Module):
        def forward(self, z):
            if z.shape[-1] == 64 and z.shape[-2] == 1031:
                return ext.softmax(
                    (z * 0.5).permute(0, 2, 1, 3).contiguous(),
                    torch.full((8,), 0.5, device="cuda"),
                    None,
                ).permute(0, 2, 1, 3)
            return torch.softmax(z, -1)

    y = a(x)
    a.softmax = Softmax()
    error_full = (a(x) - y).abs().max().item()
    assert error_full < 1e-4, error_full
    print("Original golden reference", error, error_full, flush=True)
    logits = torch.randn(1, 1031, 8, 512, device="cuda")
    temp = torch.tensor([-1.0, 0.49, 0.5, 1.0, 4.9, 5.0, 6.0, 100.0], device="cuda")
    den = torch.rand(1, 8, 512, device="cuda") + 1.0
    actual = ext.softmax(logits, temp, den)
    expected = (
        torch.softmax(logits / temp.clamp(0.5, 5).reshape(1, 1, 8, 1), -1)
        / den[:, None]
    )
    clamp_error = (actual - expected).abs().max().item()
    assert clamp_error < 1e-4, clamp_error
    print("512-slice clamp/division test", clamp_error, flush=True)


@torch.inference_mode()
def run(stage):
    golden()
    model = load()
    emb, vertices = data()
    n = len(emb)
    order = torch.randperm(n, device="cuda")
    a = model.blocks[0].Attn
    fx = torch.tensor([1.205, 30.0], device="cuda").expand(1, 200000, 2)

    def hidden(offset):
        precision("fp32")
        e = emb[order[offset : offset + 200000]][None]
        return model.blocks[0].ln_1(
            model.preprocess(torch.cat((e, fx[:, : e.shape[1]]), -1))
        )

    if stage == "probe":
        x = hidden(0)
        expected = a(x)
        base = timed(lambda: a(x))
        rows = []
        for mode in ["fp32", "tf32", "bf16x3"]:
            precision(mode)
            for tile in [1024, 4096, 8192, 32768]:
                y = forward(a, x, tile)
                err = (y - expected).abs().max().item()
                del y
                ms = timed(lambda: forward(a, x, tile))
                row = dict(mode=mode, tile=tile, error=err, ms=ms)
                rows.append(row)
                print(row, flush=True)
        valid = [r for r in rows if r["error"] < 1e-4]
        result = {
            "baseline_partition_ms": base,
            "candidates": rows,
            "selected": min(valid, key=lambda r: r["ms"]) if valid else None,
        }
        (ROOT / "v2-probe.json").write_text(json.dumps(result, indent=2))
        return
    config = json.loads((ROOT / "v2-probe.json").read_text())["selected"]
    assert config, "No accurate candidate"
    tile = config["tile"]
    mode = config["mode"]
    x = hidden(0)
    precision("fp32")
    a(x)
    precision(mode)
    forward(a, x, tile)
    torch.cuda.synchronize()
    del x
    maxerr = 0.0
    times = {"PyTorch": 0.0, "Tiled CUDA": 0.0}
    peaks = {"PyTorch": 0, "Tiled CUDA": 0}
    begin = int(sys.argv[2]) if stage == "sanitize" and len(sys.argv) > 2 else 0
    stop = min(n, int(sys.argv[3])) if stage == "sanitize" and len(sys.argv) > 3 else n
    for offset in range(begin, stop, 200000):
        x = hidden(offset)
        if stage == "sanitize":
            precision(mode)
            out = forward(a, x, tile)
            torch.cuda.synchronize()
            assert torch.isfinite(out).all()
            del out
        else:
            precision("fp32")
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            expected = a(x)
            end.record()
            end.synchronize()
            times["PyTorch"] += start.elapsed_time(end)
            peaks["PyTorch"] = max(peaks["PyTorch"], torch.cuda.max_memory_allocated())
            # Reference output is copied off GPU so candidate memory excludes it.
            reference = expected.cpu()
            del expected
            precision(mode)
            torch.cuda.reset_peak_memory_stats()
            start.record()
            actual = forward(a, x, tile)
            end.record()
            end.synchronize()
            times["Tiled CUDA"] += start.elapsed_time(end)
            peaks["Tiled CUDA"] = max(
                peaks["Tiled CUDA"], torch.cuda.max_memory_allocated()
            )
            err = (actual.cpu() - reference).abs().max().item()
            maxerr = max(maxerr, err)
            del actual, reference
        print(
            stage,
            offset,
            min(200000, n - offset),
            "max_error",
            maxerr if stage != "sanitize" else "not compared in memcheck",
            flush=True,
        )
        if stage == "sanitize":
            del x
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    result = dict(
        stage=stage,
        vertices=vertices,
        cells=n,
        partition=200000,
        config=config,
        max_abs_error=maxerr if stage != "sanitize" else None,
        checked_cell_start=begin,
        checked_cell_stop=stop,
        numerical_comparison_performed=stage != "sanitize",
        latency_ms=times,
        peak_allocated_vram_mb={k: v / 1e6 for k, v in peaks.items()},
        scope="Layer0 attention across full car; same 200000-cell partitions, not global attention",
        passed=maxerr < 1e-4
        and times["Tiled CUDA"] < 1039
        and times["Tiled CUDA"] < times["PyTorch"],
    )
    (ROOT / ("v2-" + stage + ".json")).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
