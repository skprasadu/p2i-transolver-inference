"""GPU-only inputs, upstream equivalence and synchronized inference benchmark."""

import json
import time
import statistics
import torch
from .ref_physics_attention import Physics_Attention
from .upstream.Physics_Attention import Physics_Attention_Irregular_Mesh


def wing_points(chord_samples=1000, span_samples=500):
    # Closed trailing-edge NACA 0012 extruded to a span of four chords.
    theta = torch.arange(chord_samples, device="cuda", dtype=torch.float32) * (
        2 * torch.pi / chord_samples
    )
    x = (1 - torch.cos(theta)) / 2
    z = (
        5
        * 0.12
        * (
            0.2969 * torch.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x * x
            + 0.2843 * x**3
            - 0.1036 * x**4
        )
    )
    z = torch.where(theta <= torch.pi, z, -z)
    span = torch.linspace(-2, 2, span_samples, device="cuda")
    return torch.stack(
        (
            x[None, :].expand(span_samples, -1),
            span[:, None].expand(-1, chord_samples),
            z[None, :].expand(span_samples, -1),
        ),
        dim=-1,
    ).reshape(1, -1, 3)


def setup():
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return (
        Physics_Attention(dim=3, heads=8, dim_head=64, slice_num=64, dropout=0)
        .cuda()
        .eval()
    )


@torch.inference_mode()
def equivalence():
    model = setup()
    reference = (
        Physics_Attention_Irregular_Mesh(dim=3, heads=8, dim_head=64, slice_num=64)
        .cuda()
        .eval()
    )
    reference.load_state_dict(model.state_dict())
    torch.manual_seed(42)
    x = torch.randn(1, 4096, 3, device="cuda")
    actual = model(x)
    expected = reference(x)
    error = (actual - expected).abs().max().item()
    assert error < 1e-4, error
    assert torch.isfinite(actual).all().item()
    return {"max_abs_error_fp32": error, "threshold": 1e-4, "points": 4096}


@torch.inference_mode()
def benchmark():
    check = equivalence()
    model = setup()
    x = wing_points()
    for _ in range(3):
        model(x)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    wall = []
    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t = time.perf_counter()
        start.record()
        out = model(x)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
        wall.append((time.perf_counter() - t) * 1000)
        del out
    return {
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "batch": 1,
        "points": x.shape[1],
        "input_channels": 3,
        "heads": 8,
        "head_dim": 64,
        "slices_per_head": 64,
        "dtype": "float32",
        "training": False,
        "tf32": False,
        "warmups": 3,
        "repetitions": 10,
        "cuda_event_ms": samples,
        "median_cuda_event_ms": statistics.median(samples),
        "synchronized_wall_ms": wall,
        "median_wall_ms": statistics.median(wall),
        "peak_allocated_MB": torch.cuda.max_memory_allocated() / 1e6,
        "peak_reserved_MB": torch.cuda.max_memory_reserved() / 1e6,
        "memory_definition": "decimal MB; PyTorch allocator, including input/model; excludes CUDA context",
        "equivalence": check,
        "scope": "one untrained attention block; no CFD solve or training/backward",
    }


if __name__ == "__main__":
    print(json.dumps(equivalence()))
