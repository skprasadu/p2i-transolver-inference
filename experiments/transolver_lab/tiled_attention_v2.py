"""Three-pass attention: global partition norm, normalized GEMM, deslice GEMM.

FP64 norm accumulation is rounded to FP32 BEFORE adding NVIDIA's 0.01 epsilon.
Tiling changes reduction order; equivalence must be measured, not assumed.
"""

import torch
import p2i_tiled_softmax as ext


def forward(a, x, tile=1024):
    xm, fm = a.project_input_onto_slices(x)
    n = x.shape[1]
    h = xm.shape[2]
    s = a.in_project_slice.out_features
    norm = torch.zeros((1, h, s), device=x.device, dtype=torch.float64)
    for start in range(0, n, tile):
        w = ext.softmax(
            project(a, xm[:, start : start + tile]), a.temperature.contiguous(), None
        )
        norm.add_(w.sum(1, dtype=torch.float64))
    denom = (norm.float() + 1e-2).contiguous()
    tokens = torch.zeros((1, h, s, fm.shape[-1]), device=x.device, dtype=torch.float32)
    for start in range(0, n, tile):
        w = ext.softmax(
            project(a, xm[:, start : start + tile]), a.temperature.contiguous(), denom
        )
        tokens.add_(
            gemm(w.permute(0, 2, 3, 1), fm[:, start : start + tile].permute(0, 2, 1, 3))
        )
    attended = a._compute_slice_attention_sdpa(tokens)
    out = torch.empty_like(xm)
    for start in range(0, n, tile):
        w = ext.softmax(
            project(a, xm[:, start : start + tile]), a.temperature.contiguous(), None
        )
        out[:, start : start + tile] = gemm(w.permute(0, 2, 1, 3), attended).permute(
            0, 2, 1, 3
        )
    return a.out_dropout(a.out_linear(out.reshape(x.shape)))


# Three BF16 Tensor Core products reconstruct FP32 products to substantially
# better precision than one TF32 product. Output accumulation remains FP32.
# No assertion of equivalence: the reference gate decides whether it is usable.
use_corrected_tensor_cores = False


def gemm(left, right):
    if not use_corrected_tensor_cores:
        return torch.matmul(left, right)
    ah = left.to(torch.bfloat16)
    al = (left - ah.float()).to(torch.bfloat16)
    bh = right.to(torch.bfloat16)
    bl = (right - bh.float()).to(torch.bfloat16)

    def multiply(a, b):
        if a.ndim == 2:
            return torch.mm(a, b, out_dtype=torch.float32)
        shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        aa = a.expand(*shape, *a.shape[-2:]).reshape(-1, *a.shape[-2:])
        bb = b.expand(*shape, *b.shape[-2:]).reshape(-1, *b.shape[-2:])
        return torch.bmm(aa, bb, out_dtype=torch.float32).reshape(
            *shape, a.shape[-2], b.shape[-1]
        )

    return multiply(ah, bh) + multiply(ah, bl) + multiply(al, bh)


def project(a, x):
    if not use_corrected_tensor_cores:
        return a.in_project_slice(x)
    shape = x.shape
    out = gemm(x.reshape(-1, shape[-1]), a.in_project_slice.weight.t())
    return (out + a.in_project_slice.bias).reshape(*shape[:-1], -1)
