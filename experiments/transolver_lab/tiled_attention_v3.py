"""V3: one TF32 tiled-attention experiment with FP32 normalization and graphs.

Only slice projection / aggregation / deslicing use TF32. Input/output
projections and slice-to-slice attention retain reference FP32 arithmetic.
The CUDA kernel is the unchanged, verified v2 softmax kernel.
"""

import copy
import gc
import time
import torch
import p2i_tiled_softmax as ext


def forward(a, x, tile=32768):
    torch.set_float32_matmul_precision("highest")
    xm, fm = a.project_input_onto_slices(x)
    n, h = x.shape[1], xm.shape[2]
    s = a.in_project_slice.out_features
    norm = torch.zeros((1, h, s), device=x.device, dtype=torch.float32)
    torch.set_float32_matmul_precision("high")
    for start in range(0, n, tile):
        weights = ext.softmax(
            a.in_project_slice(xm[:, start : start + tile]),
            a.temperature.contiguous(),
            None,
        )
        norm.add_(weights.sum(1))
    denom = (norm + 1e-2).contiguous()
    tokens = torch.zeros((1, h, s, fm.shape[-1]), device=x.device, dtype=torch.float32)
    for start in range(0, n, tile):
        weights = ext.softmax(
            a.in_project_slice(xm[:, start : start + tile]),
            a.temperature.contiguous(),
            denom,
        )
        tokens.add_(
            torch.matmul(
                weights.permute(0, 2, 3, 1),
                fm[:, start : start + tile].permute(0, 2, 1, 3),
            )
        )
    torch.set_float32_matmul_precision("highest")
    attended = a._compute_slice_attention_sdpa(tokens)
    out = torch.empty_like(xm)
    torch.set_float32_matmul_precision("high")
    for start in range(0, n, tile):
        weights = ext.softmax(
            a.in_project_slice(xm[:, start : start + tile]),
            a.temperature.contiguous(),
            None,
        )
        out[:, start : start + tile] = torch.matmul(
            weights.permute(0, 2, 1, 3), attended
        ).permute(0, 2, 1, 3)
    torch.set_float32_matmul_precision("highest")
    return a.out_dropout(a.out_linear(out.reshape(x.shape)))


class SharedGraph:
    """One graph per active shape, shared across all 20 trained layers.

    Copy trained parameters into stable buffers, then replay. Never keep 20
    separate graph pools. A shape change releases the prior graph and pool.
    Capture cost is reported separately; replay timing includes all copies.
    """

    def __init__(self):
        self.graph = None
        self.shape = None
        self.capture_ms = 0.0

    def prepare(self, attention, x):
        if self.shape == tuple(x.shape):
            return
        self.release()
        start = time.perf_counter()
        self.attention = copy.deepcopy(attention)
        self.static_x = x.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward(self.attention, self.static_x)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = forward(self.attention, self.static_x)
        torch.cuda.synchronize()
        self.graph.replay()
        torch.cuda.synchronize()
        self.shape = tuple(x.shape)
        self.capture_ms += (time.perf_counter() - start) * 1000

    def __call__(self, attention, x):
        assert self.shape == tuple(x.shape), "Prepare graph outside timed forward"
        self.static_x.copy_(x)
        for destination, source in zip(
            self.attention.parameters(), attention.parameters(), strict=True
        ):
            destination.copy_(source)
        self.graph.replay()
        return self.output

    def release(self):
        torch.cuda.synchronize()
        for name in ["graph", "output", "static_x", "attention"]:
            if hasattr(self, name):
                delattr(self, name)
        self.graph = None
        self.shape = None
        gc.collect()
        torch.cuda.empty_cache()
