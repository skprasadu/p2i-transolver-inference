"""Profiler-only second pass: count HBM bytes, never use instrumented latency."""

import sys
import torch
from experiments.transolver_lab.drivaer_cuda_worker import load, data, switch


@torch.inference_mode()
def main():
    model = load()
    emb, _ = data()
    n = len(emb)
    batch = 200000
    order = torch.randperm(n, device="cuda")
    fx = torch.tensor([1.205, 30.0], device="cuda").expand(1, batch, 2)
    switch(model, "--fused" in sys.argv)
    model(embedding=emb[order[:batch]][None], fx=fx)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for offset in range(0, n, batch):
        idx = order[offset : offset + batch]
        out = model(embedding=emb[idx][None], fx=fx[:, : len(idx)])
        del out
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
