# Milestone 3B-v2: tiled attention, Gate 1

Status: numerical accuracy passes for Layer 0; performance fails. Gate 2 is blocked.

## Scope and reference corrections

The car has 8,150,920 mesh vertices and 8,140,177 cell-centered model inputs. All cells are covered, using the validated deterministic 200,000-cell partitions (last partition 140,177). This is not a single global 8-million-token attention call. Changing the partition changes the model computation.

NVIDIA PhysicsNeMo 2.1.1's trained checkpoint uses 512 slices, 8 heads, 32 channels per head, clamped temperature [0.5, 5], and `slice_weights.sum(1) + 1e-2`. It divides weights before multiplying them by features. The original THUML reference uses a different epsilon/order. V2 follows NVIDIA's division order, accumulates norm sums in FP64 across tiles, rounds the norm to FP32 and then adds 0.01. This still changes floating-point reduction order and requires validation.

## Implementation

- [CUDA vectorized softmax](../../experiments/transolver_lab/cuda/tiled_softmax.cu): float4 input/output, temperature clamp, stable softmax, optional global normalization. No scalar dense projections.
- [Build](../../experiments/transolver_lab/cuda/setup_v2.py): CUDA 12.8 devel image on Modal H100; no fast-math.
- [Tiled attention](../../experiments/transolver_lab/tiled_attention_v2.py): three passes (global partition norm, normalized aggregation GEMM, deslice GEMM). Dense arithmetic uses PyTorch/cuBLAS. Original weights are recomputed for deslicing.
- [Gate worker](../../experiments/transolver_lab/drivaer_v2_worker.py) and [Modal runner](../../experiments/transolver_lab/benchmark_drivaer_v2.py).

Tested tiles: 1,024, 4,096, 8,192, 32,768 cells. At eight heads and 512 slices, one FP32 tile occupies 16 MiB, 64 MiB, 128 MiB and 512 MiB respectively. Multiple buffers are live. The fastest valid tile was 32,768; **it is not L2-resident**. Small tiles alone do not establish FlashAttention-like cache behavior.

Three arithmetic modes were measured:

1. Strict FP32, TF32 disabled: accurate, slower than reference.
2. Ordinary TF32: faster, but Layer 0 error approximately 0.00086–0.00111, exceeding 0.0001.
3. Residual-corrected Tensor Core GEMMs: split inputs into BF16 high/residual parts; combine three BF16 products accumulated/output in FP32. Small tiles pass Layer 0 accuracy but cost approximately 85–147 ms per 200,000-cell partition. Largest tile costs approximately 80 ms and fails accuracy. This mode is experimental and not selected.

No claim of Tensor Core saturation or successful FlashAttention architecture is supported by these measurements.

## Measured full-car Layer 0 results

H100, warmed attention calls, CUDA event timing summed over all 41 reference partitions. Input preparation, CPU error comparison and dataset loading excluded. Reference precision unchanged. Peak allocated memory includes resident model/geometry; expected outputs are transferred to CPU before measuring the candidate.

| Metric | Unmodified PyTorch | Selected tiled FP32 |
|---|---:|---:|
| Layer 0 attention across complete car | 893.47 ms | 1,337.43 ms |
| Speedup | 1.00× | 0.668× |
| Peak allocated VRAM, decimal MB | 10,782.64 | 2,836.06 |
| Maximum absolute output error vs reference | — | 0.0000217287 |

Numerical gate passes. Speed fails both the requested 1,039 ms cap and the measured 893.47 ms baseline. Memory falls 73.70%, not the requested 82%. These are Layer 0 measurements; do not compare their speedup to the previous 20,792 ms whole-model result.

The unchanged original THUML module with only the softmax replaced passes at maximum absolute error 7.45e-8 (seed 42, 1,031 tokens). An additional 512-slice temperature-clamp and division test passes at 8.94e-8.

## Memory safety and evidence

The initial single-process full-car memcheck completed 8,000,000 cells but ran out of memory on the last 140,177-cell partition. Its four reported errors were CUDA allocation failures; it is a failed check, not a pass. The retained [initial log](../../experiments/transolver_lab/evidence/v2-initial-memcheck.log) records that failure.

A fresh-process grouped rerun covers intervals [0,2000000), [2000000,4000000), [4000000,6000000), [6000000,8140177), preserving each reference partition and including the final partial tile. Memory safety is separate from numerical comparison: the memcheck worker checks finite outputs; the uninstrumented paired gate measures numerical error against PyTorch.

**Final grouped outcome: PASS.** All four processes exited successfully with `ERROR SUMMARY: 0 errors`, covering every one of the 8,140,177 inputs, including the final 140,177-cell partition. No Modal containers remained running at completion. The runner exits nonzero intentionally because the speed gate fails. Evidence is recorded in [gate evidence](../../experiments/transolver_lab/evidence/v2-gate.json) and [memcheck log](../../experiments/transolver_lab/evidence/v2-memcheck.log).

## Decision

Do not run Gate 2, profile the full 20-layer model, deploy this candidate, or replace the validated viewer. No full-model speedup or HBM-bandwidth figure was measured for v2.

The proposed composition of separate tiled GEMMs and softmax kernels did not achieve the combined accuracy, latency and memory targets. Further work would require a different implementation experiment with an explicit cost/benefit hypothesis; larger paid full-model runs cannot fix a failed Layer 0 gate.
