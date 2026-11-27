# Milestone 3B: trained NVIDIA attention CUDA candidate

Status: **rejected as a replacement**. The implementation builds and passes the bounded memory-safety check, but does not pass full-model FP32 equivalence and is slower than the unmodified model. No production/viewer integration or deployment was made.

## Reproducible scope

- Official `Transolver.0.501.mdlus`, same checkpoint and run-4 data as [Milestone 3A](../../experiments/transolver_lab/README.md).
- H100 80GB; PyTorch 2.10.0+cu128; PhysicsNeMo 2.1.1; FP32 with TF32 disabled; seed 42.
- Geometry: 8,150,920 vertices, **8,140,177 cell-centered model inputs**. All cells processed in the same shuffled 200,000-cell partitions, including the 140,177-cell remainder. This is not a single globally coupled 8-million-token attention call.
- All 20 trained layers replaced for the candidate; checkpoint weights and preprocessing unchanged.
- Warmup followed by one complete timed pass per implementation. Timings include indexing and model inference, excluding model/data loading, visualization, slice-ID extraction, denormalization and writing. These paired timings are more directly comparable to each other than to 3A's slightly broader timing scope.

## Result

| Implementation | Complete inference (ms) | Speedup vs PyTorch | Peak allocated VRAM (decimal MB) | Effective HBM GB/s |
|---|---:|---:|---:|---:|
| Unmodified PyTorch | 20,792.189 | 1.000 | 10,992.240 | 1,659.454 |
| Experimental fused CUDA | 190,564.713 | 0.109 | 2,004.550 | 180.198 |

The candidate reduces peak allocated memory by approximately 81.8%, but is about 9.17 times slower. This is not a speedup, nor evidence of improved CFD accuracy.

**Full-case maximum absolute final-output error: 0.0082206726**, compared with the required **0.0001**. Error is measured on the normalized four-channel network output, not CFD relative L2 or pressure in physical units. Per-layer cumulative errors are retained in the [machine-readable evidence](../../experiments/transolver_lab/evidence/drivaer-cuda-benchmark.json). The worker asserts the release gate after saving complete diagnostic measurements; a failed diagnostic is never promoted to a valid replacement.

## Kernel and verification details

The original 64-slice/64-feature THUML path is retained. The new experimental path supports 512 slices and 32 features, clamps trained head temperatures to [0.5, 5], and uses a slice normalization denominator with epsilon 0.01. Projection and softmax use tile-local shared memory; partial slice aggregates are stored globally; weights are recomputed during deslicing. No complete B×H×N×512 weight tensor is materialized by the new path.

**Important difference:** the candidate divides aggregated partial sums by the slice norm after aggregation. NVIDIA divides weights by the norm before its matrix multiplication. These operations are algebraically equivalent but not floating-point identical. Projection, softmax and reduction orders also differ. The candidate therefore does not satisfy the requested numerical replacement gate; it must not be described as NVIDIA-exact FP32 arithmetic. The available evidence does not isolate which operation contributes most to the end-to-end discrepancy.

Compute Sanitizer reported **zero errors** for all 20 trained layers on real-data subsets of 37 and 1,025 cells, plus the first trained layer on 16,385 cells. Isolated attention errors were below 2.9e-6. This is a bounded instrumented test, **not memcheck of the complete car**. A larger instrumented attempt exhausted memory and was rejected. A separate small 1,025-cell end-to-end test also failed equivalence (approximately 5.74e-4); it was not hidden or counted as passing.

Full-car numerical testing is independent of the bounded sanitizer cases and covers every production partition. Passing isolated layer comparisons does not establish equivalence through 20 layers.

HBM counter profiling completed in a separate pass using Nsight Compute `dram__bytes.sum`, with clock and cache control disabled. Instrumented profiler runtime is not used as inference latency. Bandwidth is the full-case measured DRAM read-plus-write byte count divided by uninstrumented full-case wall time. These are one-pass effective measurements, not peak hardware specifications; uncontrolled clocks/caches can introduce variability.

## Commands and files

```sh
.venv/bin/modal run experiments/transolver_lab/benchmark_drivaer_cuda.py
.venv/bin/modal run experiments/transolver_lab/benchmark_drivaer_cuda.py --profile-only
```

- [CUDA extension](../../experiments/transolver_lab/cuda/physics_attention_kernel.cu)
- [Build configuration](../../experiments/transolver_lab/setup.py)
- [Modal test runner](../../experiments/transolver_lab/benchmark_drivaer_cuda.py)
- [Trained-data worker](../../experiments/transolver_lab/drivaer_cuda_worker.py)
- [HBM profiler worker](../../experiments/transolver_lab/profile_drivaer_cuda.py)
- [Memcheck evidence](../../experiments/transolver_lab/evidence/drivaer-cuda-memcheck.log)

Keep the validated PyTorch model as the reference and viewer implementation. Any next kernel design must retain efficient tiled matrix multiplication, explicitly address NVIDIA's pre-aggregation normalization, and pass a production-sized end-to-end check before paying for another full-car benchmark. Lower temporary memory by itself is not an optimization win.

The original THUML golden-reference regression was rerun on the remote H100 with
this combined extension and PyTorch 2.10.0. Full-module maximum absolute error at
500,000 deterministic points was 2.8517097e-6 (pass). Its already documented
standalone aggregation-versus-FP32-reference discrepancy remains; the separate
FP64 aggregation diagnostic is not substituted for the trained NVIDIA model's
failed FP32 gate.

Measured source SHA-256 hashes:

- CUDA kernel: `4879374eb9b40bf87075999796c21e1a61c189bf39477c64c662400a07ce1b0a`
- Setup: `8eb74dc0eecbf0e21124a720dd5e4624a9d4573288b0fda6edf6a0321e71cf09`
- Trained-data worker: `de0d8529850df646f5e34739b6a84870d033536b67b7baf4798deab5cf8b2c2b`

The local entrypoint saves evidence and prints the comparison before asserting
the numerical gate. A failed gate must produce a nonzero command exit status,
even when the diagnostic benchmark itself completed.

Baseline HBM profiling completed: 20,787 kernel records, 34,503,682,521,600
recorded DRAM bytes, yielding 1,659.454 GB/s when divided by the uninstrumented
20.792189-second full-case latency. This is read-plus-write traffic, not unique
input dataset size, and is measured in a separate instrumented pass.

Precision disclosure: model inputs, outputs and stored tensors are FP32, and
TF32 is disabled. The candidate's final reduction of tile partials uses FP64
accumulators before storing FP32 tokens. It is therefore not a pure-FP32
arithmetic implementation. This does not waive its comparison against the
unmodified FP32 reference; that required comparison failed.

Reference repeatability control: two unmodified PyTorch passes on the identical
200,000-cell seeded run-4 input produced **maximum absolute difference 0.0** on
H100. See [control evidence](../../experiments/transolver_lab/evidence/reference-repeat.json)
and [control runner](../../experiments/transolver_lab/check_reference_repeat.py).
This rules out observed reference run-to-run variation as the explanation for
the candidate's error on that block.

## Full HBM result and concrete bottleneck

Both complete-case counter passes finished:

- PyTorch: **34,503,682,521,600 bytes**, 20,787 kernel records, **1,659.454 GB/s**.
- Candidate: **34,339,374,049,280 bytes**, 15,867 kernel records, **180.198 GB/s**.

Peak allocated VRAM fell 81.8%, but total DRAM traffic fell only about 0.48%.
Allocation footprint and total bytes transferred are distinct measurements.
The custom `nv::aggregate` kernel accounts for **26,506,439,356,416 bytes**,
approximately **77.2%** of candidate DRAM traffic. This is a measured hotspot;
register/local-array placement and memory-access behavior require inspection
before attributing the traffic to a specific compiler mechanism. No spill-count
measurement was collected, so this report does not assert that spills are proven.

See [per-kernel breakdown](../../experiments/transolver_lab/evidence/hbm-kernel-breakdown.json),
[PyTorch raw counter log](../../experiments/transolver_lab/evidence/hbm-PyTorch.log),
and [candidate raw counter log](../../experiments/transolver_lab/evidence/hbm-Fused-CUDA.log).

Next bounded gate, if continuing development: preserve normalization before
aggregation, inspect the aggregation kernel's generated-code/register behavior,
and test one fixed 200,000-cell trained batch for both error and latency before
another full-car run. The current candidate remains rejected. All Modal
containers from this milestone were stopped when evidence collection completed.
