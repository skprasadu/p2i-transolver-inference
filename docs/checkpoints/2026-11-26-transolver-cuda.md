# Fused Transolver slice operation

## Implementation contract

[CUDA extension](../../experiments/transolver_lab/cuda/physics_attention_kernel.cu) and [build script](../../experiments/transolver_lab/setup.py): FP32 inference, contiguous CUDA tensors, feature width and slice count fixed at 64. Arbitrary supported batch/head counts and nonempty point counts, including tail tiles. No backward implementation or BF16 support is claimed.

Input feature projections remain PyTorch linear layers. The custom operation fuses **slice-logit projection, stable softmax, and weighted aggregation**. Each block handles up to 1,024 points in 32-point shared-memory tiles. It writes partial sums, then a deterministic reduction computes normalized slice tokens. This retains the upstream `+1e-5` denominator. No global `[B,H,N,M]` weights are allocated. Deslicing recomputes weights in shared memory rather than retaining them globally. Slice-token attention and output projection remain upstream PyTorch operations.

## Verification and measurement

[Modal runner](../../experiments/transolver_lab/test_cuda_kernel.py) builds with nvcc inside CUDA 12.6 devel on H100, then executes a separate compute-sanitizer subprocess. Deterministic seed-42 cases cover `(B,N)=(2,37),(1,1025),(1,500000)`, on a non-default stream with explicit producer-stream synchronization. Complete block outputs must have maximum absolute error below `1e-4` against the unchanged reference. Small-case slice tokens also use this gate. At 500,000 points, slice tokens are checked against an FP64 aggregation of the reference weights, with the intermediate FP32 discrepancy explicitly recorded.

The original extra intermediate gate failed at 500,000 points: custom versus reference FP32 slice tokens differed by 0.000825882, although complete outputs differed by only 0.000002852. A diagnostic found custom tokens within 0.000002384 of FP64 aggregation, while reference FP32 tokens differed by 0.000826359. This supports accumulation-order rounding in the long reference aggregation. The custom algorithm and full-module `1e-4` gate were unchanged; the additional large intermediate gate now checks FP64 aggregation instead of insisting on reproducing that FP32 rounding. This is not a claim of improved CFD accuracy or general equivalence for all trained weights.

Benchmarks are outside the sanitizer, inference mode, FP32/TF32 disabled, three warmups and ten CUDA-event samples. Slice-operation and complete-block latencies are reported separately. Memory is peak allocator increase above already resident inputs/model. Bandwidth is **effective logical input/output GB/s**, using the same byte denominator for both slice implementations; it is not measured physical DRAM traffic and does not count rereads or partial buffers. This permits a fair latency-derived comparison without inventing profiler measurements.

## Completed run

`modal run experiments/transolver_lab/test_cuda_kernel.py` completed successfully on NVIDIA H100 80GB HBM3. [Modal run](https://modal.com/apps/praskrishna/main/ap-oDtepehNrLccV6E2cPiYzZ). Full 500,000-point checks ran under compute-sanitizer with **zero memory errors**. Complete block maximum error was **2.8517097e-6**, below `1e-4`.

| Implementation | Median ms | Effective logical GB/s | Incremental peak MB |
|---|---:|---:|---:|
| PyTorch slice | 20.0711 | 102.0447 | 2048.00 |
| Custom CUDA slice | 76.0354 | 26.9368 | 65.23 |
| PyTorch full block | 23.9867 | N/A | 4102.92 |
| Custom CUDA full block | 153.5955 | N/A | 4103.70 |

The slice stage saves about 96.8% of temporary allocation but is about 3.79 times slower. The complete block is about 6.40 times slower with no measured peak-memory benefit: projected features and output/layout tensors remain live. These results do not justify replacing the production/reference path. The scalar dot products, serial per-point softmax, and weight recomputation are candidates for profiling; no bottleneck attribution is claimed without profiling. No tensor-core speedup or physical DRAM-bandwidth claim is made.

Evidence: [raw timings and errors](../../experiments/transolver_lab/evidence/cuda-benchmark.json), [build log](../../experiments/transolver_lab/evidence/cuda-build.txt), [sanitizer log](../../experiments/transolver_lab/evidence/cuda-memcheck.txt). The application completed and stopped; no production viewer changes or deployment were made.
