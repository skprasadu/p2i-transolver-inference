# Milestone 3B-v3 — graph-captured tiled TF32 attention

## Decision and scope

One targeted implementation was tested on an H100 80 GB: capture the tiled
attention computation in a CUDA graph, allow TF32 for slice projection,
aggregation and deslicing, and keep FP32 norm accumulation, input/output
projections and slice attention. The user's v3 accuracy criteria supersede the
older strict FP32 equivalence gate for this experiment only. The unchanged CUDA
softmax still passes the original strict golden-reference checks.

This is the complete DrivAer run 4 geometry: 8,150,920 vertices and 8,140,177
cell-centered model inputs, with the original deterministic 200,000-cell
partitions and a final 140,177-cell partition. It is not global attention over
8 million tokens. Checkpoint, data preprocessing, normalization, CFD reference,
seed 42, and NVIDIA epsilon 0.01 are unchanged.

The 32,768-cell tile is 512 MiB at eight heads and 512 slices for one FP32 weight
buffer. It is bounded-memory tiling, **not proven L2-resident attention**. No claim
of Tensor Core saturation is made.

## Implementation

- [Tiled TF32 forward and shared graph](../../experiments/transolver_lab/tiled_attention_v3.py).
- [Remote validation worker](../../experiments/transolver_lab/drivaer_v3_worker.py).
- [Modal entrypoint](../../experiments/transolver_lab/benchmark_drivaer_v3.py).
- [Saved-result table formatter](../../experiments/transolver_lab/report_v3.py).

A single attention graph is reused across the 20 layers. Each call copies that
layer's trained parameters into stable graph buffers, copies its input, and
replays the captured computation. These copies are included in measured forward
latency. A change in partition shape releases the previous graph and captures a
new one, avoiding 20 retained graph pools. All compilation and GPU execution
occurred on Modal; no local CUDA execution or production deployment occurred.

## Measured results

One warmed complete-car pass, H100, PyTorch 2.10.0+cu128. Whole-model timing includes
indexing and forward, excludes loading, graph preparation, denormalization,
CPU result transfer and validation calculations. The paired fresh baseline
agrees closely with the previous 20,792 ms baseline.

| Scope | Latency ms | Speedup vs previous full-model 20,792 ms | Peak allocated MB | Pressure relative L2 % |
|---|---:|---:|---:|---:|
| Layer 0 v3, complete car | 757.919 | Not comparable | 1,023.430 | Not applicable |
| Full 20 layers, unchanged PyTorch | 20,796.695 | 0.99977x | 10,992.240 | 3.394399 |
| Full 20 layers, v3 | 17,699.752 | 1.17471x | 2,252.230 | 3.394170 |

Layer 0 is approximately 1.18x faster than the previous measured 893.47 ms Layer 0
baseline, with maximum absolute drift **0.000749588**, below the requested 0.001.
The full-model measured speedup against the paired baseline is **1.17497x**.
Graph preparation (warmup, capture and first replay for both shapes) adds
**175.92 ms** to the full-model path, excluded from warmed inference time.
This is not a complete cold-start runtime estimate; model/data loading and
container startup are additional.

Pressure L2 changes by **-0.000228973 percentage points**, or **-0.00674561% relative**
to the measured baseline. It passes both interpretations of the user's allowance:

- absolute difference at most 0.05 percentage points;
- relative change at most 0.05% of baseline error.

The small decrease is not evidence of improved physical fidelity. These are
surrogate predictions compared with existing RANS CFD, not a new CFD solve.
Wall-shear vector relative L2 is 6.267464%, versus 6.266735% for the baseline.
Pressure validation on this one case is not broad validation across geometries,
flow conditions or all engineering quantities of interest.

## Memory interpretation

Full-model peak CUDA **reserved** memory is **4,640.997 MB**, versus **11,427.381 MB**
for the paired FP32 baseline. Captured graphs retain reusable scratch storage in
private allocator pools. Consequently, the 2,252 MB allocated counter alone does
**not** establish a 2,252 MB physical working set or preservation of a 2,836 MB
physical-memory cap. Report allocated and reserved figures together.

Layer 0's paired numerical-comparison process reserves 14,424 MB because it also
runs the large unchanged reference attention while retaining the graph. That is
not the isolated full-model candidate's allocator footprint. Do not advertise its
1,023 MB allocated counter as a 90% resident-memory reduction.

## Verification and evidence

- Original THUML module golden check: maximum absolute error 7.45e-8.
- 512-slice clamp/division check: maximum absolute error 8.94e-8.
- Uninstrumented Layer 0 comparison covers every cell.
- Memcheck scope: four fresh-process Layer 0 groups covering every cell; then all
  20 trained layers on the first full-size partition and irregular final partition
  (340,177 inputs). Sanitizer timing/memory is excluded from benchmark results.

**Final status: speed and physics-accuracy gates PASS; all five memcheck processes report zero errors. No Modal containers remained running at completion. The 2,836 MB physical-memory cap is not established (see memory interpretation above).** Evidence is recorded in [results](../../experiments/transolver_lab/evidence/v3-results.json)
and [memcheck output](../../experiments/transolver_lab/evidence/v3-memcheck.log).
Predictions are saved separately in the existing Modal volume as
`v3-baseline-predictions.npy` and `v3-full-predictions.npy`; the validated production
VTP/viewer has not been overwritten.

Reproduce:

```sh
.venv/bin/modal run experiments/transolver_lab/benchmark_drivaer_v3.py
.venv/bin/python experiments/transolver_lab/report_v3.py
```

The second command only formats saved GPU measurements. It performs no local
GPU work. No HBM bandwidth, sustained Tensor Core utilization or multi-case
scientific generalization was measured in this targeted experiment.
