# Transolver reference attention milestone

Exact `Physics_Attention_Irregular_Mesh` from [thuml/Transolver](https://github.com/thuml/Transolver/blob/75e0f67643806a81cd1d3f6adc88dd8c02416fe7/Physics_Attention.py), commit `75e0f67643806a81cd1d3f6adc88dd8c02416fe7`. MIT copyright and license retained in [upstream/LICENSE](upstream/LICENSE). [Paper: Transolver, ICML 2024](https://arxiv.org/abs/2402.02366).

This is one attention block, not a trained Transolver network or CFD solver. Input channels are the three analytical NACA 0012 wing coordinates. There is no feature encoder, training, checkpoint, pressure prediction or stress prediction. 64 slices exist **per head**; random initial assignments need not occupy every dominant ID. Select soft weights to inspect participation beyond argmax coloring. The geometric wing is an analytical extruded airfoil, not a measured aircraft or CFD volume mesh.

## Remote execution only

```sh
.venv/bin/modal run ops/modal_transolver_lab.py
.venv/bin/modal serve --timeout 900 ops/modal_transolver_lab.py
```

`modal run` runs a deterministic 4096-point upstream equivalence case under compute-sanitizer, then benchmarks 500,000 points on H100 and checks NVIDIA EGL rendering on L4. FP32, TF32 disabled, inference mode, eight heads of width 64, 64 slices/head. Three warmups and ten timed passes. Report all CUDA-event samples and synchronized wall times, medians, and peak PyTorch allocated/reserved decimal MB. These exclude cold start and geometry generation; allocator memory excludes the CUDA context. No custom CUDA C++ kernel is introduced. No BF16 claim.

The protected `@modal.web_server(8000)` preview uses `requires_proxy_auth=True`, one L4 container and `scaledown_window=300`. Existing `PrivateViewer` and its auth are unchanged. Use a temporary Modal workspace proxy token kept outside the repo and the local bridge:

```sh
.venv/bin/python ops/transolver_preview_bridge.py --url MODAL_DEV_URL --credentials /path/to/private-token.json
```

Open http://127.0.0.1:8887 in Safari. Credentials never enter the browser. Revoke the temporary proxy token after preview use. `modal serve --timeout 900` ends the ephemeral app; unlike a detached production deployment, local process interruption can end this development session. No `modal deploy` is needed for this milestone.

## Visual contract

Backend `vtkEGLRenderWindow` must report NVIDIA; fail rather than fall back to CPU. Integer remote ratios: interactive 1, still 1. GPU-generated dominant IDs and soft slice weights are the displayed fields. No physical meaning is inferred from random colors. Orbit/zoom, head selection, individual slice-weight selection and fit view are provided. This dev viewer is single-state, not multi-user isolated.

## Experimental fused CUDA extension

Run `.venv/bin/modal run experiments/transolver_lab/test_cuda_kernel.py`. Compilation and compute-sanitizer execute on a remote H100 in CUDA 12.6 devel. [setup.py](setup.py) builds [the kernel](cuda/physics_attention_kernel.cu); no local CUDA is needed. FP32 inference only, D=M=64, no autograd/backward implementation. The wrapper must be used under `torch.inference_mode()`.

Projection to slice logits, stable softmax, and weighted aggregation are fused with shared-memory weights and compact partial reductions. Deslicing recomputes weights. Full module outputs are compared to the unchanged upstream reference on 500,000 points with `max_abs_error < 1e-4`. At that size the reference FP32 intermediate aggregation differs from FP64 by more than this threshold; the test records that discrepancy and separately checks custom slice tokens against FP64 aggregation. See [the complete methodology and results](../../docs/checkpoints/2026-11-26-transolver-cuda.md).

Measured outcome: zero memcheck errors, full-output error 2.85e-6. Slice temporary memory falls from 2048 MB to 65.23 MB, but slice latency increases from 20.07 ms to 76.04 ms. Complete-block latency increases from 23.99 ms to 153.60 ms, with no peak-memory improvement. This is an experimental correctness/memory baseline, not a faster production replacement. Reported GB/s is logical bytes divided by elapsed time, not measured DRAM bandwidth.

## Milestone 3A — pretrained DrivAerML surface validation

[validate_drivaerml.py](validate_drivaerml.py) runs the official NVIDIA model unchanged using PhysicsNeMo 2.1.1, PyTorch 2.10.0/CUDA 12.8 and strict checkpoint loading. The earlier experimental custom CUDA kernel is **not** used. The checkpoint has 20 layers, 8 heads and 512 slices per head. [Worker](drivaer_worker.py) and [viewer](drivaer_viewer.py) are separate to allow inspection without rerunning inference.

```sh
.venv/bin/modal run experiments/transolver_lab/validate_drivaerml.py
# Render saved results only:
.venv/bin/modal run experiments/transolver_lab/validate_drivaerml.py --view-only
.venv/bin/modal serve --timeout 900 experiments/transolver_lab/validate_drivaerml.py
```

Assets and results live in the `p2i-transolver-drivaerml` Modal Volume. Only run 4's surface VTP, STL, native cell areas and reference CSV are downloaded; no volume-flow dataset is fetched. `provenance.json` records file SHA256 hashes and immutable dataset/model revisions. Run 4 is in [NVIDIA's pinned validation manifest](https://github.com/NVIDIA/physicsnemo-cfd/blob/06846c9fbc08b8e82551841897b5294f7e5ef82d/workflows/benchmarking/drivaer_ml_files/validation.csv). This relies on the publisher's split rather than an independent audit of training records.

Measured full-case result on H100: **8,150,920 vertices / 8,140,177 native surface cells**, **20,916.87 ms**, **11,142.36 MB peak PyTorch allocated GPU memory** (decimal MB; excludes CUDA context). One warmed full pass, seeded shuffled 200,000-cell blocks, FP32 with TF32 disabled. Timing includes model forward, slice-ID extraction, denormalization and scatter, but excludes loading, meshing, VTP writing, cloud startup and transfer. This is blocked attention, not a single global attention pass over eight million cells; batch composition is part of the reproducibility contract.

| Field | Cell-unweighted relative L2 | MAE (native units) | Area-weighted relative L2 |
| --- | ---: | ---: | ---: |
| Pressure | 3.3944% | 7.67867 | 3.7160% |
| Wall-shear vector | 6.2667% | 0.056939 | 5.6820% |

WSS relative L2 uses the complete vector; its MAE averages absolute component errors. These global errors do not guarantee accurate local extrema. No acceptance threshold or arbitrary-geometry accuracy claim is inferred from this one case. Exact results: [validation JSON](evidence/drivaer-validation.json).

Normalization follows NVIDIA's surface inference: STL-cell-center mean subtraction; coordinate division by `[12, 4.5, 3.25]`; unit normals; model features `[1.205, 30.0]`; output mean/std from `global_stats.json`, followed by multiplication by `1.205 * 30²`. These are **checkpoint preprocessing conventions**. The original DrivAerML CFD speed is 38.889 m/s. Replacing the trained input with 38.889 is not a faithful reference validation. We compare decoded output numerically to native `pMeanTrim` and `wallShearStressMeanTrim`, with no fitted offset, sign flip, or output rescaling to minimize error. Native OpenFOAM kinematic fields are labeled m²/s², not Pa.

`drivaer_validated.vtp` preserves the original polygon mesh and cell association. Required fields:

- `Pred_Pressure`, `GT_Pressure`, `Abs_Error`: native pressure and absolute pressure difference.
- `Wall_Shear_Stress`: predicted WSS **magnitude**; `Pred_WSS_Vector`, `GT_WSS_Vector`, `GT_WSS_Magnitude` retain the vectors/reference.
- `Physical_Slice_ID`: final attention block (19), head 0, winning ID in 0–511. Positive per-head temperature and softmax preserve the logit argmax. These are learned groups, not physical segmentation or uncertainty.

The viewer uses identical full-range color limits for predicted/reference pressure, no mesh decimation, NVIDIA EGL and integer pixel ratios (1,1). It is proxy-authenticated using the same loopback bridge pattern above. It displays saved inference results; it is not a live CFD solver or a volume-velocity prediction. Production PrivateViewer/auth are unchanged.

Sources: [official checkpoint/model card](https://huggingface.co/nvidia/transolver_drivaerml), [NVIDIA inference wrapper](https://github.com/NVIDIA/physicsnemo-cfd/blob/06846c9fbc08b8e82551841897b5294f7e5ef82d/physicsnemo/cfd/evaluation/models/wrappers/transolver/wrapper.py), [DrivAerML dataset](https://huggingface.co/datasets/neashton/drivaerml), [Ashton et al. dataset paper](https://arxiv.org/abs/2408.11969). NVIDIA Open Model License and DrivAerML CC-BY-SA-4.0 attribution apply to the respective assets.

The initial first-layer/head-0 probe had one winning ID (39) across the case. The saved final result uses the final attention layer/head 0 instead, without changing model predictions. This avoids presenting a uniform early-layer assignment as meaningful geometric segmentation. Viewer allocation accepts L4, T4 or A10 fallbacks with 16 GB host RAM; strict L4 scheduling stalled during validation.

## Milestone 3B — experimental CUDA candidate (not accepted)

The trained NVIDIA 512-slice/32-feature CUDA candidate was tested on the complete
run-4 surface. It reduced peak allocated VRAM from 10,992 MB to 2,005 MB, but took
190,565 ms versus PyTorch's 20,792 ms and failed the strict FP32 equivalence gate:
maximum normalized model-output difference 0.00822067, required <0.0001.
It is **not integrated into the validated viewer**. Do not use it as a numerically
validated replacement or claim a speedup. The original THUML kernel remains a
separate path.

See [the complete report](../../docs/checkpoints/2026-11-26-transolver-3b.md),
[benchmark evidence](evidence/drivaer-cuda-benchmark.json), and
[bounded memcheck log](evidence/drivaer-cuda-memcheck.log).

```sh
.venv/bin/modal run experiments/transolver_lab/benchmark_drivaer_cuda.py
.venv/bin/modal run experiments/transolver_lab/benchmark_drivaer_cuda.py --profile-only
```

## Milestone 3B-v2: tiled GEMM candidate — Gate 1 failed

The vectorized CUDA softmax and three-pass cuBLAS implementation pass the trained
Layer 0 numerical tolerance across the full car, but remain slower than unmodified
PyTorch (1,337 ms versus 893 ms). Allocated VRAM falls 73.7%, below the 82% goal.
TF32 fails accuracy; residual-corrected Tensor Core products do not improve speed.
Gate 2 is blocked. See the [v2 report](../../docs/checkpoints/2026-11-26-transolver-3b-v2.md)
and [machine-readable gate evidence](evidence/v2-gate.json).

```sh
.venv/bin/modal run experiments/transolver_lab/benchmark_drivaer_v2.py
```

The command saves evidence and exits nonzero when Gate 1 fails. It never launches
an ungated full-model benchmark or deployment.

## Milestone 3B-v3: CUDA graphs + tiled TF32 — speed/physics gates pass

One shared attention graph across 20 trained layers reduces full-car inference
from 20,796.70 ms to 17,699.75 ms (1.175x). Layer 0 takes 757.92 ms with max drift
0.000749588 (<0.001). Pressure relative L2 is 3.394170% versus baseline 3.394399%,
passing both percentage-point and relative interpretations of the 0.05% allowance.
All five scoped memcheck processes report zero errors.

Peak allocated memory is 2,252 MB; peak **reserved** memory is 4,641 MB. Graph pools
retain storage, so allocated memory alone does not prove a 2,836 MB physical cap.
See [the v3 report](../../docs/checkpoints/2026-11-26-transolver-3b-v3.md) for scope,
setup cost, limits and [saved evidence](evidence/v3-results.json).

```sh
.venv/bin/modal run experiments/transolver_lab/benchmark_drivaer_v3.py
.venv/bin/python experiments/transolver_lab/report_v3.py
```

## Checkpoint: complete V3 inference and protected viewer

See the [technical white paper](../../docs/papers/p2i_l2_tiled_transolver_whitepaper.md)
for measured ablations, precision contracts and limitations. V3 uses FP32 norms;
FP64 norms belong to V2. Historical failed kernels remain for reproducibility.

Host prerequisite: authenticated Modal CLI (recorded version 1.6.4), Python virtual
environment, and `aiohttp` for the local preview bridge. Remote dependencies and
CUDA development images are pinned in the Modal entrypoints. No local CUDA is
required. Run from the repository root. On a fresh Modal volume, execute in order:

```sh
.venv/bin/modal run experiments/transolver_lab/validate_drivaerml.py
.venv/bin/modal run experiments/transolver_lab/benchmark_drivaer_v3.py
.venv/bin/python experiments/transolver_lab/report_v3.py
.venv/bin/modal run ops/modal_private_viewer.py
.venv/bin/modal serve --timeout 900 ops/modal_private_viewer.py
```

The first command downloads the checkpoint/reference case and saves the baseline
VTP; the second saves V3 predictions and runs verification gates; the fourth
compiles the extension remotely into the image and saves EGL smoke evidence.
These commands incur cloud charges and overwrite local reports with fresh results.
Preserve the tagged evidence when comparing reruns. Do not use `modal deploy` for
this inspection workflow.

In a second terminal, create a temporary proxy token without printing its secret:

```sh
umask 077
.venv/bin/modal workspace proxy-tokens create --name p2i-aero-inspection --json > /tmp/p2i-aero-inspection-token.json
.venv/bin/python ops/transolver_preview_bridge.py \
  --url 'PASTE_THE_AERO_PREVIEW_URL_PRINTED_BY_MODAL_SERVE' \
  --credentials /tmp/p2i-aero-inspection-token.json --port 8888 --revoke-on-exit
```

Open `http://127.0.0.1:8888` in Safari. The bridge binds only to loopback, runs for
15 minutes and revokes its token on orderly exit. Never commit the credentials.
After inspection, stop the bridge and `modal serve` with Ctrl-C. Confirm shutdown:

```sh
.venv/bin/modal app list --json
# Substitute the preview App ID shown above, if still live:
.venv/bin/modal app stop APP_ID --yes
.venv/bin/modal container list --json
```

An empty container list confirms no active containers in the selected environment;
persistent volumes are retained and may continue to incur storage charges. If the
bridge is forcibly killed, explicitly delete its temporary token with
`modal workspace proxy-tokens delete TOKEN_ID --yes` and remove its credential file.

Evidence SHA-256 checksums are in `evidence/SHA256SUMS`. Verify locally with:

```sh
cd experiments/transolver_lab/evidence
shasum -a 256 -c SHA256SUMS
```

The checkpoint formats Python with Black while retaining the upstream reference
source unchanged. CUDA source is preserved as executed. Local checks:

```sh
.venv/bin/python tests/protected_gateway.py
.venv/bin/python -m compileall -q experiments/transolver_lab ops/modal_private_viewer.py scripts/transolver_aero_viewer.py
```

## Publication follow-up (2026-11-27)

The original tagged evidence is immutable. Additional results live in
`evidence/publication/`: cases 4, 17 and 18, strict FP32 / global TF32 / unchanged
V3, five complete-case repetitions per method. Run detached to survive local
connection loss:

```sh
.venv/bin/modal run --detach experiments/transolver_lab/publication_benchmark.py
.venv/bin/python experiments/transolver_lab/publication_report.py
```

The runner resumes configurations with five saved samples in the Modal volume.
It must not reuse those files after changing model, kernel, preprocessing or
measurement protocol; use a new output namespace in that situation. One H100
worker is used, automatic retries are disabled, with a 3,600-second timeout.
Data preparation runs without a GPU. If the local client disconnects, retrieve
`publication/results.json` from the volume after completion; do not start another
job while the detached worker is still active.

```sh
.venv/bin/modal volume get p2i-transolver-drivaerml publication/results.json experiments/transolver_lab/evidence/publication/results.json --force
```

Memory statistics include maxima across repeated runs, not just first-pass peaks.
A fresh selective-precision non-tiled baseline remains outside this bounded pass.
