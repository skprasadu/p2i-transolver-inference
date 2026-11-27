# Transolver reference milestone — 2026-11-26

Implemented the pinned upstream irregular-mesh Physics Attention block, with its MIT license, in [the experiment](../../experiments/transolver_lab/README.md). No custom CUDA kernel or trained CFD predictor was introduced.

## Executed evidence

- `modal run ops/modal_transolver_lab.py` completed on Modal. H100 80GB HBM3, FP32, TF32 disabled, 500,000 analytical NACA 0012 surface points, three input channels, eight heads, head width 64, 64 slices per head.
- Ten timed passes after three warmups: median CUDA-event time **24.149024 ms**, median synchronized wall time **24.171746 ms**. These measure one forward attention block, excluding cold start and geometry creation.
- Peak PyTorch allocated **4142.56384 decimal MB**, reserved **4154.458112 MB**; CUDA context is excluded.
- Deterministic 4096-point comparison against vendored upstream: maximum absolute FP32 error **0.0**. NVIDIA compute-sanitizer reported **zero memory errors** on that equivalence case, not on the full 500,000-point benchmark.
- L4 visual smoke test confirmed NVIDIA `vtkEGLRenderWindow` and 500,000 points. Integer interactive/still ratios are both 1.
- `modal serve --timeout 900 ops/modal_transolver_lab.py` started successfully. Safari rendered the wing through the loopback authenticated proxy; head selection, soft assignment mode, slice 32 selection, and orbit interaction updated the view. Fit control was exercised.
- Unauthenticated access to the Modal development endpoint returned HTTP 401. Production private viewer code was unchanged.

Raw evidence: [benchmark](../../experiments/transolver_lab/evidence/benchmark.json), [memory checker](../../experiments/transolver_lab/evidence/memcheck.txt), [EGL report](../../experiments/transolver_lab/evidence/preview.json), [render](../../experiments/transolver_lab/evidence/preview.png).

## Scope and lifecycle

The colors are seed-42 **untrained assignment weights**, not pressure, stress, aerodynamic accuracy, or a scientific breakthrough. Dominant coloring uses only 3–8 winning slices per head; all 64 have soft participation. This is the golden reference and measured baseline for later kernel work.

The development preview and local bridge are bounded to 900 seconds. The bridge was started with `--revoke-on-exit` to delete its temporary proxy token; credentials remain outside the repository. A development session depends on its local launcher. No production deployment was performed.

Known visual limitation: the scalar-bar text has low contrast against the dark background. It does not affect the computed assignments or benchmark.
