# Prompt 3 Step 1 — official pretrained model discovery

## Decision

Use the official NVIDIA pretrained Transolver DrivAerML route for the next reproduction. Do not begin AirfRANS training: downloadable trained 3D models exist. No inference, GPU execution, training, CFD dataset download, or physical-field export was performed during this discovery step.

## Verified downloadable assets

[NVIDIA Hugging Face repository](https://huggingface.co/nvidia/transolver_drivaerml), pinned revision `96477aeb86d24c26ccf0797bca1b3851268017d0`. The unauthenticated [API](https://huggingface.co/api/models/nvidia/transolver_drivaerml) reported `gated=false`, `private=false`. Anonymous range requests returned HTTP 206 and actual bytes for both model archives. Subsequently both complete archives were downloaded into temporary local storage and only tar JSON metadata was inspected; no pickle or model execution occurred.

| Asset | Bytes | Purpose |
|---|---:|---|
| `transolver_drivaerml_surface_checkpoint/Transolver.0.501.mdlus` | 39,188,235 | Surface pressure and three wall-shear components |
| `transolver_drivaerml_volume_checkpoint/Transolver.0.501.mdlus` | 39,191,307 | Volume flow fields; five output channels in saved architecture |
| `global_stats.json` in each directory | 1,375 / 1,532 | Training normalization statistics |

Surface SHA256: `eb98f399a050a8f8a24919335c61642e4a835bd4044f7e21abec231aa31fd82c`.
Volume SHA256: `436804a02c1fe4cc3f70072826df933a39b0e98bc88bd4edf231eeb1d10e55d3`.

The model card identifies the NVIDIA Open Model Agreement and commercial availability. There is no manual download gate. It describes a DrivAer notchback-car training domain, not general wing or Ahmed-body validation. The model card describes eight layers, but **the saved archives specify 20**; implementation must follow checkpoint metadata, not that prose.

## Actual checkpoint contract and CUDA compatibility

Both archive `args.json` files specify `n_layers=20`, `n_hidden=256`, `n_head=8`, `slice_num=512`, `functional_dim=2`, `use_te=false`, `plus=false`. Therefore per-head width is 32. Surface has `embedding_dim=6,out_dim=4`; volume has `embedding_dim=7,out_dim=5`. Both `metadata.json` files report PhysicsNeMo 1.3.0. The volume statistics include velocity, pressure_volume and turbulent_viscosity; exact channel interpretation and dimensional decoding must be checked against the matching inference implementation before export.

Our [CUDA kernel](../../experiments/transolver_lab/cuda/physics_attention_kernel.cu) hardcodes D=M=64 and is inference-only. It cannot load these attention shapes unchanged. Moreover [PhysicsNeMo 1.3.0 attention](https://github.com/NVIDIA/physicsnemo/blob/v1.3.0/physicsnemo/models/transolver/Physics_Attention.py) clamps temperature to [0.5,5] and adds `1e-2` to the slice normalization denominator. Our THUML reference uses an unclamped temperature and `1e-5`. An adapter requires new deterministic numerical and sanitizer qualification against this exact NVIDIA implementation. Existing kernel tests do not establish equivalence for these checkpoints.

## Inference and ground-truth hooks

- [Official Python recipe](https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/external_aerodynamics/transformer_models): training, preprocessing and validation inference. Current code/configuration can differ from the saved checkpoint version.
- [Official standalone inference workflow](https://github.com/NVIDIA/physicsnemo-serve/tree/main/native-inference/workflows/transolver): references the same pinned model revision and raw DrivAerML VTP/STL inputs. This is a reproducibility lead, not a workflow tested by P2I yet.
- [Preprocessing](https://github.com/NVIDIA/physicsnemo/blob/main/examples/cfd/external_aerodynamics/transformer_models/src/preprocess.py) supplies air density and stream velocity as functional inputs. No explicit angle-of-attack input was found there. Geometry rotation is not evidence of validated angle-of-attack generalization.
- [DrivAerML dataset](https://huggingface.co/datasets/neashton/drivaerml): API listing for example `run_1` reports boundary VTP 659,606,189 bytes and STL 142,385,186 bytes (~802 MB together). Its complete volume file is split across two parts totaling 49,052,296,843 bytes (~49.05 GB). Do not download the full dataset to obtain one surface case. `run_1` is an example, not verified here to be held out from training; identify the official test split before claiming test error.

Pressure coefficient must be derived only after matching physical units and reference pressure/dynamic pressure. Surface pressure and volume velocity belong to different domains. Prefer car-surface Cp plus a surrounding velocity slice; do not paint volume speed on the wall without an explicit sampling definition. No claim that this checkpoint supports arbitrary speed/AoA/geometry sweeps is justified by download availability alone.

## Other search results and fallback

The public HF `thuml` model listing and Transolver/AirfRANS model searches did not reveal a THUML-owned AirfRANS or ShapeNet-Car checkpoint. Third-party repositories exist, including OneScience-Group and zenteiq-ai, but were not treated as official upstream checkpoints. This is a bounded search finding, not proof that none exists elsewhere.

See [AirfRANS hooks](2026-11-26-airfrans-hooks.md) for official data, conditioning, training and validation paths. AirfRANS is a 2D fallback. Our present inference-only CUDA extension has no backward operation and cannot be used to train the official model unchanged.

## Proposed next execution gate

1. Pin a compatible PhysicsNeMo environment; load the official surface checkpoint with its own normalization and a documented held-out DrivAer case.
2. Run the unchanged official path on Modal and measure ground-truth pressure error. Establish that baseline before substituting kernels.
3. Adapt and verify the CUDA operation for D=32/M=512 and NVIDIA normalization semantics; compare real checkpoint outputs against the unchanged path.
4. Export validated pressure and slice assignments in Trame. Add the volume model with a bounded volume-data/sampling plan for velocity. Keep speed/AoA controls limited to substantiated model support.
