# Step 4 — protected V3 aerodynamic viewer

## Implementation

- [Viewer](../../scripts/transolver_aero_viewer.py) loads the full 8,150,920-vertex surface mesh and replaces baseline display pressure/shear with `v3-full-predictions.npy` from the existing Modal volume. It independently recomputes pressure relative L2 against the CFD field before starting.
- [Modal launcher](../../ops/modal_private_viewer.py) retains the existing engine server and adds a separate session-protected AeroViewer using the existing gateway. Temporary `aero_preview` requires Modal proxy authentication. Idle scaledown is 300 seconds.
- The unchanged V3 extension is compiled during a CUDA-devel image build on an H100, for sm89/sm90, and imported in the remote smoke test. Opening the viewer does not compile or rerun inference.
- PBR geometry, shared-scale predicted/reference pressure, absolute error, WSS magnitude, reference slice IDs, orthographic camera presets, and X-axis cutaway/intersection are available at full resolution. Remote image ratios are integer 1/1.

## Scientific scope

Native pressure and shear are kinematic quantities in m²/s². The optional Cp view explicitly assumes zero reference pressure and uses the dataset's 38.889 m/s speed; the reference-pressure assumption is not independently established. Do not present this as certified Cp validation.

Slice IDs are the previously saved official FP32 final-layer/head-0 grouping, not newly computed V3 IDs or semantic components. There are 512 possible states, not necessarily 512 occupied regions.

The cut plane intersects the car surface. It does not expose internal CAD components or a boundary-layer volume; neither exists in this surface dataset.

HUD latency and memory are saved H100 benchmark measurements, not viewer-frame telemetry. Allocated and reserved VRAM differ (2.25 GB versus 4.64 GB for V3). Rendering GPU is reported separately from the H100 inference GPU. No claim of a fresh CFD solve or improved physics accuracy is made.

## Verification

Existing protected-gateway tests: 3 passed, covering anonymous denial, session replacement/revocation and WebSocket expiration. Python syntax checks passed. Anonymous Modal preview HTTP requests returned 401.

Full-resolution EGL smoke test passed on a Tesla T4: all seven field modes, all four camera presets, and cutaway on/off; 1,557 mid-plane intersection cells. Recomputed pressure L2 was 3.394169735931131%. Geometry was not decimated. Authenticated preview startup returned HTTP 200 (initial request about 33 seconds). See [viewer evidence](../../experiments/transolver_lab/evidence/viewer/aero-smoke.json). Browser automation was unavailable in this session; interactive Safari verification must not be claimed from server screenshots alone.

Preview commands: `modal run ops/modal_private_viewer.py` performs the remote smoke test; `modal serve --timeout 900 ops/modal_private_viewer.py` exposes the proxy-authenticated development preview. A loopback bridge can inject a short-lived Modal proxy token for Safari without placing credentials in URLs. The existing sessioned AeroViewer retains gateway expiry/revocation semantics; no production deployment was performed.

Scientific color maps disable metallic lighting. Default color limits use the 1st–99th percentiles (error/shear: zero–99th), with a visible full-range toggle. Predicted and GT pressure use identical shared limits. Saturation affects colors only; no values are removed or changed.

The visible cutaway uses VTK mapper/GPU clipping to avoid rebuilding an 8M-cell polydata on each slider change. `vtkPlaneCutter` separately computes the surface intersection; the remote smoke report records the measured first cut update time.

Final GPU-clipping smoke passed on Tesla T4: 0.109632665 seconds for the first cut update including render; 8,150,920 vertices, 8,140,177 cells, 1,557 cutter cells, pressure L2 3.3941697359322363%. All modes and camera presets exercised. Final screenshots were visually inspected.
