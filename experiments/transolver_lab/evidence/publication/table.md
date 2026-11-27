## Publication validation: three cases and repeated measurements

Cases 4, 17 and 18 were selected from the pinned publisher validation split before
examining their predictions. Each method ran in a fresh process, five sequential
complete-case repetitions; execution order was rotated between cases. All used
the same checkpoint, seed 42 and 200,000-cell partition convention. Original V3
CUDA code was unchanged. This follow-up reuses the earlier scoped sanitizer evidence;
it does not claim additional sanitizer coverage on cases 17 and 18.

| Case | Method | Median ms [min, max], n=5 | Peak allocated / reserved MB | Pressure L2 % | 0.05 pp gate |
|---|---|---:|---:|---:|---|
| 4 | fp32 | 20737.5 [20733.6, 20761.3] | 10992.2 / 11427.4 | 3.394399 | Pass |
| 4 | tf32 | 12614.9 [12612.1, 12617.4] | 10992.2 / 11427.4 | 3.626330 | Fail |
| 4 | v3 | 17805.0 [17759.3, 17815.4] | 2529.8 / 5008.0 | 3.394170 | Pass |
| 17 | fp32 | 24843.4 [24842.1, 24844.7] | 11042.1 / 11530.1 | 3.024468 | Pass |
| 17 | tf32 | 15009.3 [15004.8, 15014.4] | 11042.1 / 11530.1 | 3.289615 | Fail |
| 17 | v3 | 21305.2 [21211.3, 21313.1] | 2578.9 / 4972.3 | 3.023651 | Pass |
| 18 | fp32 | 18710.1 [18709.3, 18713.5] | 10966.6 / 11360.3 | 5.782420 | Pass |
| 18 | tf32 | 11461.5 [11460.5, 11463.1] | 10966.6 / 11360.3 | 5.943229 | Fail |
| 18 | v3 | 16094.8 [16034.6, 16110.2] | 2504.2 / 4930.4 | 5.781841 | Pass |

Memory columns use the maximum over all five repetitions, including graph-pool
behavior across repeated runs. They must not be replaced by the smaller first-pass
numbers. These runs test repeatability within each method process. A disconnected job was
resumed in a second H100 allocation, retaining completed configurations; hardware
identity across that boundary was not recorded. This is not a systematic
cross-host variability study. Per-repetition measurements and standard deviations
are available in the accompanying JSON files.

Globally enabling TF32 is a distinct accuracy/performance trade-off, not automatically
a valid replacement under the declared pressure gate. A same-selective-precision,
non-tiled implementation and separate graph-only control remain untested; the
present evidence cannot isolate the speed contribution of each optimization.
