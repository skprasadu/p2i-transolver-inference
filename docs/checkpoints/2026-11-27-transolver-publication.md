# Transolver publication-readiness checkpoint — 2026-11-27

## Outcome

Completed the bounded follow-up: 3 preselected validation cases × 3 methods × 5
complete-case repetitions = 45 measured forwards, with no kernel changes.
Cases: 4, 17, 18. All V3 pressure checks pass both the 0.05 percentage-point and
0.05%-relative gates; global TF32 fails both on every case.

- Median V3 speedup versus strict FP32: 1.162–1.166x.
- Maximum-over-five allocated-memory reduction: 76.6–77.2%.
- V3 allocated peaks grow over repetitions; no fixed persistent-serving bound.
- No new sanitizer claim: original scoped V3 coverage is retained explicitly.
- Global TF32 is faster than V3 but falls outside the predeclared pressure gate.
- Selective-precision non-tiled and graph-only controls remain untested.

## Execution and recovery

First job: `ap-M6DWQ2617uxYmEqe2Ht7mi`; local connection loss interrupted execution.
Completed configurations were committed to the Modal volume. Detached resume:
`ap-e3CvTFpbK2nWTQyQlygjAC`. It retained completed results and finished the remaining
matrix. GPU hardware identity across allocations was not recorded. Do not claim
all nine configurations ran on the same physical GPU. All jobs are now completed;
container list is empty. No production deployment or audited cost claim.

## Deliverables

- [Raw measurements](../../experiments/transolver_lab/evidence/publication/results.json)
- [Computed summary](../../experiments/transolver_lab/evidence/publication/summary.json)
- [Updated paper](../papers/p2i_l2_tiled_transolver_whitepaper.md)
- [Compiled review PDF](../papers/arxiv/main.pdf)
- [LaTeX source archive](../papers/p2i-transolver-arxiv-source.zip)
- [Outreach drafts](../papers/transolver_outreach_drafts.md)
- Private staging repository: https://github.com/skprasadu/p2i-transolver-inference

The paper removes the unverified L2-residency implication from its title, adds
Transolver++ / FlashAttention context, and corrects the DrivAerML CFD description
to hybrid RANS–LES. Old tagged measurements remain unchanged.

## Release state

The GitHub repository is private, ready for review. No outreach messages sent;
no arXiv submission performed. Author approval of the final manuscript, identity,
license and arXiv declarations remains necessary before author submission. Source
MIT licensing is separated from upstream licenses and CC BY-SA dataset-derived
figures. No model weights or raw CFD meshes are redistributed.

The PDF is generated with Pandoc/Tectonic and inspected with Poppler renders.
Local syntax/format checks and all three existing gateway tests passed. The
standalone Modal entrypoint imports from its own checkout. This is not a claim
that the entire GPU matrix was rerun from the exported checkout.

Reproduction and export commands are in the experiment README and
[release tools](../../tools/release/). Future runs must change the output namespace
if measurement code or model changes; cached five-sample configurations are otherwise
reused intentionally. Do not overwrite the original V3 checkpoint tag.
