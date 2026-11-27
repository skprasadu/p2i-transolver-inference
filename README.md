# P2I Transolver inference

Reproducible, bounded-memory inference for NVIDIA's pretrained DrivAerML surface
Transolver. Research engineering by Krishna Srinivasamurthy / Power2Idea Lab.

**Release candidate for review, not a claim of state-of-the-art performance.**
The five-repeat test also observes increasing V3 allocator peaks; persistent
service memory behavior has not been established.
See the [technical report](docs/papers/p2i_l2_tiled_transolver_whitepaper.md) and
[publication benchmark evidence](experiments/transolver_lab/evidence/publication/).

The original run-4 experiment measured 17.70 seconds, 2.25 GB peak allocated and
4.64 GB peak reserved memory on H100. It processes all 8.14 million surface cells
in independent 200,000-cell partitions. This is pretrained surrogate inference,
not a new CFD simulation. The publication benchmark compares FP32, globally
TF32-enabled PyTorch and V3 on three preselected cases with five repeats each.
Use the latest report for matched-precision results and accuracy limits.

## Reproduce on Modal

Python 3.12+ on the host and a Modal account are required. The host needs no CUDA.
GPU compilation and execution occur remotely. Downloads persist in the Modal
volume; compute and storage incur charges. The bounded publication job uses one
H100, automatic retries disabled, and a one-hour function timeout.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-host.txt
.venv/bin/modal setup
# First run creates the pinned checkpoint/data cache and baseline artifacts.
.venv/bin/modal run experiments/transolver_lab/validate_drivaerml.py
# Original strict golden tests, scoped memcheck and case-level V3 validation.
.venv/bin/modal run experiments/transolver_lab/benchmark_drivaer_v3.py
# Three cases, three methods, five repetitions per method.
.venv/bin/modal run experiments/transolver_lab/publication_benchmark.py
```

The first command also performs a remote baseline visualization smoke check.
No command deploys a persistent service. Check `modal container list` after jobs
finish. The complete private P2I authentication/viewer infrastructure is not
part of this kernel release. Saved viewer screenshots document the separate demo.

## Implementation

- `tiled_attention_v3.py`: selective TF32, FP32 slice norms and shared CUDA graph.
- `cuda/tiled_softmax.cu`: unchanged verified CUDA softmax/normalization kernel.
- `publication_worker.py`: fresh process per case/method, five timed repetitions.
- `evidence/`: successful and failed historical measurements, clearly scoped.

The tile is not proven L2-resident. No backward/training implementation is claimed.
The strict FP32 golden softmax check is separate from V3's explicitly relaxed
Layer 0 and CFD-accuracy gates. See the report for sanitizer coverage.

## Attribution and citation

Upstream authors created Transolver, PhysicsNeMo and DrivAerML; P2I provides this
inference implementation and evaluation. Cite [Transolver](https://arxiv.org/abs/2402.02366)
and [DrivAerML](https://arxiv.org/abs/2408.11969) when using this experiment.
Read [NOTICE.md](NOTICE.md) for licenses. No pretrained checkpoint or raw CFD mesh
is redistributed. No affiliation or endorsement by upstream organizations is implied.
