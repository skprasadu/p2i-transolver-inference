"""Bounded release validation: 3 cases × 3 implementations × 5 repeats, one H100."""

import json
from pathlib import Path
import modal
from experiments.transolver_lab.benchmark_drivaer_cuda import image, volume

app = modal.App("p2i-transolver-publication")


@app.function(
    image=image, volumes={"/data": volume}, cpu=4, memory=16384, timeout=1800, retries=0
)
def prepare():
    import csv, hashlib, urllib.request
    from huggingface_hub import hf_hub_download
    from experiments.transolver_lab.drivaer_worker import ROOT, DATA_REV

    url = "https://raw.githubusercontent.com/NVIDIA/physicsnemo-cfd/06846c9fbc08b8e82551841897b5294f7e5ef82d/workflows/benchmarking/drivaer_ml_files/validation.csv"
    split = urllib.request.urlopen(url, timeout=30).read().decode()
    ids = sorted({int(row["run_idx"]) for row in csv.DictReader(split.splitlines())})
    assert 4 in ids
    cases = [4] + [i for i in ids if i != 4][:2]
    files = []
    for case in cases:
        for name in [
            f"run_{case}/boundary_{case}.vtp",
            f"run_{case}/drivaer_{case}.stl",
        ]:
            p = Path(
                hf_hub_download(
                    "neashton/drivaerml",
                    name,
                    revision=DATA_REV,
                    repo_type="dataset",
                    local_dir=str(ROOT),
                )
            )
            with p.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            files.append(dict(file=name, bytes=p.stat().st_size, sha256=digest))
            print("Cached", name, flush=True)
    result = dict(
        cases=cases,
        dataset_revision=DATA_REV,
        split_source=url,
        selection="run4 plus first two other validation IDs sorted numerically; no outcome-based selection",
        files=files,
    )
    (ROOT / "publication").mkdir(exist_ok=True)
    (ROOT / "publication/provenance.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="H100!",
    cpu=8,
    memory=65536,
    timeout=3600,
    retries=0,
    max_containers=1,
)
def benchmark(cases):
    import subprocess, sys

    root = "/app/experiments/transolver_lab"
    subprocess.run(
        [sys.executable, "cuda/setup_v2.py", "build_ext", "--inplace"],
        cwd=root,
        check=True,
        timeout=600,
    )
    reports = []
    for j, case in enumerate(cases):
        # Rotate mode order between cases to reduce systematic order bias.
        modes = ["fp32", "tf32", "v3"]
        modes = modes[j:] + modes[:j]
        for mode in modes:
            saved = Path(f"/data/publication/run-{case}-{mode}.json")
            if saved.exists():
                cached = json.loads(saved.read_text())
                if len(cached.get("samples", [])) == 5:
                    print("Resume: retaining completed", case, mode, flush=True)
                    reports.append(cached)
                    continue
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.transolver_lab.publication_worker",
                    str(case),
                    mode,
                    "5",
                ],
                check=True,
                timeout=600,
            )
            volume.commit()
            reports.append(
                json.loads(
                    Path(f"/data/publication/run-{case}-{mode}.json").read_text()
                )
            )
    result = dict(
        reports=reports,
        existing_kernel="unchanged v3; prior scoped sanitizer evidence retained",
    )
    Path("/data/publication/results.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    out = Path("experiments/transolver_lab/evidence/publication")
    out.mkdir(parents=True, exist_ok=True)
    provenance = prepare.remote()
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    result = benchmark.remote(provenance["cases"])
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
