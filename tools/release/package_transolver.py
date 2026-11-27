"""Export an allowlisted standalone research repository, without lab credentials.

Usage: .venv/bin/python tools/release/package_transolver.py /absolute/export/path
The destination must not exist. No network calls or publication are performed.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def package(destination):
    destination.mkdir(parents=True, exist_ok=False)
    lab = Path("experiments/transolver_lab")
    files = list((ROOT / lab).glob("*.py")) + list((ROOT / lab / "cuda").glob("*"))
    files += list((ROOT / lab / "upstream").glob("*"))
    files += [p for p in (ROOT / lab / "evidence").rglob("*") if p.is_file()]
    files += list((ROOT / "docs/checkpoints").glob("2026-11-26-transolver-*.md"))
    files += [ROOT / "docs/papers/p2i_l2_tiled_transolver_whitepaper.md"]
    files += list((ROOT / "docs/papers/arxiv").glob("*"))
    files += [
        ROOT / lab / "README.md",
        ROOT / "docs/checkpoints/2026-11-27-transolver-publication.md",
        ROOT / "docs/papers/p2i-transolver-arxiv-source.zip",
    ]
    files += [p for p in (ROOT / "tools/release").glob("*") if p.is_file()]

    manifest = []
    for source in files:
        if not source.is_file() or source.suffix in {".pyc", ".so", ".o"}:
            continue
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest.append(
            dict(
                path=str(relative),
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            )
        )
    for name in ["README.md", "LICENSE", "NOTICE.md"]:
        shutil.copy2(
            ROOT / "tools/release" / ("transolver-" + name), destination / name
        )
    (destination / ".gitignore").write_text(
        "__pycache__/\n*.pyc\n*.so\nbuild/\n.venv/\n.env*\n*.npz\n*.npy\n*.mdlus\n.DS_Store\n"
    )
    (destination / "requirements-host.txt").write_text("modal==1.6.4\n")
    (destination / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(manifest)} allowlisted files to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    package(parser.parse_args().destination.resolve())
