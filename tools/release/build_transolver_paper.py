"""Build a reviewable LaTeX/PDF source bundle from the canonical Markdown report.

Requires pandoc, tectonic and Poppler. This performs no GPU work or submission.
"""

from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / "docs/papers/p2i_l2_tiled_transolver_whitepaper.md"
out = ROOT / "docs/papers/arxiv"
out.mkdir(exist_ok=True)
text = source.read_text()
title = text.splitlines()[0][2:]
text = text[text.index("## Abstract") :]
# Local repository links are source-navigation aids; keep readable labels in PDF.
text = re.sub(r"(?<!!)\[([^\]]+)\]\((?!https?://)([^)]+)\)", r"\1", text)
for name in ["aero-pressure.png", "aero-cut.png"]:
    shutil.copy2(ROOT / "experiments/transolver_lab/evidence/viewer" / name, out / name)
    text = text.replace(
        "../../experiments/transolver_lab/evidence/viewer/" + name, name
    )
text = re.sub(r"^## \d+\. ", "## ", text, flags=re.M)
text = re.sub(r"^### \d+\.\d+ ", "### ", text, flags=re.M)
text = re.sub(
    r"`([0-9a-f]{40,64})`",
    lambda m: "`" + " ".join(m[1][i : i + 16] for i in range(0, len(m[1]), 16)) + "`",
    text,
)
text = text.replace("`100 × ||p-hat − p||₂ / ||p||₂`", r"$100\,\|\hat p-p\|_2/\|p\|_2$")
text = (
    text.replace("PyTorch FP32, V3 paired baseline", "FP32 reference")
    .replace("CUDA V1, scalar fusion", "V1 scalar")
    .replace("CUDA V2, tiled FP32 / FP64 norms", "V2 tiled")
    .replace("CUDA V3, tiled TF32 / FP32 norms / graph", "V3 graph")
)
text = re.sub(r"(!\[[^\]]*\]\(aero-[^)]+\))", r"\1{width=65%}", text)
text = text.replace("(arxiv/benchmark.png)", "(benchmark.png)")
(out / "manuscript.md").write_text(text)
subprocess.run(
    [
        "pandoc",
        "manuscript.md",
        "--from=markdown-implicit_figures+tex_math_single_backslash+raw_tex",
        "--standalone",
        "--to=latex",
        "--output=main.tex",
        "-V",
        "geometry:margin=0.75in",
        "-V",
        "fontsize=10pt",
        "-V",
        "colorlinks=true",
        "-M",
        "title=" + title,
        "-M",
        "author=Krishna Srinivasamurthy — Power2Idea Lab",
        "-M",
        "date=27 November 2026 — Technical report draft",
    ],
    cwd=out,
    check=True,
)
subprocess.run(["tectonic", "main.tex"], cwd=out, check=True)
for name in ["main.aux", "main.log", "main.out"]:
    (out / name).unlink(missing_ok=True)
print(out / "main.pdf")
