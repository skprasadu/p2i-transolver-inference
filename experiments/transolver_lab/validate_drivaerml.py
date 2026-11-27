"""Milestone 3A: official pretrained Transolver, native DrivAerML validation.

modal run experiments/transolver_lab/validate_drivaerml.py
modal serve --timeout 900 experiments/transolver_lab/validate_drivaerml.py
"""

from pathlib import Path
import json
import subprocess
import modal

app = modal.App("p2i-transolver-drivaerml")
volume = modal.Volume.from_name("p2i-transolver-drivaerml", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("libegl1", "libgl1", "libxext6", "libxrender1")
    .pip_install(
        "torch==2.10.0",
        "torchvision==0.25.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "nvidia-physicsnemo==2.1.1",
        "huggingface_hub==0.36.0",
        "pyvista==0.49.2",
        "vtk==9.7.2",
        "trame==4.0.0",
        "trame-vtk==2.11.17",
    )
    .env(
        {
            "PYTHONPATH": "/app",
            "VTK_DEFAULT_OPENGL_WINDOW": "vtkEGLRenderWindow",
            "HF_HUB_DISABLE_XET": "1",
        }
    )
    .add_local_dir(
        "experiments/transolver_lab",
        "/app/experiments/transolver_lab",
        copy=True,
        ignore=["__pycache__", "evidence"],
    )
)


@app.function(image=image, volumes={"/data": volume}, cpu=4, memory=32768, timeout=900)
def prepare():
    from experiments.transolver_lab.drivaer_worker import download

    result = download()
    volume.commit()
    return result


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="H100!",
    cpu=8,
    memory=65536,
    timeout=1800,
    retries=0,
)
def validate(batch_size: int = 200000):
    from experiments.transolver_lab.drivaer_worker import solve

    result = solve(batch_size)
    volume.commit()
    return result


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu=["L4", "T4", "A10"],
    cpu=4,
    memory=16384,
    timeout=600,
    retries=0,
)
def visual_check():
    subprocess.run(
        ["python", "-m", "experiments.transolver_lab.drivaer_viewer", "--smoke"],
        check=True,
    )
    return (
        json.loads(Path("/tmp/drivaer-view.json").read_text()),
        Path("/tmp/drivaer-view.png").read_bytes(),
    )


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu=["L4", "T4", "A10"],
    cpu=4,
    memory=16384,
    timeout=900,
    scaledown_window=300,
    max_containers=1,
)
@modal.concurrent(max_inputs=20)
@modal.web_server(8000, startup_timeout=180, requires_proxy_auth=True)
def preview():
    subprocess.Popen(["python", "-m", "experiments.transolver_lab.drivaer_viewer"])


@app.local_entrypoint()
def main(batch_size: int = 200000, view_only: bool = False):
    out = Path("experiments/transolver_lab/evidence")
    out.mkdir(exist_ok=True)
    if not view_only:
        print(json.dumps(prepare.remote(), indent=2))
        result = validate.remote(batch_size)
        (out / "drivaer-validation.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    report, png = visual_check.remote()
    (out / "drivaer-view.json").write_text(json.dumps(report, indent=2))
    (out / "drivaer-view.png").write_bytes(png)
    print(json.dumps(report, indent=2))
