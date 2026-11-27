"""Untrained GPU slice assignments on an analytical NACA 0012 wing, remote EGL."""

import os

os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkEGLRenderWindow"
import argparse, json
from pathlib import Path
import numpy as np
import torch
import pyvista as pv
from trame.app import get_server
from trame.ui.html import DivLayout
from trame.widgets import html, vtk
from .workload import setup, wing_points


def scene():
    model = setup()
    with torch.inference_mode():
        points = wing_points()
        b, n, c = points.shape
        mid = (
            model.in_project_x(points)
            .reshape(b, n, model.heads, model.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        weights = model.softmax(model.in_project_slice(mid) / model.temperature)
        assert torch.max(torch.abs(weights.sum(-1) - 1)).item() < 1e-4
        dominant = weights.argmax(-1)[0].cpu().numpy()
        occupancy = [
            torch.unique(weights[0, h].argmax(-1)).numel() for h in range(model.heads)
        ]
        xyz = points[0].cpu().numpy()
    # Surface connectivity is geometry preparation, not an invented physics field.
    a = np.arange(499)[:, None] * 1000 + np.arange(1000)[None, :]
    b = np.arange(499)[:, None] * 1000 + (np.arange(1000)[None, :] + 1) % 1000
    faces = np.stack((np.full_like(a, 4), a, b, b + 1000, a + 1000), axis=-1).reshape(
        -1
    )
    mesh = pv.PolyData(xyz, faces)
    mesh["slice"] = dominant[0]
    plot = pv.Plotter(off_screen=True, window_size=(1500, 950))
    plot.set_background("#0b1724", top="#254359")
    actor = plot.add_mesh(
        mesh,
        scalars="slice",
        clim=(0, 63),
        cmap="turbo",
        n_colors=64,
        smooth_shading=True,
        show_scalar_bar=True,
        scalar_bar_args={"title": "Dominant slice ID (head 0)"},
    )
    plot.camera_position = ((4, 6, 3), (0.5, 0, 0), (0, 0, 1))
    plot.reset_camera()
    plot.show(auto_close=False, interactive=False)
    caps = plot.render_window.ReportCapabilities()
    assert (
        plot.render_window.GetClassName() == "vtkEGLRenderWindow" and "NVIDIA" in caps
    )
    report = {
        "renderer": plot.render_window.GetClassName(),
        "nvidia": True,
        "points": int(points.shape[1]),
        "gpu": torch.cuda.get_device_name(),
        "dominant_slices_per_head": occupancy,
        "scope": "untrained seed 42 assignments; not pressure or stress",
    }
    return plot, mesh, actor, weights, dominant, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    plot, mesh, actor, weights, dominant, report = scene()
    if args.smoke:
        plot.screenshot("/tmp/transolver-preview.png")
        Path("/tmp/transolver-preview.json").write_text(json.dumps(report, indent=2))
        return
    server = get_server(client_type="vue3")
    st = server.state
    st.update(
        head=0,
        slice_id=0,
        mode="dominant",
        occupied=report["dominant_slices_per_head"][0],
    )

    def update(**_):
        h = int(st.head)
        s = int(st.slice_id)
        if st.mode == "dominant":
            values = dominant[h]
            limits = (0, 63)
            title = f"Dominant slice ID (head {h})"
        else:
            values = weights[0, h, :, s].cpu().numpy()
            limits = (0, float(weights[0, h, :, s].max().item()))
            title = f"Slice {s} assignment weight"
        mesh["slice"] = values
        actor.mapper.scalar_range = limits
        plot.update_scalar_bar_range(limits)
        plot.scalar_bar.SetTitle(title)
        st.occupied = report["dominant_slices_per_head"][h]
        if server.controller.update.exists():
            server.controller.update()

    st.change("head", "slice_id", "mode")(update)

    def fit():
        plot.reset_camera()
        server.controller.update()

    with DivLayout(server):
        with html.Div(
            style="display:flex;height:100vh;background:#0b1724;color:#dfedf5;font-family:Arial"
        ):
            with html.Div(style="width:290px;padding:24px;flex-shrink:0;overflow:auto"):
                html.P("P2I LAB / TRANSOLVER")
                html.H1("How attention groups a wing")
                html.P(
                    "NACA 0012 · 500,000 surface points. Initial, untrained slice assignments—not a CFD result."
                )
                html.Label("Attention head")
                with html.Select(
                    v_model=("head",), style="display:block;margin:12px 0"
                ):
                    for i in range(8):
                        html.Option(str(i), value=i)
                with html.Select(
                    v_model=("mode",), style="display:block;margin:12px 0"
                ):
                    html.Option("Dominant slice", value="dominant")
                    html.Option("Soft assignment weight", value="weight")
                html.Label("Slice {{ slice_id }}")
                html.Input(
                    type="range",
                    min=0,
                    max=63,
                    step=1,
                    v_model=("slice_id",),
                    style="width:100%",
                )
                html.P(
                    "{{ occupied }} slices win at least one point in this head. All 64 participate through soft weights."
                )
                html.Button("Fit wing", click=fit)
                html.P("Drag to orbit; scroll to zoom. NVIDIA EGL remote rendering.")
                html.A(
                    "Transolver · Wu et al. · ICML 2024",
                    href="https://arxiv.org/abs/2402.02366",
                    target="_blank",
                    style="color:#57d7ca",
                )
            with html.Div(style="flex:1;min-width:0"):
                view = vtk.VtkRemoteView(
                    plot.render_window, interactive_ratio=1, still_ratio=1
                )
                server.controller.update = view.update
    server.start(host="0.0.0.0", port=8000, open_browser=False, timeout=0)


if __name__ == "__main__":
    main()
