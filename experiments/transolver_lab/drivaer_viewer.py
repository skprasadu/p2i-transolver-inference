"""Protected remote EGL viewer for full-resolution, cell-associated CFD comparison."""

import os

os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkEGLRenderWindow"
import argparse
import json
from pathlib import Path
import numpy as np
import pyvista as pv
from trame.app import get_server
from trame.ui.html import DivLayout
from trame.widgets import html, vtk

FIELDS = [
    "Pred_Pressure",
    "GT_Pressure",
    "Abs_Error",
    "Wall_Shear_Stress",
    "Physical_Slice_ID",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    metrics = json.loads(Path("/data/validation.json").read_text())
    mesh = pv.read("/data/drivaer_validated.vtp")
    pv.global_theme.font.color = "#e0edf5"
    plot = pv.Plotter(off_screen=True, window_size=(1500, 950))
    plot.set_background("#0b1724", top="#254359")
    pressure_limits = (
        min(mesh["Pred_Pressure"].min(), mesh["GT_Pressure"].min()),
        max(mesh["Pred_Pressure"].max(), mesh["GT_Pressure"].max()),
    )
    actor = plot.add_mesh(
        mesh,
        scalars="Pred_Pressure",
        preference="cell",
        cmap="coolwarm",
        clim=pressure_limits,
        show_edges=False,
        scalar_bar_args={"title": "Pressure · native units", "color": "white"},
    )
    plot.camera_position = [(9, 7, 4), (1.5, 0, 0.3), (0, 0, 1)]
    plot.reset_camera()
    plot.show(auto_close=False, interactive=False)
    caps = plot.render_window.ReportCapabilities()
    assert (
        plot.render_window.GetClassName() == "vtkEGLRenderWindow" and "NVIDIA" in caps
    )

    def select(field):
        mesh.set_active_scalars(field, preference="cell")
        actor.mapper.array_name = field
        actor.mapper.scalar_map_mode = "cell"
        limits = (
            pressure_limits
            if field in FIELDS[:2]
            else (float(mesh[field].min()), float(mesh[field].max()))
        )
        if field == "Physical_Slice_ID":
            limits = (0, 511)
        actor.mapper.scalar_range = limits
        actor.mapper.lookup_table.cmap = (
            "turbo"
            if field == "Physical_Slice_ID"
            else ("coolwarm" if field in FIELDS[:2] else "inferno")
        )
        plot.update_scalar_bar_range(limits)
        plot.scalar_bar.SetTitle(
            field + (" · ID" if field == "Physical_Slice_ID" else " · native units")
        )
        plot.render()

    if args.smoke:
        for field in FIELDS:
            select(field)
            assert np.isfinite(mesh[field]).all()
        select("Pred_Pressure")
        plot.screenshot("/tmp/drivaer-view.png")
        Path("/tmp/drivaer-view.json").write_text(
            json.dumps(
                {
                    "renderer": plot.render_window.GetClassName(),
                    "NVIDIA": True,
                    "vertices": mesh.n_points,
                    "cells": mesh.n_cells,
                    "fields_checked": FIELDS,
                    "geometry_decimation": False,
                    "pixel_ratios": [1, 1],
                },
                indent=2,
            )
        )
        return
    server = get_server(client_type="vue3")
    state = server.state
    state.field = FIELDS[0]

    @state.change("field")
    def update(field, **_):
        select(field)
        if server.controller.update.exists():
            server.controller.update()

    def fit():
        plot.reset_camera()
        server.controller.update()

    with DivLayout(server):
        with html.Div(
            style="display:flex;height:100vh;background:#0b1724;color:#e0edf5;font-family:Arial"
        ):
            with html.Div(style="width:285px;padding:24px;flex-shrink:0;overflow:auto"):
                html.P("P2I LAB / REAL CFD COMPARISON")
                html.H1("DrivAer · predicted vs CFD")
                html.P(
                    f"Run 4 · {mesh.n_cells:,} surface cells. Official NVIDIA pretrained Transolver."
                )
                with html.Select(v_model=("field",), style="width:100%;padding:10px"):
                    for field in FIELDS:
                        html.Option(field.replace("_", " "), value=field)
                html.P(
                    f"Pressure relative L2: {metrics['pressure']['relative_l2']:.2%}"
                )
                html.P(
                    f"Wall-shear vector relative L2: {metrics['wall_shear_vector']['relative_l2']:.2%}"
                )
                html.P(
                    "Pressure views share the same color scale. Error is the absolute pressure difference."
                )
                html.P(
                    "Wall shear shows predicted vector magnitude. Slice IDs show learned grouping: final layer, head 0—not physical regions."
                )
                html.P(
                    "Native CFD units: m²/s². Saved neural predictions compared with reference CFD; this is not a live CFD solve."
                )
                html.Button("Fit car", click=fit)
                html.P("Drag to orbit · scroll to zoom")
                html.A(
                    "Model & attribution",
                    href="https://huggingface.co/nvidia/transolver_drivaerml",
                    target="_blank",
                    style="color:#50dbc7",
                )
                html.Br()
                html.A(
                    "DrivAerML · Ashton et al.",
                    href="https://huggingface.co/datasets/neashton/drivaerml",
                    target="_blank",
                    style="color:#50dbc7",
                )
            with html.Div(style="flex:1;min-width:0"):
                view = vtk.VtkRemoteView(
                    plot.render_window, interactive_ratio=1, still_ratio=1
                )
                server.controller.update = view.update
    server.start(host="0.0.0.0", port=8000, open_browser=False, timeout=0)


if __name__ == "__main__":
    main()
