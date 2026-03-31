"""
Interactive Hypersphere Visualization — served on a port.
Usage: uv run hypersphere_server.py [port]
"""

import sys
import numpy as np
import dash
from dash import html, dcc
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def sample_hypersphere(n_dim: int, n_points: int, radius: float = 1.0) -> np.ndarray:
    points = np.random.randn(n_points, n_dim)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points * radius


def stereographic_projection(points: np.ndarray) -> np.ndarray:
    projected = np.zeros((points.shape[0], points.shape[1] - 1))
    for i, p in enumerate(points):
        denom = 1.0 - p[-1]
        if abs(denom) < 1e-10:
            projected[i] = np.sign(p[:-1]) * 10.0
        else:
            projected[i] = p[:-1] / denom
    return projected


def parametric_torus_knot(n_points: int = 2000) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n_points)
    p, q = 2, 3
    eta = np.pi / 2 * np.ones_like(t)
    x = np.cos(eta / 2) * np.cos(p * t)
    y = np.cos(eta / 2) * np.sin(p * t)
    z = np.sin(eta / 2) * np.cos(q * t)
    w = np.sin(eta / 2) * np.sin(q * t)
    return np.column_stack([x, y, z, w])


def build_stereographic_trace(n_points=10000):
    pts4d = sample_hypersphere(4, n_points)
    proj = stereographic_projection(pts4d)
    mask = np.all(np.abs(proj) < 5, axis=1)
    proj = proj[mask]
    colors = np.linalg.norm(proj, axis=1)
    return go.Scatter3d(
        x=proj[:, 0], y=proj[:, 1], z=proj[:, 2],
        mode="markers", marker=dict(size=1, color=colors, colorscale="Plasma", opacity=0.5),
        name="Stereographic",
    )


def build_slices_trace(n_points=5000):
    traces = []
    w_values = np.linspace(-0.9, 0.9, 12)
    for w in w_values:
        r = np.sqrt(1.0 - w ** 2)
        pts = np.random.randn(n_points // 12, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        pts *= r
        color_val = 0.5 + 0.5 * w / 0.9
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers", marker=dict(size=1, opacity=0.25,
                                        color=[[0, color_val, 1 - color_val]]),
            name=f"w={w:.2f}", showlegend=False,
        ))
    return traces


def build_hopf_trace(n_fibers=60, n_pts=150):
    traces = []
    t = np.linspace(0, 2 * np.pi, n_pts)
    for i in range(n_fibers):
        phi = np.arccos(1 - 2 * (i + 0.5) / n_fibers)
        theta = np.pi * (1 + 5 ** 0.5) * i
        cp, sp = np.cos(phi / 2), np.sin(phi / 2)
        fiber4d = np.column_stack([
            cp * np.cos(t), cp * np.sin(t),
            sp * np.cos(t + theta), sp * np.sin(t + theta),
        ])
        fiber3d = stereographic_projection(fiber4d)
        mask = np.all(np.abs(fiber3d) < 8, axis=1)
        fiber3d = fiber3d[mask]
        if len(fiber3d) > 1:
            traces.append(go.Scatter3d(
                x=fiber3d[:, 0], y=fiber3d[:, 1], z=fiber3d[:, 2],
                mode="lines", line=dict(width=1.5,
                                        color=f"hsl({int(360 * i / n_fibers)}, 80%, 55%)"),
                showlegend=False,
            ))
    return traces


def build_torus_knot_trace():
    knot4d = parametric_torus_knot(3000)
    knot3d = stereographic_projection(knot4d)
    t = np.linspace(0, 2 * np.pi, len(knot3d))
    return go.Scatter3d(
        x=knot3d[:, 0], y=knot3d[:, 1], z=knot3d[:, 2],
        mode="markers", marker=dict(size=2, color=t, colorscale="Twilight", opacity=0.8),
        name="Torus Knot (2,3)",
    )


def build_cross_section_trace():
    """2D cross-sections at constant w — shown as concentric circles."""
    traces = []
    w_values = np.linspace(-0.95, 0.95, 15)
    for w in w_values:
        r = np.sqrt(1.0 - w ** 2)
        theta = np.linspace(0, 2 * np.pi, 200)
        traces.append(go.Scatter(
            x=r * np.cos(theta), y=r * np.sin(theta),
            mode="lines", line=dict(width=1.5),
            showlegend=False,
        ))
    return traces


def build_great_circles_trace():
    axes_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    labels = ["xy", "xz", "xw", "yz", "yw", "zw"]
    traces = []
    t = np.linspace(0, 2 * np.pi, 300)
    for (a, b), label in zip(axes_pairs, labels):
        traces.append(go.Scatter(
            x=np.cos(t), y=np.sin(t),
            mode="lines", line=dict(width=2), name=label,
        ))
    return traces


def make_figure():
    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
            [{"type": "scene"}, {"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=[
            "Stereographic Projection S³→R³",
            "S³ Slices at Constant w",
            "Hopf Fibration",
            "Torus Knot (2,3) on S³",
            "Great Circles (2D Projections)",
            "Cross-Sections at Constant w",
        ],
    )

    # Panel 1
    fig.add_trace(build_stereographic_trace(), row=1, col=1)

    # Panel 2
    for tr in build_slices_trace():
        fig.add_trace(tr, row=1, col=2)

    # Panel 3
    for tr in build_hopf_trace():
        fig.add_trace(tr, row=1, col=3)

    # Panel 4
    fig.add_trace(build_torus_knot_trace(), row=2, col=1)

    # Panel 5 — great circles
    for tr in build_great_circles_trace():
        fig.add_trace(tr, row=2, col=2)

    # Panel 6 — cross sections
    for tr in build_cross_section_trace():
        fig.add_trace(tr, row=2, col=3)

    fig.update_layout(
        height=900, width=1400,
        title_text="Interactive Hypersphere (S³) Visualization",
        title_font_size=20,
        showlegend=False,
        paper_bgcolor="#0e0e0e",
        font_color="white",
    )

    scene_cfg = dict(
        bgcolor="#0e0e0e",
        xaxis=dict(showbackground=False, showgrid=True, gridcolor="#333"),
        yaxis=dict(showbackground=False, showgrid=True, gridcolor="#333"),
        zaxis=dict(showbackground=False, showgrid=True, gridcolor="#333"),
    )
    for s in ["scene", "scene2", "scene3", "scene4"]:
        fig.layout[s].update(**scene_cfg)
    fig.layout["xaxis2"].update(showgrid=True, gridcolor="#333", zeroline=False)
    fig.layout["yaxis2"].update(showgrid=True, gridcolor="#333", zeroline=False, scaleanchor="x2", scaleratio=1)
    fig.layout["xaxis3"].update(showgrid=True, gridcolor="#333", zeroline=False)
    fig.layout["yaxis3"].update(showgrid=True, gridcolor="#333", zeroline=False, scaleanchor="x3", scaleratio=1)

    return fig


app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id="hypersphere", figure=make_figure(), style={"height": "95vh"}),
], style={"backgroundColor": "#0e0e0e", "height": "100vh", "padding": 0, "margin": 0})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8050
    print(f"\n  ➜  Hypersphere viz: http://0.0.0.0:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
