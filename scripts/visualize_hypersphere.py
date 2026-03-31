"""
Hypersphere Visualization
=========================
Visualizes an n-dimensional hypersphere using multiple projection techniques:
  1. Stereographic projection from 4D -> 3D
  2. Parallel coordinate slices (great circles)
  3. 3D scatter of uniformly sampled surface points (projected)

Run: uv run visualize_hypersphere.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial import ConvexHull


def sample_hypersphere(n_dim: int, n_points: int, radius: float = 1.0) -> np.ndarray:
    """Uniformly sample points on the surface of an n-sphere."""
    # Sample from standard normal, then normalize to the sphere surface
    points = np.random.randn(n_points, n_dim)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points * radius


def stereographic_projection(points: np.ndarray) -> np.ndarray:
    """
    Stereographic projection from S^n to R^n.
    Projects from the 'north pole' (0,...,0,1) onto the hyperplane x_{n+1} = 0.
    """
    n_dim = points.shape[1]
    north_pole = np.zeros(n_dim)
    north_pole[-1] = 1.0

    projected = np.zeros((points.shape[0], n_dim - 1))
    for i, p in enumerate(points):
        denom = 1.0 - p[-1]
        if abs(denom) < 1e-10:
            # Point is at/near the north pole — project to infinity, skip or clamp
            projected[i] = np.sign(p[:-1]) * 10.0
        else:
            projected[i] = p[:-1] / denom
    return projected


def parametric_torus_knot(n_points: int = 2000) -> np.ndarray:
    """
    Generate a torus knot on S^3 using Hopf coordinates.
    Returns points on the 3-sphere in R^4.
    """
    t = np.linspace(0, 2 * np.pi, n_points)
    p, q = 2, 3  # torus knot parameters

    eta = np.pi * np.ones_like(t) / 2  # on the equator of S^3
    xi1 = p * t
    xi2 = q * t

    # Hopf coordinate -> Cartesian in R^4
    x = np.cos(eta / 2) * np.cos(xi1)
    y = np.cos(eta / 2) * np.sin(xi1)
    z = np.sin(eta / 2) * np.cos(xi2)
    w = np.sin(eta / 2) * np.sin(xi2)

    return np.column_stack([x, y, z, w])


def plot_stereographic(ax, points_4d: np.ndarray, title: str = "Stereographic Projection S³ → R³"):
    """Plot stereographic projection of 4D sphere points as 3D scatter."""
    proj = stereographic_projection(points_4d)
    # Clamp extreme values (near north pole)
    mask = np.all(np.abs(proj) < 5, axis=1)
    proj = proj[mask]

    colors = np.linalg.norm(proj, axis=1)
    sc = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
                    c=colors, cmap="plasma", s=1, alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title, fontsize=10)
    return sc


def plot_rotating_slices(ax, n_points: int = 5000):
    """
    Visualize S³ by showing several great-circle slices at different 'w' values,
    projected into xyz. Each slice is a 2-sphere of radius sqrt(1 - w²).
    """
    cmap = plt.cm.cool
    w_values = np.linspace(-0.9, 0.9, 12)

    for w in w_values:
        r = np.sqrt(1.0 - w ** 2)
        # Sample a 2-sphere of radius r
        pts = np.random.randn(n_points // len(w_values), 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        pts *= r

        color = cmap(0.5 + 0.5 * w / 0.9)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=[color], s=0.5, alpha=0.25)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("S³ Slices at Constant w", fontsize=10)


def plot_great_circles_2d(ax):
    """
    Plot pairwise great circles of S³ projected onto 2D planes.
    Shows the 6 pairwise projections (xy, xz, xw, yz, yw, zw).
    """
    t = np.linspace(0, 2 * np.pi, 300)
    # Unit circle in each coordinate pair
    pairs = [("x", "y"), ("x", "z"), ("x", "w"), ("y", "z"), ("y", "w"), ("z", "w")]
    for i, (a, b) in enumerate(pairs):
        theta = np.pi * i / 6
        # Rotate the circle in 4D
        p = np.zeros((len(t), 4))
        p[:, i % 4] = np.cos(t)
        p[:, (i + 1) % 4] = np.sin(t)
        ax.plot(p[:, 0], p[:, 1], p[:, 2], alpha=0.3)  # placeholder

    # Simpler: just plot great circles in 6 2D planes
    ax.clear()
    t = np.linspace(0, 2 * np.pi, 300)

    colors = plt.cm.Set2(np.linspace(0, 1, 6))
    labels = ["xy", "xz", "xw", "yz", "yw", "zw"]
    axes_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    for idx, ((a, b), label) in enumerate(zip(axes_pairs, labels)):
        circle = np.zeros((len(t), 4))
        circle[:, a] = np.cos(t)
        circle[:, b] = np.sin(t)
        ax.plot(circle[:, a], circle[:, b], color=colors[idx], label=label, linewidth=1.5)

    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("Great Circles of S³ (2D Projections)", fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_hopf_fibration(ax, n_fibers: int = 40, n_points_per_fiber: int = 100):
    """
    Visualize the Hopf fibration: each fiber is a great circle on S³.
    We stereographically project the fibers to R³.
    """
    cmap = plt.cm.hsv
    for i in range(n_fibers):
        # Parametrize a point on S² (base space)
        phi = np.arccos(1 - 2 * (i + 0.5) / n_fibers)
        theta = np.pi * (1 + 5 ** 0.5) * i  # golden angle spacing

        # For each base point, generate the fiber (circle) on S³
        t = np.linspace(0, 2 * np.pi, n_points_per_fiber)

        # Hopf fiber coordinates
        cos_p = np.cos(phi / 2)
        sin_p = np.sin(phi / 2)

        x = cos_p * np.cos(t)
        y = cos_p * np.sin(t)
        z = sin_p * np.cos(t + theta)
        w = sin_p * np.sin(t + theta)

        fiber_4d = np.column_stack([x, y, z, w])
        fiber_3d = stereographic_projection(fiber_4d)

        # Clamp
        mask = np.all(np.abs(fiber_3d) < 8, axis=1)
        fiber_3d = fiber_3d[mask]

        if len(fiber_3d) > 1:
            color = cmap(i / n_fibers)
            ax.plot(fiber_3d[:, 0], fiber_3d[:, 1], fiber_3d[:, 2],
                    color=color, alpha=0.5, linewidth=0.5)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Hopf Fibration (Stereographic)", fontsize=10)


def main():
    np.random.seed(42)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Hypersphere (S³) Visualization in 4D", fontsize=16, fontweight="bold")
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # --- 1. Stereographic projection of random S³ points ---
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    points_4d = sample_hypersphere(n_dim=4, n_points=10000)
    plot_stereographic(ax1, points_4d)

    # --- 2. Slices of S³ at constant w ---
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    plot_rotating_slices(ax2)

    # --- 3. Hopf fibration ---
    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    plot_hopf_fibration(ax3)

    # --- 4. Torus knot on S³, stereographically projected ---
    ax4 = fig.add_subplot(gs[1, 0], projection="3d")
    knot_4d = parametric_torus_knot(3000)
    knot_3d = stereographic_projection(knot_4d)
    t_param = np.linspace(0, 2 * np.pi, len(knot_3d))
    ax4.scatter(knot_3d[:, 0], knot_3d[:, 1], knot_3d[:, 2],
                c=t_param, cmap="twilight", s=2, alpha=0.7)
    ax4.set_xlabel("x"); ax4.set_ylabel("y"); ax4.set_zlabel("z")
    ax4.set_title("Torus Knot (2,3) on S³", fontsize=10)

    # --- 5. Great circles 2D projections ---
    ax5 = fig.add_subplot(gs[1, 1])
    plot_great_circles_2d(ax5)

    # --- 6. Cross-sections: 2D slices at different w values ---
    ax6 = fig.add_subplot(gs[1, 2])
    cmap = plt.cm.viridis
    w_values = np.linspace(-0.95, 0.95, 15)
    for w in w_values:
        r = np.sqrt(1.0 - w ** 2)
        circle = plt.Circle((0, 0), r, fill=False,
                            color=cmap(0.5 + 0.5 * w / 0.95), linewidth=1, alpha=0.7)
        ax6.add_patch(circle)
    ax6.set_xlim(-1.2, 1.2)
    ax6.set_ylim(-1.2, 1.2)
    ax6.set_aspect("equal")
    ax6.set_title("S³ Cross-Sections (xy-plane at constant w)", fontsize=10)
    ax6.set_xlabel("x")
    ax6.set_ylabel("y")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hypersphere_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: hypersphere_visualization.png")


if __name__ == "__main__":
    main()
