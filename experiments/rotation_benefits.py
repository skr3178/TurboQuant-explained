"""
Rotation Benefits: 4 empirical tests proving random rotation enables scalar quantization.

Proves the geometric trick: random rotation Π makes any worst-case vector look the same
to a scalar quantizer — coordinates become near-independent, near-Gaussian.

Tests:
  1. Energy spreading — spiky inputs flatten after rotation
  2. Distribution fit — rotated coordinates match Beta marginal / Gaussian
  3. Near-independence — correlation matrix becomes identity after rotation
  4. Quantization MSE payoff — rotation closes the gap between worst/best-case inputs

All on synthetic adversarial inputs at d=128 (also d=8, 32, 512 for dimension scaling).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from scipy.special import gammaln

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_input_vectors(d: int, device: str = "cuda") -> dict[str, torch.Tensor]:
    """Generate 5 adversarial input types, all L2-normalized to unit sphere."""
    vectors = {}

    # 1. One-hot: all energy in coordinate 0
    v = torch.zeros(d, device=device)
    v[0] = 1.0
    vectors["one-hot"] = v

    # 2. Sparse: heavy tail on first few coords
    v = torch.zeros(d, device=device)
    v[0], v[1], v[2] = 0.9, 0.3, 0.1
    v /= v.norm()
    vectors["sparse"] = v

    # 3. Smooth: all equal (constant vector)
    v = torch.ones(d, device=device)
    v /= v.norm()
    vectors["smooth"] = v

    # 4. Decaying: exponentially decreasing
    v = torch.tensor([0.7**i for i in range(d)], device=device, dtype=torch.float32)
    v /= v.norm()
    vectors["decaying"] = v

    # 5. Random Gaussian
    torch.manual_seed(42)
    v = torch.randn(d, device=device)
    v /= v.norm()
    vectors["random"] = v

    return vectors


def make_rotation(d: int, seed: int, device: str = "cuda") -> torch.Tensor:
    """Random orthogonal matrix via QR (no caching — we need many distinct ones)."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    M = torch.randn(d, d, generator=generator, dtype=torch.float32)
    Q, R = torch.linalg.qr(M)
    signs = torch.sign(torch.diag(R))
    Q = Q * signs.unsqueeze(0)
    return Q.to(device)


def beta_marginal_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """Theoretical beta marginal PDF for coordinate on d-dimensional sphere."""
    log_coeff = gammaln(d / 2) - math.log(math.pi) / 2 - gammaln((d - 1) / 2)
    coeff = math.exp(log_coeff)
    return coeff * (1 - x**2) ** ((d - 3) / 2)


def uniform_quantize(x: torch.Tensor, b: int) -> torch.Tensor:
    """Uniform scalar quantizer: divide [-1, 1] into 2^b equal buckets."""
    n_levels = 2**b
    step = 2.0 / n_levels
    # Centroids at midpoints of each bucket
    centroids = torch.linspace(
        -1 + step / 2, 1 - step / 2, n_levels, device=x.device, dtype=x.dtype
    )
    # Find nearest centroid
    idx = torch.argmin(torch.abs(x.unsqueeze(-1) - centroids), dim=-1)
    return centroids[idx]


def dark_theme(ax):
    """Apply dark theme to an axes."""
    fig = ax.get_figure()
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#0e0e0e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#333")


# ---------------------------------------------------------------------------
# Test 1: Energy Spreading
# ---------------------------------------------------------------------------


def test1_energy_spreading(
    d: int = 128, seed: int = 0, results_dir: str = "experiments/results"
):
    """Show coordinate magnitudes flatten after rotation for spiky inputs."""
    print("\n" + "=" * 60)
    print("TEST 1: Energy Spreading (d={})".format(d))
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = make_input_vectors(d, device)
    Pi = make_rotation(d, seed, device)

    input_names = ["one-hot", "sparse", "decaying", "random", "smooth"]
    colors = ["#f87171", "#fb923c", "#a78bfa", "#60a5fa", "#6ee7b7"]

    fig, axes = plt.subplots(2, len(input_names), figsize=(20, 8))
    fig.suptitle(
        f"Test 1: Energy Spreading Before vs After Rotation (d={d})",
        fontsize=15, fontweight="bold", color="white",
    )

    cv_results = {}

    for col, name in enumerate(input_names):
        x = inputs[name]
        y = Pi @ x  # rotated

        x_abs = x.abs().cpu().numpy()
        y_abs = y.abs().cpu().numpy()

        # Coefficient of variation (std/mean of |coord|)
        cv_before = x_abs.std() / (x_abs.mean() + 1e-10)
        cv_after = y_abs.std() / (y_abs.mean() + 1e-10)
        cv_results[name] = (cv_before, cv_after)

        # Top row: before rotation
        ax = axes[0, col]
        ax.bar(range(d), x_abs, color=colors[col], alpha=0.7, width=1.0)
        ax.set_title(f"{name}\n(before)", fontsize=11, color="white")
        ax.set_ylabel("|coord|" if col == 0 else "", color="white")
        ax.set_ylim(0, max(x_abs.max(), y_abs.max()) * 1.1)
        dark_theme(ax)

        # Bottom row: after rotation
        ax = axes[1, col]
        ax.bar(range(d), y_abs, color=colors[col], alpha=0.7, width=1.0)
        ax.set_title(f"(after rotation)\nCV: {cv_before:.3f} → {cv_after:.3f}", fontsize=10, color="white")
        ax.set_ylabel("|coord|" if col == 0 else "", color="white")
        ax.set_ylim(0, max(x_abs.max(), y_abs.max()) * 1.1)
        dark_theme(ax)

    plt.tight_layout()
    path = os.path.join(results_dir, "test1_energy_spreading.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    print("\nCoefficient of Variation (std/mean of |coord|):")
    print(f"  {'Input':<12} {'Before':>8} {'After':>8} {'Reduction':>10}")
    for name in input_names:
        before, after = cv_results[name]
        reduction = (1 - after / before) * 100 if before > 0 else 0
        print(f"  {name:<12} {before:>8.4f} {after:>8.4f} {reduction:>9.1f}%")


# ---------------------------------------------------------------------------
# Test 2: Distribution Fit
# ---------------------------------------------------------------------------


def test2_distribution_fit(
    d_values: list[int] = [4, 16, 64, 128],
    n_rotations: int = 10_000,
    results_dir: str = "experiments/results",
):
    """Show rotated coordinate matches Beta marginal / Gaussian regardless of input."""
    print("\n" + "=" * 60)
    print("TEST 2: Distribution Fit")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_names = ["one-hot", "sparse", "smooth", "decaying", "random"]
    input_colors = {
        "one-hot": "#f87171",
        "sparse": "#fb923c",
        "smooth": "#6ee7b7",
        "decaying": "#a78bfa",
        "random": "#60a5fa",
    }

    fig, axes = plt.subplots(len(d_values), 1, figsize=(12, 4 * len(d_values)))
    if len(d_values) == 1:
        axes = [axes]
    fig.suptitle(
        "Test 2: Distribution of Rotated Coordinate vs Theoretical PDF",
        fontsize=15, fontweight="bold", color="white",
    )

    for row, d in enumerate(d_values):
        ax = axes[row]
        inputs = make_input_vectors(d, device)

        xs = np.linspace(-1, 1, 500)
        # Beta marginal
        beta_pdf = beta_marginal_pdf(xs, d)
        # Gaussian approximation N(0, 1/d)
        gauss_pdf = stats.norm.pdf(xs, 0, 1 / math.sqrt(d))

        ax.plot(xs, beta_pdf, "w-", linewidth=2.5, label="Beta marginal", zorder=5)
        ax.plot(
            xs, gauss_pdf, "w--", linewidth=1.5, alpha=0.7,
            label=f"Gaussian N(0, 1/d)", zorder=4,
        )

        for name in input_names:
            # Collect first coordinate from n_rotations different rotations
            coords = []
            for i in range(n_rotations):
                Pi = make_rotation(d, seed=i, device=device)
                y = Pi @ inputs[name]
                coords.append(y[0].item())
            coords = np.array(coords)

            ax.hist(
                coords, bins=150, density=True, alpha=0.3,
                color=input_colors[name], label=name, histtype="stepfilled",
            )

        ax.set_title(f"d = {d}", fontsize=13, color="white")
        ax.set_xlabel("Coordinate value", color="white")
        ax.set_ylabel("Density", color="white")
        ax.legend(fontsize=9, facecolor="#1a1a1a", edgecolor="#333", labelcolor="white", ncol=2)
        dark_theme(ax)

    plt.tight_layout()
    path = os.path.join(results_dir, "test2_distribution_fit.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # KS test against Gaussian
    print("\nKolmogorov-Smirnov test against N(0, 1/d):")
    print(f"  {'d':>5} {'Input':<12} {'KS stat':>10} {'p-value':>10} {'Gaussian?':>10}")
    for d in d_values:
        inputs = make_input_vectors(d, device)
        for name in input_names:
            coords = []
            for i in range(n_rotations):
                Pi = make_rotation(d, seed=i, device=device)
                y = Pi @ inputs[name]
                coords.append(y[0].item())
            coords = np.array(coords)
            ks_stat, p_val = stats.kstest(coords, "norm", args=(0, 1 / math.sqrt(d)))
            is_gaussian = "YES" if p_val > 0.05 else "no"
            print(f"  {d:>5} {name:<12} {ks_stat:>10.6f} {p_val:>10.6f} {is_gaussian:>10}")


# ---------------------------------------------------------------------------
# Test 3: Near-Independence
# ---------------------------------------------------------------------------


def test3_near_independence(
    d: int = 128,
    n_rotations: int = 5_000,
    results_dir: str = "experiments/results",
):
    """Show correlation matrix becomes identity after rotation."""
    print("\n" + "=" * 60)
    print("TEST 3: Near-Independence (d={})".format(d))
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_names = ["one-hot", "sparse", "smooth", "decaying", "random"]

    fig, axes = plt.subplots(2, len(input_names), figsize=(20, 8))
    fig.suptitle(
        f"Test 3: Correlation Matrix Before vs After Rotation (d={d})",
        fontsize=15, fontweight="bold", color="white",
    )

    for col, name in enumerate(input_names):
        inputs = make_input_vectors(d, device)
        x = inputs[name]

        # Collect rotated vectors across many rotations
        rotated = []
        for i in range(n_rotations):
            Pi = make_rotation(d, seed=i, device=device)
            y = (Pi @ x).cpu().numpy()
            rotated.append(y)
        rotated = np.array(rotated)  # (n_rotations, d)

        # Before rotation: trivially, single vector has no correlation
        # Show the |coord| pattern as a proxy for "energy structure"
        ax = axes[0, col]
        x_np = x.cpu().numpy()
        # Outer product shows energy concentration
        energy = np.outer(x_np, x_np)
        im = ax.imshow(energy[:32, :32], cmap="inferno", aspect="auto")
        ax.set_title(f"{name} (before)\nx·xᵀ (first 32×32)", fontsize=10, color="white")
        dark_theme(ax)

        # After rotation: correlation matrix
        ax = axes[1, col]
        corr_matrix = np.corrcoef(rotated.T)  # (d, d)
        im = ax.imshow(corr_matrix[:32, :32], cmap="inferno", aspect="auto", vmin=-1, vmax=1)
        max_off_diag = np.max(np.abs(corr_matrix - np.eye(d)))
        ax.set_title(
            f"(after rotation)\nmax|off-diag| = {max_off_diag:.4f}",
            fontsize=10, color="white",
        )
        dark_theme(ax)

    plt.tight_layout()
    path = os.path.join(results_dir, "test3_near_independence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # Dimension scaling of off-diagonal correlation
    print("\nMax off-diagonal correlation vs dimension:")
    print(f"  {'d':>5} {'Input':<12} {'max|off-diag|':>15}")
    d_scaling = [8, 16, 32, 64, 128, 256, 512]
    scaling_results = {}
    for d_test in d_scaling:
        inputs = make_input_vectors(d_test, device)
        for name in ["one-hot"]:  # Just worst case
            rotated = []
            for i in range(min(n_rotations, 2000)):
                Pi = make_rotation(d_test, seed=i, device=device)
                y = (Pi @ inputs[name]).cpu().numpy()
                rotated.append(y)
            rotated = np.array(rotated)
            corr = np.corrcoef(rotated.T)
            max_off = np.max(np.abs(corr - np.eye(d_test)))
            scaling_results[d_test] = max_off
            print(f"  {d_test:>5} {name:<12} {max_off:>15.6f}")

    # Plot scaling
    fig, ax = plt.subplots(figsize=(8, 5))
    ds = list(scaling_results.keys())
    offs = list(scaling_results.values())
    ax.loglog(ds, offs, "o-", color="#6ee7b7", linewidth=2, markersize=8, label="Empirical max|off-diag|")
    ax.loglog(ds, [1.0 / d for d in ds], "s--", color="#f87171", linewidth=1.5, label="O(1/d) reference")
    ax.set_xlabel("Dimension d", fontsize=13)
    ax.set_ylabel("Max off-diagonal |correlation|", fontsize=13)
    ax.set_title("Test 3b: Off-diagonal Correlation Scales as O(1/d)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    dark_theme(ax)
    plt.tight_layout()
    path = os.path.join(results_dir, "test3b_correlation_scaling.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Test 4: Quantization MSE Payoff
# ---------------------------------------------------------------------------


def test4_quantization_mse(
    d_values: list[int] = [8, 32, 128, 512],
    bit_widths: list[int] = [1, 2, 3, 4],
    n_rotations: int = 50,
    results_dir: str = "experiments/results",
):
    """Show rotation dramatically reduces MSE for adversarial inputs."""
    print("\n" + "=" * 60)
    print("TEST 4: Quantization MSE Payoff")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_names = ["one-hot", "sparse", "smooth", "decaying", "random"]
    input_colors = {
        "one-hot": "#f87171",
        "sparse": "#fb923c",
        "smooth": "#6ee7b7",
        "decaying": "#a78bfa",
        "random": "#60a5fa",
    }

    d_primary = 128
    inputs = make_input_vectors(d_primary, device)

    # --- Panel 1: MSE vs bit-width for d=128 ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Test 4: Quantization MSE With vs Without Rotation (d={d_primary})",
        fontsize=14, fontweight="bold", color="white",
    )

    # Without rotation
    ax = axes[0]
    ax.set_title("Without Rotation (naive quantize)", fontsize=12, color="white")
    for name in input_names:
        x = inputs[name]
        mses = []
        for b in bit_widths:
            x_q = uniform_quantize(x.unsqueeze(0), b).squeeze(0)
            mse = ((x - x_q) ** 2).sum().item()
            mses.append(mse)
        ax.semilogy(bit_widths, mses, "o-", color=input_colors[name],
                     linewidth=2, markersize=8, label=name)
    ax.set_xlabel("Bit-width b", fontsize=13)
    ax.set_ylabel("MSE (||x - x̃||²)", fontsize=13)
    ax.set_xticks(bit_widths)
    ax.legend(fontsize=10, facecolor="#1a1a1a", edgecolor="#333", labelcolor="white")
    ax.grid(True, alpha=0.3)
    dark_theme(ax)

    # With rotation (average over n_rotations)
    ax = axes[1]
    ax.set_title("With Random Rotation", fontsize=12, color="white")
    for name in input_names:
        x = inputs[name]
        mses = []
        for b in bit_widths:
            total_mse = 0.0
            for i in range(n_rotations):
                Pi = make_rotation(d_primary, seed=i, device=device)
                y = Pi @ x
                y_q = uniform_quantize(y.unsqueeze(0), b).squeeze(0)
                x_hat = Pi.T @ y_q
                total_mse += ((x - x_hat) ** 2).sum().item()
            mses.append(total_mse / n_rotations)
        ax.semilogy(bit_widths, mses, "o-", color=input_colors[name],
                     linewidth=2, markersize=8, label=name)
    ax.set_xlabel("Bit-width b", fontsize=13)
    ax.set_ylabel("MSE (||x - x̃||²)", fontsize=13)
    ax.set_xticks(bit_widths)
    ax.legend(fontsize=10, facecolor="#1a1a1a", edgecolor="#333", labelcolor="white")
    ax.grid(True, alpha=0.3)
    dark_theme(ax)

    plt.tight_layout()
    path = os.path.join(results_dir, "test4_mse_payoff.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # --- Panel 2: MSE ratio (without/with rotation) across dimensions ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in input_names:
        ratios = []
        for d in d_values:
            inputs_d = make_input_vectors(d, device)
            x = inputs_d[name]
            b = 2  # Use b=2 as representative
            # Without rotation
            x_q = uniform_quantize(x.unsqueeze(0), b).squeeze(0)
            mse_no_rot = ((x - x_q) ** 2).sum().item()
            # With rotation (average)
            total_mse = 0.0
            for i in range(n_rotations):
                Pi = make_rotation(d, seed=i, device=device)
                y = Pi @ x
                y_q = uniform_quantize(y.unsqueeze(0), b).squeeze(0)
                x_hat = Pi.T @ y_q
                total_mse += ((x - x_hat) ** 2).sum().item()
            mse_rot = total_mse / n_rotations
            ratios.append(mse_no_rot / (mse_rot + 1e-10))
        ax.semilogx(d_values, ratios, "o-", color=input_colors[name],
                     linewidth=2, markersize=8, label=name)

    ax.axhline(y=1, color="white", linestyle="--", alpha=0.3, label="No benefit")
    ax.set_xlabel("Dimension d", fontsize=13)
    ax.set_ylabel("MSE Reduction Factor (no-rot / rot)", fontsize=13)
    ax.set_title("Test 4b: Rotation Benefit Increases with Dimension (b=2)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, facecolor="#1a1a1a", edgecolor="#333", labelcolor="white")
    ax.grid(True, alpha=0.3)
    dark_theme(ax)
    plt.tight_layout()
    path = os.path.join(results_dir, "test4b_dimension_scaling.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # Print detailed MSE table for d=128
    print(f"\nMSE Table (d={d_primary}, averaged over {n_rotations} rotations):")
    print(f"  {'Input':<12} {'b':>3} {'No Rot':>12} {'With Rot':>12} {'Factor':>10}")
    for name in input_names:
        x = inputs[name]
        for b in bit_widths:
            mse_no = ((x - uniform_quantize(x.unsqueeze(0), b).squeeze(0)) ** 2).sum().item()
            total_mse = 0.0
            for i in range(n_rotations):
                Pi = make_rotation(d_primary, seed=i, device=device)
                y = Pi @ x
                y_q = uniform_quantize(y.unsqueeze(0), b).squeeze(0)
                x_hat = Pi.T @ y_q
                total_mse += ((x - x_hat) ** 2).sum().item()
            mse_rot = total_mse / n_rotations
            factor = mse_no / (mse_rot + 1e-10)
            print(f"  {name:<12} {b:>3} {mse_no:>12.6f} {mse_rot:>12.6f} {factor:>10.1f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    print("Rotation Benefits Experiments")
    print("=" * 60)

    test1_energy_spreading(d=128, results_dir=results_dir)
    test2_distribution_fit(d_values=[4, 16, 64, 128], results_dir=results_dir)
    test3_near_independence(d=128, results_dir=results_dir)
    test4_quantization_mse(d_values=[8, 32, 128, 512], results_dir=results_dir)

    print("\n" + "=" * 60)
    print("All tests complete. Results in experiments/results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
