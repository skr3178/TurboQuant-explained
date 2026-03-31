"""
Surface area of a unit hypersphere vs dimension d.

A(d) = 2 * pi^{(d+1)/2} / Gamma((d+1)/2)

where d is the dimension of the sphere S^d embedded in R^{d+1}.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

dimensions = np.arange(2, 129)
surface_area = 2 * np.pi ** ((dimensions + 1) / 2) / gamma((dimensions + 1) / 2)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(dimensions, surface_area, "o-", color="#6ee7b7", markersize=3, linewidth=1.5)
ax.fill_between(dimensions, surface_area, alpha=0.15, color="#6ee7b7")

peak_dim = dimensions[np.argmax(surface_area)]
peak_val = surface_area.max()
ax.annotate(f"Peak: d={peak_dim}, A={peak_val:.2f}",
            xy=(peak_dim, peak_val), xytext=(peak_dim + 8, peak_val * 0.9),
            arrowprops=dict(arrowstyle="->", color="white"), color="white", fontsize=11)

ax.set_xlabel("Dimension d", fontsize=13)
ax.set_ylabel("Surface Area A(d)", fontsize=13)
ax.set_title("Surface Area of the Unit Hypersphere vs Dimension (d = 2..128)", fontsize=15, fontweight="bold")
ax.set_yscale("log")
ax.set_xlim(2, 128)
ax.grid(True, alpha=0.3)
fig.patch.set_facecolor("#0e0e0e")
ax.set_facecolor("#0e0e0e")
ax.tick_params(colors="white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.title.set_color("white")
for spine in ax.spines.values():
    spine.set_color("#333")

plt.tight_layout()
plt.savefig("hypersphere_surface_area.png", dpi=150, bbox_inches="tight")
print("Saved: hypersphere_surface_area.png")
