from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon


OUT_DIR = Path("images")

POINTS = np.array(
    [
        [0.50, 0.78],
        [0.72, 0.66],
        [0.78, 0.42],
        [0.58, 0.24],
        [0.34, 0.26],
        [0.20, 0.48],
        [0.30, 0.70],
        [0.50, 0.50],
        [0.39, 0.52],
        [0.48, 0.91],
        [0.91, 0.50],
    ]
)

EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 0),
    (0, 7),
    (1, 7),
    (2, 7),
    (3, 7),
    (4, 7),
    (5, 8),
    (6, 8),
    (7, 8),
]
TRIANGLES = [(0, 1, 7), (2, 3, 7), (4, 5, 8), (6, 0, 8)]


def _style_axis(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_points(ax):
    _style_axis(ax)
    ax.scatter(POINTS[:, 0], POINTS[:, 1], s=46, color="#111827", zorder=3)


def draw_vr(ax):
    _style_axis(ax)
    radius = 0.155
    for x, y in POINTS:
        ax.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor="#bfdbfe",
                edgecolor="#60a5fa",
                linewidth=2.0,
                alpha=0.58,
                zorder=1,
            )
        )
    for tri in TRIANGLES:
        vertices = POINTS[list(tri)]
        ax.add_patch(
            Polygon(
                vertices,
                closed=True,
                facecolor="#fbbf24",
                edgecolor="none",
                alpha=0.46,
                zorder=2,
            )
        )
    for i, j in EDGES:
        ax.plot(
            [POINTS[i, 0], POINTS[j, 0]],
            [POINTS[i, 1], POINTS[j, 1]],
            color="#111827",
            linewidth=1.9,
            alpha=1.0,
            zorder=3,
        )
    ax.scatter(POINTS[:, 0], POINTS[:, 1], s=42, color="#111827", zorder=4)


def draw_diagram(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1.4)
    ax.set_xlabel("Birth", fontsize=12)
    ax.set_ylabel("Death", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af", linewidth=1.5)

    h0 = np.array(
        [
            [0.03, 0.16],
            [0.03, 0.21],
            [0.03, 0.27],
            [0.03, 0.32],
            [0.03, 0.38],
            [0.03, 0.45],
        ]
    )
    h1 = np.array([[0.32, 0.82]])
    ax.scatter(h0[:, 0], h0[:, 1], s=42, color="#2563eb", label=r"$H_0$", zorder=3)
    ax.scatter(h1[:, 0], h1[:, 1], s=48, color="#dc2626", label=r"$H_1$", zorder=3)
    ax.legend(frameon=False, loc="lower right", fontsize=11)


def save_single(name: str, draw_fn):
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    draw_fn(ax)
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_combined():
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.2))
    draw_points(axes[0])
    axes[0].set_title("Point cloud", fontsize=13, fontweight="bold")
    draw_vr(axes[1])
    axes[1].set_title("Vietoris--Rips complex", fontsize=13, fontweight="bold")
    draw_diagram(axes[2])
    axes[2].set_title("Persistence diagram", fontsize=13, fontweight="bold")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT_DIR / "vr_persistence_illustration.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "vr_persistence_illustration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_single("vr_points", draw_points)
    save_single("vr_complex", draw_vr)
    save_single("vr_persistence_diagram", draw_diagram)
    save_combined()
    for path in [
        OUT_DIR / "vr_points.pdf",
        OUT_DIR / "vr_complex.pdf",
        OUT_DIR / "vr_persistence_diagram.pdf",
        OUT_DIR / "vr_persistence_illustration.pdf",
    ]:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
