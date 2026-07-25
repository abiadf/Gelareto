from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


OUT_DIR = Path("images")
OUT_DIR.mkdir(exist_ok=True)
DATA_PATH = Path(
    "datasets/2D/bouncing_rings/processed/"
    "test_N128_T50_H64_W64_shapering_balls5-8_r4_dt0.018_pulse0.45-0.35_"
    "thick1-3_overlap0.3_dtypeuint8_seed50000.pt"
)


def cuboid_faces(x0, x1, y0, y1, z0, z1):
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]


def draw_grid_face(ax, x=0.0, n=8, color="#61717f"):
    for i in range(n + 1):
        y = i / n
        ax.plot([x, x], [y, y], [0, 1], color=color, lw=0.55, alpha=0.55)
        z = i / n
        ax.plot([x, x], [0, 1], [z, z], color=color, lw=0.55, alpha=0.55)


def clip_motion_score(clip):
    frames = clip[:, 0].float().numpy()
    masks = frames > max(float(frames.max()) * 0.25, 1.0)
    coords = np.argwhere(masks)
    if coords.size == 0:
        return -np.inf
    _, ys, xs = coords[:, 0], coords[:, 1], coords[:, 2]
    bbox_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
    centroids = []
    for mask in masks:
        pts = np.argwhere(mask)
        if len(pts):
            centroids.append(pts.mean(axis=0))
    if len(centroids) < 2:
        displacement = 0.0
    else:
        c = np.asarray(centroids)
        displacement = np.linalg.norm(c.max(axis=0) - c.min(axis=0))
    return float(bbox_area + 20.0 * displacement)


def load_video_frames():
    import torch

    video = torch.load(DATA_PATH, map_location="cpu")
    # Tensor layout: batch, time, channel, height, width.
    scores = [clip_motion_score(video[i]) for i in range(min(len(video), 128))]
    clip_idx = int(np.argmax(scores))
    print(f"Selected {DATA_PATH.name}, clip {clip_idx}, motion score {scores[clip_idx]:.1f}")
    frames = video[clip_idx, :, 0].float().numpy()
    frames = (frames - frames.min()) / max(frames.max() - frames.min(), 1e-8)
    return frames


def frame_facecolors(frame):
    # Transparent background plus semi-opaque ring pixels. Full opacity makes
    # overlapping time slices collapse into an unreadable blue mass.
    frame = np.clip(frame, 0, 1) ** 0.70
    bg = np.array([0.90, 0.96, 1.00, 0.035])
    fg = np.array([0.00, 0.24, 0.78, 0.68])
    img = frame[..., None]
    colors = bg * (1 - img) + fg * img
    colors[..., 3] = 0.01 + 0.62 * frame
    return colors


def add_textured_frame(ax, x, frame, thickness=0.018, is_front=False):
    h, w = frame.shape
    y = np.linspace(0, 1, w)
    z = np.linspace(1, 0, h)
    yy, zz = np.meshgrid(y, z)
    x0 = x
    x1 = x + thickness
    xx = np.full_like(yy, x0)
    xx_back = np.full_like(yy, x1)

    colors = frame_facecolors(frame)
    ax.plot_surface(
        xx,
        yy,
        zz,
        rstride=1,
        cstride=1,
        facecolors=colors,
        shade=False,
        linewidth=0,
        antialiased=False,
    )
    ax.plot_surface(
        xx_back,
        yy,
        zz,
        rstride=1,
        cstride=1,
        facecolors=colors,
        shade=False,
        linewidth=0,
        antialiased=False,
    )

    # Uniform thin slab sides for every frame.
    face = [[(x0, 0, 0), (x1, 0, 0), (x1, 1, 0), (x0, 1, 0)]]
    face += [[(x0, 1, 0), (x1, 1, 0), (x1, 1, 1), (x0, 1, 1)]]
    face += [[(x0, 0, 1), (x1, 0, 1), (x1, 1, 1), (x0, 1, 1)]]
    face += [[(x0, 0, 0), (x1, 0, 0), (x1, 0, 1), (x0, 0, 1)]]
    slab = Poly3DCollection(
        face,
        facecolors="#e7f4ff",
        edgecolors="#425968",
        linewidths=0.30,
        alpha=0.12,
    )
    ax.add_collection3d(slab)
    ax.plot([x0, x0, x0, x0, x0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="#425968", lw=0.35, alpha=0.42)
    ax.plot([x1, x1, x1, x1, x1], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="#425968", lw=0.35, alpha=0.42)
    if is_front:
        draw_grid_face(ax, x=x0 - 0.001, n=8)


def main():
    frames = load_video_frames()
    frame_ids = np.linspace(2, len(frames) - 4, 11).astype(int)

    fig = plt.figure(figsize=(5.2, 3.3))

    ax3d = fig.add_axes([0.00, 0.00, 1.00, 1.00], projection="3d")
    ax3d.set_proj_type("persp")
    ax3d.view_init(elev=18, azim=-60)

    length = 1.52
    slab_thickness = 0.018
    xs = np.linspace(0.08, length - 0.08, len(frame_ids))

    # Light outer cubical volume.
    faces = cuboid_faces(0, length + 0.02, 0, 1, 0, 1)
    volume = Poly3DCollection(
        faces,
        facecolors="#d7ecff",
        edgecolors="#2f4858",
        linewidths=0.75,
        alpha=0.08,
    )
    ax3d.add_collection3d(volume)

    for i, (x, idx) in enumerate(zip(xs, frame_ids)):
        add_textured_frame(ax3d, x, frames[idx], thickness=slab_thickness, is_front=(i == 0))

    # Cubical grid hints on top and side.
    for x in xs:
        for edge_x in (x, x + slab_thickness):
            ax3d.plot([edge_x, edge_x], [0, 1], [1, 1], color="#6c7f8d", lw=0.35, alpha=0.30)
            ax3d.plot([edge_x, edge_x], [1, 1], [0, 1], color="#6c7f8d", lw=0.35, alpha=0.30)
    for y in np.linspace(0, 1, 6):
        ax3d.plot([0, length], [y, y], [1, 1], color="#6c7f8d", lw=0.35, alpha=0.22)
    for z in np.linspace(0, 1, 6):
        ax3d.plot([0, length], [1, 1], [z, z], color="#6c7f8d", lw=0.35, alpha=0.22)

    ax3d.set_xlim(-0.08, length + 0.10)
    ax3d.set_ylim(-0.08, 1.08)
    ax3d.set_zlim(-0.08, 1.08)
    ax3d.set_axis_off()

    fig.savefig(OUT_DIR / "video3d_tda_diagram.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "video3d_tda_diagram.png", bbox_inches="tight", dpi=300)
    print(OUT_DIR / "video3d_tda_diagram.pdf")
    print(OUT_DIR / "video3d_tda_diagram.png")


if __name__ == "__main__":
    main()
