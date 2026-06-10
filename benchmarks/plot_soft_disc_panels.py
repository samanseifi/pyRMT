"""Plot the soft disc in the lid-driven cavity at the Jain et al. (2019) Fig. 16
time instances, for one or both grid resolutions.

Reads the `snap_t*.h5` field snapshots written by
`soft_disc_in_lid_driven.run(..., snapshot_times=[...])` and produces:

  1. A panel grid per resolution: velocity magnitude (solid masked white) +
     phi=0 interface (black) + reference-map contours, one panel per time.
  2. An interface-only overlay comparing N=64 vs N=128 at each time
     (grid-convergence of the deformed shape).

Usage:
    python benchmarks/plot_soft_disc_panels.py
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def grid(N):
    x = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, x)
    return X, Y


def load_snaps(out_dir):
    snaps = []
    for path in sorted(glob.glob(os.path.join(out_dir, "snap_t*.h5"))):
        with h5py.File(path, "r") as f:
            d = {k: f[k][:] for k in f.keys()}
            d["t"] = float(f.attrs.get("t", np.nan))
            d["t_target"] = float(f.attrs.get("t_target", np.nan))
        snaps.append(d)
    return snaps


def panel_grid(out_dir, title, save):
    snaps = load_snaps(out_dir)
    if not snaps:
        print(f"  no snapshots in {out_dir}")
        return None
    N = snaps[0]["phi"].shape[0]
    X, Y = grid(N)
    n = len(snaps)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, s in zip(axes, snaps):
        phi = s["phi"]; a = s["a"]; b = s["b"]; X1 = s["X1"]; X2 = s["X2"]
        umag = np.ma.masked_where(phi <= 0, np.hypot(a, b))
        ax.contourf(X, Y, umag, levels=40, cmap="Spectral_r")
        ax.contourf(X, Y, (phi <= 0).astype(float), levels=[0.5, 1], colors="white", zorder=2)
        X1m = np.where(phi <= 0, X1, np.nan)
        X2m = np.where(phi <= 0, X2, np.nan)
        ax.contour(X, Y, X1m, levels=12, colors="0.4", linewidths=0.4, zorder=3)
        ax.contour(X, Y, X2m, levels=12, colors="0.4", linewidths=0.4, linestyles="dashed", zorder=3)
        ax.contour(X, Y, phi, levels=[0], colors="black", linewidths=1.6, zorder=4)
        ax.set_title(f"t = {s['t']:.2f}", fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save, dpi=140)
    plt.close(fig)
    print(f"  saved {save}")
    return snaps


def overlay_compare(dir64, dir128, save):
    s64 = load_snaps(dir64)
    s128 = load_snaps(dir128)
    if not s64 or not s128:
        print("  overlay skipped (missing snapshots)")
        return
    X64, Y64 = grid(s64[0]["phi"].shape[0])
    X128, Y128 = grid(s128[0]["phi"].shape[0])
    n = min(len(s64), len(s128))
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for k in range(n):
        ax = axes[k]
        ax.contour(X64, Y64, s64[k]["phi"], levels=[0], colors="tab:blue", linewidths=1.8)
        ax.contour(X128, Y128, s128[k]["phi"], levels=[0], colors="tab:red",
                   linewidths=1.4, linestyles="dashed")
        ax.set_title(f"t = {s128[k]['t']:.2f}", fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    # legend proxy
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([0], [0], color="tab:blue", lw=1.8, label="N=64"),
                            Line2D([0], [0], color="tab:red", lw=1.4, ls="--", label="N=128")],
                   fontsize=8, loc="upper right")
    fig.suptitle("Soft disc in lid-driven cavity — interface: N=64 vs N=128", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save, dpi=140)
    plt.close(fig)
    print(f"  saved {save}")


if __name__ == "__main__":
    out = os.path.join(ROOT, "outputs")
    d64 = os.path.join(out, "soft_disc_lid_N64_semilagrangian")
    d128 = os.path.join(out, "soft_disc_lid_N128_semilagrangian")
    panel_grid(d64, "Soft disc in lid-driven cavity (N=64)",
               os.path.join(d64, "disc_panels_N64.png"))
    panel_grid(d128, "Soft disc in lid-driven cavity (N=128)",
               os.path.join(d128, "disc_panels_N128.png"))
    overlay_compare(d64, d128, os.path.join(out, "disc_interface_N64_vs_N128.png"))
