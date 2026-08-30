"""Snapshots of the soft disc in a lid-driven cavity (Jain et al. 2019, Fig. 16 style):
the solid interface (reference-map level set phi=0) over the flow streamlines, at a
sequence of times. Writes paper/figures/soft_disc_lid_snapshots.png.

Usage: python benchmarks/soft_disc_lid_snapshots.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.mac_soft_disc_lid import run


def main():
    import matplotlib; matplotlib.use("Agg")
    from _plotstyle import apply as _style; _style()
    import matplotlib.pyplot as plt

    N = 192
    # match Jain et al. (2019) Fig. 16 time instants for direct visual comparison
    times = [1.17, 2.34, 3.52, 4.69, 5.86, 7.03, 8.20]
    traj, snaps = run(N=N, t_end=times[-1] + 0.05, scheme="semilagrangian_cubic",
                      integrator="imex-sl", write=False, snap_times=times)
    # cache the snapshots so the figure can be restyled without re-running
    cache = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "outputs", "overlay", "soft_disc_lid_snaps.npz")
    np.savez(cache, **{f"phi_{i}": s["phi"] for i, s in enumerate(snaps)},
             **{f"uc_{i}": s["uc"] for i, s in enumerate(snaps)},
             **{f"vc_{i}": s["vc"] for i, s in enumerate(snaps)},
             **{f"X1_{i}": s["X1"] for i, s in enumerate(snaps)},
             **{f"X2_{i}": s["X2"] for i, s in enumerate(snaps)},
             t=np.array([s["t"] for s in snaps]),
             xc=snaps[0]["Xc"][0, :], yc=snaps[0]["Yc"][:, 0])
    _render(snaps)


def _render(snaps):
    import matplotlib; matplotlib.use("Agg")
    from _plotstyle import apply as _style; _style()
    import matplotlib.pyplot as plt

    # two rows: 4 on top, 3 on bottom
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.6))
    axes = axes.ravel()
    # material-coordinate levels for the reference-map grid (disc: (0.6,0.5), R=0.2)
    lv1 = np.linspace(0.42, 0.78, 10); lv2 = np.linspace(0.32, 0.68, 10)
    for k, (ax, s) in enumerate(zip(axes, snaps)):
        xc = s["Xc"][0, :]; yc = s["Yc"][:, 0]
        inside = s["phi"] <= 0
        X1 = np.where(inside, s["X1"], np.nan); X2 = np.where(inside, s["X2"], np.nan)
        # grey streamlines FIRST, then the OPAQUE solid fill on top (no flow shows through)
        ax.streamplot(xc, yc, s["uc"], s["vc"], color="0.6", linewidth=0.4,
                      density=0.9, arrowsize=0.5, arrowstyle="->", zorder=1)
        ax.contourf(xc, yc, s["phi"], levels=[-10, 0.0], colors=["#cfe3f2"], zorder=2)
        # deformed reference-map grid, dashed (the material lines carried by the flow)
        ax.contour(xc, yc, X1, levels=lv1, colors="k", linewidths=0.7, linestyles="--", zorder=3)
        ax.contour(xc, yc, X2, levels=lv2, colors="k", linewidths=0.7, linestyles="--", zorder=3)
        ax.contour(xc, yc, s["phi"], levels=[0.0], colors="k", linewidths=1.8, zorder=4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(rf"$t={s['t']:.2f}$", fontsize=12)
        for sp in ax.spines.values():
            sp.set_edgecolor("0.6")
    axes[-1].axis("off")   # 8th cell empty (4+3)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.02, wspace=0.06, hspace=0.12)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "paper", "figures", "soft_disc_lid_snapshots.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    main()
