"""Fig.: snapshots of the standard-linear-solid disc carried and deformed by a single
Taylor-Green vortex -- the solid interface (reference-map level set) over the flow
streamlines (coloured by speed), at a sequence of times. Companion to the stored-energy
figure. Writes paper/figures/ve_snapshots.png.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.viscoelastic_disc_tg import run_one

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    import matplotlib; matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    from _plotstyle import apply as _style; _style()
    import matplotlib.pyplot as plt

    times = [0.4, 0.8, 1.2, 1.6, 2.0]
    snaps = run_one(N=128, t_end=times[-1] + 0.02, model="sls", tau=1.0,
                    mu_s=0.5, G_ve=1.5, kw=2, x0=0.5, y0=0.5, R=0.2, U0=0.2, integrator="imex",
                    snap_times=times)["snaps"]

    n = len(snaps)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.7))
    for ax, s in zip(np.atleast_1d(axes), snaps):
        xc = s["Xc"][0, :]; yc = s["Yc"][:, 0]
        ax.contourf(xc, yc, s["phi"], levels=[-10, 0.0], colors=["#f2c0c0"])
        ax.streamplot(xc, yc, s["uc"], s["vc"], color=s["speed"], cmap="viridis",
                      linewidth=0.5, density=0.9, arrowsize=0.5, arrowstyle="->")
        ax.contour(xc, yc, s["phi"], levels=[0.0], colors="k", linewidths=1.6)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(rf"$t={s['t']:.1f}$", fontsize=12)
        for sp in ax.spines.values():
            sp.set_edgecolor("0.6")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.9, bottom=0.02, wspace=0.06)
    out = os.path.join(ROOT, "paper", "figures", "ve_snapshots.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    main()
