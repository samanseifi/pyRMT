"""Supplementary figure: the developed lid-driven cavity flow (pure-fluid MAC solver),
pressure field with velocity streamlines, at Re=100 and Re=1000 (N=128, steady state).
Complements the Ghia centreline validation. Writes paper/figures/lid_developed.png.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.mac_lid_driven import run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    import matplotlib; matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    from _plotstyle import apply as _style; _style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6))
    for ax, Re in zip(axes, (100, 1000)):
        f = run(Re=Re, N=128, imex=True, return_fields=True)
        u, v, p, N = f["u"], f["v"], f["p"], f["N"]
        uc = 0.5 * (u[:, :-1] + u[:, 1:])       # -> cell centres
        vc = 0.5 * (v[:-1, :] + v[1:, :])
        xc = (np.arange(N) + 0.5) / N
        pp = p - p.mean()                        # pressure is defined up to a constant
        lim = np.percentile(np.abs(pp), 99)
        pc = ax.pcolormesh(xc, xc, pp, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
        spd = np.sqrt(uc**2 + vc**2)
        ax.streamplot(xc, xc, uc, vc, color="k", linewidth=0.6, density=1.1,
                      arrowsize=0.6, arrowstyle="->")
        ax.set_title(rf"$\mathrm{{Re}}={Re}$", fontsize=11)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("pressure", fontsize=9)
    fig.tight_layout()
    out = os.path.join(ROOT, "paper", "figures", "lid_developed.png")
    fig.savefig(out, dpi=200); print("saved", out)


if __name__ == "__main__":
    main()
