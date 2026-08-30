"""Contact demonstration figures: two soft discs colliding head-on in a viscous fluid with
the trace-free contact stress, in the elastic and viscoelastic (SLS) regimes, integrated
explicitly and with the IMEX scheme.
  contact_collision.png  -- centroid gap vs time (elastic vs viscoelastic, explicit vs IMEX)
  contact_snapshots.png  -- interfaces of the two discs over the fluid streamlines, in time
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.contact_collision import run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N, T, V0, MUF, ETA, SEP, R = 128, 1.6, 0.6, 0.005, 2.0, 0.30, 0.12
GTOT = 0.8
EL = dict(model="elastic", mu_s=GTOT)
VE = dict(model="sls", mu_s=0.2, G_ve=0.6, tau=0.2)


def main():
    import matplotlib; matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    from _plotstyle import apply as _style; _style()
    import matplotlib.pyplot as plt

    common = dict(N=N, t_end=T, V0=V0, mu_f=MUF, eta=ETA, sep=SEP, R=R)
    runs = {
        ("elastic", "explicit"): run(**EL, integrator="explicit", **common),
        ("elastic", "imex"):     run(**EL, integrator="imex", **common),
        ("sls", "explicit"):     run(**VE, integrator="explicit", **common),
        ("sls", "imex"):         run(**VE, integrator="imex", **common),
    }
    np.savez(os.path.join(ROOT, "outputs", "overlay", "contact_collision.npz"),
             **{f"{m}_{i}": h for (m, i), h in runs.items()})

    # ---- gap vs time: elastic vs viscoelastic (IMEX; the explicit comparison is in the supp) ----
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.plot(runs[("elastic", "imex")][:, 0], runs[("elastic", "imex")][:, 3], "-",
            color="blue", lw=2.0, label="elastic")
    ax.plot(runs[("sls", "imex")][:, 0], runs[("sls", "imex")][:, 3], "-",
            color="red", lw=2.0, label="viscoelastic (SLS)")
    ax.axhline(2 * R, color="0.5", ls=":", lw=1.2, label=r"$2R$ (touching)")
    ax.set_xlabel("time $t$"); ax.set_ylabel("centroid gap")
    ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "paper", "figures", "contact_collision.png"), dpi=200)
    print("saved contact_collision.png")

    # ---- snapshot montage: the discs colliding in the fluid ----
    times = [0.15, 0.35, 0.55, 0.90, 1.40]
    _, snaps = run(**EL, integrator="explicit", snap_times=times, **common)
    fig, axes = plt.subplots(1, len(snaps), figsize=(2.3 * len(snaps), 2.6))
    for ax, s in zip(np.atleast_1d(axes), snaps):
        xc = s["Xc"][0, :]; yc = s["Yc"][:, 0]
        ax.streamplot(xc, yc, s["uc"], s["vc"], color="0.6", linewidth=0.4,
                      density=0.8, arrowsize=0.5, arrowstyle="->", zorder=1)
        for phi, col in ((s["pa"], "#bcd4e6"), (s["pb"], "#f2c0c0")):
            ax.contourf(xc, yc, phi, levels=[-10, 0.0], colors=[col], zorder=2)
            ax.contour(xc, yc, phi, levels=[0.0], colors="k", linewidths=1.4, zorder=3)
        ax.set_xlim(0, 1); ax.set_ylim(0.5 - 0.28, 0.5 + 0.28); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(rf"$t={s['t']:.2f}$", fontsize=11)
        for sp in ax.spines.values():
            sp.set_edgecolor("0.6")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.88, bottom=0.02, wspace=0.06)
    fig.savefig(os.path.join(ROOT, "paper", "figures", "contact_snapshots.png"),
                dpi=200, bbox_inches="tight")
    print("saved contact_snapshots.png")
    for key, h in runs.items():
        print(f"  {key}: end gap {h[-1,3]:.3f}")


if __name__ == "__main__":
    main()
