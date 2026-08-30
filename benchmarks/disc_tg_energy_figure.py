"""Standalone re-plot of the disc-in-Taylor-Green energy-validation figure (Fig. 3) with
standard colours, without re-running the full spatial-convergence sweep. One N=128 run
(Jain parameters) plus the digitized reference methods (PCHIP-smoothed).
Writes paper/figures/disc_tg_energy.png.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.mac_disc_in_tg_convergence import run_one
from benchmarks.disc_tg_fsi_convergence import JAIN, _load_jain_methods

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    import matplotlib; matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    from _plotstyle import apply as _style; _style()
    import matplotlib.pyplot as plt
    from scipy.interpolate import PchipInterpolator

    hist = run_one(128, 1.0, integrator="imex", **JAIN)["hist"]
    methods = _load_jain_methods()

    def smooth(t, y, n=400):
        t = np.asarray(t, float); y = np.asarray(y, float); o = np.argsort(t)
        t, y = t[o], y[o]; keep = np.concatenate(([True], np.diff(t) > 1e-9))
        t, y = t[keep], y[keep]; tt = np.linspace(t.min(), t.max(), n)
        return tt, PchipInterpolator(t, y)(tt)

    # standard colours
    refstyle = {"Jain": ("black", "-", "Jain et al."),
                "Zhao": ("blue", "--", "Zhao et al."),
                "Mosher": ("green", "-.", "Robinson-Mosher")}
    ddir = os.path.join(ROOT, "benchmarks", "data")
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.3))
    for j, comp in enumerate(("ke", "se")):
        for m, (c, ls, lab) in refstyle.items():
            if comp == "ke":   # high-resolution curves extracted from Jain Fig. 14b (vector)
                p = os.path.join(ddir, f"jain2019_fig14b_{m}_ke.csv")
                if os.path.isfile(p):
                    d = np.loadtxt(p, delimiter=",", skiprows=1)
                    ax[j].plot(d[:, 0], d[:, 1], color=c, ls=ls, lw=1.4, label=lab); continue
            key = f"{m}_{comp}"
            if methods is not None and key in methods:
                ax[j].plot(*smooth(*methods[key]), color=c, ls=ls, lw=1.4, label=lab)
        ax[j].plot(hist["t"], hist[comp], color="red", lw=2.2, label=r"ours ($N{=}128$)")
        ax[j].set_xlabel("time $t$"); ax[j].set_xlim(0, 1)
    ax[0].set_ylabel("kinetic energy"); ax[0].set_ylim(0.010, 0.026)
    ax[0].set_title("(a) kinetic energy"); ax[0].legend(loc="upper right")
    ax[1].set_ylabel("strain energy"); ax[1].set_ylim(0, 0.011)
    ax[1].set_title("(b) strain energy"); ax[1].legend(loc="upper right")
    fig.tight_layout()
    out = os.path.join(ROOT, "paper", "figures", "disc_tg_energy.png")
    fig.savefig(out, dpi=300); print("saved", out)


if __name__ == "__main__":
    main()
