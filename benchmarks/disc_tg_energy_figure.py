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

    # Present result = the model's tau->inf elastic limit, carried by the log-conformation
    # branch (sigma = G dev(b_e), b_e -> F F^T as tau->inf). This recovers the elastic
    # benchmark within the unified viscoelastic formulation (no separate neo-Hookean run).
    from benchmarks.viscoelastic_disc_tg import run_one as run_ve
    h = run_ve(128, 1.0, model="maxwell", G_ve=1.0, tau=np.inf, integrator="imex",
               scheme="semilagrangian_cubic", R=0.2, x0=0.5, y0=0.5, mu_s=1.0,
               mu_f=1e-3, U0=0.05, kw=2, history=True)["hist"]
    hist = {"t": h["t"], "ke": h["KE"], "se": h["Eel"]}
    ddir = os.path.join(ROOT, "benchmarks", "data")

    def refcurve(m, comp):
        """Load the vector-extracted Jain-Fig-14b reference (both ke and se now digitized
        directly from the source), sort, and bin-average the scattered dots to a clean curve."""
        p = os.path.join(ddir, f"jain2019_fig14b_{m}_{comp}.csv")
        if not os.path.isfile(p):
            return None
        d = np.loadtxt(p, delimiter=",", skiprows=1)
        t, y = d[:, 0], d[:, 1]
        if comp == "se":
            # the SE csv is already a clean bin-median curve anchored at (0,0); smooth
            # it directly (no re-binning, which would average the zero start away)
            return smooth(t, y)
        bins = np.linspace(t.min(), t.max(), 61)
        idx = np.digitize(t, bins)
        tb, yb = [], []
        for b in range(1, 61):
            msk = idx == b
            if msk.any():
                tb.append(t[msk].mean()); yb.append(y[msk].mean())
        tt, yy = np.array(tb), np.array(yb)
        return smooth(tt, yy) if len(tt) > 4 else (tt, yy)

    def smooth(t, y, n=400):
        t = np.asarray(t, float); y = np.asarray(y, float); o = np.argsort(t)
        t, y = t[o], y[o]; keep = np.concatenate(([True], np.diff(t) > 1e-9))
        t, y = t[keep], y[keep]; tt = np.linspace(t.min(), t.max(), n)
        return tt, PchipInterpolator(t, y)(tt)

    # standard (classic) colours + distinct line styles
    refstyle = {"Jain": ("black", "--", "Jain et al."),
                "Zhao": ("blue", "-.", "Zhao et al."),
                "Mosher": ("darkgreen", ":", "Robinson-Mosher")}
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.3))
    for j, comp in enumerate(("ke", "se")):
        for m, (c, ls, lab) in refstyle.items():
            rc = refcurve(m, comp)
            if rc is not None:
                ax[j].plot(rc[0], rc[1], color=c, ls=ls, lw=1.6, label=lab)
        ax[j].plot(hist["t"], hist[comp], color="red", lw=2.4, ls="-",
                   label=r"present, $\tau\!\to\!\infty$ ($N{=}128$)")
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
