"""Shear-benchmark figure from the 2026-11 re-runs (no simulation here).

Data: outputs/rerun_2026-11/shear.npz (all cases to t=8) and shear_long.npz (Wi = 2, 3, 4, 8 to
t=20), both produced with bicubic transport and the IMEX integrator, N=96 on a 2x1 cell, with the
D(t) history kept. "Steady" D and theta are the time averages over a late window (t in [6,8], or
t >= 10 for the long runs); the error bars span the min-max of D in that window, i.e. the residual
oscillation. The Wi->0 point is a genuine SLS run with tau=1e-8, not the elastic run.

Usage: python benchmarks/disc_in_shear_figure.py  ->  paper/figures/disc_in_shear.png
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "outputs", os.environ.get("RERUN_DIR", "rerun_2026-11"))
CAS = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
WIS = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
TH_MIN_D = 0.08          # below this the two axes differ by < ~3 cells: orientation not resolved


def steady(h, t0, t1=np.inf):
    t, d, th = h[:, 0], h[:, 1], h[:, 2]
    m = (t >= t0) & (t <= t1)
    return d[m].mean(), d[m].min(), d[m].max(), np.degrees(th[m].mean())


def load():
    s = np.load(os.path.join(D, "shear.npz")); lg = np.load(os.path.join(D, "shear_long.npz"))
    el = [steady(s[f"el_Ca{c}"], 6.0, 8.0) for c in CAS]
    ve = []
    for w in WIS:
        k = f"sls_Wi{w}"
        ve.append(steady(lg[k], 10.0) if k in lg.files else steady(s[k], 6.0, 8.0))
    return el, ve


def main():
    import matplotlib; matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks")); from _plotstyle import apply as _s; _s()
    import matplotlib.pyplot as plt
    el, ve = load()
    Del = np.array([e[0] for e in el]); th = np.array([e[3] for e in el])
    Dve = np.array([v[0] for v in ve]); lo = Dve - [v[1] for v in ve]; hi = np.array([v[2] for v in ve]) - Dve
    D_lo = Del[CAS.index(0.1)]; D_hi = Del[CAS.index(0.2)]
    print("elastic D:", dict(zip(CAS, np.round(Del, 4))))
    print("elastic theta:", dict(zip(CAS, np.round(th, 1))))
    print("SLS D:", dict(zip(WIS, np.round(Dve, 4))), " osc +-", np.round((lo + hi) / 2, 4))
    print(f"anchors: D_el(Ca=0.2)={D_hi:.4f} (Wi->0), D_el(Ca=0.1)={D_lo:.4f} (Wi->inf)")

    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.2))
    ax[0].plot(CAS, Del, "o-", color="b", lw=1.4, label="RMT (elastic disc)")
    ax[0].plot([0, max(CAS)], [0, max(CAS)], "k--", lw=1, label=r"linear limit $D=\mathrm{Ca}$")
    ax[0].set_xlabel(r"$\mathrm{Ca}=\mu_f\dot\gamma/G$"); ax[0].set_ylabel(r"deformation $D=(a-b)/(a+b)$")
    ax[0].set_title("(a) elastic disc in shear"); ax[0].legend(loc="lower right", frameon=False)
    axr = ax[0].twinx(); mk = Del >= TH_MIN_D
    axr.plot(np.array(CAS)[mk], th[mk], "s:", color="r", ms=4, alpha=0.85)
    axr.set_ylabel(r"orientation $\theta$ (deg)", color="r"); axr.set_ylim(0, 50)
    axr.axhline(45, ls=":", color="r", lw=0.7)

    ax[1].axhline(D_hi, ls="--", color="0.5", lw=1.0); ax[1].axhline(D_lo, ls="--", color="0.5", lw=1.0)
    ax[1].errorbar(WIS, Dve, yerr=[lo, hi], fmt="^-", color="g", lw=1.6, ms=6, capsize=3, zorder=3)
    ax[1].set_xlabel(r"$\mathrm{Wi}=\dot\gamma\,\tau$"); ax[1].set_ylabel(r"steady $D$ at $\mathrm{Ca}=0.2$")
    ax[1].set_title("(b) viscoelastic (SLS) disc")
    ax[1].text(max(WIS) * 0.97, D_hi, r"$\mathrm{Wi}\!\to\!0$: elastic $D(\mathrm{Ca}{=}0.2)$",
               ha="right", va="bottom", fontsize=9, color="0.35")
    ax[1].text(max(WIS) * 0.97, D_lo * 0.93, r"$\mathrm{Wi}\!\to\!\infty$: elastic $D(\mathrm{Ca}{=}0.1)$",
               ha="right", va="top", fontsize=9, color="0.35")
    ax[1].set_ylim(0.0, D_hi * 1.3)
    fig.tight_layout()
    out = os.path.join(ROOT, "paper", "figures", "disc_in_shear.png"); fig.savefig(out, dpi=200)
    print("saved", out)


if __name__ == "__main__":
    main()
