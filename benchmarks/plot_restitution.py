"""Re-plot the coefficient-of-restitution figure (paper Fig. 12) from the swept data, on log axes.
Panel (a): e vs impact Stokes number St = rho*V0*R/mu_f, swept to high St by decreasing mu_f at a
fixed impact speed (the mu_f -> 0, St -> infinity limit). St is also the collision Reynolds number,
so beyond St ~ 90 the near-contact flow is under-resolved at N=128 and the ELASTIC collision
diverges; the relaxation-stabilized SLS survives to St ~ 360. Both rise monotonically toward the
lossless elastic limit e -> 1. Panel (b): grid convergence at St = 22 (log-x resolution axis).
Writes paper/figures/contact_restitution.png.
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAN = np.nan

# --- swept data (from benchmarks/contact_restitution.py; elastic diverges for St >= 90) ---
St_a = np.array([9.0, 13.5, 21.6, 36.0, 54.0, 90.0, 135.0, 216.0, 360.0])
e_el = np.array([0.146, 0.202, 0.270, 0.350, 0.457, NAN, NAN, NAN, NAN])
e_sls = np.array([0.062, 0.088, 0.121, 0.186, 0.223, 0.249, 0.306, 0.380, 0.411])

# --- grid convergence at St = 21.6 (mu_f = 0.005, V0 = 0.9); N=160 filled from n160.log ---
Nb = np.array([64, 96, 128, 160])
cb_el = np.array([0.504, 0.310, 0.270, NAN])
cb_sls = np.array([0.383, 0.183, 0.121, NAN])


def _load_n160():
    p = os.path.join("/tmp/claude-1000/-home-samanseifi-codes-pyrmt/"
                     "65b84d1e-eba2-4c1b-a2aa-0cc69e21cc0d/scratchpad", "n160.log")
    if not os.path.isfile(p):
        return
    for line in open(p):
        if line.startswith("N160 elastic:") and "e=" in line:
            cb_el[3] = float(line.split("e=")[1])
        if line.startswith("N160 sls:") and "e=" in line:
            cb_sls[3] = float(line.split("e=")[1])


def main():
    import matplotlib; matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks")); from _plotstyle import apply as _s; _s()
    import matplotlib.pyplot as plt
    _load_n160()
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.4))
    # (a) e vs St, log-x, toward the e -> 1 elastic limit
    ax[0].axhline(1.0, color="0.5", ls="--", lw=1.2)
    ax[0].text(St_a.min(), 0.965, r"lossless elastic limit  $e\!\to\!1$  ($\mathrm{St}\!\to\!\infty$)",
               fontsize=10, color="0.4", va="top")
    ax[0].semilogx(St_a, e_el, "o-", color="blue", lw=1.8, ms=7, label="elastic")
    ax[0].semilogx(St_a, e_sls, "s-", color="red", lw=1.8, ms=6, label="viscoelastic (SLS)")
    # mark where the elastic collision can no longer be resolved
    stx = np.sqrt(54.0 * 90.0)
    ax[0].axvline(stx, color="blue", ls=":", lw=1.0)
    ax[0].text(stx, 0.30, "  elastic diverges\n  (under-resolved,\n  $\\mathrm{St}\\gtrsim90$)",
               color="blue", fontsize=8.5, va="center", ha="left")
    ax[0].set_xlabel(r"impact Stokes number  $\mathrm{St}=\rho V_0 R/\mu_f$")
    ax[0].set_ylabel("coefficient of restitution  $e$")
    ax[0].set_ylim(0.0, 1.05); ax[0].set_title(r"(a) restitution vs.\ Stokes number")
    ax[0].legend(loc="upper left")
    # (b) grid convergence, log-x
    m = np.isfinite(cb_el); ms = np.isfinite(cb_sls)
    ax[1].semilogx(Nb[m], cb_el[m], "o-", color="blue", lw=1.8, ms=7, label="elastic")
    ax[1].semilogx(Nb[ms], cb_sls[ms], "s-", color="red", lw=1.8, ms=6, label="viscoelastic (SLS)")
    ax[1].set_xlabel("grid resolution  $N$"); ax[1].set_ylabel("coefficient of restitution  $e$")
    ax[1].set_xticks(Nb); ax[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax[1].set_ylim(bottom=0.0)
    ax[1].set_title(r"(b) grid convergence  ($\mathrm{St}=22$)")
    ax[1].legend(loc="best")
    fig.tight_layout()
    out = os.path.join(ROOT, "paper", "figures", "contact_restitution.png")
    fig.savefig(out, dpi=200); print("saved", out, "N160 el/sls =", cb_el[3], cb_sls[3])


if __name__ == "__main__":
    main()
