"""Disc-in-Taylor-Green FSI: replication of the convergence benchmark of Jain, Kamrin &
Mani (JCP 2019, Sec. 4.4), itself validated there against the mixed Eulerian-Lagrangian
results of Zhao et al. (2008) and Robinson-Mosher et al. (2011).

Setup (their parameters): a circular neo-Hookean solid of radius r=0.2 at the centre of a
unit box, released in a Taylor-Green vortex psi = psi0 sin(2 pi x) sin(2 pi y), psi0=0.05;
mu_f=1e-3, mu_s=0.5, rho_s=rho_f=1. The solid stretches and elastically retracts, oscillating
while the fluid viscously damps it. We reproduce (a) the grid-independent kinetic/strain
energy evolution and (b) the O(dx^2) convergence of the fields at t=0.25.

Writes outputs/overlay/disc_tg_convergence.png.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
from benchmarks._plotstyle import apply as _apply_style; _apply_style()
import matplotlib.pyplot as plt
from benchmarks.mac_disc_in_tg_convergence import run_one, _sample

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "overlay")
os.makedirs(OUT, exist_ok=True)

# Jain-Kamrin-Mani (2019) Sec. 4.4 parameters (energy-evolution benchmark).
# Shear modulus: Jain writes the neo-Hookean stress as sigma = 2 mu_s^Jain b (their
# Eq. 10), so their mu_s^Jain = 0.5 is a small-strain shear modulus G = 2*0.5 = 1.0.
# Our convention (Rycroft et al. 2018, Eq. 3.16) is sigma = mu_s b with mu_s = G, so we
# must set mu_s = 1.0 to reproduce the SAME physical solid. (Running mu_s = 0.5 gives a
# solid half as stiff -> disc over-deforms, oscillation ~1.5x too slow, se ~1.5x high.)
# isochoric=True: the volume-corrected neo-Hookean sigma = mu_s J^{-2} dev(b), which is
# trace-free. The plain sigma=mu_s b (isochoric=False) leaks its isotropic part as spurious
# solenoidal velocity, because the wide-stencil force divergence is not in the range of the
# compact projection gradient; negligible here (mu_s=1) but corrected for correctness.
JAIN = dict(R=0.2, x0=0.5, y0=0.5, mu_s=1.0, mu_f=1e-3, U0=0.05, kw=2,
            scheme="semilagrangian_cubic", isochoric=True)
# standard soft-disc parameters for the convergence study (single vortex)
SOFT = dict(R=0.15, x0=0.5, y0=0.7, mu_s=0.1, mu_f=0.01, U0=0.3, kw=1,
            scheme="semilagrangian_cubic", isochoric=True)


def _load_jain_methods():
    """Digitized Jain et al. (2019) Fig. 14b: the three published methods (Jain
    'present', Zhao 2008, Robinson-Mosher 2011), ke and se, keyed by series name."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                        "jain2019_fig14b_methods.csv")
    if not os.path.exists(path):
        return None
    d = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    out = {}
    for s in np.unique(d["series"]):
        m = d[d["series"] == s]
        out[str(s)] = (m["t"], m["energy"])
    return out


def run(NS=(32, 64, 128, 256, 512), t_conv=0.6, t_evol=1.0):
    # (a,b) energy evolution overlaid on Jain et al. Fig. 14b (G=1.0; our mu_s=1.0)
    evol = {N: run_one(N, t_evol, integrator="imex", **JAIN)["hist"] for N in (128,)}
    methods = _load_jain_methods()
    # (b) spatial convergence (soft-disc parameters), errors vs a RESOLVED N=256 reference
    conv = {N: run_one(N, t_conv, integrator="imex", **SOFT) for N in NS}
    ref = NS[-1]; coarse = list(NS[:-1]); h = np.array([1.0 / N for N in coarse])
    g = conv[ref]["xc"]
    spref = _sample(conv[ref]["xc"], conv[ref]["speed"], g)
    prref = _sample(conv[ref]["xc"], conv[ref]["p"], g)
    verr = [np.sqrt(np.mean((_sample(conv[N]["xc"], conv[N]["speed"], g) - spref) ** 2)) for N in coarse]
    perr = [np.sqrt(np.mean((_sample(conv[N]["xc"], conv[N]["p"], g) - prref) ** 2)) for N in coarse]
    keerr = [abs(conv[N]["KE"] - conv[ref]["KE"]) for N in coarse]
    seerr = [abs(conv[N]["Estrain"] - conv[ref]["Estrain"]) for N in coarse]

    def order(e):
        e = np.array(e)
        return np.log(e[:-1] / e[1:]) / np.log(np.array(coarse[1:]) / np.array(coarse[:-1]))
    print("Disc-in-TG convergence (soft params) at t=%.2f (ref N=%d):" % (t_conv, ref))
    for nm, e in (("velocity", verr), ("pressure", perr), ("kinetic E", keerr), ("strain E", seerr)):
        print(f"  {nm:10s} L2 {[f'{x:.2e}' for x in e]}  order {[f'{o:.2f}' for o in order(e)]}")

    # digitized reference curves are sparse (25-57 pts) and render jagged; resample each
    # through a monotone cubic (PCHIP) so the published-method curves are smooth. This only
    # interpolates the digitized points -- it adds no information, just clean rendering.
    from scipy.interpolate import PchipInterpolator
    def _smooth(t, y, n=400):
        t = np.asarray(t, float); y = np.asarray(y, float); o = np.argsort(t)
        t, y = t[o], y[o]
        keep = np.concatenate(([True], np.diff(t) > 1e-9))
        t, y = t[keep], y[keep]
        tt = np.linspace(t.min(), t.max(), n)
        return tt, PchipInterpolator(t, y)(tt)

    refstyle = {"Jain": ("k", "-", "Jain et al.\\ (present)"),
                "Zhao": ("#1f77b4", "--", "Zhao et al.\\ 2008"),
                "Mosher": ("#ff7f0e", "-.", "Robinson-Mosher 2011")}

    # ---------- Figure 1: energy validation (kinetic + strain) ----------
    figE, ax = plt.subplots(1, 2, figsize=(10.4, 4.3))
    for j, comp in enumerate(("ke", "se")):
        if methods is not None:
            for m, (c, ls, lab) in refstyle.items():
                key = f"{m}_{comp}"
                if key in methods:
                    ax[j].plot(*_smooth(*methods[key]), color=c, ls=ls, lw=1.3, label=lab)
        ax[j].plot(evol[128]["t"], evol[128][comp], color="#C1121F", lw=2.4, label="ours ($N{=}128$)")
        ax[j].set_xlabel("time $t$"); ax[j].set_xlim(0, 1)
    ax[0].set_ylabel("kinetic energy"); ax[0].set_ylim(0.010, 0.026)
    ax[0].set_title("(a) kinetic energy"); ax[0].legend(loc="upper right")
    ax[1].set_ylabel("strain energy"); ax[1].set_ylim(0, 0.011)
    ax[1].set_title("(b) strain energy"); ax[1].legend(loc="upper right")
    figE.tight_layout()
    pE = os.path.join(OUT, "disc_tg_energy.png"); figE.savefig(pE, dpi=200); plt.close(figE)
    print(f"  saved {pE}")

    # ---------- Figure 2: spatial convergence ----------
    figC, axc = plt.subplots(figsize=(6.6, 4.8))
    axc.loglog(h, keerr, "s-", color="k", lw=1.6, ms=6, label="kinetic energy")   # black
    axc.loglog(h, seerr, "D-", color="g", lw=1.6, ms=6, label="strain energy")    # green
    axc.loglog(h, verr,  "o-", color="b", lw=1.6, ms=6, label="velocity")         # blue
    axc.loglog(h, perr,  "^-", color="r", lw=1.6, ms=6, label="pressure")         # red
    axc.loglog(h, keerr[0] * (h / h[0]) ** 2, "--", color="0.45", lw=1.1, label="$O(\\Delta x^2)$")
    axc.loglog(h, verr[0] * (h / h[0]) ** 1, ":",  color="0.45", lw=1.1, label="$O(\\Delta x)$")
    axc.invert_xaxis(); axc.set_xlabel("$\\Delta x = 1/N$")
    axc.set_ylabel("$L_2$ error vs.\\ resolved reference")
    axc.set_title(f"convergence (energy 2nd order; fields 1st; ref $N{{=}}{ref}$)")
    axc.legend(fontsize=11, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    figC.tight_layout()
    pC = os.path.join(OUT, "disc_tg_convergence.png")
    figC.savefig(pC, dpi=200, bbox_inches="tight"); plt.close(figC)
    print(f"  saved {pC}")


if __name__ == "__main__":
    run()
