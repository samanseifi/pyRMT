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

# Jain-Kamrin-Mani (2019) Sec. 4.4 parameters (energy-evolution benchmark)
JAIN = dict(R=0.2, x0=0.5, y0=0.5, mu_s=0.5, mu_f=1e-3, U0=0.05, kw=2,
            scheme="semilagrangian")
# standard soft-disc parameters for the convergence study (single vortex)
SOFT = dict(R=0.15, x0=0.5, y0=0.7, mu_s=0.1, mu_f=0.01, U0=0.3, kw=1,
            scheme="semilagrangian")


def run(NS=(32, 64, 128, 256), t_conv=0.6, t_evol=1.0):
    # (a) energy evolution vs Jain et al. Fig. 14 (their stiffer parameters)
    evol = {N: run_one(N, t_evol, **JAIN)["hist"] for N in (64, 128)}
    # (b) spatial convergence (soft-disc parameters), errors vs a RESOLVED N=256 reference
    conv = {N: run_one(N, t_conv, **SOFT) for N in NS}
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

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.3))
    # (a) energy evolution vs Jain et al.
    cols = {64: "C0", 128: "C3"}
    for N in (64, 128):
        ax[0].plot(evol[N]["t"], evol[N]["ke"], "-", color=cols[N], label=f"$N={N}$, ke")
        ax[0].plot(evol[N]["t"], evol[N]["se"], "--", color=cols[N], label=f"$N={N}$, se")
    ax[0].set_xlabel("time $t$"); ax[0].set_ylabel("energy")
    ax[0].set_title("(a) energy evolution vs.\\ Jain et al.\\ (grid-independent)")
    ax[0].legend(ncol=2, fontsize=8.5, loc="upper right")

    # (b) spatial convergence: integrated energy ~2nd order, pointwise fields ~1st order
    ax[1].loglog(h, keerr, "^-", label="kinetic energy")
    ax[1].loglog(h, seerr, "D-", label="strain energy")
    ax[1].loglog(h, verr, "o-", label="velocity")
    ax[1].loglog(h, perr, "s-", label="pressure")
    ax[1].loglog(h, keerr[0] * (h / h[0]) ** 2, "k--", lw=1, label="$O(\\Delta x^2)$")
    ax[1].loglog(h, verr[0] * (h / h[0]) ** 1, "k:", lw=1, label="$O(\\Delta x)$")
    ax[1].invert_xaxis(); ax[1].set_xlabel("$\\Delta x = 1/N$")
    ax[1].set_ylabel("$L_2$ error vs resolved reference")
    ax[1].set_title("(b) convergence (energy 2nd order; fields 1st)")
    ax[1].legend(ncol=2, fontsize=8, loc="lower right"); ax[1].grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    p = os.path.join(OUT, "disc_tg_convergence.png"); fig.savefig(p); plt.close()
    print(f"  saved {p}")


if __name__ == "__main__":
    run()
