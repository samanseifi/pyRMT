"""Spatial convergence of the disc-in-Taylor-Green at the tau->inf elastic limit
(log-conformation, bicubic transport), replacing the neo-Hookean Fig. 3 (disc_tg_convergence).
Errors vs a resolved N=512 reference; velocity/pressure are the diffuse-interface-limited fields,
kinetic/strain energies converge at ~2nd order. Writes paper/figures/disc_tg_convergence.png.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.viscoelastic_disc_tg import run_one
from benchmarks.mac_disc_in_tg_convergence import _sample
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOFT = dict(R=0.15, x0=0.5, y0=0.7, mu_f=0.01, U0=0.3, kw=1)   # single vortex, soft disc

def main():
    import matplotlib; matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks")); from _plotstyle import apply as _s; _s()
    import matplotlib.pyplot as plt
    NS = (32, 64, 128, 256, 512); tconv = 0.6
    conv = {}
    for N in NS:
        r = run_one(N, tconv, model="maxwell", G_ve=0.1, tau=np.inf, integrator="imex",
                    scheme="semilagrangian_cubic", mu_s=0.1, history=True, **SOFT)
        conv[N] = dict(xc=r["xc"], speed=r["speed"], p=r["p"], KE=r["KE"],
                       SE=float(r["hist"]["Eel"][-1]))
        print(f"  N={N} done KE={conv[N]['KE']:.6f} SE={conv[N]['SE']:.6f}", flush=True)
    ref = NS[-1]; coarse = list(NS[:-1]); h = np.array([1.0 / N for N in coarse])
    g = conv[ref]["xc"]; spref = conv[ref]["speed"]; prref = conv[ref]["p"]
    verr = [np.sqrt(np.mean((_sample(conv[N]["xc"], conv[N]["speed"], g) - spref) ** 2)) for N in coarse]
    perr = [np.sqrt(np.mean((_sample(conv[N]["xc"], conv[N]["p"], g) - prref) ** 2)) for N in coarse]
    # energies converge to the ~1e-6 discretization floor by N~128, so a vs-reference error
    # crosses the reference's own error there; report the monotone successive-grid (Richardson)
    # difference instead, which measures the convergence without the reference-crossing artifact.
    KEa = np.array([conv[N]["KE"] for N in NS]); SEa = np.array([conv[N]["SE"] for N in NS])
    keerr = np.abs(KEa[:-1] - KEa[1:]); seerr = np.abs(SEa[:-1] - SEa[1:])   # at N in coarse
    np.savez(os.path.join(ROOT, "outputs", "overlay", "tg_conv_tauinf.npz"),
             h=h, verr=verr, perr=perr, keerr=keerr, seerr=seerr, coarse=coarse, ref=ref)
    def order(e):
        e = np.array(e); return np.log(e[:-1] / e[1:]) / np.log(np.array(coarse[1:]) / np.array(coarse[:-1]))
    for nm, e in (("velocity", verr), ("pressure", perr), ("kinetic E", keerr), ("strain E", seerr)):
        print(f"  {nm:10s} order {[f'{o:.2f}' for o in order(e)]}", flush=True)
    figC, axc = plt.subplots(figsize=(6.6, 4.8))
    axc.loglog(h, keerr, "s-", color="black", lw=1.6, ms=6, label="kinetic energy")
    axc.loglog(h, seerr, "D-", color="darkgreen", lw=1.6, ms=6, label="strain energy")
    axc.loglog(h, verr, "o-", color="blue", lw=1.6, ms=6, label="velocity")
    axc.loglog(h, perr, "^-", color="red", lw=1.6, ms=6, label="pressure")
    axc.loglog(h, keerr[0] * (h / h[0]) ** 2, "--", color="0.45", lw=1.1, label="$O(\\Delta x^2)$")
    axc.loglog(h, verr[0] * (h / h[0]) ** 1, ":", color="0.45", lw=1.1, label="$O(\\Delta x)$")
    axc.set_xlabel("$\\Delta x = 1/N$")
    axc.set_ylabel(f"$L_2$ error (fields vs.\\ $N{{=}}{ref}$; energy successive-grid)")
    axc.set_title("convergence, $\\tau\\!\\to\\!\\infty$ (energy 2nd order; fields 1st)")
    axc.legend(fontsize=11, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    figC.tight_layout()
    out = os.path.join(ROOT, "paper", "figures", "disc_tg_convergence.png")
    figC.savefig(out, dpi=200, bbox_inches="tight"); print("saved", out)

if __name__ == "__main__":
    main()
