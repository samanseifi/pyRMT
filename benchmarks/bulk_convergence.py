"""Bulk-vs-band spatial convergence of the disc-in-Taylor-Green fields (tau->inf elastic limit).
The one-fluid fields are only first order at the diffuse interface (the blended stress smears the
traction jump over a ~2dx band). Here we separate the error INSIDE each grid's own blend band
(|phi| < 2 dx_N, where the stress is blended) from the BULK (|phi| > 2 dx_N, where the stress is
the pure fluid/solid stress). The bulk should recover second order; the full-domain error is
first order, capped by the band. Reference = resolved N=512. Caches fields to
outputs/overlay/bulk_conv.npz and writes paper/figures/disc_tg_convergence.png.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.viscoelastic_disc_tg import run_one
from benchmarks.mac_disc_in_tg_convergence import _sample
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOFT = dict(R=0.15, x0=0.5, y0=0.7, mu_f=0.01, U0=0.3, kw=1)
NPZ = os.path.join(ROOT, "outputs", "overlay", "bulk_conv.npz")


def compute():
    NS = (32, 64, 128, 256, 512); tconv = 0.6
    conv = {}
    for N in NS:
        r = run_one(N, tconv, model="maxwell", G_ve=0.1, tau=np.inf, integrator="imex",
                    scheme="semilagrangian_cubic", mu_s=0.1, history=True, **SOFT)
        conv[N] = dict(xc=r["xc"], speed=r["speed"], p=r["p"], phi=r["phi"],
                       KE=r["KE"], SE=float(r["hist"]["Eel"][-1]))
        print(f"  N={N} done", flush=True)
    ref = NS[-1]; coarse = list(NS[:-1])
    g = conv[ref]["xc"]; spref = conv[ref]["speed"]; prref = conv[ref]["p"]
    phiref = conv[ref]["phi"]                      # reference level set on the ref grid
    save = dict(ref=ref, coarse=np.array(coarse), g=g, spref=spref, prref=prref, phiref=phiref)
    for N in coarse:
        save[f"sp_{N}"] = _sample(conv[N]["xc"], conv[N]["speed"], g)
        save[f"pr_{N}"] = _sample(conv[N]["xc"], conv[N]["p"], g)
    save["KE"] = np.array([conv[N]["KE"] for N in NS])
    save["SE"] = np.array([conv[N]["SE"] for N in NS])
    save["NS"] = np.array(NS)
    np.savez(NPZ, **save)
    return save


def main(recompute=True):
    import matplotlib; matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks")); from _plotstyle import apply as _s; _s()
    import matplotlib.pyplot as plt
    d = compute() if (recompute or not os.path.isfile(NPZ)) else dict(np.load(NPZ))
    coarse = list(d["coarse"]); ref = int(d["ref"])
    spref, prref, phiref = d["spref"], d["prref"], d["phiref"]
    h = np.array([1.0 / N for N in coarse])
    delta = 0.06                                # fixed interfacial exclusion band (same for all N)
    mbulk = np.abs(phiref) > delta
    full_v, full_p, bulk_v, bulk_p = [], [], [], []
    for N in coarse:
        sp, pr = d[f"sp_{N}"], d[f"pr_{N}"]
        dv2 = (sp - spref) ** 2; dp2 = (pr - prref) ** 2
        full_v.append(np.sqrt(dv2.mean())); full_p.append(np.sqrt(dp2.mean()))
        bulk_v.append(np.sqrt(dv2[mbulk].mean())); bulk_p.append(np.sqrt(dp2[mbulk].mean()))
    # energies: error vs the N=512 reference. The integrated energies super-converge and reach the
    # reference's own ~1e-6 (KE) / ~1e-7 (SE) floor by N~128-192, so we plot each up to the last
    # grid above its floor (monotone regime); both track O(dx^2).
    KE = dict(zip(d["NS"], d["KE"])); SE = dict(zip(d["NS"], d["SE"]))
    keN = [N for N in coarse if N <= 192]; seN = [N for N in coarse if N <= 128]
    hke = np.array([1.0 / N for N in keN]); hse = np.array([1.0 / N for N in seN])
    eKE = np.array([abs(KE[N] - KE[ref]) for N in keN])
    eSE = np.array([abs(SE[N] - SE[ref]) for N in seN])
    def _ord(Ns, e):
        Ns = np.array(Ns, float); e = np.array(e)
        return [f"{x:.2f}" for x in np.log(e[:-1] / e[1:]) / np.log(Ns[1:] / Ns[:-1])]
    for nm, e in (("full velocity", full_v), ("bulk velocity", bulk_v),
                  ("full pressure", full_p), ("bulk pressure", bulk_p)):
        print(f"  {nm:14s} order {_ord(coarse, e)}", flush=True)
    print(f"  kinetic E      order {_ord(keN, eKE)}", flush=True)
    print(f"  strain E       order {_ord(seN, eSE)}", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.5))
    # (a) integrated energies vs the reference -- second order
    ax[0].loglog(hke, eKE, "s-", color="black", lw=1.6, ms=6, label="kinetic energy")
    ax[0].loglog(hse, eSE, "D-", color="darkgreen", lw=1.6, ms=6, label="strain energy")
    ax[0].loglog(hke, eKE[0] * (hke / hke[0]) ** 2, "--", color="0.5", lw=1.0, label="$O(\\Delta x^2)$")
    ax[0].set_xlabel("$\\Delta x = 1/N$"); ax[0].set_ylabel(f"energy error (vs.\\ $N{{=}}{ref}$)")
    ax[0].set_title("(a) integrated energies"); ax[0].legend(fontsize=10)
    # (b) fields: full-domain vs bulk (outside a fixed interfacial band |phi|>delta).
    # Pressure's low order is interfacial: the bulk pressure recovers toward 2nd order. Velocity
    # is global: the bulk barely differs from the full domain.
    ax[1].loglog(h, full_v, "o--", color="blue", lw=1.4, ms=6, alpha=0.6, label="velocity, full")
    ax[1].loglog(h, bulk_v, "o-", color="blue", lw=1.9, ms=6, label="velocity, bulk")
    ax[1].loglog(h, full_p, "^--", color="red", lw=1.4, ms=6, alpha=0.6, label="pressure, full")
    ax[1].loglog(h, bulk_p, "^-", color="red", lw=1.9, ms=6, label="pressure, bulk")
    ax[1].loglog(h, np.array(bulk_p)[0] * (h / h[0]) ** 2, "--", color="0.5", lw=1.0, label="$O(\\Delta x^2)$")
    ax[1].loglog(h, np.array(full_v)[0] * (h / h[0]) ** 1, ":", color="0.5", lw=1.0, label="$O(\\Delta x)$")
    ax[1].set_xlabel("$\\Delta x = 1/N$"); ax[1].set_ylabel(f"$L_2$ field error (vs.\\ $N{{=}}{ref}$)")
    ax[1].set_title(rf"(b) fields: full vs.\ bulk ($|\phi|>{delta}$)")
    ax[1].legend(fontsize=8.5, ncol=1, loc="lower left")
    fig.tight_layout()
    out = os.path.join(ROOT, "paper", "figures", "disc_tg_convergence.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    main(recompute="--plot-only" not in sys.argv)
