"""Paper figures for the sharp-interface variant.

  sharp_static.png        exact static test (eigenstress ellipse): (a) max velocity error and
                          near-interface pressure error vs N, blended and sharp; (b) pressure along
                          the line through the ellipse centre (N = 64), with the exact pressure.
  disc_tg_convergence.png disc in the Taylor-Green vortex, successive-grid differences, blended and
                          sharp: (a) velocity, (b) pressure (cells more than 1.5 cells from the interface, and the bulk),
                          (c) energies and disc position.

Reads the cached vortex runs of iim_vortex_disc.py (run that first, up to N = 256).
Usage: python benchmarks/iim_paper_figures.py [output directory]
"""
import os, sys
import numpy as np
from scipy import ndimage as ndi
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iim_eigenstress as E
import iim_eigenstress_pressure as EP
import iim_vortex_disc as V

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "paper", "figures")
CB, CS = "blue", "red"                            # blended, sharp (classic colours, as in the other paper figures)


def static_data():
    ur, vr = E.reference(); ur -= ur.mean(); vr -= vr.mean(); pr = EP.reference_p(); M = E.M
    res = {}
    for meth in ("smeared", "sharp"):
        rows = []
        for N in (32, 64, 128, 256):
            u, v, p, phi = EP.steady(N, meth); st = M // N
            e = np.concatenate([(u - ur[st // 2::st, 0::st]).ravel(), (v - vr[0::st, st // 2::st]).ravel()])
            pex = pr[st // 2::st, st // 2::st]; s = phi <= 0
            dist = np.where(s, ndi.distance_transform_edt(s), ndi.distance_transform_edt(~s)) / N
            far = dist > 0.04; near = (dist > 1.5 / N) & (dist > 6 * EP.SIGMA)
            d = (p - p[far].mean()) - (pex - pex[far].mean())
            rows.append((N, np.abs(e).max(), np.sqrt(np.mean(e ** 2)), np.abs(d[near]).max(), np.sqrt(np.mean(d[far] ** 2))))
            if N == 64:
                j = N // 2; res[meth + "_line"] = ((np.arange(N) + 0.5) / N, (p - p[far].mean())[j, :])
                res["exact_line"] = (np.arange(M) / M, (pr - pex[far].mean())[st // 2 + j * st, :])
        res[meth] = np.array(rows)
    return res


def vortex_data(nmax=256):
    NS = [n for n in (32, 64, 128, 256) if n <= nmax]; out = {}
    for meth in ("smeared", "sharp"):
        F = {N: V.cached(meth, N) for N in NS}
        D = [V.differences(F[a], F[2 * a], a) for a in NS[:-1]]
        out[meth] = {q: np.array([x[q] for x in D]) for q in D[0]}
        out[meth]["KEv"] = [float(F[N]["KE"]) for N in NS]; out[meth]["Wv"] = [float(F[N]["W"]) for N in NS]
        out[meth]["c"] = [(float(F[N]["cx"]), float(F[N]["cy"])) for N in NS]
    out["h"] = np.array([1.0 / a for a in NS[:-1]]); out["NS"] = NS
    return out


def report(st, vx):
    od = lambda e: " ".join(f"{np.log2(e[i] / e[i + 1]):.2f}" for i in range(len(e) - 1))
    print("STATIC (N = 32, 64, 128, 256)")
    for m in ("smeared", "sharp"):
        r = st[m]
        for k, nm in ((1, "u max"), (2, "u L2"), (3, "p near max"), (4, "p far L2")):
            print(f"  {m:8s} {nm:11s} " + " ".join(f"{x:.2e}" for x in r[:, k]) + "  orders " + od(r[:, k]))
    print("VORTEX differences " + ", ".join(f"{a}/{2*a}" for a in vx["NS"][:-1]))
    for m in ("smeared", "sharp"):
        for q in ("vel_L2", "vel_max", "vel_bulk", "p_L2", "p_off", "p_bulk", "KE", "W", "centroid"):
            e = vx[m][q]; print(f"  {m:8s} {q:9s} " + " ".join(f"{x:.2e}" for x in e) + "  orders " + od(e))
        print(f"  {m:8s} KE " + " ".join(f"{x:.6f}" for x in vx[m]["KEv"]) + " | W " + " ".join(f"{x:.4e}" for x in vx[m]["Wv"])
              + " | centroid " + " ".join(f"({a:.5f},{b:.5f})" for a, b in vx[m]["c"]))


def main():
    import matplotlib; matplotlib.use("Agg")
    from _plotstyle import apply as _s; _s()
    import matplotlib.pyplot as plt
    st = static_data(); vx = vortex_data(); report(st, vx)
    lab = {"smeared": "blended", "sharp": "sharp"}; col = {"smeared": CB, "sharp": CS}

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.4))
    a = ax[0]
    for m in ("smeared", "sharp"):
        r = st[m]; h = 1.0 / r[:, 0]
        a.loglog(h, r[:, 1], "o-", color=col[m], label=f"velocity, {lab[m]}")
        a.loglog(h, r[:, 3], "s--", color=col[m], mfc="none", label=f"pressure near interface, {lab[m]}")
    h = 1.0 / st["sharp"][:, 0]
    a.loglog(h, 0.6 * st["sharp"][0, 1] * (h / h[0]) ** 2, "k:", lw=1, label="second order")
    a.loglog(h, 2.2 * st["smeared"][0, 1] * (h / h[0]), "k-.", lw=1, label="first order")
    a.invert_xaxis(); a.set_xlabel(r"$\Delta x$"); a.set_ylabel("maximum error"); a.set_title("(a) exact static test")
    a.legend(fontsize=9, loc="lower left")
    a = ax[1]
    x, pe = st["exact_line"]; a.plot(x, pe, "k-", lw=1.0, label="exact")
    for m, mk in (("smeared", "o"), ("sharp", "s")):
        x, p = st[m + "_line"]; a.plot(x, p, mk, color=col[m], ms=3.6, mfc="none", label=lab[m] + r" ($N=64$)")
    a.set_xlim(0.2, 0.8); a.set_xlabel(r"$x$"); a.set_ylabel(r"$p$ on the line through the centre")
    a.set_title("(b) pressure across the solid"); a.legend(fontsize=10)
    fig.tight_layout(); out = os.path.join(OUTDIR, "sharp_static.png"); fig.savefig(out); plt.close(fig); print("saved", out)

    h = vx["h"]
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 5.6))
    for a, items, ttl in ((ax[0], (("vel_L2", "o-", "whole domain"), ("vel_bulk", "s--", "bulk")), "(a) velocity"),
                          (ax[1], (("p_off", "o-", "off the interface"), ("p_bulk", "s--", "bulk")), "(b) pressure"),
                          (ax[2], (("KE", "o-", "kinetic energy"), ("W", "s--", "strain energy"), ("centroid", "^:", "disc position")),
                           "(c) energies and position")):
        for m in ("smeared", "sharp"):
            for q, sty, nm in items:
                a.loglog(h, vx[m][q], sty, color=col[m], mfc=("none" if "--" in sty or ":" in sty else None), label=f"{nm}, {lab[m]}")
        ref = vx["smeared"][items[0][0]][0]
        a.loglog(h, 1.6 * ref * (h / h[0]) ** 2, "k:", lw=1, label="second order")
        a.loglog(h, 1.6 * ref * (h / h[0]), "k-.", lw=1, label="first order")
        a.invert_xaxis(); a.set_xlabel(r"$\Delta x$ of the coarser grid"); a.set_title(ttl)
        a.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax[0].set_ylabel(r"$\|f_N-f_{2N}\|$")
    fig.tight_layout(); out = os.path.join(OUTDIR, "disc_tg_convergence.png"); fig.savefig(out); plt.close(fig); print("saved", out)


if __name__ == "__main__":
    main()
