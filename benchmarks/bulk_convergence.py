"""Bulk-vs-band spatial convergence of the disc-in-Taylor-Green fields (tau->inf elastic limit).

Orders are measured REFERENCE-FREE, from successive-grid (Cauchy) differences ||f_N - f_2N|| over
the doubling pairs (32,64) (48,96) (64,128) (96,192) (128,256) (256,512). An error measured against
the N=512 run is biased upward on the fine grids (the reference is then only 2-4x finer than the
grid being tested: a first-order quantity shows an apparent order ~1.8 on the 192->256 pair), so
it is not used for the quoted orders. All fields are sampled on the N=512 cell centres.

"Bulk" = cells farther than DELTA from the N=512 interface, by true distance (Euclidean distance
transform of the solid indicator). The level set itself cannot be used for this: phi = phi0(xi) is
only defined inside the 3-layer extrapolation band and is a constant outside it.

Caches fields to outputs/overlay/bulk_conv.npz (compute() + add_convergence_points.py) and writes
paper/figures/disc_tg_convergence.png. Re-plot with `--plot-only`.
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


DELTA = 0.06                                    # bulk = farther than this from the interface
PAIRS = [(32, 64), (48, 96), (64, 128), (96, 192), (128, 256), (256, 512)]


def analyse(d):
    """Successive-grid differences and their observed orders (reference-free)."""
    from scipy import ndimage as ndi
    ref = int(d["ref"])
    f = {int(N): (d[f"sp_{N}"], d[f"pr_{N}"]) for N in d["coarse"]}
    f[ref] = (d["spref"], d["prref"])
    solid = d["phiref"] <= 0.0
    dist = np.where(solid, ndi.distance_transform_edt(solid), ndi.distance_transform_edt(~solid)) / ref
    mbulk = dist > DELTA
    KE = {int(N): float(v) for N, v in zip(d["NS"], d["KE"])}
    SE = {int(N): float(v) for N, v in zip(d["NS"], d["SE"])}
    rms = lambda a: float(np.sqrt((a ** 2).mean()))
    out = dict(h=[], full_v=[], bulk_v=[], full_p=[], bulk_p=[], dKE=[], dSE=[], bulk_frac=float(mbulk.mean()))
    for a, b in PAIRS:
        if a not in f or b not in f:
            continue
        dv = f[a][0] - f[b][0]; dp = f[a][1] - f[b][1]
        out["h"].append(1.0 / a)
        out["full_v"].append(rms(dv)); out["bulk_v"].append(rms(dv[mbulk]))
        out["full_p"].append(rms(dp)); out["bulk_p"].append(rms(dp[mbulk]))
        out["dKE"].append(KE[a] - KE[b]); out["dSE"].append(SE[a] - SE[b])
    for k in ("h", "full_v", "bulk_v", "full_p", "bulk_p", "dKE", "dSE"):
        out[k] = np.array(out[k])
    return out


def _orders(h, e):
    """Observed order between difference points whose grids are a factor 2 apart."""
    res = []
    for i, hi in enumerate(h):
        for k, hk in enumerate(h):
            if abs(hi / hk - 2.0) < 1e-12:
                res.append((round(1 / hi), round(1 / hk), float(np.log2(abs(e[i]) / abs(e[k])))))
    return res


def from_conv(path):
    """Adapt a sweep file written by benchmarks/reruns/convergence_after_fix.py (fields sampled on
    its finest grid) to the layout analyse() expects. The strain energy is the smooth-weighted one
    (Eel_s), which measures the solid without cell-quantization noise."""
    c = np.load(path); Ns = sorted(int(k[3:]) for k in c.files if k.startswith("sp_"))
    ref = Ns[-1]; d = dict(ref=ref, coarse=np.array(Ns[:-1]), spref=c[f"sp_{ref}"], prref=c[f"pr_{ref}"],
                          phiref=c["phig"], NS=np.array(Ns),
                          KE=np.array([float(c[f"KE_{N}"]) for N in Ns]),
                          SE=np.array([float(c[f"SEs_{N}"] if f"SEs_{N}" in c.files else c[f"SE_{N}"]) for N in Ns]))
    for N in Ns[:-1]:
        d[f"sp_{N}"] = c[f"sp_{N}"]; d[f"pr_{N}"] = c[f"pr_{N}"]
    return d


def main(recompute=True, conv_path=None):
    import matplotlib; matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks")); from _plotstyle import apply as _s; _s()
    import matplotlib.pyplot as plt
    if conv_path:
        d = from_conv(conv_path)
    else:
        d = compute() if (recompute or not os.path.isfile(NPZ)) else dict(np.load(NPZ))
    r = analyse(d); h = r["h"]
    print(f"  bulk (> {DELTA} from the interface) = {100 * r['bulk_frac']:.1f}% of the domain")
    for nm, key in (("full velocity", "full_v"), ("bulk velocity", "bulk_v"), ("full pressure", "full_p"),
                    ("bulk pressure", "bulk_p"), ("kinetic E", "dKE"), ("strain E", "dSE")):
        o = _orders(h, r[key])
        print(f"  {nm:14s} " + "  ".join(f"{a}/{b}:{p:5.2f}" for a, b, p in o)
              + f"   mean {np.mean([p for *_, p in o]):.2f}", flush=True)
    print("  signed dKE", r["dKE"]); print("  signed dSE", r["dSE"])

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.5))
    # (a) integrated energies: successive-grid differences, every pair shown
    ax[0].loglog(h, np.abs(r["dKE"]), "s-", color="black", lw=1.6, ms=6, label="kinetic energy")
    ax[0].loglog(h, np.abs(r["dSE"]), "D-", color="darkgreen", lw=1.6, ms=6, label="strain energy")
    ax[0].loglog(h, abs(r["dKE"][0]) * (h / h[0]) ** 2, "--", color="0.5", lw=1.0, label="$O(\\Delta x^2)$")
    ax[0].set_xlabel("$\\Delta x = 1/N$"); ax[0].set_ylabel("$|E_N - E_{2N}|$")
    ax[0].set_title("(a) integrated energies"); ax[0].legend(fontsize=12)
    # (b) fields: full domain vs bulk
    ax[1].loglog(h, r["full_v"], "o--", color="blue", lw=1.4, ms=6, alpha=0.6, label="velocity, full")
    ax[1].loglog(h, r["bulk_v"], "o-", color="blue", lw=1.9, ms=6, label="velocity, bulk")
    ax[1].loglog(h, r["full_p"], "^--", color="red", lw=1.4, ms=6, alpha=0.6, label="pressure, full")
    ax[1].loglog(h, r["bulk_p"], "^-", color="red", lw=1.9, ms=6, label="pressure, bulk")
    ax[1].loglog(h, r["full_v"][0] * (h / h[0]) ** 1, ":", color="0.5", lw=1.0, label="$O(\\Delta x)$")
    ax[1].loglog(h, r["full_p"][0] * (h / h[0]) ** 0.5, "-.", color="0.5", lw=1.0, label="$O(\\Delta x^{1/2})$")
    ax[1].set_xlabel("$\\Delta x = 1/N$"); ax[1].set_ylabel("$\\|f_N - f_{2N}\\|_{L_2}$")
    ax[1].set_title("(b) fields: full domain vs.\\ bulk")
    ax[1].legend(fontsize=11, ncol=1, loc="lower right")
    fig.tight_layout()
    out = os.path.join(ROOT, "paper", "figures", "disc_tg_convergence.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    conv = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--from-conv=")), None)
    main(recompute="--plot-only" not in sys.argv, conv_path=conv)
