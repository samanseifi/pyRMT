"""Lid-driven soft disc: blended vs sharp interface against Sugiyama et al. (2011), 1024^2.

Reads the runs of iim_lid_disc.py (sharp, N = 128 and 192) and the blended runs (N = 128 from the
paper's regeneration, outputs/overlay/lid_stab_tauinf.npz in the main repository; N = 192 from
iim_lid_disc.py if present). Sugiyama's trajectory is a set of points without time stamps, so each
point is given the time of the nearest point of the N = 192 sharp trajectory, and the distance of
each run to it is the nearest-point distance.

Usage: python benchmarks/iim_lid_compare.py <output png>
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iim_lid_disc import sugiyama, OUT

MAIN = os.path.join(os.path.dirname(OUT), "..", "..", "pyrmt", "outputs", "overlay", "lid_stab_tauinf.npz")
MAIN = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pyrmt", "outputs",
                                     "overlay", "lid_stab_tauinf.npz"))


def runs():
    r = {}
    r["blended 128"] = np.load(MAIN)["sls"]
    for lab, fn in (("blended 192", "blended_N192_f4_t24.npy"), ("sharp 128", "sharp_N128_f4_t24.npy"),
                    ("sharp 192", "sharp_N192_f4_t24.npy")):
        p = os.path.join(OUT, fn)
        if os.path.isfile(p):
            r[lab] = np.load(p)
    return r


def near(tr, pts):
    d = np.sqrt(((pts[:, None, :] - tr[None, :, 1:3]) ** 2).sum(-1)); i = d.argmin(1)
    return d[np.arange(len(pts)), i], tr[i, 0]


def matched(a, b):
    t = a[:, 0]; d = np.hypot(a[:, 1] - np.interp(t, b[:, 0], b[:, 1]), a[:, 2] - np.interp(t, b[:, 0], b[:, 2]))
    return d.mean(), d.max()


def main():
    sug = sugiyama(); R = runs()
    ref = R["sharp 192"] if "sharp 192" in R else R["sharp 128"]
    _, tsug = near(ref, sug); order = np.argsort(tsug); tsug = tsug[order]; sug = sug[order]
    windows = ((0, 6, "first turn (t < 6)"), (6, 12, "second turn (6-12)"), (12, 24, "inner turns (t > 12)"))
    print("mean nearest-point distance to Sugiyama 1024^2 (all points; and by turn)")
    D = {}
    for k, tr in R.items():
        d, _ = near(tr, sug); D[k] = d
        print(f"  {k:12s} all {d.mean():.2e} (max {d.max():.2e}) | " +
              " | ".join(f"{nm} {d[(tsug >= a) & (tsug < b)].mean():.2e}" for a, b, nm in windows))
    print("time-matched centroid differences")
    for a, b in (("sharp 128", "sharp 192"), ("blended 128", "blended 192"), ("blended 128", "sharp 128"),
                 ("blended 192", "sharp 192")):
        if a in R and b in R:
            print(f"  {a} vs {b}: mean %.2e max %.2e" % matched(R[a], R[b]))
    import matplotlib; matplotlib.use("Agg")
    from _plotstyle import apply as _s; _s()
    import matplotlib.pyplot as plt
    sty = {"blended 128": ("blue", "-", 1.4), "blended 192": ("blue", "--", 1.4),
           "sharp 128": ("red", "-", 1.4), "sharp 192": ("red", "--", 1.4)}
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for a, lims in ((ax[0], None), (ax[1], ((0.40, 0.66), (0.60, 0.83)))):
        a.plot(sug[:, 0], sug[:, 1], "o", mfc="none", mec="k", ms=5, label="Sugiyama et al. $1024^2$")
        for k, tr in R.items():
            c, ls, lw = sty[k]; a.plot(tr[:, 1], tr[:, 2], ls, color=c, lw=lw, label=f"{k.split()[0]}, $N={k.split()[1]}$")
        a.set_aspect("equal"); a.set_xlabel("centroid $x$"); a.set_ylabel("centroid $y$")
        if lims:
            a.set_xlim(*lims[0]); a.set_ylim(*lims[1])
    ax[0].set_title("(a) centroid trajectory"); ax[1].set_title("(b) inner turns")
    ax[0].legend(fontsize=9, loc="lower left")
    a = ax[2]
    for k in R:
        c, ls, lw = sty[k]; a.plot(tsug, D[k], ls, color=c, lw=lw, marker="o", ms=3, label=f"{k.split()[0]}, $N={k.split()[1]}$")
    a.set_xlabel("time of the nearest trajectory point"); a.set_ylabel("distance to Sugiyama point")
    a.set_title("(c) distance along the trajectory"); a.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(sys.argv[1]); print("saved", sys.argv[1])


if __name__ == "__main__":
    main()
