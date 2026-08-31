"""Coefficient of restitution of the colliding-disc contact vs. impact Stokes number, for
elastic and viscoelastic (SLS) discs -- a semi-quantitative comparison with the
elastohydrodynamic-collision picture (Davis, Serayssol & Hinch, JFM 1986): e rises with the
Stokes number St from a critical value below which the fluid fully damps the impact (e=0),
and the viscoelastic discs, which dissipate collision energy in the Maxwell branch, have a
lower e than the elastic ones at the same St.

e is measured from the centroid-gap trajectory as e = |max rebound rate| / |max approach rate|
(the separation vs. approach speed of the two centroids across the collision). The impact
Stokes number is St = rho * V0 * R / mu_f.

Writes paper/figures/contact_restitution.png.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.contact_collision import run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N, MUF, ETA, SEP, R, RHO = 96, 0.005, 2.0, 0.30, 0.12, 1.0
# clean single-bounce regime (V0 >~ 1.3 becomes violent / diverges -- not a clean restitution)
V0S = [0.3, 0.45, 0.6, 0.75, 0.9, 1.05]
EL = dict(model="elastic", mu_s=0.8)
VE = dict(model="sls", mu_s=0.2, G_ve=0.6, tau=0.2)


def restitution(h):
    """e = |max separation rate after closest approach| / |max approach rate before it|,
    measured on a smoothed gap trajectory (removes staircase-centroid noise)."""
    from scipy.ndimage import uniform_filter1d
    t, gap = h[:, 0], h[:, 3]
    gs = uniform_filter1d(gap, size=9, mode="nearest")
    dg = np.gradient(gs, t)
    imin = int(np.argmin(gs))
    if imin < 5 or imin > len(gs) - 5:
        return np.nan
    v_in = -np.min(dg[:imin])                 # fastest approach (gap decreasing)
    v_out = np.max(dg[imin:])                 # fastest separation (gap increasing)
    if v_in <= 1e-6:
        return np.nan
    return max(0.0, v_out) / v_in


def main():
    import matplotlib; matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    from _plotstyle import apply as _style; _style()
    import matplotlib.pyplot as plt

    res = {"elastic": [], "sls": []}; traj = {}
    for V0 in V0S:
        t_end = min(2.4, 0.6 + 1.2 / V0)      # enough time for the slow low-V0 rebound
        for key, cfg in (("elastic", EL), ("sls", VE)):
            try:
                h = run(N=N, t_end=t_end, V0=V0, mu_f=MUF, eta=ETA, sep=SEP, R=R,
                        integrator="imex", **cfg)
                e = restitution(h); traj[f"{key}_{V0}"] = h[:, [0, 3]]
            except Exception as ex:
                e = np.nan; print(f"  V0={V0} {key}: diverged ({type(ex).__name__})", flush=True)
            res[key].append(e)
            print(f"  V0={V0} {key:8s}: e={e:.3f}", flush=True)
    np.savez(os.path.join(ROOT, "outputs", "overlay", "contact_restitution_traj.npz"), **traj)
    St = np.array(V0S) * RHO * R / MUF
    np.savez(os.path.join(ROOT, "outputs", "overlay", "contact_restitution.npz"),
             V0=np.array(V0S), St=St, e_elastic=np.array(res["elastic"]),
             e_sls=np.array(res["sls"]))

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.plot(St, res["elastic"], "o-", color="blue", lw=1.8, ms=7, label="elastic")
    ax.plot(St, res["sls"], "s-", color="red", lw=1.8, ms=6, label="viscoelastic (SLS)")
    ax.set_xlabel(r"impact Stokes number  $\mathrm{St}=\rho V_0 R/\mu_f$")
    ax.set_ylabel("coefficient of restitution  $e$")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="lower right"); fig.tight_layout()
    out = os.path.join(ROOT, "paper", "figures", "contact_restitution.png")
    fig.savefig(out, dpi=200); print("saved", out)


if __name__ == "__main__":
    main()
