"""Coefficient of restitution of the colliding-disc contact vs. impact Stokes number, for
elastic and viscoelastic (SLS) discs -- a semi-quantitative comparison with the
elastohydrodynamic-collision picture (Davis, Serayssol & Hinch, JFM 1986).

Theory / asymptotic targets the test approaches:
  * e -> 1 as St -> infinity: a lossless elastic collision conserves kinetic energy, so as
    the fluid becomes negligible (mu_f -> 0, high Stokes) the elastic restitution approaches
    unity. This is the upper bound the elastic curve rises toward.
  * e -> 0 below a critical Stokes number St_c: the lubrication/fluid dissipation fully damps
    a slow impact (no rebound).
  * viscoelastic (SLS) e < elastic e at the same St: the Maxwell branch dissipates part of
    the impact; the two converge only at high St, where the collision is too brief to relax.

The centroid gap is measured with a smooth (Heaviside-weighted) centroid (see
contact_collision.run), so e = |max separation rate| / |max approach rate| is free of the
staircase-mask noise that a hard phi<=0 mask would inject. St = rho * V0 * R / mu_f.

Panel (a): e vs St (elastic + SLS) at the fine grid, with the e=1 asymptote.
Panel (b): grid convergence of e (N = 64, 96, 128, 160) at a fixed impact -- e approaches a
           grid-independent limit, i.e. the measured number is a real limit, not grid noise.

Writes paper/figures/contact_restitution.png.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.contact_collision import run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_MAIN, ETA, SEP, R, RHO = 128, 2.0, 0.30, 0.12, 1.0
# The impact speed cannot be pushed past V0 ~ 1.3 (the collision becomes violent and diverges),
# so the Stokes number St = rho*V0*R/mu_f is swept to high values by DECREASING the fluid
# viscosity mu_f at a fixed, stable impact speed -- physically the mu_f -> 0 (St -> infinity)
# limit in which the fluid dissipation vanishes and the elastic restitution approaches unity.
V0_FIX = 0.9
MUF_SWEEP = [0.012, 0.008, 0.005, 0.003, 0.002, 0.0012, 0.0008, 0.0005, 0.0003]  # -> St ~ 9..360
MUF_CONV = 0.005            # grid-convergence sweep at this mu_f (St ~ 22)
NS = [64, 96, 128, 160]
EL = dict(model="elastic", mu_s=0.8)
VE = dict(model="sls", mu_s=0.2, G_ve=0.6, tau=0.2)


def restitution(h):
    """e = |max separation rate after closest approach| / |max approach rate before it|,
    on the smooth-centroid gap (contact_collision uses a Heaviside-weighted centroid).
    Edges are trimmed so the closest-approach index is found in the interior -- robust for
    the weak near-critical collisions, where the rebound is slow but real."""
    t, gap = h[:, 0], h[:, 3]
    n = len(gap)
    if n < 12:
        return np.nan
    lo, hi = 3, n - 3
    imin = lo + int(np.argmin(gap[lo:hi]))    # closest approach, interior
    dg = np.gradient(gap, t)
    v_in = -np.min(dg[lo:imin + 1]) if imin > lo else np.nan
    v_out = np.max(dg[imin:hi]) if imin < hi - 1 else np.nan
    if not np.isfinite(v_in) or v_in <= 1e-4:
        return np.nan
    return max(0.0, v_out) / max(v_in, 1e-12)


def one(N, V0, muf, cfg):
    t_end = min(2.4, 0.6 + 1.2 / V0)
    try:
        h = run(N=N, t_end=t_end, V0=V0, mu_f=muf, eta=ETA, sep=SEP, R=R,
                integrator="imex", stab=False, **cfg)
        return restitution(h), h[:, [0, 3]]
    except Exception as ex:
        print(f"    N={N} V0={V0} muf={muf}: diverged ({type(ex).__name__})", flush=True)
        return np.nan, None


def main():
    import matplotlib; matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    from _plotstyle import apply as _style; _style()
    import matplotlib.pyplot as plt

    # ---- panel (a): e vs St (swept to high St via decreasing mu_f) at the fine grid ----
    res = {"elastic": [], "sls": []}; traj = {}
    for muf in MUF_SWEEP:
        St = RHO * V0_FIX * R / muf
        for key, cfg in (("elastic", EL), ("sls", VE)):
            e, tg = one(N_MAIN, V0_FIX, muf, cfg)
            res[key].append(e)
            if tg is not None:
                traj[f"{key}_muf{muf}"] = tg
            print(f"  [a] muf={muf} St={St:.0f} {key:8s}: e={e:.3f}", flush=True)

    # ---- panel (b): grid convergence at a fixed impact ----
    conv = {"elastic": [], "sls": []}
    for N in NS:
        for key, cfg in (("elastic", EL), ("sls", VE)):
            e, tg = one(N, V0_FIX, MUF_CONV, cfg)
            conv[key].append(e)
            if tg is not None:
                traj[f"conv_{key}_{N}"] = tg
            print(f"  [b] N={N} {key:8s}: e={e:.3f}", flush=True)

    St = RHO * V0_FIX * R / np.array(MUF_SWEEP)
    np.savez(os.path.join(ROOT, "outputs", "overlay", "contact_restitution_traj.npz"), **traj)
    np.savez(os.path.join(ROOT, "outputs", "overlay", "contact_restitution.npz"),
             muf=np.array(MUF_SWEEP), St=St, e_elastic=np.array(res["elastic"]),
             e_sls=np.array(res["sls"]), NS=np.array(NS), V0_fix=V0_FIX,
             conv_elastic=np.array(conv["elastic"]), conv_sls=np.array(conv["sls"]))

    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.4))
    # (a) -- log-x St axis, extended toward the St -> infinity elastic limit
    ax[0].axhline(1.0, color="0.5", ls="--", lw=1.2)
    ax[0].text(St.min(), 0.965, r"lossless elastic limit  $e\!\to\!1$  ($\mathrm{St}\!\to\!\infty$)",
               fontsize=10, color="0.4", va="top")
    ax[0].semilogx(St, res["elastic"], "o-", color="blue", lw=1.8, ms=7, label="elastic")
    ax[0].semilogx(St, res["sls"], "s-", color="red", lw=1.8, ms=6, label="viscoelastic (SLS)")
    ax[0].set_xlabel(r"impact Stokes number  $\mathrm{St}=\rho V_0 R/\mu_f$")
    ax[0].set_ylabel("coefficient of restitution  $e$")
    ax[0].set_ylim(0.0, 1.05); ax[0].set_title(r"(a) restitution vs.\ Stokes number")
    ax[0].legend(loc="upper left")
    # (b) -- log-x resolution axis
    ax[1].semilogx(NS, conv["elastic"], "o-", color="blue", lw=1.8, ms=7, label="elastic")
    ax[1].semilogx(NS, conv["sls"], "s-", color="red", lw=1.8, ms=6, label="viscoelastic (SLS)")
    ax[1].set_xlabel("grid resolution  $N$"); ax[1].set_ylabel("coefficient of restitution  $e$")
    ax[1].set_xticks(NS); ax[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax[1].set_ylim(bottom=0.0)
    stc = RHO * V0_FIX * R / MUF_CONV
    ax[1].set_title(rf"(b) grid convergence  ($\mathrm{{St}}={stc:.0f}$)")
    ax[1].legend(loc="best")
    fig.tight_layout()
    out = os.path.join(ROOT, "paper", "figures", "contact_restitution.png")
    fig.savefig(out, dpi=200); print("saved", out)


if __name__ == "__main__":
    main()
