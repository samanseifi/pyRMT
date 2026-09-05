"""Temporal convergence of the coupled IMEX FSI scheme (referee C2). At a fixed spatial
resolution we step the semi-Lagrangian IMEX integrator (imex-sl, which lifts the viscous AND
advective CFLs) at a geometric sequence of time steps and measure the maximum centroid deviation
from the finest-step run of the SAME scheme at the SAME resolution -- a self-convergence study that
isolates the temporal error. The fitted slope is the observed temporal order. This is the study
that decides whether a higher-order elastic-wave sub-integrator (generalized-alpha / HHT) would
change the coupled order. Writes paper/figures/temporal_convergence.png.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.mac_soft_disc_lid import run
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T_END = 1.5


def _traj(N, dt, integrator="imex-sl"):
    return run(t_end=T_END, write=False, N=N, dt=dt, integrator=integrator)


def _cerr(tr, ref):
    tt = tr[:, 0]
    cx = np.interp(tt, ref[:, 0], ref[:, 1]); cy = np.interp(tt, ref[:, 0], ref[:, 2])
    return float(np.max(np.hypot(tr[:, 1] - cx, tr[:, 2] - cy)))


def main():
    import matplotlib; matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks")); from _plotstyle import apply as _s; _s()
    import matplotlib.pyplot as plt
    N = 96
    dx = 1.0 / N
    dt0 = min(0.3 * dx, 0.2 * dx * dx / 0.01)      # explicit binding CFL as the base step
    facs = np.array([1, 2, 4, 8, 16])
    dt_ref = dt0 / 2.0                              # finest = self-convergence reference
    print(f"N={N} dt0={dt0:.3e} dt_ref={dt_ref:.3e}", flush=True)
    ref = _traj(N, dt_ref)
    dts, errs = [], []
    for f in facs:
        dt = f * dt0
        tr = _traj(N, dt)
        e = _cerr(tr, ref)
        dts.append(dt); errs.append(e)
        print(f"  dt={dt:.3e} ({f:2d}x dt0)  err={e:.3e}", flush=True)
    dts = np.array(dts); errs = np.array(errs)
    np.savez(os.path.join(ROOT, "outputs", "overlay", "temporal_conv.npz"),
             dts=dts, errs=errs, N=N, dt_ref=dt_ref)
    good = errs > 1e-12
    p = np.polyfit(np.log(dts[good]), np.log(errs[good]), 1)[0]
    print(f"  fitted temporal order p = {p:.2f}", flush=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.loglog(dts, errs, "o-", color="red", lw=1.7, ms=7, label=f"imex-sl ($N={N}$), $p\\approx{p:.2f}$")
    ax.loglog(dts, errs[0] * (dts / dts[0]) ** 1, "--", color="0.45", lw=1.1, label="$O(\\Delta t)$")
    ax.loglog(dts, errs[0] * (dts / dts[0]) ** 2, ":", color="0.45", lw=1.1, label="$O(\\Delta t^2)$")
    ax.set_xlabel("time step $\\Delta t$")
    ax.set_ylabel("max centroid error vs.\\ fine-$\\Delta t$ reference")
    ax.set_title(f"Temporal convergence of the coupled IMEX FSI scheme ($N={N}$)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = os.path.join(ROOT, "paper", "figures", "temporal_convergence.png")
    fig.savefig(out, dpi=200); print("saved", out)


if __name__ == "__main__":
    main()
