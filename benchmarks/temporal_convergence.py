"""Temporal convergence of the FSI integrators on the soft-disc-in-lid case at fixed
resolution (N=64): max centroid error vs a fine-time-step explicit reference, as a
function of Delta t. Explicit runs only up to its CFL; IMEX-SL runs well beyond it.
The split IMEX-SL is first order in Delta t and levels off at the semi-Lagrangian
numerical-diffusion floor at large steps.

Usage: python benchmarks/temporal_convergence.py  ->  data/temporal_convergence.png
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.mac_soft_disc_lid import run

N, T_END = 64, 3.0


def _traj(**kw):
    return run(N=N, t_end=T_END, write=False, **kw)


def _cerr(tr, ref):
    tt = tr[:, 0]
    cx = np.interp(tt, ref[:, 0], ref[:, 1]); cy = np.interp(tt, ref[:, 0], ref[:, 2])
    return float(np.max(np.hypot(tr[:, 1] - cx, tr[:, 2] - cy)))


def main():
    import matplotlib; matplotlib.use("Agg")
    from _plotstyle import apply as _style; _style()
    import matplotlib.pyplot as plt

    dx = 1.0 / N
    dt_ex = min(0.3 * dx, 0.2 * dx * dx / 0.01)      # explicit CFL
    ref = _traj(dt=0.125 * dt_ex, integrator="explicit")   # fine-time reference

    expl = [(f, _cerr(_traj(dt=f * dt_ex, integrator="explicit"), ref)) for f in (0.25, 0.5, 1.0)]
    imexsl = [(f, _cerr(_traj(dt=f * dt_ex, integrator="imex-sl"), ref)) for f in (1.0, 2.0, 4.0, 8.0)]
    print("explicit:", [(f, f"{e:.2e}") for f, e in expl])
    print("imex-sl :", [(f, f"{e:.2e}") for f, e in imexsl])

    fig, ax = plt.subplots(figsize=(6.2, 4.7))
    fe, ee = zip(*expl); fs, es = zip(*imexsl)
    dte = np.array(fe) * dt_ex; dts = np.array(fs) * dt_ex
    ax.loglog(dte, ee, "o-", color="C0", label="explicit")
    ax.loglog(dts, es, "^-", color="C3", label="IMEX-SL")
    # O(dt) and O(dt^2) guides anchored at the IMEX-SL mid-point
    d0 = dts[1]; e0 = es[1]
    ax.loglog(dts, e0 * (dts / d0)**1, "k--", lw=1, label="$O(\\Delta t)$")
    ax.loglog(dte, ee[0] * (dte / dte[0])**2, "k:", lw=1, label="$O(\\Delta t^2)$")
    ax.set_xlabel(r"$\Delta t$"); ax.set_ylabel("max centroid error vs fine-$\\Delta t$ reference")
    ax.set_title(f"Temporal convergence, soft disc in lid ($N={N}$)")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data, exist_ok=True)
    out = os.path.join(data, "temporal_convergence.png"); fig.savefig(out); print("saved", out)
    return dict(explicit=expl, imexsl=imexsl)


if __name__ == "__main__":
    main()
