"""Work-precision (error vs wall-clock) for the FSI integrators on the validated
soft-disc-in-lid-driven-cavity case (paper Sec. 8.3). For each resolution the accuracy
is measured as the maximum centroid deviation from a fine-time-step explicit reference
at the SAME resolution (so the plot isolates the temporal error each integrator
controls), against the measured wall-clock time.

Explicit carries every CFL; IMEX lifts the viscous limit (which binds at the finer grid,
where dx^2/nu < dx/U); IMEX-SL additionally lifts the advective limit and runs at several
times the explicit step. The semi-monolithic JFNK is velocity-pressure only (pure-fluid
accuracy reported separately in Sec. 9.3) and is not part of this FSI comparison.

Usage: python benchmarks/work_precision.py  ->  data/work_precision.png
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.mac_soft_disc_lid import run

T_END = 1.5


def _traj(**kw):
    t0 = time.perf_counter()
    tr = run(t_end=T_END, write=False, **kw)
    return tr, time.perf_counter() - t0


def _cerr(tr, ref):
    """max centroid deviation of tr vs reference ref (both (t,cx,cy,...)) on common t."""
    tt = tr[:, 0]
    cx = np.interp(tt, ref[:, 0], ref[:, 1]); cy = np.interp(tt, ref[:, 0], ref[:, 2])
    return float(np.max(np.hypot(tr[:, 1] - cx, tr[:, 2] - cy)))


def collect(N, ref):
    dx = 1.0 / N
    dt_ex = min(0.3 * dx, 0.2 * dx * dx / 0.01)     # explicit binding CFL (adv or visc)
    pts = []
    tr, wt = _traj(N=N, dt=dt_ex, integrator="explicit"); pts.append(("explicit", 1, _cerr(tr, ref), wt))
    tr, wt = _traj(N=N, integrator="imex"); pts.append(("imex", 0, _cerr(tr, ref), wt))
    for fac in (2, 4, 6, 8):
        tr, wt = _traj(N=N, dt=fac * dt_ex, integrator="imex-sl")
        pts.append(("imex-sl", fac, _cerr(tr, ref), wt))
    return pts


def main():
    import matplotlib; matplotlib.use("Agg")
    from _plotstyle import apply as _style; _style()
    import matplotlib.pyplot as plt

    # common fine reference: explicit at the finest resolution, natural step
    dxr = 1.0 / 128; dt_ref = min(0.3 * dxr, 0.2 * dxr * dxr / 0.01)
    ref, _ = _traj(N=128, dt=dt_ref, integrator="explicit")
    res = {N: collect(N, ref) for N in (64, 128)}
    for N in res:
        print(f"N={N}:")
        for nm, fac, e, wt in res[N]:
            print(f"  {nm:9s} fac/steps={fac:<6} err={e:.2e}  wall={wt:.2f}s")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    mk = {"explicit": "o", "imex": "s", "imex-sl": "^"}
    col = {64: "C0", 128: "C3"}
    for N in (64, 128):
        for scheme in ("explicit", "imex", "imex-sl"):
            P = [(e, wt) for nm, fac, e, wt in res[N] if nm == scheme and e > 1e-10]
            if not P:
                continue
            e, wt = zip(*sorted(P))
            ax.plot(wt, e, mk[scheme] + "-", color=col[N], ms=7, lw=1.3,
                    label=f"{scheme}, $N={N}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("wall-clock time (s)"); ax.set_ylabel("max centroid error vs fine reference")
    ax.legend(ncol=2, fontsize=9); ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data, exist_ok=True)
    out = os.path.join(data, "work_precision.png"); fig.savefig(out); print("saved", out)
    return res


if __name__ == "__main__":
    main()
