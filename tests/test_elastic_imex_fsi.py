"""Stage-2b: trapezoidal implicit-elastic coupling wired into the soft-disc FSI loop.

The elastic force enters the implicit solve and the reference map is advanced with the
implicit-midpoint velocity (required for stability, per the wave analysis), solved with
the preconditioned CG. Two checks:

  1. CORRECTNESS/REGRESSION -- at matched dt the trapezoidal implicit-elastic integrator
     reproduces the explicit soft-disc centroid trajectory (the Sugiyama-validated case).
  2. STABILITY -- it runs a stiff disc to completion.

Honest scope note: the lid-driven case is *advection*-limited (the lid drives flow at
U=1 and momentum advection stays explicit), so lifting the elastic-wave CFL does not by
itself reduce dt here; the elastic-CFL lift is demonstrated in the quiescent standing
wave (tests/test_implicit_elastic.py). Realising a net speedup in flow-driven FSI
additionally requires implicit advection (future work).
"""
import numpy as np
from benchmarks.mac_soft_disc_lid import run


def _reached(tr, t_end):
    return len(tr) > 0 and tr[-1, 0] >= t_end - 1e-9


def test_elastic_imex_fsi_reproduces_explicit():
    """REGRESSION: trapezoidal implicit-elastic reproduces explicit centroid at matched dt."""
    N, t_end = 48, 0.8
    dx = 1.0 / N
    dt = 0.2 * dx * dx / 0.01                       # matched (viscous-limited) dt
    te = run(N=N, t_end=t_end, dt=dt, integrator="explicit", write=False)
    tl = run(N=N, t_end=t_end, dt=dt, integrator="imex-elastic", write=False)
    n = min(len(te), len(tl))
    assert n > 5, "a run terminated early"
    dcx = np.abs(te[:n, 1] - tl[:n, 1]).max()
    dcy = np.abs(te[:n, 2] - tl[:n, 2]).max()
    assert dcx < 5e-3 and dcy < 5e-3, f"centroid drift dcx={dcx:.2e} dcy={dcy:.2e}"


def test_elastic_imex_fsi_stable_stiff_disc():
    """The trapezoidal implicit-elastic integrator runs a stiff disc stably to t_end
    (energy-conserving elastic coupling; no artificial damping blow-up)."""
    N, t_end = 48, 0.5
    dx = 1.0 / N
    tl = run(N=N, t_end=t_end, dt=0.3 * dx / 1.0, integrator="imex-elastic",
             mu_s=8.0, write=False)
    assert _reached(tl, t_end) and np.all(np.isfinite(tl[-1]))
