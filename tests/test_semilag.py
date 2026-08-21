"""Stage-2c: semi-Lagrangian (cubic) momentum advection -- lifts the last explicit CFL
(advection). Combined with implicit viscosity it is unconditionally stable on the
periodic Taylor-Green vortex; cubic interpolation keeps the SL numerical diffusion low.
"""
import numpy as np
from benchmarks.mac_taylor_green_convergence import run_one

L = 2 * np.pi


def test_periodic_cubic_interp_exact_at_nodes():
    """The periodic cubic interpolator reproduces a field at integer indices exactly."""
    from pyRMT.mac import _interp_per
    N = 32
    rng = np.random.default_rng(0)
    f = rng.standard_normal((N, N))
    I, J = np.meshgrid(np.arange(N), np.arange(N))
    got = _interp_per(f, I.astype(float), J.astype(float))
    assert np.max(np.abs(got - f)) < 1e-9


def test_semilag_accurate_at_small_dt():
    """imex-sl (SL advection + implicit viscosity) reproduces the Taylor-Green decay
    accurately at small dt -- cubic SL is not over-diffusive."""
    err, dmax = run_one(64, nu=0.05, T=1.0, dt=0.02, integrator="imex-sl")
    assert err < 8e-3, f"imex-sl small-dt error {err:.2e}"
    assert dmax < 1e-12, "projection not divergence-free"


def test_semilag_stable_above_advection_cfl():
    """At a dt about 2x the advection CFL (dx/U_max) imex-sl stays stable and accurate,
    where explicit central advection is CFL-restricted."""
    N, nu = 64, 0.05
    dx = L / N
    dt = 2.0 * dx / 1.0                         # ~2x the advection CFL (U_max ~ 1)
    err, dmax = run_one(N, nu=nu, T=1.0, dt=dt, integrator="imex-sl")
    assert np.isfinite(err) and err < 6e-2, f"imex-sl large-dt error {err:.2e}"
    assert dmax < 1e-12, "projection not divergence-free"
