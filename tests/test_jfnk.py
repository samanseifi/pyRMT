"""Stage-1 proof of the fully-monolithic route: Jacobian-free Newton-Krylov (JFNK)
backward-Euler incompressible Navier-Stokes on Taylor-Green.

Unlike the semi-Lagrangian monolithic (which has an O(1) diffusion floor at large dt), the
JFNK solve treats advection implicitly and consistently, so it is accurate AND
unconditionally stable -- the property the fully-monolithic viscoelastic solver needs.
"""
import numpy as np
from pyRMT.jfnk import step

L = 2 * np.pi


def _faces(N):
    dx = L / N; i = np.arange(N); xf = i * dx; xc = (i + 0.5) * dx
    return dx, np.meshgrid(xf, xc), np.meshgrid(xc, xf)


def _tg(g, nu, t, c):
    X, Y = g; e = np.exp(-2 * nu * t)
    return (-np.cos(X) * np.sin(Y) * e) if c == "u" else (np.sin(X) * np.cos(Y) * e)


def _run(N, nu, T, dt):
    dx, gu, gv = _faces(N)
    u = _tg(gu, nu, 0.0, "u"); v = _tg(gv, nu, 0.0, "v")
    ns = int(round(T / dt)); last = {}
    for _ in range(ns):
        u, v, _p = step(u, v, nu, dx, dx, dt, info=last)
    ue = _tg(gu, nu, ns * dt, "u"); ve = _tg(gv, nu, ns * dt, "v")
    err = np.sqrt(np.mean((u - ue) ** 2 + (v - ve) ** 2)) if np.all(np.isfinite(u)) else np.inf
    return err, last


def test_jfnk_matches_analytic_small_dt():
    """JFNK reproduces the analytic Taylor-Green decay and Newton drives the residual down."""
    err, info = _run(64, nu=0.05, T=0.5, dt=0.05)
    assert err < 5e-4, f"JFNK error {err:.2e}"
    assert info["res"] <= 1e-6 * max(info["res0"], 1.0), "Newton did not converge"


def test_jfnk_accurate_and_stable_far_above_advection_cfl():
    """At 5x the advection CFL the JFNK solve stays accurate (error ~1e-3, NOT the O(1)
    semi-Lagrangian diffusion floor) and stable -- implicit advection done consistently."""
    dx = L / 64
    dt = 5.0 * dx / 1.0                          # ~5x the advection CFL (U_max ~ 1)
    err, info = _run(64, nu=0.05, T=1.0, dt=dt)
    assert np.isfinite(err) and err < 5e-3, f"JFNK large-dt error {err:.2e}"
    assert info["newton_iters"] <= 8, "Newton should converge in a few iterations"


def test_jfnk_newton_quadratic_ish_convergence_one_step():
    """A single JFNK step reduces the nonlinear residual by many orders (Newton works)."""
    dx = L / 48
    u = _tg(_faces(48)[1], 0.05, 0.0, "u"); v = _tg(_faces(48)[2], 0.05, 0.0, "v")
    info = {}
    step(u, v, 0.05, dx, dx, 0.1, info=info)
    assert info["res"] < 1e-6 * info["res0"], "residual not reduced enough"


def test_jfnk_elastic_coupled_newton_converges_one_step():
    """Stage 2d-2 (WIP): the fully-coupled (u,v,p,xi1,xi2) neo-Hookean Newton solve
    converges for a SINGLE step from rest. NOTE: sustained runs currently stall (Newton
    stops reducing the residual) because the block-independent preconditioner is too weak
    for the coupled elastic+xi system; a proper physics-based preconditioner (one
    operator-split sweep) + line-search Newton is required. This test only guards the
    single-step convergence, not sustained integration."""
    from pyRMT.jfnk import step_elastic
    N = 16; dx = L / N
    xc = (np.arange(N) + 0.5) * dx; Xc, Yc = np.meshgrid(xc, xc)
    x1 = Xc.copy(); x2 = Yc - 0.05 * np.sin(Xc)
    u = np.zeros((N, N)); v = np.zeros((N, N))
    info = {}
    u, v, p, x1, x2 = step_elastic(u, v, x1, x2, 0.0, dx, dx, 0.05, mu_s=1.0, info=info)
    assert info["res"] < 1e-6 * info["res0"], "coupled elastic Newton did not converge"
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(x1))
