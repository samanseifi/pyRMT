"""IMEX wall/lid implicit viscosity (#14.1 extension to Dirichlet BCs via a
matrix-free CG Helmholtz) and its FSI regression.

  * the CG Helmholtz is consistent with the ghost-cell viscous stencil;
  * on the lid-driven cavity the IMEX integrator reproduces the Ghia-validated
    steady state (and stays correct above the explicit viscous CFL);
  * on the soft-disc-in-lid-driven FSI case the IMEX integrator reproduces the
    explicit (Sugiyama-validated) centroid trajectory -- the regression requested
    for the wall-bounded solver.
"""
import numpy as np
import pytest


def test_cg_helmholtz_lid_consistency():
    """(I - c*Lap_hom) applied to the CG solution recovers the rhs (SPD solve)."""
    from pyRMT.mac import _lap_u_lid_hom, _cg_helmholtz
    N = 24; dx = dy = 1.0 / N
    rng = np.random.default_rng(2)
    rhs = rng.standard_normal((N, N - 1))
    embed = lambda x: np.pad(x, ((0, 0), (1, 1)))
    c = 0.5
    x = _cg_helmholtz(rhs, lambda w: _lap_u_lid_hom(w, dx, dy), embed, c)
    resid = x - c * _lap_u_lid_hom(embed(x), dx, dy) - rhs
    assert np.abs(resid).max() < 1e-8


def test_pcg_matches_cg_and_cuts_iterations():
    """Preconditioned CG solves the same system as CG (the DST preconditioner does not
    change the answer) but in far fewer iterations, and the iteration count stays small
    as N grows (the fix for the net-speedup bottleneck)."""
    from pyRMT.mac import _lap_u_lid_hom, _cg_helmholtz, _pcg_helmholtz
    embed = lambda x: np.pad(x, ((0, 0), (1, 1)))
    prev_iters = None
    for N in (64, 128):
        dx = dy = 1.0 / N
        rng = np.random.default_rng(4)
        rhs = rng.standard_normal((N, N - 1))
        lap = lambda w: _lap_u_lid_hom(w, dx, dy)
        coef = 2.0e-3 * 0.01                              # dt*nu, a representative value
        x_cg = _cg_helmholtz(rhs, lap, embed, coef, rtol=1e-8)
        cnt = []
        x_pcg = _pcg_helmholtz(rhs, lap, embed, coef, dx, dy, rtol=1e-8, count=cnt)
        assert np.abs(x_cg - x_pcg).max() < 1e-6         # same solution
        assert cnt[0] <= 12                              # few iterations
        prev_iters = cnt[0]
    # N-independent conditioning: iteration count does not blow up with resolution
    assert prev_iters <= 12


def test_imex_lid_reproduces_ghia_and_lifts_cfl():
    """IMEX lid-driven cavity reproduces the explicit Ghia RMS (regression) and
    stays correct at a dt above the explicit viscous CFL."""
    from benchmarks.mac_lid_driven import run
    out = "/tmp/_pyrmt_test_lid"
    rms_exp = run(Re=100, N=32, max_steps=20000, out_root=out, imex=False)
    rms_imex = run(Re=100, N=32, max_steps=20000, out_root=out, imex=True)
    rms_imex_big = run(Re=100, N=32, max_steps=20000, out_root=out, imex=True, dt_fac=5.0)
    assert rms_exp is not None and rms_imex is not None
    assert abs(rms_imex - rms_exp) < 5e-3, f"IMEX RMS {rms_imex:.2e} vs explicit {rms_exp:.2e}"
    assert rms_imex_big < rms_exp + 5e-3, "IMEX at 5x dt drifted from Ghia"


def test_dst_direct_solve_exact_and_matches_cg():
    """The direct DST wall-Helmholtz solve is EXACT for constant viscosity (machine-precision
    operator residual) and agrees with the CG fallback. A rectangular grid (Nx != Ny) guards
    against an axis / transform-type mix-up (u: DST-I in x, DST-II in y; v: swapped)."""
    from pyRMT.mac import (_dst_solve_u, _dst_solve_v, _pcg_helmholtz,
                           _lap_u_lid_hom, _lap_v_lid_hom)
    rng = np.random.default_rng(7)
    for (Ny, Nx) in ((32, 32), (48, 64)):
        dx, dy = 1.0 / Nx, 1.0 / Ny
        coef = 3.7e-2                                    # dt*nu (+ elastic), constant
        # u-component: interior faces (Ny, Nx-1); walls i=0,Nx zeroed
        rhs_u = rng.standard_normal((Ny, Nx - 1))
        emb_u = lambda x: np.pad(x, ((0, 0), (1, 1)))
        xu = _dst_solve_u(rhs_u, coef, dx, dy)
        res_u = xu - coef * _lap_u_lid_hom(emb_u(xu), dx, dy) - rhs_u
        assert np.abs(res_u).max() / np.abs(rhs_u).max() < 1e-10   # exact
        xu_cg = _pcg_helmholtz(rhs_u, lambda w: _lap_u_lid_hom(w, dx, dy), emb_u, coef, dx, dy)
        assert np.abs(xu - xu_cg).max() < 1e-6                     # matches CG fallback
        # v-component: interior faces (Ny-1, Nx); walls j=0,Ny zeroed
        rhs_v = rng.standard_normal((Ny - 1, Nx))
        emb_v = lambda x: np.pad(x, ((1, 1), (0, 0)))
        xv = _dst_solve_v(rhs_v, coef, dx, dy)
        res_v = xv - coef * _lap_v_lid_hom(emb_v(xv), dx, dy) - rhs_v
        assert np.abs(res_v).max() / np.abs(rhs_v).max() < 1e-10
        xv_cg = _pcg_helmholtz(rhs_v, lambda w: _lap_v_lid_hom(w, dx, dy), emb_v, coef, dx, dy)
        assert np.abs(xv - xv_cg).max() < 1e-6


def test_imex_softdisc_reproduces_explicit_centroid():
    """REGRESSION: IMEX reproduces the explicit soft-disc-in-lid centroid trajectory
    (the Sugiyama-validated FSI result) to within grid accuracy."""
    from benchmarks.mac_soft_disc_lid import run
    N, t_end = 48, 1.0
    dt = 0.2 * (1.0 / N) ** 2 / 0.01                # matched (viscous-limited) dt
    te = run(N=N, t_end=t_end, imex=False, dt=dt, write=False)
    ti = run(N=N, t_end=t_end, imex=True, dt=dt, write=False)
    n = min(len(te), len(ti))
    assert n > 5, "trajectory too short (a run diverged)"
    dcx = np.abs(te[:n, 1] - ti[:n, 1]).max()
    dcy = np.abs(te[:n, 2] - ti[:n, 2]).max()
    assert dcx < 6e-3 and dcy < 6e-3, f"centroid drift dcx={dcx:.2e} dcy={dcy:.2e}"
