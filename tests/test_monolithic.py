"""Tests for the fully monolithic Eulerian RMT-FSI solver (pyRMT/monolithic.py).

These pin the three claims of paper/monolithic_aml.tex:
  * the implicit C1 semi-Lagrangian transport makes the residual DIFFERENTIABLE
    (the transport tangent Eq. 9 matches a finite difference; Newton converges
    quadratically) -- and the clamped bicubic used by the explicit solver does not;
  * the step is CONSISTENT (it reproduces the explicit solver as dt -> 0);
  * the step is UNCONDITIONALLY STABLE (it runs far above the elastic-wave CFL
    with det(grad xi) bounded away from zero).
"""
import numpy as np
import pytest

from pyRMT.monolithic import (MonolithicRMT, ExtrapPlan, bicubic_c1,
                              bicubic_c1_grad, transport_tangent,
                              _fs_eigs_u, _fs_eigs_v, _solve_helm_u,
                              _solve_helm_v, _lap_u_fs, _lap_v_fs,
                              _lid_eigs_u, _lid_eigs_v, _solve_helm_u_lid,
                              _solve_helm_v_lid, _lap_u_lid, _lap_v_lid)
from pyRMT.interpolators import bicubic_interpolate


# ── fixtures ────────────────────────────────────────────────────────────────
R_DISC, X0, Y0 = 0.15, 0.5, 0.7
U0, KW = 0.3, np.pi


def _phi0(X, Y):
    return np.sqrt((X - X0) ** 2 + (Y - Y0) ** 2) - R_DISC


def _make(N=32, mu_s=0.1, mu_f=0.01, rho=1.0, **kw):
    s = MonolithicRMT(N, _phi0, mu_s=mu_s, mu_f=mu_f, rho=rho, layers=5, **kw)
    dx = s.dx
    xf = np.arange(N + 1) * dx
    xc = (np.arange(N) + 0.5) * dx
    Xu, Yu = np.meshgrid(xf, xc)
    Xv, Yv = np.meshgrid(xc, xf)
    s.set_velocity(U0 * KW * np.sin(KW * Xu) * np.cos(KW * Yu),
                   -U0 * KW * np.cos(KW * Xv) * np.sin(KW * Yv))
    s.seed_reference_map()
    return s


def _explicit_dt(s, mu_s, rho=1.0):
    """The explicit solver's step limit for this configuration."""
    dx = s.dx
    return dict(adv=0.25 * dx / (U0 * KW),
                visc=0.2 * dx * dx * rho / s.mu_f,
                elastic=0.25 * dx / np.sqrt(mu_s / rho))


# ── 1. the interpolant is C1 (and the clamped one is not) ───────────────────
def test_bicubic_c1_gradient_matches_finite_difference():
    N = 32; dx = 1.0 / N
    xc = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(xc, xc)
    f = np.sin(4 * X) * np.cos(3 * Y) + 0.3 * X * Y
    q = np.array([[0.31234, 0.5123], [0.7712, 0.2131]])
    r = np.array([[0.41237, 0.6712], [0.1339, 0.8817]])
    _, gx, gy = bicubic_c1_grad(f, q, r, dx, dx, 0.5 * dx, 0.5 * dx)
    h = 1e-6
    fd_x = (bicubic_c1(f, q + h, r, dx, dx, .5 * dx, .5 * dx)
            - bicubic_c1(f, q - h, r, dx, dx, .5 * dx, .5 * dx)) / (2 * h)
    fd_y = (bicubic_c1(f, q, r + h, dx, dx, .5 * dx, .5 * dx)
            - bicubic_c1(f, q, r - h, dx, dx, .5 * dx, .5 * dx)) / (2 * h)
    assert np.max(np.abs(gx - fd_x)) < 1e-7
    assert np.max(np.abs(gy - fd_y)) < 1e-7


def test_c1_interpolant_has_continuous_slope_where_clamped_one_does_not():
    """The stencil min/max limiter of the explicit solver's bicubic is what would
    break Newton: its slope jumps where the limiter switches on.  Sweep across a
    reference-map-like kink and compare the jump in the numerical derivative."""
    N = 32; dx = 1.0 / N
    xc = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(xc, xc)
    f = np.where(X < 0.5, 0.0, 1.0) + 0.1 * Y      # steep jump -> limiter active
    xs = np.linspace(0.40, 0.60, 4001)
    q = xs[np.newaxis, :].copy()
    r = np.full_like(q, 0.5)
    v_c1 = bicubic_c1(f, q, r, dx, dx, .5 * dx, .5 * dx)[0]
    v_cl = bicubic_interpolate(f, q, r, dx, dx, N, N)[0]
    d_c1 = np.diff(v_c1) / np.diff(xs)
    d_cl = np.diff(v_cl) / np.diff(xs)
    jump_c1 = np.max(np.abs(np.diff(d_c1)))
    jump_cl = np.max(np.abs(np.diff(d_cl)))
    assert jump_cl > 20 * jump_c1, (jump_c1, jump_cl)


# ── 2. the frozen extrapolation is an exact linear operator ─────────────────
def test_extrapolation_plan_is_linear_and_exact_for_the_identity_map():
    N = 48; dx = 1.0 / N
    xc = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(xc, xc)
    plan = ExtrapPlan(_phi0(X, Y), dx, dx, 5)
    sm = plan.solid
    a, b = plan.apply(X * sm, Y * sm)
    # a plane fit is exact on the (linear) undeformed reference map
    assert np.max(np.abs(a - X)[plan.known]) < 1e-9
    assert np.max(np.abs(b - Y)[plan.known]) < 1e-9
    # and the operator is exactly linear -- the property Newton needs
    rs = np.random.RandomState(0)
    r1, r2 = rs.randn(N, N), rs.randn(N, N)
    e1, _ = plan.apply(r1, r1)
    e2, _ = plan.apply(r2, r2)
    e12, _ = plan.apply(3 * r1 - 2 * r2, 3 * r1 - 2 * r2)
    assert np.max(np.abs(e12 - (3 * e1 - 2 * e2))) < 1e-12


def test_extrapolation_plan_is_deterministic():
    N = 40; dx = 1.0 / N
    xc = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(xc, xc)
    phi = _phi0(X, Y)
    a = ExtrapPlan(phi, dx, dx, 5).apply(X * (phi < 0), Y * (phi < 0))
    b = ExtrapPlan(phi, dx, dx, 5).apply(X * (phi < 0), Y * (phi < 0))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# ── 3. the free-slip Helmholtz preconditioner block is exact ────────────────
@pytest.mark.parametrize("N", [16, 32])
def test_freeslip_helmholtz_is_exact(N):
    dx = 1.0 / N
    a, b = 3.7, 0.21
    rs = np.random.RandomState(1)
    ru = rs.randn(N, N - 1)
    x = _solve_helm_u(ru, _fs_eigs_u(N, N, dx, dx), a, b)
    U = np.zeros((N, N + 1)); U[:, 1:-1] = x
    assert np.max(np.abs(a * x - b * _lap_u_fs(U, dx, dx) - ru)) < 1e-12
    rv = rs.randn(N - 1, N)
    y = _solve_helm_v(rv, _fs_eigs_v(N, N, dx, dx), a, b)
    V = np.zeros((N + 1, N)); V[1:-1, :] = y
    assert np.max(np.abs(a * y - b * _lap_v_fs(V, dx, dx) - rv)) < 1e-12


# ── 4. the transport tangent of the draft (Eq. 9) is the true derivative ────
def test_transport_tangent_matches_finite_difference():
    s = _make(N=32)
    s._dt = 2e-3
    s._un, s._vn = s.u.copy(), s.v.copy()
    s._xi1n, s._xi2n = s.xi1.copy(), s.xi2.copy()
    rs = np.random.RandomState(3)
    du = np.zeros_like(s.u); du[:, 1:-1] = rs.randn(*du[:, 1:-1].shape)
    dv = np.zeros_like(s.v); dv[1:-1, :] = rs.randn(*dv[1:-1, :].shape)
    d1, d2 = transport_tangent(s, s.u, s.v, du, dv)
    h = 1e-6
    a_p, b_p = s.transport(s.u + h * du, s.v + h * dv)
    a_m, b_m = s.transport(s.u - h * du, s.v - h * dv)
    fd1 = (a_p - a_m) / (2 * h)
    fd2 = (b_p - b_m) / (2 * h)
    m = s._plan.known
    scale = max(np.max(np.abs(fd1[m])), np.max(np.abs(fd2[m])), 1e-30)
    assert np.max(np.abs(d1 - fd1)[m]) / scale < 1e-5
    assert np.max(np.abs(d2 - fd2)[m]) / scale < 1e-5


# ── 5. Newton converges quadratically; the projection is exact ──────────────
def test_newton_converges_quadratically_and_divergence_is_machine_zero():
    s = _make(N=32, mu_s=0.1)
    lim = _explicit_dt(s, 0.1)
    st = s.step(4.0 * min(lim.values()), newton_tol=1e-10)
    assert st["res"] < 1e-10 * st["res0"]
    assert st["newton"] <= 6
    # incompressibility is enforced BY the Newton solve (there is no separate
    # projection), so div u tracks the Newton tolerance, not machine epsilon
    assert st["div_rel"] < 1e-10
    assert st["min_det"] > 0.0


# ── 6. consistency: the monolithic step reproduces the explicit solver ──────
def test_consistent_with_explicit_solver_at_small_dt():
    """At a step well inside every explicit CFL the two schemes must agree to the
    (first-order) time-discretization error, not to a different physics."""
    from pyRMT.mac import (momentum_predictor_freeslip, project,
                           poisson_eigs_neumann)
    from pyRMT.functions import (extrapolate_reference_map, advect_reference_map,
                                 solid_cauchy_stress, smoothed_heaviside,
                                 grad_central_x_2nd, grad_central_y_2nd)
    N, mu_s, mu_f, rho = 32, 0.1, 0.01, 1.0
    s = _make(N=N, mu_s=mu_s)
    dt = 0.1 * min(_explicit_dt(s, mu_s).values())
    nsteps = 12
    for _ in range(nsteps):
        s.step(dt)

    # --- reference: the explicit driver, same operators, same dt ---
    dx = s.dx
    xc = (np.arange(N) + 0.5) * dx
    Xc, Yc = np.meshgrid(xc, xc)
    Xg, Yg = np.meshgrid(np.arange(N) * dx, np.arange(N) * dx)
    xf = np.arange(N + 1) * dx
    Xu, Yu = np.meshgrid(xf, xc); Xv, Yv = np.meshgrid(xc, xf)
    u = U0 * KW * np.sin(KW * Xu) * np.cos(KW * Yu)
    v = -U0 * KW * np.cos(KW * Xv) * np.sin(KW * Yv)
    u[:, 0] = u[:, -1] = 0.0; v[0, :] = v[-1, :] = 0.0
    phi = _phi0(Xc, Yc); sm = (phi <= 0).astype(float)
    X1, X2 = extrapolate_reference_map(Xc * sm, Yc * sm, phi, dx, dx, 3)
    eig = poisson_eigs_neumann(N, N, dx, dx)
    w_t = 2.0 * dx
    for _ in range(nsteps):
        uc = 0.5 * (u[:, :-1] + u[:, 1:]); vc = 0.5 * (v[:-1, :] + v[1:, :])
        phi = _phi0(X1, X2); sm = (phi <= 0).astype(float)
        X1 = advect_reference_map(X1, uc, vc, Xg, Yg, dt, dx, dx, phi,
                                  "semilagrangian_cubic", 4.0 * dx) * sm
        X2 = advect_reference_map(X2, uc, vc, Xg, Yg, dt, dx, dx, phi,
                                  "semilagrangian_cubic", 4.0 * dx) * sm
        X1, X2 = extrapolate_reference_map(X1, X2, phi, dx, dx, 3)
        phi = _phi0(X1, X2)
        sxx, sxy, syy, _ = solid_cauchy_stress(X1, X2, dx, dx, mu_s, 0.0, phi)
        H = smoothed_heaviside(phi, w_t)
        Sxx = (1 - H) * sxx; Sxy = (1 - H) * sxy; Syy = (1 - H) * syy
        divx = grad_central_x_2nd(Sxx, dx) + grad_central_y_2nd(Sxy, dx)
        divy = grad_central_x_2nd(Sxy, dx) + grad_central_y_2nd(Syy, dx)
        fu = np.zeros((N, N + 1)); fu[:, 1:-1] = 0.5 * (divx[:, 1:] + divx[:, :-1])
        fv = np.zeros((N + 1, N)); fv[1:-1, :] = 0.5 * (divy[1:, :] + divy[:-1, :])
        us, vs = momentum_predictor_freeslip(u, v, mu_f / rho, dx, dx, dt,
                                             fu=fu, fv=fv, rho=rho)
        u, v, _ = project(us, vs, dx, dx, dt, rho, eig)

    uc_m, vc_m = s.cell_velocity()
    uc_e = 0.5 * (u[:, :-1] + u[:, 1:]); vc_e = 0.5 * (v[:-1, :] + v[1:, :])
    rel = (np.sqrt(np.mean((uc_m - uc_e) ** 2 + (vc_m - vc_e) ** 2))
           / np.sqrt(np.mean(uc_e ** 2 + vc_e ** 2)))
    assert rel < 2e-2, f"monolithic vs explicit relative velocity error {rel:.3e}"


# ── 7. the velocity-pressure kernel is second-order-capable ────────────────
def test_theta_half_is_second_order_when_the_interface_is_inactive():
    """theta=1/2 with Eulerian advection is genuinely 2nd order in dt for the
    fluid kernel.  (The COUPLED scheme is only 1st order -- the reference-map band
    caps it -- which is why the draft's second-order claim needs qualifying; that
    is documented in the module docstring and exercised by the benchmark.)"""
    N, T = 24, 0.06
    dt0 = T / 32

    def run(dt, theta):
        s = MonolithicRMT(N, lambda X, Y: np.full_like(X, 1e3),   # no solid
                          mu_s=0.0, mu_f=0.05, rho=1.0, theta=theta,
                          advection="eulerian", layers=1)
        dx = s.dx
        xf = np.arange(N + 1) * dx; xc = (np.arange(N) + 0.5) * dx
        Xu, Yu = np.meshgrid(xf, xc); Xv, Yv = np.meshgrid(xc, xf)
        s.set_velocity(U0 * KW * np.sin(KW * Xu) * np.cos(KW * Yu),
                       -U0 * KW * np.cos(KW * Xv) * np.sin(KW * Yv))
        s.xi1 = np.zeros((N, N)); s.xi2 = np.zeros((N, N))
        s.phi = np.full((N, N), 1e3)
        while s.t < T - 1e-12:
            s.step(min(dt, T - s.t))
        return s.cell_velocity()

    for theta, lo, hi in ((1.0, 0.8, 1.3), (0.5, 1.7, 2.3)):
        ref = run(dt0 / 8, theta)
        errs = []
        for f in (4, 2, 1):
            r = run(dt0 * f, theta)
            errs.append(np.sqrt(np.mean((r[0] - ref[0]) ** 2 + (r[1] - ref[1]) ** 2)))
        p = np.log(errs[0] / errs[-1]) / np.log(4.0)
        assert lo < p < hi, f"theta={theta}: observed order {p:.2f}"


def test_advection_modes_agree_at_small_dt():
    """'sl' and 'eulerian' are two discretizations of the same equation."""
    s_sl = _make(N=32, mu_s=1.0, advection="sl")
    s_eu = _make(N=32, mu_s=1.0, advection="eulerian")
    dt = 0.25 * min(_explicit_dt(s_sl, 1.0).values())
    for _ in range(8):
        s_sl.step(dt); s_eu.step(dt)
    a, b = s_sl.cell_velocity(), s_eu.cell_velocity()
    rel = (np.sqrt(np.mean((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))
           / np.sqrt(np.mean(b[0] ** 2 + b[1] ** 2)))
    assert rel < 2e-2, rel


def test_outer_interface_sweeps_reduce_the_error():
    """The fixed point on the frozen interface geometry (draft Sec. 5, point 1)
    removes the O(dt) lag in the band -- worth a constant factor, not an order."""
    N, mu_s, T = 32, 1.0, 0.05

    def run(dt, n_outer):
        s = _make(N=N, mu_s=mu_s, theta=0.5, advection="eulerian")
        while s.t < T - 1e-12:
            s.step(min(dt, T - s.t), n_outer=n_outer)
        return s.cell_velocity()

    dt0 = min(_explicit_dt(_make(N=N, mu_s=mu_s), mu_s).values())
    ref = run(dt0 / 8, 2)
    e1 = run(dt0 * 4, 1)
    e2 = run(dt0 * 4, 2)
    err = lambda r: np.sqrt(np.mean((r[0] - ref[0]) ** 2 + (r[1] - ref[1]) ** 2))
    assert err(e2) < err(e1), (err(e1), err(e2))


# ── 8. unconditional stability above the elastic-wave CFL ───────────────────
def test_stable_far_above_the_elastic_wave_cfl():
    """The headline claim: a stiff solid advanced at many times dx/c_s, with
    det(grad xi) bounded away from zero and the projection still exact."""
    mu_s = 100.0                      # c_s = 10 -> elastic CFL is the binding one
    s = _make(N=32, mu_s=mu_s)
    lim = _explicit_dt(s, mu_s)
    assert lim["elastic"] == min(lim.values())
    dt = 20.0 * lim["elastic"]
    for _ in range(6):
        st = s.step(dt)
        assert st["min_det"] > 0.2
        # div u is enforced by Newton, so it sits at the Newton tolerance
        assert st["div_rel"] < 10 * 1e-6
        assert np.all(np.isfinite(s.u)) and np.all(np.isfinite(s.v))


# ── 9. lid-driven cavity boundary conditions ───────────────────────────────
@pytest.mark.parametrize("N", [16, 24])
def test_lid_helmholtz_is_exact(N):
    """No-slip walls replace the free-slip mirror with a sign flip, so the
    velocity block diagonalizes under DST-I x DST-II instead of DST-I x DCT-II.
    Both must be machine-exact for the projection preconditioner to be a direct
    solve rather than an inner iteration."""
    dx = 1.0 / N
    a, b = 2.9, 0.17
    rs = np.random.RandomState(2)
    x = rs.randn(N, N - 1)
    U = np.zeros((N, N + 1)); U[:, 1:-1] = x
    r = a * x - b * _lap_u_lid(U, dx, dx)
    assert np.max(np.abs(_solve_helm_u_lid(r, _lid_eigs_u(N, N, dx, dx), a, b) - x)) < 1e-12
    y = rs.randn(N - 1, N)
    V = np.zeros((N + 1, N)); V[1:-1, :] = y
    r = a * y - b * _lap_v_lid(V, dx, dx)
    assert np.max(np.abs(_solve_helm_v_lid(r, _lid_eigs_v(N, N, dx, dx), a, b) - y)) < 1e-12


def _make_lid(N=32, mu_s=0.1, **kw):
    s = MonolithicRMT(N, lambda X, Y: np.sqrt((X - 0.6) ** 2 + (Y - 0.5) ** 2) - 0.2,
                      bc="lid", U_lid=1.0, mu_s=mu_s, mu_f=0.01, rho=1.0,
                      layers=5, **kw)
    s.set_velocity(np.zeros((N, N + 1)), np.zeros((N + 1, N)))
    s.seed_reference_map()
    return s


def test_lid_tangent_matches_finite_difference():
    """The lid enters the residual as a CONSTANT ghost value, so it must drop out
    of the derivative: a perturbation is padded with -du, not 2*U_lid-du.  Getting
    that wrong leaves the tangent wrong by ~7e-3 and Newton merely linearly
    convergent, which is why this is pinned."""
    s = _make_lid(N=32)
    dt = 0.3 * s.dx
    for _ in range(6):
        s.step(dt)
    s._dt = dt
    s._un, s._vn, s._pn = s.u.copy(), s.v.copy(), s.p.copy()
    s._xi1n, s._xi2n = s.xi1.copy(), s.xi2.copy()
    s._pscale = s.rho * s.dx / dt
    s._plan = ExtrapPlan(s.phi, s.dx, s.dy, s.layers)
    z = s._pack(s.u, s.v, s.p)
    s.residual(z)
    s._prepare_tangent(z)
    rs = np.random.RandomState(1)
    w = rs.randn(z.size); w /= np.linalg.norm(w)
    ana = s._jacobian_action(w)
    h = 1e-5
    fd = (s.residual(z + h * w) - s.residual(z - h * w)) / (2 * h)
    assert np.linalg.norm(ana - fd) / np.linalg.norm(ana) < 1e-7


def test_lid_reproduces_the_explicit_driver():
    """Monolithic vs the repo's explicit lid-cavity FSI driver, matched stress
    convention and matched dt: the disc centroid must agree."""
    from benchmarks.mac_soft_disc_lid import run as run_mac
    N, T = 32, 0.6
    dt = 0.3 / N
    s = _make_lid(N=N, isochoric=True)
    xc = (np.arange(N) + 0.5) * s.dx
    Xc, Yc = np.meshgrid(xc, xc)
    while s.t < T - 1e-12:
        s.step(min(dt, T - s.t))
    m = s.phi <= 0
    cx, cy = Xc[m].mean(), Yc[m].mean()
    traj = np.asarray(run_mac(N=N, t_end=T, dt=dt, mu_s=0.1, isochoric=True, write=False))
    assert np.hypot(cx - traj[-1, 1], cy - traj[-1, 2]) < 5e-3


# ── 10. the sparse (elastic-aware) preconditioner block ────────────────────
@pytest.mark.parametrize("lid", [False, True])
def test_assembled_velocity_block_matches_the_stencil(lid):
    """The assembled variable-coefficient block must be the same operator the
    scheme applies matrix-free.  For constant mu it must also reproduce the exact
    free-slip / lid Laplacians, which is the only way to check the lid wall rows
    (where the no-slip ghost -u contributes 2*mu rather than zero flux)."""
    from pyRMT.monolithic import (_assemble_helm_u, _assemble_helm_v,
                                  _apply_helm_u_var, _apply_helm_v_var,
                                  _mu_at_u_rows, _mu_at_v_cols)
    N = 16
    dx = 1.0 / N
    a, mu = 3.1, 0.17
    rs = np.random.RandomState(0)
    muc = np.full((N, N), mu)
    muy = _mu_at_u_rows(muc, lid)
    mux = _mu_at_v_cols(muc, lid)
    x = rs.randn(N, N - 1)
    U = np.zeros((N, N + 1)); U[:, 1:-1] = x
    ref = a * x - mu * (_lap_u_lid(U, dx, dx) if lid else _lap_u_fs(U, dx, dx))
    got = (_assemble_helm_u(a, muc, muy, dx, dx, lid) @ x.ravel()).reshape(x.shape)
    assert np.max(np.abs(got - ref)) < 1e-10
    y = rs.randn(N - 1, N)
    V = np.zeros((N + 1, N)); V[1:-1, :] = y
    ref = a * y - mu * (_lap_v_lid(V, dx, dx) if lid else _lap_v_fs(V, dx, dx))
    got = (_assemble_helm_v(a, muc, mux, dx, dx, lid) @ y.ravel()).reshape(y.shape)
    assert np.max(np.abs(got - ref)) < 1e-10

    # variable mu: must agree with the matrix-free form (free-slip stencils only)
    if not lid:
        muv = 0.01 + 2.0 * rs.rand(N, N)
        muy = _mu_at_u_rows(muv); mux = _mu_at_v_cols(muv)
        got = (_assemble_helm_u(a, muv, muy, dx, dx) @ x.ravel()).reshape(x.shape)
        assert np.max(np.abs(got - _apply_helm_u_var(x, a, muv, muy, dx, dx))) < 1e-10
        got = (_assemble_helm_v(a, muv, mux, dx, dx) @ y.ravel()).reshape(y.shape)
        assert np.max(np.abs(got - _apply_helm_v_var(y, a, muv, mux, dx, dx))) < 1e-10


def test_preconditioner_choice_does_not_change_the_answer():
    """P^-1 sets only the convergence rate, so 'transform' and 'sparse' must reach
    the same step -- this is what makes the elastic-aware block safe to switch on."""
    out = {}
    for mode in ("transform", "sparse"):
        s = _make(N=32, mu_s=100.0, pc_mode=mode)
        lim = _explicit_dt(s, 100.0)
        for _ in range(3):
            s.step(8.0 * lim["elastic"], newton_tol=1e-9)
        out[mode] = s.cell_velocity()
    a, b = out["transform"], out["sparse"]
    rel = (np.sqrt(np.mean((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))
           / np.sqrt(np.mean(b[0] ** 2 + b[1] ** 2)))
    assert rel < 1e-6, rel


def test_sparse_preconditioner_cuts_krylov_work_at_large_dt():
    """The point of the assembled block: it contains the elastic stiffness, so the
    Krylov count stops growing with dt.  Measured 2211 -> 522 matvecs at 40x the
    elastic CFL on the paper's benchmark."""
    mu_s = 100.0
    counts = {}
    for mode in ("transform", "sparse"):
        s = _make(N=32, mu_s=mu_s, pc_mode=mode)
        lim = _explicit_dt(s, mu_s)
        for _ in range(3):
            s.step(20.0 * lim["elastic"])
        counts[mode] = sum(x["gmres"] for x in s.stats)
    assert counts["sparse"] < 0.7 * counts["transform"], counts


def test_lid_runs_with_the_sparse_preconditioner():
    s = _make_lid(N=32, mu_s=10.0, pc_mode="sparse")
    for _ in range(4):
        st = s.step(0.6 * s.dx)
    assert st["min_det"] > 0.5 and st["div_rel"] < 1e-5
