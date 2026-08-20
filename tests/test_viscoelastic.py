"""Unit tests for the viscoelastic (upper-convected Maxwell) constitutive update."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.viscoelastic import (ucm_local_step, viscoelastic_stress,
                                logconf_local_step, logconf_local_step_strang,
                                relax_exact, sym_exp, sym_log)


def test_step_strain_relaxation():
    """sigma_xy relaxes as G*gamma*exp(-t/tau) after a held step shear strain."""
    G, tau, gamma, dt = 1.0, 2.0, 0.5, 2e-4
    b11, b12, b22 = 1.0 + gamma**2, gamma, 1.0      # F F^T for shear gamma
    for k in range(1, int(round(8.0 / dt)) + 1):
        b11, b12, b22 = ucm_local_step(b11, b12, b22, 0., 0., 0., 0., tau, dt)
        assert abs(b12 - gamma * np.exp(-k * dt / tau)) < 1e-4


def test_steady_shear_viscometric():
    """UCM steady simple shear: b_xy=tau*gdot, b_xx=1+2 tau^2 gdot^2, N1=2 G tau^2 gdot^2."""
    G, tau, gdot, dt = 1.0, 2.0, 0.3, 2e-4
    b11, b12, b22 = 1.0, 0.0, 1.0
    for _ in range(int(round(80.0 / dt))):
        b11, b12, b22 = ucm_local_step(b11, b12, b22, 0., gdot, 0., 0., tau, dt)
    assert abs(b12 - tau * gdot) < 1e-3
    assert abs(b11 - (1.0 + 2.0 * tau**2 * gdot**2)) < 1e-3
    assert abs(b22 - 1.0) < 1e-3
    _, sxy, _ = viscoelastic_stress(b11, b12, b22, G)
    N1 = G * (b11 - b22)
    assert abs(sxy - G * tau * gdot) < 1e-3
    assert abs(N1 - 2.0 * G * tau**2 * gdot**2) < 1e-3


def test_elastic_limit():
    """tau -> infinity: b_e tracks F F^T exactly under constant L."""
    gdot, dt = 0.4, 2e-4
    b11, b12, b22 = 1.0, 0.0, 1.0
    F = np.array([[1.0, 0.0], [0.0, 1.0]]); L = np.array([[0.0, gdot], [0.0, 0.0]])
    for _ in range(int(round(2.0 / dt))):
        b11, b12, b22 = ucm_local_step(b11, b12, b22, 0., gdot, 0., 0., np.inf, dt)
        k1 = L @ F; k2 = L @ (F + dt * k1); F = F + 0.5 * dt * (k1 + k2)
    bb = F @ F.T
    assert abs(b11 - bb[0, 0]) < 1e-6
    assert abs(b12 - bb[0, 1]) < 1e-6
    assert abs(b22 - bb[1, 1]) < 1e-6


# ── log-conformation (psi = log b_e): same analytical responses, SPD-safe ──

def test_logconf_step_relaxation():
    tau, gamma, dt = 2.0, 0.5, 2e-4
    p11, p12, p22 = sym_log(1.0 + gamma**2, gamma, 1.0)
    for k in range(1, int(round(8.0 / dt)) + 1):
        p11, p12, p22 = logconf_local_step(p11, p12, p22, 0., 0., 0., 0., tau, dt)
        _, b12, _ = sym_exp(p11, p12, p22)
        assert abs(b12 - gamma * np.exp(-k * dt / tau)) < 1e-4


def test_logconf_steady_shear():
    G, tau, gdot, dt = 1.0, 2.0, 0.3, 2e-4
    p11, p12, p22 = sym_log(1.0, 0.0, 1.0)
    for _ in range(int(round(80.0 / dt))):
        p11, p12, p22 = logconf_local_step(p11, p12, p22, 0., gdot, 0., 0., tau, dt)
    b11, b12, b22 = sym_exp(p11, p12, p22)
    assert abs(b12 - tau * gdot) < 1e-3
    assert abs(b11 - (1.0 + 2.0 * tau**2 * gdot**2)) < 1e-3
    assert abs(G * (b11 - b22) - 2.0 * G * tau**2 * gdot**2) < 1e-3


def test_logconf_spd_under_extreme_shear():
    """b_e = exp(psi) stays SPD even at large strain (the reason for log-conf)."""
    p11, p12, p22 = sym_log(1.0, 0.0, 1.0)
    for _ in range(20000):
        p11, p12, p22 = logconf_local_step(p11, p12, p22, 0., 5.0, 0., 0., np.inf, 1e-3)
    b11, b12, b22 = sym_exp(p11, p12, p22)
    assert b11 > 0 and b22 > 0 and (b11 * b22 - b12**2) > 0   # SPD
    assert np.isfinite(b11) and np.isfinite(b12)


# ── Strang exact-relaxation integrator (backlog #14.2) ───────────────────────

def _psi_shear(gamma):
    """psi = log(F F^T) for a step shear strain gamma."""
    return sym_log(1.0 + gamma**2, gamma, 1.0)


def test_relax_exact_matches_closed_form():
    """relax_exact reproduces b_e <- I + (b_e-I)exp(-dt/tau) exactly (scalar and
    field), and returns psi unchanged for tau=inf."""
    b11, b12, b22 = 3.0, 0.8, 1.5
    p = sym_log(b11, b12, b22)
    dt, tau = 0.7, 0.4
    q = relax_exact(*p, tau, dt)
    e = np.exp(-dt / tau)
    exp_b = (1.0 + (b11 - 1.0) * e, b12 * e, 1.0 + (b22 - 1.0) * e)
    got_b = sym_exp(*q)
    assert all(abs(got_b[i] - exp_b[i]) < 1e-12 for i in range(3))
    assert relax_exact(*p, np.inf, dt) == p        # tau=inf -> identity


def test_strang_reduces_to_explicit_stretch_when_tau_inf():
    """tau=inf: the Strang step is byte-for-byte the explicit pure-stretch step
    (both relaxation half-steps are identities) -> exact neo-Hookean-limit."""
    rng = np.random.default_rng(3)
    p = [rng.standard_normal((8, 8)) * 0.1 for _ in range(3)]
    Lc = rng.standard_normal(4) * 0.3
    a = logconf_local_step_strang(p[0], p[1], p[2], *Lc, np.inf, 0.01)
    b = logconf_local_step(p[0], p[1], p[2], *Lc, np.inf, 0.01)
    assert all(np.max(np.abs(a[i] - b[i])) < 1e-14 for i in range(3))


def test_strang_step_strain_exact_at_any_dt():
    """Held step strain (L=0): b_xy(t)=gamma exp(-t/tau). The Strang exact-relaxation
    step reproduces it to machine precision for ANY dt -- including dt >> tau where
    the explicit step diverges."""
    gamma, tau, T = 1.0, 0.1, 1.0
    p0 = _psi_shear(gamma)
    for dt in (0.01, 0.2, 0.5):                     # dt/tau up to 5
        p = list(p0); n = int(round(T / dt))
        for _ in range(n):
            p = list(logconf_local_step_strang(p[0], p[1], p[2], 0, 0, 0, 0, tau, dt))
        assert abs(sym_exp(*p)[1] - gamma * np.exp(-n * dt / tau)) < 1e-10


def test_strang_stable_where_explicit_diverges():
    """At dt/tau=2 the explicit log-conformation relaxation is wildly inaccurate on
    a stretched state; the Strang step stays exact -- the new-capability headline."""
    gamma, tau, T, dt = 1.0, 0.1, 1.0, 0.2         # dt/tau = 2
    p0 = _psi_shear(gamma)
    exact = gamma * np.exp(-T / tau)
    pe = list(p0); ps = list(p0); n = int(round(T / dt))
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(n):
            pe = list(logconf_local_step(pe[0], pe[1], pe[2], 0, 0, 0, 0, tau, dt))
    for _ in range(n):
        ps = list(logconf_local_step_strang(ps[0], ps[1], ps[2], 0, 0, 0, 0, tau, dt))
    assert abs(sym_exp(*pe)[1] - exact) > 1e-2      # explicit inaccurate
    assert abs(sym_exp(*ps)[1] - exact) < 1e-10     # strang exact


def test_strang_steady_shear_reproduces_viscometric():
    """REGRESSION vs the explicit result: Strang steady simple shear gives the same
    UCM viscometric functions b_xy=tau gdot, b_xx=1+2 tau^2 gdot^2, b_yy=1."""
    tau, gdot, dt = 2.0, 0.3, 2e-3
    p = list(sym_log(1.0, 0.0, 1.0))
    for _ in range(int(round(80.0 / dt))):
        p = list(logconf_local_step_strang(p[0], p[1], p[2], 0.0, gdot, 0.0, 0.0, tau, dt))
    b11, b12, b22 = sym_exp(*p)
    assert abs(b12 - tau * gdot) < 2e-3
    assert abs(b11 - (1.0 + 2.0 * tau**2 * gdot**2)) < 5e-3
    assert abs(b22 - 1.0) < 2e-3


def test_strang_matches_explicit_on_taylor_green_field_small_dt():
    """REGRESSION on a real flow: on the Taylor-Green strain field the Strang and
    explicit steppers agree across the whole field at small dt (new method
    reproduces old physics), and the Strang b_e stays SPD."""
    from benchmarks.viscoelastic_taylor_green import run_compare
    diff, _me_exp, me_str = run_compare(N=48, tau=0.5, dt=5e-3, T=2.0)
    assert diff < 1e-3, f"strang vs explicit TG-field diff {diff:.2e} too large"
    assert me_str > 0.0, "strang lost positive-definiteness"


def test_strang_preserves_spd_at_large_dt_on_taylor_green():
    """NEW CAPABILITY: on the TG field at dt/tau=2.5 the Strang b_e stays SPD."""
    from benchmarks.viscoelastic_taylor_green import run_compare
    _diff, _me_exp, me_str = run_compare(N=48, tau=0.2, dt=0.5, T=2.0)
    assert me_str > 0.0
