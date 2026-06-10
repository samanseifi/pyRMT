"""Solid Cauchy stress sanity checks (neo-Hookean from the reference map)."""
import numpy as np
from pyRMT.functions import create_grid, solid_cauchy_stress


def _disc_phi(X, Y, R=0.25):
    return np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) - R


def test_undeformed_identity_zero_shear_J_one():
    """Identity reference map (xi = x) => F = I, b = I, J = 1, zero shear."""
    N = 65
    X, Y, dx, dy = create_grid(N, N, 1.0, 1.0)
    phi = _disc_phi(X, Y)
    X1, X2 = X.copy(), Y.copy()                 # identity map everywhere
    sxx, sxy, syy, J = solid_cauchy_stress(X1, X2, dx, dy, mu_s=1.0, kappa=0.0, phi=phi)
    solid = phi <= 0
    # b = I -> sigma = mu_s * I (no kappa): sxx=syy=mu_s, sxy=0
    assert np.allclose(sxx[solid], 1.0, atol=1e-6)
    assert np.allclose(syy[solid], 1.0, atol=1e-6)
    assert np.allclose(sxy[solid], 0.0, atol=1e-6)
    assert np.allclose(J[solid], 1.0, atol=1e-6)


def test_rigid_translation_unchanged_stress():
    """A rigid translation of the body (xi = x - c) still has F = I => same stress."""
    N = 65
    X, Y, dx, dy = create_grid(N, N, 1.0, 1.0)
    phi = _disc_phi(X, Y)
    X1, X2 = X - 0.1, Y + 0.05                  # translated identity
    sxx, sxy, syy, J = solid_cauchy_stress(X1, X2, dx, dy, 1.0, 0.0, phi)
    solid = phi <= 0
    assert np.allclose(J[solid], 1.0, atol=1e-6)
    assert np.allclose(sxx[solid], 1.0, atol=1e-6)
    assert np.allclose(sxy[solid], 0.0, atol=1e-6)


def test_uniform_stretch_known_stress():
    """xi = (x/lam, y) => deformation stretches x by lam. F = diag(lam,1),
    b = diag(lam^2,1), J = lam. Check sxx = mu*lam^2, syy = mu."""
    N = 81
    X, Y, dx, dy = create_grid(N, N, 1.0, 1.0)
    phi = _disc_phi(X, Y)
    lam = 1.5
    X1, X2 = X / lam, Y.copy()                  # grad xi = diag(1/lam, 1)
    sxx, sxy, syy, J = solid_cauchy_stress(X1, X2, dx, dy, mu_s=2.0, kappa=0.0, phi=phi)
    solid = phi <= 0
    assert np.allclose(J[solid], lam, atol=1e-6)
    assert np.allclose(sxx[solid], 2.0 * lam**2, atol=1e-6)
    assert np.allclose(syy[solid], 2.0, atol=1e-6)
    assert np.allclose(sxy[solid], 0.0, atol=1e-6)


def test_detg_clamp_bounds_J():
    """With a corrupt (near-singular) map, the detg_clamp bounds J in band mode."""
    N = 49
    X, Y, dx, dy = create_grid(N, N, 1.0, 1.0)
    phi = _disc_phi(X, Y)
    # strong compression xi = 10*x => detG=10 => J=0.1; clamp C=3 => J>=1/3
    X1, X2 = 10.0 * X, Y.copy()
    _, _, _, J = solid_cauchy_stress(X1, X2, dx, dy, 1.0, 0.0, phi,
                                      w_cut=2 * dx, detg_clamp=3.0)
    solid = phi <= 0
    assert J[solid].min() >= 1.0 / 3.0 - 1e-9
    assert J[solid].max() <= 3.0 + 1e-9
