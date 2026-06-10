"""Interpolators, reference-map extrapolation, and strain-energy consistency."""
import numpy as np
from pyRMT.functions import (
    create_grid, extrapolate_reference_map, solid_cauchy_stress,
)
from pyRMT.interpolators import bilinear_interpolate, bicubic_interpolate
from pyRMT.output import compute_strain_energy


def test_bilinear_exact_on_linear_field():
    N = 33
    X, Y, dx, dy = create_grid(N, N, 1.0, 1.0)
    u = 2.0 * X + 3.0 * Y
    # query at shifted interior points
    xq = X[1:-1, 1:-1] + 0.3 * dx
    yq = Y[1:-1, 1:-1] + 0.2 * dy
    out = bilinear_interpolate(u, xq, yq, dx, dy, N, N)
    exact = 2.0 * xq + 3.0 * yq
    assert np.allclose(out, exact, atol=1e-10)


def test_bicubic_exact_on_linear_field():
    N = 33
    X, Y, dx, dy = create_grid(N, N, 1.0, 1.0)
    u = 2.0 * X - 1.5 * Y
    xq = X[2:-2, 2:-2] + 0.4 * dx
    yq = Y[2:-2, 2:-2] - 0.1 * dy
    out = bicubic_interpolate(u, xq, yq, dx, dy, N, N)
    exact = 2.0 * xq - 1.5 * yq
    assert np.allclose(out, exact, atol=1e-9)


def test_extrapolation_exact_on_linear_reference_map():
    """A linear reference map should be extrapolated exactly into the band."""
    N = 65
    X, Y, dx, dy = create_grid(N, N, 1.0, 1.0)
    phi = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) - 0.25
    solid = (phi < 0).astype(float)
    # linear maps a*x+b*y+c, zeroed outside, then extrapolated
    X1 = (1.3 * X + 0.2 * Y) * solid
    X2 = (-0.4 * X + 0.9 * Y) * solid
    X1e, X2e = extrapolate_reference_map(X1, X2, phi, dx, dy, max_layers=3)
    band = (phi >= 0) & (phi < 3 * dx)
    err1 = np.abs(X1e[band] - (1.3 * X[band] + 0.2 * Y[band]))
    err2 = np.abs(X2e[band] - (-0.4 * X[band] + 0.9 * Y[band]))
    # least-squares plane fit reproduces a linear field to ~machine precision
    assert err1.max() < 1e-8
    assert err2.max() < 1e-8


def test_strain_energy_matches_stress_no_lnJ():
    """For a uniform stretch, the (lnJ-free) strain-energy density must equal
    the neo-Hookean W = (mu/2)(I1 - 2) consistent with sigma = mu*b."""
    N = 81
    X, Y, dx, dy = create_grid(N, N, 1.0, 1.0)
    phi = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) - 0.25
    lam = 1.4
    X1, X2 = X / lam, Y.copy()                  # F = diag(lam, 1)
    mu_s = 2.0
    se = compute_strain_energy(X1, X2, phi, mu_s, dx, dy, kappa=0.0)
    # I1 = lam^2 + 1; W = (mu/2)(I1 - 2) = (mu/2)(lam^2 - 1); integrate over solid area
    solid_area = np.sum(phi <= 0) * dx * dy
    expected = 0.5 * mu_s * (lam**2 - 1.0) * solid_area
    assert abs(se - expected) / expected < 0.05


def test_interpolators_handle_nonfinite_coords():
    """Non-finite query coordinates must yield NaN, not a segfault
    (int(floor(NaN)) would index out of bounds)."""
    N = 33
    X, Y, dx, dy = create_grid(N, N, 1.0, 1.0)
    u = (2.0 * X + 3.0 * Y).copy()
    xq = X.copy(); yq = Y.copy()
    xq[0, 0] = np.nan; yq[1, 1] = np.inf; xq[2, 2] = -np.inf
    for interp in (bilinear_interpolate, bicubic_interpolate):
        out = interp(u, xq, yq, dx, dy, N, N)
        assert np.isnan(out[0, 0]) and np.isnan(out[1, 1]) and np.isnan(out[2, 2])
        assert np.all(np.isfinite(out[5:, 5:]))   # finite queries still fine
