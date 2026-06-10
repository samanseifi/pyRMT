"""Poisson solvers and pressure projection: Neumann (DCT) and periodic (FFT)."""
import numpy as np
import pytest
from pyRMT.functions import (
    create_grid, _precompute_poisson_eigenvalues, _solve_poisson_dct,
    _precompute_poisson_eigenvalues_periodic, _solve_poisson_fft,
    _compute_divergence, _compute_divergence_periodic, pressure_projection_amg,
)


def test_dct_recovers_manufactured_neumann():
    """DCT solve of lap(p)=rhs recovers p (up to a constant) for a field with
    zero normal derivative on the walls."""
    N = 65
    X, Y, dx, dy = create_grid(N, N, 1.0, 1.0)
    p_true = np.cos(np.pi * X) * np.cos(np.pi * Y)   # dp/dn = 0 on [0,1]^2 walls
    lap = -2.0 * np.pi**2 * p_true
    eig = _precompute_poisson_eigenvalues(N, N, dx, dy)
    p = _solve_poisson_dct(lap, eig)
    p -= p.mean(); pt = p_true - p_true.mean()
    assert np.max(np.abs(p - pt)) < 5e-3


def test_fft_periodic_roundtrip_machine_precision():
    """solve(div(grad(p))) == p for the periodic operators (exact)."""
    N = 65
    X, Y, dx, dy = create_grid(N, N, 1.0, 1.0)
    from pyRMT.functions import _compute_pressure_gradient_periodic
    k = 2 * np.pi
    p_true = np.cos(k * X) * np.sin(k * Y) + 0.5 * np.sin(2 * k * X)
    gx, gy = _compute_pressure_gradient_periodic(p_true, dx, dy)
    lap = _compute_divergence_periodic(gx, gy, dx, dy)
    eig = _precompute_poisson_eigenvalues_periodic(N, N, dx, dy)
    p = _solve_poisson_fft(lap, eig)
    pt = p_true - p_true.mean()
    assert np.max(np.abs((p - pt)[:-1, :-1])) < 1e-10


def _wall_bc(u, v):
    u = u.copy(); v = v.copy()
    u[:, 0] = u[:, -1] = u[0, :] = u[-1, :] = 0.0
    v[:, 0] = v[:, -1] = v[0, :] = v[-1, :] = 0.0
    return u, v


def test_neumann_projection_reduces_divergence():
    N = 65
    X, Y, dx, dy = create_grid(N, N, 1.0, 1.0)
    a = np.sin(np.pi * X) * np.cos(np.pi * Y)
    b = 0.5 * np.cos(np.pi * X) * np.sin(np.pi * Y)
    a, b = _wall_bc(a, b)
    eig = _precompute_poisson_eigenvalues(N, N, dx, dy)
    d0 = np.abs(_compute_divergence(a, b, dx, dy)[2:-2, 2:-2]).max()
    an, bn, p, _, _ = pressure_projection_amg(
        a, b, dx, dy, 1e-2, 1.0, _wall_bc, p_prev=None, eigenvalues=eig, bc_type='neumann')
    d1 = np.abs(_compute_divergence(an, bn, dx, dy)[2:-2, 2:-2]).max()
    assert d1 < d0 / 50.0


def _periodic_bc(u, v):
    u = u.copy(); v = v.copy()
    u[:, -1] = u[:, 0]; v[:, -1] = v[:, 0]
    u[-1, :] = u[0, :]; v[-1, :] = v[0, :]
    return u, v


def test_periodic_projection_makes_divergence_free():
    N = 65
    X, Y, dx, dy = create_grid(N, N, 1.0, 1.0)
    k = 2 * np.pi
    a = np.sin(k * X) * np.cos(k * Y) + 0.3 * np.cos(k * X)
    b = -np.cos(k * X) * np.sin(k * Y) + 0.2 * np.sin(k * Y)
    a, b = _periodic_bc(a, b)
    eig = _precompute_poisson_eigenvalues_periodic(N, N, dx, dy)
    an, bn, p, _, _ = pressure_projection_amg(
        a, b, dx, dy, 1e-2, 1.0, _periodic_bc, p_prev=None, eigenvalues=eig, bc_type='periodic')
    d1 = np.abs(_compute_divergence_periodic(an, bn, dx, dy)[:-1, :-1]).max()
    assert d1 < 1e-9
