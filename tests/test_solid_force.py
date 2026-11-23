"""Bugs 2 and 3: how the solid stress becomes a face force.

Bug 3 (force stencil). The drivers took the stress divergence by 2dx central differences at the
cell centres and averaged it to the faces. That operator is not the MAC gradient, so even a purely
isotropic stress sigma = s I -- which should be absorbed entirely by the pressure -- leaves a
solenoidal residue that the projection cannot remove and that drives spurious flow. The compact
face divergence `solid_force_faces` reduces to the MAC pressure gradient for an isotropic stress,
so the projection removes it to round-off.

Bug 2 (interface cut-off). `solid_cauchy_stress` in its default (legacy) mode evaluates the
reference-map stress only where phi <= 0, with one-sided differences next to the interface, and
returns zero elsewhere. Multiplied by the blend (1 - H), which runs from 1 at phi = -w to 0 at
phi = +w, the blended stress then drops by about half of its value at phi = 0 instead of ramping
to zero across the band. Evaluating it across the extrapolated band (w_cut > 0) removes the jump.
"""
import numpy as np
import pytest

from pyRMT.mac import (mac_grid, project, project_per, poisson_eigs_neumann, poisson_eigs_periodic,
                       solid_force_faces)
from pyRMT.functions import (grad_central_x_2nd, grad_central_y_2nd, solid_cauchy_stress,
                             smoothed_heaviside, extrapolate_reference_map)


def _old_force(Sxx, Sxy, Syy, dx, dy):
    divx = grad_central_x_2nd(Sxx, dx) + grad_central_y_2nd(Sxy, dy)
    divy = grad_central_x_2nd(Sxy, dx) + grad_central_y_2nd(Syy, dy)
    Ny, Nx = Sxx.shape
    fu = np.zeros((Ny, Nx + 1)); fu[:, 1:-1] = 0.5 * (divx[:, 1:] + divx[:, :-1])
    fv = np.zeros((Ny + 1, Nx)); fv[1:-1, :] = 0.5 * (divy[1:, :] + divy[:-1, :])
    return fu, fv


def _blob(N):
    dx, dy = mac_grid(N, N); xc = (np.arange(N) + 0.5) * dx; X, Y = np.meshgrid(xc, xc)
    r = np.hypot(X - 0.5, Y - 0.45)
    s = np.where(r < 0.2, 1.0 + np.cos(3 * X) * np.sin(2 * Y), 0.0)    # O(1) jump on a circle
    return s, dx, dy


def test_isotropic_stress_is_absorbed_by_projection_walled():
    N = 64; s, dx, dy = _blob(N); eig = poisson_eigs_neumann(N, N, dx, dy); dt = 1e-2
    fu, fv = solid_force_faces(s, np.zeros_like(s), s, dx, dy)
    u, v, _ = project(dt * fu, dt * fv, dx, dy, dt, 1.0, eig)
    assert max(np.abs(u).max(), np.abs(v).max()) < 1e-10
    fu0, fv0 = _old_force(s, np.zeros_like(s), s, dx, dy)
    u0, v0, _ = project(dt * fu0, dt * fv0, dx, dy, dt, 1.0, eig)
    # the former stencil leaves a spurious solenoidal velocity of O(dt * |f|)
    assert max(np.abs(u0).max(), np.abs(v0).max()) > 1e-3 * dt * np.abs(fu0).max()


def test_isotropic_stress_is_absorbed_by_projection_periodic():
    N = 48; s, dx, dy = _blob(N); eig = poisson_eigs_periodic(N, N, dx, dy); dt = 1e-2
    fu, fv = solid_force_faces(s, np.zeros_like(s), s, dx, dy, periodic=True)
    u, v, _ = project_per(dt * fu, dt * fv, dx, dy, dt, 1.0, eig)
    assert max(np.abs(u).max(), np.abs(v).max()) < 1e-10


def test_compact_force_is_second_order_for_smooth_stress():
    errs = []
    for N in (32, 64, 128):
        dx, dy = mac_grid(N, N); xc = (np.arange(N) + 0.5) * dx; X, Y = np.meshgrid(xc, xc)
        Sxx = np.sin(2 * np.pi * X) * np.cos(np.pi * Y); Sxy = np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        Syy = X * Y ** 2
        fu, fv = solid_force_faces(Sxx, Sxy, Syy, dx, dy)
        xf = np.arange(N + 1) * dx; XU, YU = np.meshgrid(xf, xc)
        ex = (2 * np.pi * np.cos(2 * np.pi * XU) * np.cos(np.pi * YU)
              + 2 * np.pi * np.cos(2 * np.pi * XU) * np.cos(2 * np.pi * YU))
        errs.append(np.abs(fu[2:-2, 2:-2] - ex[2:-2, 2:-2]).max())
    assert errs[0] / errs[1] > 3.5 and errs[1] / errs[2] > 3.5


def _affine_disc(N, w_cut):
    """Disc with a uniformly stretched reference map: xi = A (x - c) + c, so the stress is
    uniform in the solid. Returns the blended stress along the horizontal centre line."""
    dx, dy = mac_grid(N, N); xc = (np.arange(N) + 0.5) * dx; X, Y = np.meshgrid(xc, xc)
    c = 0.5; R = 0.25
    A = np.array([[1.0 / 1.2, 0.0], [0.0, 1.2]])            # current = stretched by 1.2 in x
    X1 = c + A[0, 0] * (X - c); X2 = c + A[1, 1] * (Y - c)
    phi0 = lambda a, b: np.hypot(a - c, b - c) - R
    phi = phi0(X1, X2)
    m = (phi <= 0).astype(float)
    X1, X2 = extrapolate_reference_map(X1 * m, X2 * m, phi, dx, dy, 4)
    phi = phi0(X1, X2)
    sxx, _, _, _ = solid_cauchy_stress(X1, X2, dx, dy, 1.0, 0.0, phi, w_cut=w_cut)
    H = smoothed_heaviside(phi, 2 * dx)
    j = N // 2
    return (1 - H[j]) * sxx[j], (1 - H[j]), sxx[j], phi[j], dx


def test_blended_stress_has_no_jump_with_band_evaluation():
    N = 128
    blended_old, ramp, sxx_old, phi, dx = _affine_disc(N, 0.0)
    blended_new, _, sxx_new, _, _ = _affine_disc(N, 2 * dx)
    inside = sxx_new[np.argmin(np.abs(phi + 0.1))]           # value well inside the solid
    band = np.abs(phi) <= 2 * dx
    # the evaluated stress is (nearly) the uniform interior value across the whole band ...
    assert np.abs(sxx_new[band] - inside).max() < 0.02 * abs(inside)
    # ... so the blended stress ramps smoothly: no cell-to-cell drop larger than a ramp step
    assert np.abs(np.diff(blended_new)).max() < 0.4 * abs(inside)
    # the legacy evaluation drops to zero for phi > 0 and jumps by about half the stress
    assert np.abs(np.diff(blended_old)).max() > 0.4 * abs(inside)
