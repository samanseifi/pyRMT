"""Bug 1: the velocity gradient that drives the log-conformation stretch must be trace-free
whenever the MAC velocity is discretely divergence-free.

For the upper-convected stretch, d(ln det b_e)/dt = 2 tr(L). The drivers used to build L by
central differences of cell-AVERAGED velocities, whose trace is a smoothed combination of
neighbouring MAC divergences, not the divergence itself: it vanishes for smooth fields only up to
O(dx^2) and is O(1) where the velocity gradient jumps (at the interface), so det b_e drifted
(to 1.36 in the elastic limit, where it must stay 1). `cell_velocity_gradient` takes the diagonal
entries from the compact MAC differences, so tr(L) equals the MAC divergence exactly.
"""
import numpy as np
import pytest

from pyRMT.mac import mac_grid, divergence, divergence_per, cell_velocity_gradient
from pyRMT.functions import grad_central_x_2nd, grad_central_y_2nd


def _old_trace(u_c, v_c, dx, dy):
    return grad_central_x_2nd(u_c, dx) + grad_central_y_2nd(v_c, dy)


def _walled_field(N, kink, seed=0):
    """Discretely divergence-free MAC velocity from a node stream function psi:
    u = dpsi/dy on x-faces, v = -dpsi/dx on y-faces (zero normal flux on the walls)."""
    dx, dy = mac_grid(N, N)
    x = np.arange(N + 1) * dx; X, Y = np.meshgrid(x, x)            # nodes (N+1, N+1)
    psi = np.sin(np.pi * X) ** 2 * np.sin(np.pi * Y) ** 2 * np.exp(X - 0.3 * Y)   # not x-y symmetric
    if kink:                                                       # gradient jump on a circle
        r = np.hypot(X - 0.5, Y - 0.5)
        psi = psi * (1.0 + 0.5 * np.maximum(0.0, 0.25 - r))
    u = (psi[1:, :] - psi[:-1, :]) / dy                              # (N, N+1)
    v = -(psi[:, 1:] - psi[:, :-1]) / dx                             # (N+1, N)
    return u, v, dx, dy


@pytest.mark.parametrize("kink", [False, True])
def test_walled_trace_equals_mac_divergence(kink):
    u, v, dx, dy = _walled_field(64, kink)
    assert np.abs(divergence(u, v, dx, dy)).max() < 1e-11
    L11, L12, L21, L22 = cell_velocity_gradient(u, v, dx, dy)
    assert np.abs(L11 + L22).max() < 1e-11
    u_c = 0.5 * (u[:, :-1] + u[:, 1:]); v_c = 0.5 * (v[:-1, :] + v[1:, :])
    old = _old_trace(u_c, v_c, dx, dy)[2:-2, 2:-2]
    scale = np.abs(L11).max()
    # the old construction is NOT trace-free; with a gradient jump the error is O(1) of the
    # velocity-gradient scale, which is what drives the det(b_e) drift
    assert np.abs(old).max() > (0.1 if kink else 1e-5) * scale


def test_walled_off_diagonals_unchanged():
    u, v, dx, dy = _walled_field(32, False)
    L11, L12, L21, L22 = cell_velocity_gradient(u, v, dx, dy)
    u_c = 0.5 * (u[:, :-1] + u[:, 1:]); v_c = 0.5 * (v[:-1, :] + v[1:, :])
    assert np.allclose(L12, grad_central_y_2nd(u_c, dy))
    assert np.allclose(L21, grad_central_x_2nd(v_c, dx))


def test_periodic_trace_equals_mac_divergence():
    N = 48; dx, dy = mac_grid(N, N); rng = np.random.default_rng(1)
    psi = rng.standard_normal((N, N))                                # random periodic node field
    u = (np.roll(psi, -1, 0) - psi) / dy
    v = -(np.roll(psi, -1, 1) - psi) / dx
    assert np.abs(divergence_per(u, v, dx, dy)).max() < 1e-10
    L11, L12, L21, L22 = cell_velocity_gradient(u, v, dx, dy, periodic=True)
    assert np.abs(L11 + L22).max() < 1e-10
