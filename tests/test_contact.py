"""Solid-solid contact force: repulsion direction and locality."""
import numpy as np
from pyRMT.functions import create_grid, compute_contact_force


def _disc(X, Y, x0, y0, R):
    return np.sqrt((X - x0) ** 2 + (Y - y0) ** 2) - R


def test_contact_force_direction_is_repulsive():
    """Two barely-touching discs (left at 0.4, right at 0.6). Near the contact,
    material on the left side of the mid-surface is pushed left (fx<0) and on the
    right side pushed right (fx>0) — i.e. the discs are pushed apart along +/-x."""
    N = 161
    X, Y, dx, dy = create_grid(N, N, 1.0, 1.0)
    R = 0.105
    phi1 = _disc(X, Y, 0.40, 0.5, R)
    phi2 = _disc(X, Y, 0.60, 0.5, R)   # gap ~ 0.20 - 2R = -0.01 (just touching)
    w_c = 4 * dx
    fx, fy = compute_contact_force(phi1, phi2, k_rep=1.0, w_c=w_c, dx=dx, dy=dy)

    jmid = N // 2                       # y = 0.5 row
    xs = X[jmid, :]
    # a point just LEFT of the mid-surface (x=0.5) inside the contact band
    iL = np.argmin(np.abs(xs - 0.485))
    iR = np.argmin(np.abs(xs - 0.515))
    assert fx[jmid, iL] < 0.0          # left side pushed left
    assert fx[jmid, iR] > 0.0          # right side pushed right
    # the force vanishes away from the contact band
    far = (np.abs(0.5 * (phi1 - phi2)) > w_c)
    assert np.allclose(fx[far], 0.0) and np.allclose(fy[far], 0.0)


def test_contact_force_zero_when_far_apart():
    """Discs far apart (no overlap of influence regions) => no contact force."""
    N = 121
    X, Y, dx, dy = create_grid(N, N, 1.0, 1.0)
    R = 0.12
    phi1 = _disc(X, Y, 0.25, 0.5, R)
    phi2 = _disc(X, Y, 0.75, 0.5, R)
    fx, fy = compute_contact_force(phi1, phi2, k_rep=1.0, w_c=2 * dx, dx=dx, dy=dy)
    assert np.allclose(fx, 0.0) and np.allclose(fy, 0.0)


def test_two_solid_momentum_step_runs():
    """The two-solid momentum step (blended stress + contact force) runs and
    returns finite fields for two nearby discs."""
    from pyRMT.functions import (momentum_step_rk4_2solids, apply_phi_BCs,
                                 extrapolate_reference_map)
    N = 48
    X, Y, dx, dy = create_grid(N, N, 1.0, 1.0)
    R = 0.15
    pa = apply_phi_BCs(_disc(X, Y, 0.35, 0.5, R))
    pb = apply_phi_BCs(_disc(X, Y, 0.65, 0.5, R))
    ma = (pa <= 0).astype(float); mb = (pb <= 0).astype(float)
    X1a, X2a = extrapolate_reference_map(X * ma, Y * ma, pa, dx, dy, 3)
    X1b, X2b = extrapolate_reference_map(X * mb, Y * mb, pb, dx, dy, 3)
    bc = lambda u, v: (u.copy(), v.copy())
    u = np.zeros((N, N)); v = np.zeros((N, N)); p = np.zeros((N, N))
    un, vn, Jmin = momentum_step_rk4_2solids(
        u, v, p, X1a, X2a, X1b, X2b, bc, 1.0, 0.0, 0.0, dx, dy, 1e-3, 1.0, 1.0,
        pa, pb, 0.01, 2 * dx, k_rep=2.0, w_c=3 * dx)
    assert np.all(np.isfinite(un)) and np.all(np.isfinite(vn))
    assert np.all(np.isfinite(Jmin))
