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
