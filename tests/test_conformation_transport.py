"""Bug 4: the conformation psi = log(b_e) was cut with the OLD solid mask but extrapolated with the
NEW level set. Cells the interface had just moved into were then treated as known solid data while
still holding psi = 0 (unstrained), so a strip of spurious zero strain appeared at the advancing
edge every step. `advance_conformation` cuts and extrapolates with the same (new) level set.

Test: a disc carrying a uniform strain translates in a uniform flow. The strain must stay uniform
everywhere inside the solid.
"""
import numpy as np

from pyRMT.mac import mac_grid
from pyRMT.functions import (advect_reference_map, extrapolate_reference_map,
                             rebuild_phi_from_reference_map, advance_conformation)

N, R, PSI0 = 64, 0.2, 0.3


def _setup():
    dx, dy = mac_grid(N, N); xc = (np.arange(N) + 0.5) * dx; X, Y = np.meshgrid(xc, xc)
    Xg, Yg = np.meshgrid(np.arange(N) * dx, np.arange(N) * dy)
    phi0 = lambda a, b: np.hypot(a - 0.4, b - 0.5) - R
    phi = phi0(X, Y); m = (phi <= 0).astype(float)
    X1, X2 = extrapolate_reference_map(X * m, Y * m, phi, dx, dy, 3)
    p = PSI0 * m
    p, _ = extrapolate_reference_map(p, np.zeros_like(p), phi, dx, dy, 3)
    u_c = np.full((N, N), 0.7); v_c = np.full((N, N), 0.3)
    return dx, dy, Xg, Yg, phi0, X1, X2, p, u_c, v_c


def _run(fixed, steps=20):
    dx, dy, Xg, Yg, phi0, X1, X2, p, u_c, v_c = _setup()
    dt = 0.3 * dx / 0.7; worst = 0.0
    for _ in range(steps):
        phi = rebuild_phi_from_reference_map(X1, X2, phi0); sm = (phi <= 0).astype(float)
        X1 = advect_reference_map(X1, u_c, v_c, Xg, Yg, dt, dx, dy, phi, "semilagrangian_cubic", 0.0) * sm
        X2 = advect_reference_map(X2, u_c, v_c, Xg, Yg, dt, dx, dy, phi, "semilagrangian_cubic", 0.0) * sm
        X1, X2 = extrapolate_reference_map(X1, X2, phi, dx, dy, 3)
        phi_new = rebuild_phi_from_reference_map(X1, X2, phi0)
        if fixed:
            p, _, _ = advance_conformation(p, np.zeros_like(p), np.zeros_like(p), u_c, v_c, Xg, Yg,
                                           dt, dx, dy, phi_new, "semilagrangian_cubic", 0.0, 3)
        else:   # the former driver sequence: old mask, new level set
            p = advect_reference_map(p, u_c, v_c, Xg, Yg, dt, dx, dy, phi_new, "semilagrangian_cubic", 0.0) * sm
            p, _ = extrapolate_reference_map(p, np.zeros_like(p), phi_new, dx, dy, 3)
        inside = phi_new <= 0
        worst = max(worst, np.abs(p[inside] - PSI0).max())
    return worst


def test_uniform_strain_survives_translation():
    assert _run(fixed=True) < 1e-6


def test_former_sequence_leaves_unstrained_cells():
    assert _run(fixed=False) > 0.5 * PSI0          # whole cells drop to psi = 0
