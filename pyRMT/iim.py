"""Reference-map immersed-interface method (RM-IIM): sharp jump corrections on the MAC grid.

Stage 1: the pressure jump. The interface crossing on each face is located from the level set
phi = phi0(xi) rebuilt from the reference map; the jump [q] = q_out - q_in (out = phi > 0) is
imposed with ghost-fluid corrections (Liu, Fedkiw & Kang 2000). The corrections enter only the
right-hand side, so the DCT Poisson solve of the projection is unchanged, and the velocity
correction uses the same jump-corrected face gradient.

See docs/rm_iim_design.md for the plan.
"""
import numpy as np

from pyRMT.mac import solve_poisson_neumann


def _face_jumps(phi, jump):
    """Jump [q] on each interior face that the interface crosses (0 elsewhere), and the side.
    The cell-centred jump field is interpolated linearly to the crossing point.
    Returns jx (Ny, Nx-1), jy (Ny-1, Nx) and sx, sy: +1 if the left/lower cell is inside."""
    a, b = phi[:, :-1], phi[:, 1:]
    cx = (a <= 0) != (b <= 0)
    th = np.where(cx, a / np.where(cx, a - b, 1.0), 0.0)                  # crossing fraction
    jx = np.where(cx, (1 - th) * jump[:, :-1] + th * jump[:, 1:], 0.0)
    sx = np.where(cx, np.where(a <= 0, 1.0, -1.0), 0.0)
    a, b = phi[:-1, :], phi[1:, :]
    cy = (a <= 0) != (b <= 0)
    th = np.where(cy, a / np.where(cy, a - b, 1.0), 0.0)
    jy = np.where(cy, (1 - th) * jump[:-1, :] + th * jump[1:, :], 0.0)
    sy = np.where(cy, np.where(a <= 0, 1.0, -1.0), 0.0)
    return jx, sx, jy, sy


def project_with_jump(u_star, v_star, dx, dy, dt, rho, eig, phi, jump):
    """Projection with a prescribed pressure jump [q] = q(phi>0) - q(phi<=0) across phi = 0.

    Walled MAC grid: u (Ny, Nx+1), v (Ny+1, Nx); homogeneous Neumann for q.
    Ghost-fluid: on a crossing face the gradient is (q_R - q_L - s*[q]) / h, s = +1 if the left
    (lower) cell is inside. The discrete Poisson equation therefore gains a known RHS term, and
    the same corrected gradient updates the velocity, so the jump is carried sharply.
    Returns u, v, q."""
    jx, sx, jy, sy = _face_jumps(phi, jump)
    cx = sx * jx / dx; cy = sy * jy / dy                 # jump part of the face gradient
    div = (u_star[:, 1:] - u_star[:, :-1]) / dx + (v_star[1:, :] - v_star[:-1, :]) / dy
    # div(grad q) with the corrected gradient = (rho/dt) div u*  ->  move the known part right
    corr = np.zeros_like(div)
    corr[:, :-1] += cx / dx; corr[:, 1:] -= cx / dx
    corr[:-1, :] += cy / dy; corr[1:, :] -= cy / dy
    q = solve_poisson_neumann((rho / dt) * div + corr, eig)
    u = u_star.copy(); v = v_star.copy()
    gx = (q[:, 1:] - q[:, :-1]) / dx - cx
    gy = (q[1:, :] - q[:-1, :]) / dy - cy
    u[:, 1:-1] -= (dt / rho) * gx
    v[1:-1, :] -= (dt / rho) * gy
    return u, v, q
