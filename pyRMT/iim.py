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


def laplacian_jump_correction(phi, gx, gy, dx, dy, gxx=None, gyy=None):
    """Immersed-interface correction to the 5-point Laplacian of a field that is continuous
    across phi = 0 but whose first derivatives jump there (stage 2: tangential traction).

    phi, gx, gy are given at the field's own grid points (shape (Ny, Nx)); gx = [d/dx], gy =
    [d/dy] are the derivative jumps OUTSIDE minus INSIDE (phi > 0 minus phi <= 0), interpolated to
    each crossing. Along a grid line the correction needs the jump right minus left,
    g = s (out - in) with s = +1 if the left/lower point is inside. For a crossing between points
    k and k+1 at fraction theta from k:
        Lap_k     += -g (1 - theta) h / h^2      (k sees a value from the other side)
        Lap_{k+1} += -g theta h / h^2
    so that (standard 5-point Laplacian) + correction is the one-sided Laplacian on each side,
    with an O(h) local truncation error at the interface points instead of O(1/h).
    Optional gxx, gyy: jumps (outside minus inside) of the second derivatives d2/dx2, d2/dy2. With
    them the correction is exact for piecewise quadratics and the local error at the interface
    points drops from O(1) to O(h):
        Lap_k     += -g (1 - theta)/h - gxx (1 - theta)^2 / 2
        Lap_{k+1} += -g theta/h       + gxx theta^2 / 2
    Returns the additive correction C (Ny, Nx): Lap_true ~= Lap_5pt(u) + C."""
    C = np.zeros_like(phi)
    for ax, g, g2, h in ((1, gx, gxx, dx), (0, gy, gyy, dy)):
        a = phi[:, :-1] if ax == 1 else phi[:-1, :]
        b = phi[:, 1:] if ax == 1 else phi[1:, :]
        ga = g[:, :-1] if ax == 1 else g[:-1, :]
        gb = g[:, 1:] if ax == 1 else g[1:, :]
        cr = (a <= 0) != (b <= 0)
        th = np.where(cr, a / np.where(cr, a - b, 1.0), 0.0)
        sgn = np.where(a <= 0, 1.0, -1.0)                       # right-minus-left orientation
        gc = np.where(cr, sgn * ((1 - th) * ga + th * gb), 0.0)
        lo = -gc * (1 - th) / h; hi = -gc * th / h
        if g2 is not None:                                       # second-derivative jump
            g2a = g2[:, :-1] if ax == 1 else g2[:-1, :]
            g2b = g2[:, 1:] if ax == 1 else g2[1:, :]
            g2c = np.where(cr, sgn * ((1 - th) * g2a + th * g2b), 0.0)
            lo = lo - g2c * (1 - th) ** 2 / 2.0; hi = hi + g2c * th ** 2 / 2.0
        if ax == 1:
            C[:, :-1] += lo; C[:, 1:] += hi
        else:
            C[:-1, :] += lo; C[1:, :] += hi
    return C


def divergence_jump_correction_per(pu, gux, pv, gvy, dx, dy):
    """Immersed-interface correction to the periodic MAC divergence of a velocity whose normal
    derivatives jump across phi = 0 (the velocity itself is continuous).

    Cell (j, i) lies between x-faces i and i+1 (u) and y-faces j and j+1 (v). If the interface
    crosses between the two faces at fraction theta from the left face, the face difference is
    h (u_x^- theta + u_x^+ (1 - theta)); the one-sided derivative on the cell centre's side is
    recovered by adding g theta (centre on the right side) or subtracting g (1 - theta) (centre on
    the left side), g = jump of du/dx right minus left. pu, pv: level set at the u / v points;
    gux, gvy: jumps of du/dx, dv/dy outside minus inside at those points.
    Returns C (Ny, Nx): div_true ~= div_MAC + C."""
    C = np.zeros_like(pu)
    for p, g, h, ax in ((pu, gux, dx, 1), (pv, gvy, dy, 0)):
        a = p; b = np.roll(p, -1, ax)                       # the two faces of each cell
        ga = g; gb = np.roll(g, -1, ax)
        cr = (a <= 0) != (b <= 0)
        th = np.where(cr, a / np.where(cr, a - b, 1.0), 0.0)
        sgn = np.where(a <= 0, 1.0, -1.0)                    # right-minus-left orientation
        gc = np.where(cr, sgn * ((1 - th) * ga + th * gb), 0.0)
        C += np.where(cr, np.where(th < 0.5, gc * th, -gc * (1 - th)), 0.0) / 1.0
    return C


def project_per_with_jumps(u_star, v_star, dx, dy, eig, div_corr):
    """Periodic exact projection with a corrected divergence (div_MAC + div_corr) as the source.
    The returned velocity has corrected divergence zero; q is the projection scalar."""
    from pyRMT.mac import divergence_per, solve_poisson_periodic
    div = divergence_per(u_star, v_star, dx, dy) + div_corr
    q = solve_poisson_periodic(div, eig)
    u = u_star - (q - np.roll(q, 1, 1)) / dx
    v = v_star - (q - np.roll(q, 1, 0)) / dy
    return u, v, q


# ---------------------------------------------------------------------------------------------
# Stage 3: jumps from the solid stress, and one sharp step on the periodic MAC grid.
# ---------------------------------------------------------------------------------------------

def _to_u(f):   # cell centres -> periodic x-faces (face i left of cell i)
    return 0.5 * (f + np.roll(f, 1, 1))


def _to_v(f):   # cell centres -> periodic y-faces
    return 0.5 * (f + np.roll(f, 1, 0))


def traction_jumps(phi, sxx, sxy, syy, mu):
    """Interface jump conditions from the solid Cauchy stress sigma_s (cell-centred, valid in a
    band around phi = 0, e.g. extrapolated from the solid side).

    Total-traction continuity with a continuous velocity gives (outside minus inside)
        [p] = -n.sigma_s.n,        [du/dn] = (I - n n) sigma_s n / mu,
    and, since tangential derivatives are continuous, [d u_i/d x_j] = [d u_i/d n] n_j.
    Returns a dict of cell-centred fields: jp, and the derivative jumps of u and v."""
    gy, gx = np.gradient(phi)                     # periodic not needed: only used near the disc
    nrm = np.sqrt(gx ** 2 + gy ** 2) + 1e-300; nx, ny = gx / nrm, gy / nrm
    tx = sxx * nx + sxy * ny; ty = sxy * nx + syy * ny          # sigma_s n
    tn = tx * nx + ty * ny
    ttx, tty = tx - tn * nx, ty - tn * ny                        # tangential part
    return dict(jp=-tn, ux=ttx * nx / mu, uy=ttx * ny / mu, vx=tty * nx / mu, vy=tty * ny / mu)


def _pressure_face_jumps_per(phi, jump):
    """Periodic version of _face_jumps for cell-centred phi and [q]: returns (cx, cy), the jump
    part of the face gradient on x-faces (left of cell i) and y-faces (below cell j)."""
    out = []
    for ax in (1, 0):
        a = np.roll(phi, 1, ax); b = phi                         # left/lower cell, this cell
        ja = np.roll(jump, 1, ax); jb = jump
        cr = (a <= 0) != (b <= 0)
        th = np.where(cr, a / np.where(cr, a - b, 1.0), 0.0)
        jc = np.where(cr, (1 - th) * ja + th * jb, 0.0)
        s = np.where(a <= 0, 1.0, -1.0)
        out.append(np.where(cr, s * jc, 0.0))
    return out


def sharp_step_per(u, v, dx, dy, dt, nu, rho, lap_eig, peig, phi_c, J, f_u=None, f_v=None):
    """One semi-implicit step with sharp interface jumps (periodic MAC grid, constant nu, rho).

    u, v (Ny, Nx): face velocities after explicit advection (and any bulk force already added).
    phi_c: cell-centred level set; J: traction_jumps(...) evaluated on cell centres.
      1. implicit viscosity  (I - dt nu Lap) u* = u + dt nu C_u   (C: IIM Laplacian correction)
      2. projection with the jump-corrected divergence and the pressure jump [p] (ghost fluid)
    All corrections are right-hand-side terms, so both solves stay direct FFT solves."""
    from pyRMT.mac import solve_helmholtz_periodic, solve_poisson_periodic, divergence_per
    pu, pv = _to_u(phi_c), _to_v(phi_c)
    Cu = laplacian_jump_correction(pu, _to_u(J["ux"]), _to_u(J["uy"]), dx, dy)
    Cv = laplacian_jump_correction(pv, _to_v(J["vx"]), _to_v(J["vy"]), dx, dy)
    ru = u + dt * nu * Cu + (dt * f_u / rho if f_u is not None else 0.0)
    rv = v + dt * nu * Cv + (dt * f_v / rho if f_v is not None else 0.0)
    us = solve_helmholtz_periodic(ru, lap_eig, dt * nu)
    vs = solve_helmholtz_periodic(rv, lap_eig, dt * nu)
    dc = divergence_jump_correction_per(pu, _to_u(J["ux"]), pv, _to_v(J["vy"]), dx, dy)
    # p solves div(grad_c p) = (rho/dt) div_c(u*), with the corrected face gradient
    # grad_c p = (p_R - p_L - s [p]) / h on crossed faces; the known jump part goes to the RHS.
    cx, cy = _pressure_face_jumps_per(phi_c, J["jp"])
    gxj = cx / dx; gyj = cy / dy                                  # jump part of the face gradient
    div = divergence_per(us, vs, dx, dy) + dc
    corr = (np.roll(gxj, -1, 1) - gxj) / dx + (np.roll(gyj, -1, 0) - gyj) / dy
    p = solve_poisson_periodic((rho / dt) * div + corr, peig)
    un = us - (dt / rho) * ((p - np.roll(p, 1, 1)) / dx - gxj)
    vn = vs - (dt / rho) * ((p - np.roll(p, 1, 0)) / dy - gyj)
    return un, vn, p


def bulk_solid_force_per(phi, sxx, sxy, syy, dx, dy):
    """Bulk solid force chi * div(sigma_s) on the periodic MAC faces, for the sharp formulation.

    The divergence is the compact face divergence of the (smoothly extended) solid stress, kept
    only on faces whose two neighbouring cells are both inside the solid (phi <= 0). Faces that
    cross the interface carry no bulk force: there the traction enters through the jump
    conditions, so nothing is counted twice. Without this term the solid interior responds like
    fluid once its stress becomes non-uniform, and the interface traction injects energy."""
    from pyRMT.mac import solid_force_faces
    fu, fv = solid_force_faces(sxx, sxy, syy, dx, dy, periodic=True)
    ins = phi <= 0
    fu = np.where(ins & np.roll(ins, 1, 1), fu, 0.0)
    fv = np.where(ins & np.roll(ins, 1, 0), fv, 0.0)
    return fu, fv
