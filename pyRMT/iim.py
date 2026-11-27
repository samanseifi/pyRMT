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


def laplacian_jump_correction(phi, gx, gy, dx, dy, gxx=None, gyy=None, w=None):
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
    Optional w: jump (outside minus inside) of the field's VALUE. Needed for the intermediate
    velocity of a projection method, which is discontinuous by (dt/rho)[grad p] across the
    interface even though the end-of-step velocity is continuous:
        Lap_k     += -w / h^2,      Lap_{k+1} += +w / h^2
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
        if w is not None:                                        # value jump
            wa = w[:, :-1] if ax == 1 else w[:-1, :]
            wb = w[:, 1:] if ax == 1 else w[1:, :]
            wc = np.where(cr, sgn * ((1 - th) * wa + th * wb), 0.0)
            lo = lo - wc / h ** 2; hi = hi + wc / h ** 2
        if ax == 1:
            C[:, :-1] += lo; C[:, 1:] += hi
        else:
            C[:-1, :] += lo; C[1:, :] += hi
    return C


def divergence_jump_correction_per(pu, gux, pv, gvy, dx, dy, guxx=None, gvyy=None, phi_c=None):
    """Immersed-interface correction to the periodic MAC divergence of a velocity whose normal
    derivatives jump across phi = 0 (the velocity itself is continuous).

    Cell (j, i) lies between x-faces i and i+1 (u) and y-faces j and j+1 (v). If the interface
    crosses between the two faces at fraction theta from the left face, the face difference is
    h (u_x^- theta + u_x^+ (1 - theta)); the one-sided derivative on the cell centre's side is
    recovered by adding g theta (centre on the right side) or subtracting g (1 - theta) (centre on
    the left side), g = jump of du/dx right minus left. pu, pv: level set at the u / v points;
    gux, gvy: jumps of du/dx, dv/dy outside minus inside at those points.
    Optional guxx, gvyy: jumps of d2u/dx2, d2v/dy2 (outside minus inside). With g2 the jump right
    minus left, the face difference picks up g2 (1 - theta)^2 h / 2 more, so the corrections become
        centre on the right:  + g theta - g2 theta^2 h / 2
        centre on the left:   - g (1 - theta) - g2 (1 - theta)^2 h / 2
    and the divergence constraint on cut cells is second order instead of first.
    Optional phi_c: level set at the cell centres. The side of the centre is then taken from its
    sign for BOTH directions (instead of theta < 1/2 separately along x and along y, which can
    disagree when the interface passes close to the centre and leaves an O(1) error in that cell).
    Returns C (Ny, Nx): div_true ~= div_MAC + C."""
    C = np.zeros_like(pu)
    for p, g, g2, h, ax in ((pu, gux, guxx, dx, 1), (pv, gvy, gvyy, dy, 0)):
        a = p; b = np.roll(p, -1, ax)                       # the two faces of each cell
        ga = g; gb = np.roll(g, -1, ax)
        cr = (a <= 0) != (b <= 0)
        th = np.where(cr, a / np.where(cr, a - b, 1.0), 0.0)
        sgn = np.where(a <= 0, 1.0, -1.0)                    # right-minus-left orientation
        gc = np.where(cr, sgn * ((1 - th) * ga + th * gb), 0.0)
        right = (th < 0.5) if phi_c is None else ((phi_c <= 0) != (a <= 0))   # centre beyond the crossing
        lo = -gc * (1 - th); hi = gc * th
        if g2 is not None:
            g2c = np.where(cr, sgn * ((1 - th) * g2 + th * np.roll(g2, -1, ax)), 0.0)
            lo = lo - 0.5 * g2c * (1 - th) ** 2 * h; hi = hi - 0.5 * g2c * th ** 2 * h
        C += np.where(cr, np.where(right, hi, lo), 0.0)
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

PN_BULK = True           # bulk-force term -div(sigma_s).n in the jump of the normal pressure gradient
DIV_SECOND = False       # second-derivative jumps in the cut-cell divergence correction (no measurable gain)
DIV_CENTRE_SIDE = True   # side of a cut cell's centre from phi at the centre, same for x and y


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


def _pressure_face_jumps_per(phi, jump, jgrad=None, h=None):
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
        if jgrad is None:
            out.append(np.where(cr, s * jc, 0.0))
        else:
            # second order: the jump of dp/dx_ax (j1) shifts the one-sided face gradient. With
            # j0 = s[p], j1 = s[p_ax] (right minus left), the face (midway) lies on the right
            # side if theta < 1/2:  grad = (q_R - q_L - j0 + j1 theta h)/h, else
            # grad = (q_R - q_L - j0 - j1 (1 - theta) h)/h.  Returned as h * (jump part).
            g = jgrad[0] if ax == 1 else jgrad[1]
            ga = np.roll(g, 1, ax); j1 = s * np.where(cr, (1 - th) * ga + th * g, 0.0)
            j0 = s * jc
            out.append(np.where(cr, np.where(th < 0.5, j0 - j1 * th * h, j0 + j1 * (1 - th) * h), 0.0))
    return out


def _helm_freeslip(rhs, coef, dx, dy, ax):
    """(I - coef Lap) x = rhs on the free-slip box for a face field stored in the periodic layout
    (N, N): along `ax` the faces are nodes with the wall face at index 0 (value 0, Dirichlet),
    along the other axis they are cell-centred with zero normal gradient. Direct solve:
    DST-I along `ax` (interior faces 1..N-1), DCT-II along the other axis."""
    from scipy.fft import dst, idst, dct, idct
    ot = 1 - ax; N = rhs.shape[ax]; No = rhs.shape[ot]; ha, ho = (dx, dy) if ax == 1 else (dy, dx)
    r = rhs[:, 1:] if ax == 1 else rhs[1:, :]
    la = -(2.0 - 2.0 * np.cos(np.pi * np.arange(1, N) / N)) / ha ** 2
    lo = -(2.0 - 2.0 * np.cos(np.pi * np.arange(No) / No)) / ho ** 2
    lam = (la[None, :] + lo[:, None]) if ax == 1 else (la[:, None] + lo[None, :])
    xh = dct(dst(r, type=1, axis=ax, norm="ortho"), type=2, axis=ot, norm="ortho") / (1.0 - coef * lam)
    x = idst(idct(xh, type=2, axis=ot, norm="ortho"), type=1, axis=ax, norm="ortho")
    out = np.zeros_like(rhs)
    if ax == 1:
        out[:, 1:] = x
    else:
        out[1:, :] = x
    return out


def sharp_step_per(u, v, dx, dy, dt, nu, rho, lap_eig, peig, phi_c, J, f_u=None, f_v=None,
                   conserve=True, J2=None, G_old=None, bc="periodic", U_lid=0.0):
    """One semi-implicit step with sharp interface jumps (periodic MAC grid, constant nu, rho).

    u, v (Ny, Nx): face velocities after explicit advection (and any bulk force already added).
    phi_c: cell-centred level set; J: traction_jumps(...) evaluated on cell centres.
      1. implicit viscosity  (I - dt nu Lap) u* = u + dt nu C_u   (C: IIM Laplacian correction)
      2. projection with the jump-corrected divergence and the pressure jump [p] (ghost fluid)
    All corrections are right-hand-side terms, so both solves stay direct FFT solves.

    G_old = (Gx, Gy): the jump-corrected pressure gradient on the faces from the previous step
    (incremental pressure form). The viscous solve then acts on u* = u^{n+1} + (dt/rho)(G_new -
    G_old), which is continuous across the interface up to O(dt^2); without it u* carries jumps
    (dt/rho)[grad p], (dt/rho)[grad grad p], ... of every order, and the result depends on dt at
    O(dt [grad p]). With G_old given, the function returns (u, v, p, (Gx, Gy)).

    bc = "lid": no-slip box with the top wall moving at U_lid (same layout as "freeslip"; the
    viscous solves are the direct DST-I/DST-II wall solves of pyRMT.mac). Use conserve=False: the
    walls exchange momentum with the fluid.

    bc = "freeslip": the free-slip box (normal velocity zero at the walls, zero normal gradient
    of the tangential velocity), for an interface that stays away from the walls. u and v are
    passed in the same (Ny, Nx) layout, u[:, 0] and v[0, :] being the (zero) wall faces -- the
    periodic wrap then reproduces the walled divergence exactly. peig must be the Neumann
    eigenvalues (pyRMT.mac.poisson_eigs_neumann); lap_eig is not used. Only the two direct solves
    and the wall faces of the pressure gradient differ from the periodic case."""
    from pyRMT.mac import (solve_helmholtz_periodic, solve_poisson_periodic, divergence_per,
                           solve_poisson_neumann)
    if bc == "periodic":
        helm_u = helm_v = lambda r: solve_helmholtz_periodic(r, lap_eig, dt * nu)
        pois = lambda r: solve_poisson_periodic(r, peig)
    elif bc == "lid":
        # no-slip walls, top wall moving at U_lid: Dirichlet in both directions for both
        # components; the lid enters as the right-hand-side inhomogeneity of the top u row.
        from pyRMT.mac import _dst_solve_u, _dst_solve_v
        def helm_u(r):
            q = r[:, 1:].copy(); q[-1, :] += dt * nu * 2.0 * U_lid / dy ** 2
            out = np.zeros_like(r); out[:, 1:] = _dst_solve_u(q, dt * nu, dx, dy); return out
        def helm_v(r):
            out = np.zeros_like(r); out[1:, :] = _dst_solve_v(r[1:, :], dt * nu, dx, dy); return out
        pois = lambda r: solve_poisson_neumann(r - r.mean(), peig)
    else:
        helm_u = lambda r: _helm_freeslip(r, dt * nu, dx, dy, 1)
        helm_v = lambda r: _helm_freeslip(r, dt * nu, dx, dy, 0)
        pois = lambda r: solve_poisson_neumann(r - r.mean(), peig)
    pu, pv = _to_u(phi_c), _to_v(phi_c)
    k2 = {} if J2 is None else J2
    # The intermediate velocity u* = u^{n+1} + (dt/rho) grad p is DISCONTINUOUS across the
    # interface by (dt/rho)[grad p] (a splitting artifact: [grad p] != 0 whenever the traction
    # varies along the interface), so the viscous solve needs the value-jump correction too.
    inc = G_old is not None
    if inc and len(G_old) == 4 and J2 is not None:
        # Faces the interface swept across since the last step: the stored gradient there is the
        # one-sided gradient of the OTHER side. Move it to the current side with [grad p]
        # (outside minus inside), otherwise u* carries a (dt/rho)[grad p] glitch on those faces.
        su_old, sv_old = G_old[2], G_old[3]
        su, sv = pu <= 0, pv <= 0
        gxo = G_old[0] + np.where(su & ~su_old, -1.0, 0.0) * _to_u(J2["px"]) + np.where(~su & su_old, 1.0, 0.0) * _to_u(J2["px"])
        gyo = G_old[1] + np.where(sv & ~sv_old, -1.0, 0.0) * _to_v(J2["py"]) + np.where(~sv & sv_old, 1.0, 0.0) * _to_v(J2["py"])
        G_old = (gxo, gyo)
    Cu = laplacian_jump_correction(pu, _to_u(J["ux"]), _to_u(J["uy"]), dx, dy,
                                   gxx=_to_u(k2["uxx"]) if k2 else None, gyy=_to_u(k2["uyy"]) if k2 else None,
                                   w=(dt / rho) * _to_u(k2["px"]) if (k2 and not inc) else None)
    Cv = laplacian_jump_correction(pv, _to_v(J["vx"]), _to_v(J["vy"]), dx, dy,
                                   gxx=_to_v(k2["vxx"]) if k2 else None, gyy=_to_v(k2["vyy"]) if k2 else None,
                                   w=(dt / rho) * _to_v(k2["py"]) if (k2 and not inc) else None)
    ru = u + dt * nu * Cu + (dt * f_u / rho if f_u is not None else 0.0)
    rv = v + dt * nu * Cv + (dt * f_v / rho if f_v is not None else 0.0)
    # p solves div(grad_c p) = (rho/dt) div_c(u*), with the corrected face gradient
    # grad_c p = (p_R - p_L - s [p]) / h on crossed faces; the known jump part goes to the RHS.
    cx, cy = _pressure_face_jumps_per(phi_c, J["jp"], None if J2 is None else (J2["px"], J2["py"]), dx)
    gxj = cx / dx; gyj = cy / dy                                  # jump part of the face gradient
    if conserve:
        # The interface corrections do not sum to exactly zero net force on the discrete grid,
        # which makes the body and the periodic flow drift (a grid-dependent spurious momentum).
        # Their net effect over the step is known in advance: the forcing in ru/rv plus the jump
        # part of the pressure gradient, which enters the velocity update as +dt/rho * gxj.
        # Remove it uniformly over the solid's faces, before the solves (the Helmholtz solve and
        # the raw projection gradient both preserve the mean), so total momentum is conserved.
        for r, g, ph, du in ((ru, gxj, pu, u), (rv, gyj, pv, v)):
            net = np.sum(r - du) + (dt / rho) * np.sum(g)
            m = ph <= 0
            if m.any():
                r[m] -= net / m.sum()
    if inc:        # incremental form: old pressure gradient inside the viscous solve ...
        us = helm_u(ru - (dt / rho) * G_old[0]) + (dt / rho) * G_old[0]
        vs = helm_v(rv - (dt / rho) * G_old[1]) + (dt / rho) * G_old[1]
    else:          # ... and added back, so the projection below still solves for the total pressure
        us = helm_u(ru)
        vs = helm_v(rv)
    dc = divergence_jump_correction_per(pu, _to_u(J["ux"]), pv, _to_v(J["vy"]), dx, dy,
                                        guxx=_to_u(k2["uxx"]) if (k2 and DIV_SECOND) else None,
                                        gvyy=_to_v(k2["vyy"]) if (k2 and DIV_SECOND) else None,
                                        phi_c=phi_c if DIV_CENTRE_SIDE else None)
    div = divergence_per(us, vs, dx, dy) + dc
    corr = (np.roll(gxj, -1, 1) - gxj) / dx + (np.roll(gyj, -1, 0) - gyj) / dy
    p = pois((rho / dt) * div + corr)
    Gx = (p - np.roll(p, 1, 1)) / dx - gxj
    Gy = (p - np.roll(p, 1, 0)) / dy - gyj
    if bc != "periodic":                                          # no correction of the wall faces
        Gx[:, 0] = 0.0; Gy[0, :] = 0.0
    un = us - (dt / rho) * Gx
    vn = vs - (dt / rho) * Gy
    if inc:
        return un, vn, p, (Gx, Gy, pu <= 0, pv <= 0)
    return un, vn, p


def bulk_solid_force_per(phi, sxx, sxy, syy, dx, dy, mode="cells"):
    """Bulk solid force chi * div(sigma_s) on the periodic MAC faces, for the sharp formulation.

    The divergence is the compact face divergence of the (smoothly extended) solid stress, kept
    only on faces whose two neighbouring cells are both inside the solid (phi <= 0). Faces that
    cross the interface carry no bulk force: there the traction enters through the jump
    conditions, so nothing is counted twice. Without this term the solid interior responds like
    fluid once its stress becomes non-uniform, and the interface traction injects energy."""
    from pyRMT.mac import solid_force_faces
    fu, fv = solid_force_faces(sxx, sxy, syy, dx, dy, periodic=True)
    if mode == "face":            # the face point itself is inside (uses the extended stress)
        fu = np.where(_to_u(phi) <= 0, fu, 0.0)
        fv = np.where(_to_v(phi) <= 0, fv, 0.0)
        return fu, fv
    ins = phi <= 0
    fu = np.where(ins & np.roll(ins, 1, 1), fu, 0.0)
    fv = np.where(ins & np.roll(ins, 1, 0), fv, 0.0)
    return fu, fv


def second_order_jumps(phi, J, divsig_x=None, divsig_y=None, mu=1.0, h=1.0, dp=None):
    """Higher-order jump conditions (outside minus inside), cell-centred, for second-order IIM.

    Inputs: phi (level set), J = traction_jumps(...) (first-derivative jumps of u, v and [p]),
    optionally the solid's bulk force div(sigma_s) evaluated inside and extended (it vanishes in
    the fluid, so [f] = -div(sigma_s)), and dp = the tangential traction G.tau (if None it is
    recovered from J). With n the outward normal, tau = (-n_y, n_x), kappa = div(n),
    A_i = [d u_i/dn] (continuous velocity => tangential derivatives continuous), and
    [p_n] = -d/ds (G.tau) - div(sigma_s).n,
    [grad p] = [p_n] n + d[p]/ds tau,  [Lap u_i] = ([d_i p] + div(sigma_s)_i) / mu,
    the Cartesian second-derivative jumps are
        [u_nn] = [Lap u] - kappa A,  [u_tt] = kappa A,  [u_nt] = dA/ds,
        [u_xx] = n_x^2 [u_nn] + 2 n_x t_x [u_nt] + t_x^2 [u_tt]   (and likewise for yy).
    Returns dict with uxx, uyy, vxx, vyy (velocity) and px, py ([d p/dx], [d p/dy])."""
    gy, gx = np.gradient(phi, h)
    nr = np.sqrt(gx ** 2 + gy ** 2) + 1e-300; nx, ny = gx / nr, gy / nr
    tx, ty = -ny, nx
    kap = np.gradient(nx, h, axis=1) + np.gradient(ny, h, axis=0)
    dds = lambda f: tx * np.gradient(f, h, axis=1) + ty * np.gradient(f, h, axis=0)
    if dp is None:                       # G.tau = mu * (A . tau) since mu A = G_tau tau
        dp = mu * (J["ux"] * nx * tx + J["vx"] * nx * ty + J["uy"] * ny * tx + J["vy"] * ny * ty)
    fx = divsig_x if divsig_x is not None else 0.0
    fy = divsig_y if divsig_y is not None else 0.0
    # [p_n] = mu n.[Lap u] + [f.n]: the kinematic part is -d(G.tau)/ds (from div u = 0 on both
    # sides), and the bulk force exists only inside, so [f.n] = -div(sigma_s).n.
    pn = -dds(dp) - ((fx * nx + fy * ny) if PN_BULK else 0.0)
    ps = dds(J["jp"])
    px = pn * nx + ps * tx; py = pn * ny + ps * ty
    out = dict(px=px, py=py)
    for c, gpi, fi in (("u", px, fx), ("v", py, fy)):
        A = J[c + "x"] * nx + J[c + "y"] * ny          # [d u_c / dn]
        lap = (gpi + fi) / mu
        unn = lap - kap * A; utt = kap * A; unt = dds(A)
        out[c + "xx"] = nx ** 2 * unn + 2 * nx * tx * unt + tx ** 2 * utt
        out[c + "yy"] = ny ** 2 * unn + 2 * ny * ty * unt + ty ** 2 * utt
    return out
