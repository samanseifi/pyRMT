"""Staggered (Marker-and-Cell) operators and projection.

Layout for Nx x Ny cells on [0,Lx] x [0,Ly], dx=Lx/Nx, dy=Ly/Ny:

    p : (Ny, Nx)      cell centres,  xc_i=(i+0.5)dx,  yc_j=(j+0.5)dy
    u : (Ny, Nx+1)    x-faces,       x_i = i dx,      yc_j
    v : (Ny+1, Nx)    y-faces,       xc_i,            y_j = j dy

u[:,0]/u[:,Nx] and v[0,:]/v[Ny,:] are the domain-boundary (wall) faces.

The discrete divergence (cell centres), pressure gradient (to faces) and Laplacian
(div of grad) are consistent by construction, so the projection is exact: the
projected velocity is divergence-free to machine precision (no checkerboard, no
Rhie-Chow needed).
"""

import numpy as np
from scipy.fft import dctn, idctn


def mac_grid(Nx, Ny, Lx=1.0, Ly=1.0):
    return Lx / Nx, Ly / Ny


def divergence(u, v, dx, dy):
    """Cell-centred divergence of a staggered velocity. u:(Ny,Nx+1), v:(Ny+1,Nx)
    -> (Ny, Nx)."""
    return (u[:, 1:] - u[:, :-1]) / dx + (v[1:, :] - v[:-1, :]) / dy


def gradient_p_u(p, dx):
    """d p/dx at u-faces. p:(Ny,Nx) -> (Ny,Nx+1). Wall faces (i=0,Nx) are 0
    (no pressure correction of the zero wall-normal velocity)."""
    Ny, Nx = p.shape
    g = np.zeros((Ny, Nx + 1))
    g[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / dx
    return g


def gradient_p_v(p, dy):
    """d p/dy at v-faces. p:(Ny,Nx) -> (Ny+1,Nx). Wall faces (j=0,Ny) are 0."""
    Ny, Nx = p.shape
    g = np.zeros((Ny + 1, Nx))
    g[1:-1, :] = (p[1:, :] - p[:-1, :]) / dy
    return g


def poisson_eigs_neumann(Nx, Ny, dx, dy):
    """Eigenvalues of the cell-centred Neumann Laplacian (dp/dn=0 at the walls),
    diagonalised by DCT-II.  lambda_k = -2(1-cos(pi k/N))/h^2.  The (0,0) constant
    mode is pinned to 1 (handled in the solve)."""
    kx = np.arange(Nx)
    ky = np.arange(Ny)
    lx = -2.0 * (1.0 - np.cos(np.pi * kx / Nx)) / dx**2
    ly = -2.0 * (1.0 - np.cos(np.pi * ky / Ny)) / dy**2
    eig = lx[np.newaxis, :] + ly[:, np.newaxis]
    eig = eig.copy()
    eig[0, 0] = 1.0
    return eig


def solve_poisson_neumann(rhs, eig):
    """Solve lap(p) = rhs with homogeneous-Neumann BC via DCT-II (mean removed)."""
    rhat = dctn(rhs, type=2, norm='ortho')
    phat = rhat / eig
    phat[0, 0] = 0.0
    return idctn(phat, type=2, norm='ortho')


def project(u_star, v_star, dx, dy, dt, rho, eig):
    """Project a staggered velocity onto the divergence-free space.

    Solves lap(phi) = (rho/dt) div(u*) and returns
    (u, v, p) with u = u* - (dt/rho) grad(phi), divergence-free to machine
    precision.  `rho` is a scalar (constant density).
    """
    div = divergence(u_star, v_star, dx, dy)
    rhs = (rho / dt) * div
    rhs = rhs - rhs.mean()                      # enforce solvability (zero mean)
    phi = solve_poisson_neumann(rhs, eig)
    u = u_star - (dt / rho) * gradient_p_u(phi, dx)
    v = v_star - (dt / rho) * gradient_p_v(phi, dy)
    return u, v, phi


# ── Lid-driven cavity: momentum (advection + diffusion) with ghost-cell BCs ───
# Walls: no-slip. Top lid moves at U_lid (tangential to the x-walls... at the top
# y-wall). Normal velocity at every wall is zero (u[:,0]=u[:,-1]=0, v[0,:]=v[-1,:]=0);
# tangential no-slip / lid is imposed via reflected ghost rows/cols.

def _u_ghost_y(u, U_lid):
    """Pad u (Ny,Nx+1) with one ghost row top & bottom enforcing tangential BC:
    bottom wall u=0 -> ghost=-u[0]; top lid u=U_lid -> ghost=2*U_lid-u[-1]."""
    Ny, Nxp1 = u.shape
    up = np.empty((Ny + 2, Nxp1))
    up[1:-1, :] = u
    up[0, :] = -u[0, :]                 # bottom no-slip
    up[-1, :] = 2.0 * U_lid - u[-1, :]  # top lid
    return up


def _v_ghost_x(v):
    """Pad v (Ny+1,Nx) with one ghost col left & right enforcing no-slip (v=0)."""
    Nyp1, Nx = v.shape
    vp = np.empty((Nyp1, Nx + 2))
    vp[:, 1:-1] = v
    vp[:, 0] = -v[:, 0]
    vp[:, -1] = -v[:, -1]
    return vp


def _v_at_u(v):
    """Interpolate v (Ny+1,Nx) to the interior u-faces (Ny, Nx-1)."""
    # u-face (i, j) sits between v[j,i-1],v[j,i],v[j+1,i-1],v[j+1,i]
    return 0.25 * (v[:-1, :-1] + v[:-1, 1:] + v[1:, :-1] + v[1:, 1:])


def _u_at_v(u):
    """Interpolate u (Ny,Nx+1) to the interior v-faces (Ny-1, Nx)."""
    return 0.25 * (u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:])


def interfacial_force_faces(kappa, H, gamma, dx, dy):
    """Continuum-surface-force at faces: f = -gamma * kappa * grad(H), with kappa
    (cell centres) interpolated to faces and grad(H) the SAME compact face gradient
    as the pressure gradient -> balanced-force by construction. Returns (fu, fv)
    on the u-faces (Ny,Nx+1) and v-faces (Ny+1,Nx); wall faces are 0."""
    Ny, Nx = H.shape
    fu = np.zeros((Ny, Nx + 1))
    fv = np.zeros((Ny + 1, Nx))
    # u-faces (interior i=1..Nx-1): kappa interp in x, grad H compact in x
    ku = 0.5 * (kappa[:, 1:] + kappa[:, :-1])
    fu[:, 1:-1] = -gamma * ku * (H[:, 1:] - H[:, :-1]) / dx
    # v-faces (interior j=1..Ny-1)
    kv = 0.5 * (kappa[1:, :] + kappa[:-1, :])
    fv[1:-1, :] = -gamma * kv * (H[1:, :] - H[:-1, :]) / dy
    return fu, fv


def momentum_predictor(u, v, nu, dx, dy, dt, U_lid, fu=None, fv=None, rho=1.0):
    """One explicit predictor step (central advection + diffusion) for the
    lid-driven cavity, plus optional face body forces fu (Ny,Nx+1)/fv (Ny+1,Nx)
    (e.g. surface tension) added as fu/rho. Returns u*, v* with wall faces zeroed."""
    Ny, Nxp1 = u.shape
    Nx = Nxp1 - 1
    up = _u_ghost_y(u, U_lid)            # (Ny+2, Nx+1)
    vp = _v_ghost_x(v)                   # (Ny+1, Nx+2)

    # --- u-momentum at interior u-faces i=1..Nx-1 ---
    uc = u[:, 1:-1]                                  # (Ny, Nx-1)
    dudx = (u[:, 2:] - u[:, :-2]) / (2 * dx)         # (Ny, Nx-1)
    dudy = (up[2:, 1:-1] - up[:-2, 1:-1]) / (2 * dy) # (Ny, Nx-1)
    lapu = ((u[:, 2:] - 2 * uc + u[:, :-2]) / dx**2
            + (up[2:, 1:-1] - 2 * up[1:-1, 1:-1] + up[:-2, 1:-1]) / dy**2)
    v_u = _v_at_u(v)                                 # (Ny, Nx-1)
    rhs_u = -(uc * dudx + v_u * dudy) + nu * lapu
    if fu is not None:
        rhs_u = rhs_u + fu[:, 1:-1] / rho
    ustar = u.copy()
    ustar[:, 1:-1] = uc + dt * rhs_u
    ustar[:, 0] = 0.0; ustar[:, -1] = 0.0

    # --- v-momentum at interior v-faces j=1..Ny-1 ---
    vc = v[1:-1, :]                                  # (Ny-1, Nx)
    dvdy = (v[2:, :] - v[:-2, :]) / (2 * dy)
    dvdx = (vp[1:-1, 2:] - vp[1:-1, :-2]) / (2 * dx)
    lapv = ((vp[1:-1, 2:] - 2 * vp[1:-1, 1:-1] + vp[1:-1, :-2]) / dx**2
            + (v[2:, :] - 2 * vc + v[:-2, :]) / dy**2)
    u_v = _u_at_v(u)                                 # (Ny-1, Nx)
    rhs_v = -(u_v * dvdx + vc * dvdy) + nu * lapv
    if fv is not None:
        rhs_v = rhs_v + fv[1:-1, :] / rho
    vstar = v.copy()
    vstar[1:-1, :] = vc + dt * rhs_v
    vstar[0, :] = 0.0; vstar[-1, :] = 0.0
    return ustar, vstar


# ── Periodic MAC (for Taylor-Green convergence) ──────────────────────────────
# Periodic layout (Nx x Ny): u,v both (Ny,Nx); u at x-faces (i dx,(j+.5)dy),
# v at y-faces ((i+.5)dx, j dy), p at centres. All operators use np.roll.

def divergence_per(u, v, dx, dy):
    return (np.roll(u, -1, 1) - u) / dx + (np.roll(v, -1, 0) - v) / dy


def grad_p_u_per(p, dx):
    return (p - np.roll(p, 1, 1)) / dx           # d/dx to x-face


def grad_p_v_per(p, dy):
    return (p - np.roll(p, 1, 0)) / dy           # d/dy to y-face


def poisson_eigs_periodic(Nx, Ny, dx, dy):
    kx = np.arange(Nx); ky = np.arange(Ny)
    lx = -4.0 * np.sin(np.pi * kx / Nx) ** 2 / dx**2
    ly = -4.0 * np.sin(np.pi * ky / Ny) ** 2 / dy**2
    eig = lx[np.newaxis, :] + ly[:, np.newaxis]
    eig = eig.copy(); eig[0, 0] = 1.0
    return eig


def solve_poisson_periodic(rhs, eig):
    rhat = np.fft.fft2(rhs)
    phat = rhat / eig; phat[0, 0] = 0.0
    return np.real(np.fft.ifft2(phat))


def project_per(u_star, v_star, dx, dy, dt, rho, eig):
    div = divergence_per(u_star, v_star, dx, dy)
    phi = solve_poisson_periodic((rho / dt) * div, eig)
    return (u_star - (dt / rho) * grad_p_u_per(phi, dx),
            v_star - (dt / rho) * grad_p_v_per(phi, dy), phi)


def _v_at_u_per(v):
    return 0.25 * (v + np.roll(v, 1, 1) + np.roll(v, -1, 0)
                   + np.roll(np.roll(v, -1, 0), 1, 1))


def _u_at_v_per(u):
    return 0.25 * (u + np.roll(u, -1, 1) + np.roll(u, 1, 0)
                   + np.roll(np.roll(u, 1, 0), -1, 1))


def momentum_predictor_periodic(u, v, nu, dx, dy, dt):
    """Forward-Euler predictor (central advection + diffusion) on the periodic
    staggered grid. Returns u*, v*."""
    dudx = (np.roll(u, -1, 1) - np.roll(u, 1, 1)) / (2 * dx)
    dudy = (np.roll(u, -1, 0) - np.roll(u, 1, 0)) / (2 * dy)
    lapu = ((np.roll(u, -1, 1) - 2 * u + np.roll(u, 1, 1)) / dx**2
            + (np.roll(u, -1, 0) - 2 * u + np.roll(u, 1, 0)) / dy**2)
    ru = -(u * dudx + _v_at_u_per(v) * dudy) + nu * lapu

    dvdx = (np.roll(v, -1, 1) - np.roll(v, 1, 1)) / (2 * dx)
    dvdy = (np.roll(v, -1, 0) - np.roll(v, 1, 0)) / (2 * dy)
    lapv = ((np.roll(v, -1, 1) - 2 * v + np.roll(v, 1, 1)) / dx**2
            + (np.roll(v, -1, 0) - 2 * v + np.roll(v, 1, 0)) / dy**2)
    rv = -(_u_at_v_per(u) * dvdx + v * dvdy) + nu * lapv
    return u + dt * ru, v + dt * rv


# ── Free-slip box momentum (for the disc-in-Taylor-Green benchmark) ──────────
# Normal velocity zero at walls; tangential free (zero normal-gradient ghosts).

def momentum_predictor_freeslip(u, v, nu, dx, dy, dt, fu=None, fv=None, rho=1.0):
    """Explicit predictor (central advection + diffusion) on a free-slip box,
    plus optional face body forces fu/fv. Returns u*, v* with wall (normal)
    faces zeroed."""
    Ny, Nxp1 = u.shape
    # free-slip tangential ghosts: mirror (+interior)
    up = np.empty((Ny + 2, Nxp1)); up[1:-1] = u; up[0] = u[0]; up[-1] = u[-1]
    Nyp1, Nx = v.shape
    vp = np.empty((Nyp1, Nx + 2)); vp[:, 1:-1] = v; vp[:, 0] = v[:, 0]; vp[:, -1] = v[:, -1]

    uc = u[:, 1:-1]
    dudx = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    dudy = (up[2:, 1:-1] - up[:-2, 1:-1]) / (2 * dy)
    lapu = ((u[:, 2:] - 2 * uc + u[:, :-2]) / dx**2
            + (up[2:, 1:-1] - 2 * up[1:-1, 1:-1] + up[:-2, 1:-1]) / dy**2)
    v_u = _v_at_u(v)
    rhs_u = -(uc * dudx + v_u * dudy) + nu * lapu
    if fu is not None:
        rhs_u = rhs_u + fu[:, 1:-1] / rho
    ustar = u.copy(); ustar[:, 1:-1] = uc + dt * rhs_u
    ustar[:, 0] = 0.0; ustar[:, -1] = 0.0

    vc = v[1:-1, :]
    dvdy = (v[2:, :] - v[:-2, :]) / (2 * dy)
    dvdx = (vp[1:-1, 2:] - vp[1:-1, :-2]) / (2 * dx)
    lapv = ((vp[1:-1, 2:] - 2 * vp[1:-1, 1:-1] + vp[1:-1, :-2]) / dx**2
            + (v[2:, :] - 2 * vc + v[:-2, :]) / dy**2)
    u_v = _u_at_v(u)
    rhs_v = -(u_v * dvdx + vc * dvdy) + nu * lapv
    if fv is not None:
        rhs_v = rhs_v + fv[1:-1, :] / rho
    vstar = v.copy(); vstar[1:-1, :] = vc + dt * rhs_v
    vstar[0, :] = 0.0; vstar[-1, :] = 0.0
    return ustar, vstar


# ── Conservative reference-map advection with the divergence-free face velocity ──
# Jain 2019 Eq. 26: d(xi)/dt + H div(u xi) = 0. Using the MAC FACE velocity (which
# is discretely divergence-free) makes div(u xi) = u.grad(xi) exactly -- no spurious
# xi*div(u) source (the cell-interpolated velocity is NOT divergence-free and folds
# the map). H gates the update to the solid; the band is filled by extrapolation.

def _xi_flux_div_faces(xi, u, v, dx, dy):
    """div(u xi) at cell centres using face velocities u:(Ny,Nx+1), v:(Ny+1,Nx)
    and xi:(Ny,Nx) interpolated to the faces (edge values at domain walls)."""
    Ny, Nx = xi.shape
    xi_uf = np.empty((Ny, Nx + 1))
    xi_uf[:, 1:-1] = 0.5 * (xi[:, :-1] + xi[:, 1:])
    xi_uf[:, 0] = xi[:, 0]; xi_uf[:, -1] = xi[:, -1]
    fx = u * xi_uf
    xi_vf = np.empty((Ny + 1, Nx))
    xi_vf[1:-1, :] = 0.5 * (xi[:-1, :] + xi[1:, :])
    xi_vf[0, :] = xi[0, :]; xi_vf[-1, :] = xi[-1, :]
    fy = v * xi_vf
    return (fx[:, 1:] - fx[:, :-1]) / dx + (fy[1:, :] - fy[:-1, :]) / dy


def advect_xi_conservative(xi, u, v, dx, dy, dt, phi, w_cut=0.0):
    """SSP-RK3 conservative advection of a cell-centred reference-map component
    with the divergence-free MAC face velocity, gated to phi<=w_cut (Jain Eq. 26)."""
    H = (phi <= w_cut).astype(float)
    def rhs(q):
        return -H * _xi_flux_div_faces(q, u, v, dx, dy)
    q1 = xi + dt * rhs(xi)
    q2 = 0.75 * xi + 0.25 * (q1 + dt * rhs(q1))
    return (1.0 / 3.0) * xi + (2.0 / 3.0) * (q2 + dt * rhs(q2))


# ── Solid-solid contact STRESS (Rycroft et al. 2018, arXiv 1810.03015 Eq.4.10-4.12) ──
# Unlike a repulsive BODY force (which is curl-free -> nullified by the projection),
# contact is added as a trace-free STRESS tensor; its divergence is a momentum-
# conserving force that survives the exact projection.

def contact_stress(phi_a, phi_b, eta, Gsum, eps, dx, dy):
    """Trace-free contact stress for a pair of solids whose blur zones overlap.

      f(phi)  = 1/2 (1 - phi/eps) for phi < eps, else 0   (contact intensity)
      n       = grad(phi_a - phi_b)/|grad(phi_a - phi_b)|  (pair normal)
      tau_col = -eta * min{f_a,f_b} * Gsum * (n⊗n - 1/2 I)   (2D, trace-free)

    Returns the cell-centred components (txx, txy, tyy), to be ADDED to the solid
    stress before taking its divergence. Gsum = G_a + G_b, eps ~ contact width.
    """
    fa = np.where(phi_a < eps, 0.5 * (1.0 - phi_a / eps), 0.0)
    fb = np.where(phi_b < eps, 0.5 * (1.0 - phi_b / eps), 0.0)
    fc = np.minimum(fa, fb)                       # active only where both overlap
    d = phi_a - phi_b
    dpx = np.zeros_like(d); dpy = np.zeros_like(d)
    dpx[:, 1:-1] = (d[:, 2:] - d[:, :-2]) / (2 * dx)
    dpy[1:-1, :] = (d[2:, :] - d[:-2, :]) / (2 * dy)
    mag = np.sqrt(dpx * dpx + dpy * dpy) + 1e-12
    nx = dpx / mag; ny = dpy / mag
    s = -eta * fc * Gsum
    return s * (nx * nx - 0.5), s * (nx * ny), s * (ny * ny - 0.5)
