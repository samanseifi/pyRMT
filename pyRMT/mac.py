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
