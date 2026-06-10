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
