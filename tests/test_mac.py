"""MAC staggered operators + projection: consistency and exactness.

The headline property (which the collocated solver cannot achieve): the discrete
divergence, pressure gradient and Laplacian are consistent, so the projection
makes the velocity divergence-free to machine precision.
"""
import numpy as np
import pytest

from pyRMT.mac import (
    mac_grid, divergence, gradient_p_u, gradient_p_v,
    poisson_eigs_neumann, solve_poisson_neumann, project,
)


def _centres(Nx, Ny, dx, dy):
    xc = (np.arange(Nx) + 0.5) * dx
    yc = (np.arange(Ny) + 0.5) * dy
    return np.meshgrid(xc, yc)            # (Ny, Nx)


def _u_faces(Nx, Ny, dx, dy):
    xf = np.arange(Nx + 1) * dx
    yc = (np.arange(Ny) + 0.5) * dy
    return np.meshgrid(xf, yc)            # (Ny, Nx+1)


def _v_faces(Nx, Ny, dx, dy):
    xc = (np.arange(Nx) + 0.5) * dx
    yf = np.arange(Ny + 1) * dy
    return np.meshgrid(xc, yf)            # (Ny+1, Nx)


def test_divergence_exact_on_linear():
    Nx, Ny = 16, 20
    dx, dy = mac_grid(Nx, Ny)
    Xu, Yu = _u_faces(Nx, Ny, dx, dy)
    Xv, Yv = _v_faces(Nx, Ny, dx, dy)
    u = 3.0 * Xu + 2.0 * Yu          # du/dx = 3
    v = -1.5 * Xv + 4.0 * Yv         # dv/dy = 4
    div = divergence(u, v, dx, dy)
    assert np.allclose(div, 3.0 + 4.0, atol=1e-10)


def test_divergence_second_order():
    """divergence of a smooth field converges at 2nd order."""
    errs = []
    for N in (24, 48):
        dx, dy = mac_grid(N, N)
        Xu, Yu = _u_faces(N, N, dx, dy)
        Xv, Yv = _v_faces(N, N, dx, dy)
        k = 2 * np.pi
        u = np.sin(k * Xu) * np.cos(k * Yu)
        v = np.cos(k * Xv) * np.sin(k * Yv)
        Xc, Yc = _centres(N, N, dx, dy)
        exact = k * np.cos(k * Xc) * np.cos(k * Yc) + k * np.cos(k * Xc) * np.cos(k * Yc)
        errs.append(np.max(np.abs(divergence(u, v, dx, dy) - exact)))
    order = np.log(errs[0] / errs[1]) / np.log(2)
    assert order > 1.8


def test_div_grad_equals_laplacian_consistency():
    """THE consistency property: div(grad p) computed with the MAC operators
    equals the DCT Neumann Laplacian applied to p (to machine precision). This
    is what makes the projection exact."""
    Nx, Ny = 24, 32
    dx, dy = mac_grid(Nx, Ny)
    rng_x = (np.arange(Nx) + 0.5) / Nx
    rng_y = (np.arange(Ny) + 0.5) / Ny
    Xc, Yc = np.meshgrid(rng_x, rng_y)
    # a Neumann-compatible field (cosines -> zero normal gradient at walls)
    p = np.cos(2 * np.pi * Xc) * np.cos(3 * np.pi * Yc)
    lap_mac = divergence(gradient_p_u(p, dx), gradient_p_v(p, dy), dx, dy)
    # apply the DCT Laplacian: idct( eig * dct(p) )
    from scipy.fft import dctn, idctn
    eig = poisson_eigs_neumann(Nx, Ny, dx, dy)
    eig_true = eig.copy(); eig_true[0, 0] = 0.0
    lap_dct = idctn(eig_true * dctn(p, type=2, norm='ortho'), type=2, norm='ortho')
    assert np.max(np.abs(lap_mac - lap_dct)) < 1e-9


def test_poisson_roundtrip():
    """solve(lap(p)) == p (mean-removed) to machine precision."""
    Nx, Ny = 32, 40
    dx, dy = mac_grid(Nx, Ny)
    rng_x = (np.arange(Nx) + 0.5) / Nx
    rng_y = (np.arange(Ny) + 0.5) / Ny
    Xc, Yc = np.meshgrid(rng_x, rng_y)
    p_true = np.cos(2 * np.pi * Xc) * np.cos(np.pi * Yc) + 0.3 * np.cos(4 * np.pi * Xc)
    lap = divergence(gradient_p_u(p_true, dx), gradient_p_v(p_true, dy), dx, dy)
    eig = poisson_eigs_neumann(Nx, Ny, dx, dy)
    p = solve_poisson_neumann(lap, eig)
    pt = p_true - p_true.mean()
    assert np.max(np.abs(p - pt)) < 1e-10


def test_projection_makes_divergence_machine_zero():
    """THE headline: projecting any (wall-BC) velocity yields machine-zero
    divergence -- exact, unlike the collocated approximate projection."""
    Nx, Ny = 48, 48
    dx, dy = mac_grid(Nx, Ny)
    Xu, Yu = _u_faces(Nx, Ny, dx, dy)
    Xv, Yv = _v_faces(Nx, Ny, dx, dy)
    rng = np.random.default_rng(0)
    u = np.sin(2 * np.pi * Xu) * np.cos(2 * np.pi * Yu) + 0.2 * rng.standard_normal((Ny, Nx + 1))
    v = np.cos(2 * np.pi * Xv) * np.sin(2 * np.pi * Yv) + 0.2 * rng.standard_normal((Ny + 1, Nx))
    # enforce no-penetration at walls
    u[:, 0] = 0.0; u[:, -1] = 0.0
    v[0, :] = 0.0; v[-1, :] = 0.0
    eig = poisson_eigs_neumann(Nx, Ny, dx, dy)
    un, vn, p = project(u, v, dx, dy, dt=0.01, rho=1.0, eig=eig)
    assert np.max(np.abs(divergence(un, vn, dx, dy))) < 1e-11


def test_interfacial_force_balanced_for_constant_curvature():
    """For constant curvature, f = -gamma*kappa*grad(H) must be the discrete
    gradient of (-gamma*kappa*H) at faces -- i.e. exactly balanceable by pressure
    (this is why MAC surface tension has small parasitic currents)."""
    from pyRMT.mac import mac_grid, interfacial_force_faces, gradient_p_u, gradient_p_v
    Nx, Ny = 40, 40
    dx, dy = mac_grid(Nx, Ny)
    xc = (np.arange(Nx) + 0.5) * dx
    Xc, Yc = np.meshgrid(xc, xc)
    H = 0.5 * (1 + np.tanh((np.sqrt((Xc-.5)**2+(Yc-.5)**2)-.25)/(2*dx)))  # smooth band
    gamma, kR = 0.1, 4.0
    kappa = np.full((Ny, Nx), kR)
    fu, fv = interfacial_force_faces(kappa, H, gamma, dx, dy)
    # potential phi = -gamma*kR*H ; grad(phi) at faces should equal (fu,fv)
    phi = -gamma * kR * H
    gu = gradient_p_u(phi, dx); gv = gradient_p_v(phi, dy)
    assert np.allclose(fu, gu, atol=1e-12) and np.allclose(fv, gv, atol=1e-12)
