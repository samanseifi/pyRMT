# pyRMT.py
#
# Copyright (c) 2025 Saman Seifi, PhD
#
# This code contains all the functionalities needed to deveklop simulations 
# using Reference Map Technique (RMT).
#

import numpy as np
from numba import njit, prange
import pyamg
from scipy.ndimage import gaussian_filter
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import LinearOperator, cg
from scipy.fft import dctn, idctn

from pyRMT.interpolators import bilinear_interpolate, bicubic_interpolate
from pyRMT.utils import (
    diff_upwind_3rd,
    grad_central_x_2nd,
    grad_central_y_2nd,
    lap_2nd,
    fast_solve_3x3,
)

def create_grid(Nx, Ny, Lx, Ly):
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    return X, Y, dx, dy

def apply_phi_BCs(phi):
    """
    Apply periodic boundary conditions to phi (3-cell periodic BCs).
    """
    phi[0:3, :] = phi[-6:-3, :]
    phi[-3:, :] = phi[3:6, :]
    phi[:, 0:3] = phi[:, -6:-3]
    phi[:, -3:] = phi[:, 3:6]
    
    # phi[0, :] = phi[1, :]
    # phi[-1, :] = phi[-2, :]
    # phi[:, 0] = phi[:, 1]
    # phi[:, -1] = phi[:, -2]
    return phi

@njit
def extrapolate_transverse_layers_2field(X1, X2, phi, dx, dy, band_width, max_layers):
    """
    Extrapolate solid reference maps (X1, X2) into fluid region.
    Iteratively extrapolates layer-by-layer near the interface.
    
    Seed known values:
        1.  Copy solid data (phi < 0) into X_ext and mark those cells as “known.”
    Identify frontier cells:
        2.  Find unknown cells adjacent (3×3 neighborhood) to any known cell—these form the next “layer.”
    Weighted local fit:
        3.  For each frontier cell, gather known neighbors within a radius, weight them by distance,
            and fit a plane f(x,y) = a + b*x + c*y. Assign f at the cell’s (x, y).
    Iterate layers:
        4.  Repeat “identify → fit → mark known” up to max_layers, stopping early if no new cells appear.
    
    Result:
        A smooth extension of your solid’s reference map into the fluid region.
    
    """
    Ny, Nx = X1.shape
    X1_ext = X1.copy()
    X2_ext = X2.copy()

    solid_flag = phi < 0
    known_flag = solid_flag.copy()

    stencil_radius_sq = (4 * np.sqrt(dx**2 + dy**2))**2

    for layer in range(max_layers):
        target_flag = np.zeros((Ny, Nx), dtype=np.bool_)

        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                if not known_flag[j, i]:
                    for dj in range(-1, 2):
                        for di in range(-1, 2):
                            if known_flag[j + dj, i + di]:
                                target_flag[j, i] = True
                                break
                        if target_flag[j, i]:
                            break

        if not np.any(target_flag):
            break

        for j in prange(1, Ny - 1):
            for i in range(1, Nx - 1):
                if target_flag[j, i]:
                    x0 = dx * i
                    y0 = dy * j

                    A = np.zeros((100, 3))
                    b1 = np.zeros(100)
                    b2 = np.zeros(100)
                    w = np.zeros(100)
                    count = 0

                    for jj in range(max(0, j - 4), min(Ny, j + 5)):
                        for ii in range(max(0, i - 4), min(Nx, i + 5)):
                            if known_flag[jj, ii]:
                                xi = dx * ii
                                yi = dy * jj
                                dist_sq = (xi - x0)**2 + (yi - y0)**2

                                if dist_sq <= stencil_radius_sq:
                                    A[count, 0] = 1.0
                                    A[count, 1] = xi
                                    A[count, 2] = yi
                                    b1[count] = X1_ext[jj, ii]
                                    b2[count] = X2_ext[jj, ii]
                                    w[count] = np.exp(-dist_sq / stencil_radius_sq)
                                    count += 1

                    if count >= 3:
                        Aw = np.zeros((3, 3))
                        Bw1 = np.zeros(3)
                        Bw2 = np.zeros(3)

                        for n in range(count):
                            a0 = A[n, 0]
                            a1 = A[n, 1]
                            a2 = A[n, 2]
                            wgt = w[n]
                            Bw1[0] += wgt * a0 * b1[n]
                            Bw1[1] += wgt * a1 * b1[n]
                            Bw1[2] += wgt * a2 * b1[n]
                            Bw2[0] += wgt * a0 * b2[n]
                            Bw2[1] += wgt * a1 * b2[n]
                            Bw2[2] += wgt * a2 * b2[n]

                            Aw[0, 0] += wgt * a0 * a0
                            Aw[0, 1] += wgt * a0 * a1
                            Aw[0, 2] += wgt * a0 * a2
                            Aw[1, 1] += wgt * a1 * a1
                            Aw[1, 2] += wgt * a1 * a2
                            Aw[2, 2] += wgt * a2 * a2

                        Aw[1, 0] = Aw[0, 1]
                        Aw[2, 0] = Aw[0, 2]
                        Aw[2, 1] = Aw[1, 2]

                        det = (Aw[0,0]*(Aw[1,1]*Aw[2,2] - Aw[1,2]*Aw[2,1])
                             - Aw[0,1]*(Aw[1,0]*Aw[2,2] - Aw[1,2]*Aw[2,0])
                             + Aw[0,2]*(Aw[1,0]*Aw[2,1] - Aw[1,1]*Aw[2,0]))

                        if np.abs(det) > 1e-10:
                            coeffs1 = fast_solve_3x3(Aw, Bw1)
                            coeffs2 = fast_solve_3x3(Aw, Bw2)

                            X1_ext[j, i] = coeffs1[0] + coeffs1[1] * x0 + coeffs1[2] * y0
                            X2_ext[j, i] = coeffs2[0] + coeffs2[1] * x0 + coeffs2[2] * y0
                            known_flag[j, i] = True

    return X1_ext, X2_ext

def compute_timestep(a, b, dx, dy, CFL, dt_min_cap, mu_s, rho_s, gamma, rho_f, mu_f=0.0, eta_s=0.0, kappa=0.0):
    # 1. Solid Wave Speed (P-wave: includes bulk modulus contribution)
    cs_solid = np.sqrt((kappa + mu_s * 4.0 / 3.0) / (rho_s + 1e-12))
    dt_solid = CFL * dx / (cs_solid + 1e-14)

    # 2. Fluid Advection Speed
    u_max = np.max(np.sqrt(a**2 + b**2))
    dt_fluid = CFL * dx / (u_max + 1e-6)

    # 3. Surface Tension Capillary Wave Speed
    dt_st = 1.0
    if gamma > 1e-12:
        # Approximate capillary timestep constraint (Brackbill)
        # dt < sqrt( (rho * dx^3) / (2 * pi * gamma) )
        rho_avg = 0.5 * (rho_s + rho_f)
        dt_st = np.sqrt( (rho_avg * dx**3) / (2 * np.pi * gamma) ) * 0.5 # Safety factor 0.5

    # 4. Viscous Diffusion Constraint
    # dt < rho_min * dx^2 / (2 * d * mu_max), where d = number of dimensions
    dt_visc = 1.0
    mu_max = max(mu_f, eta_s)
    rho_min = min(rho_s, rho_f)
    if mu_max > 1e-12 and rho_min > 1e-12:
        dt_visc = CFL * rho_min * dx**2 / (4.0 * mu_max)  # 4 = 2*d for 2D

    dt = min(dt_solid, dt_fluid, dt_st, dt_visc, dt_min_cap)

    return dt

@njit
def advect_semi_lagrangian_rk4(q, a, b, X, Y, dt, dx, dy):
    Ny, Nx = q.shape

    def interp(u, xq, yq):
        return bilinear_interpolate(u, xq, yq, dx, dy, Nx, Ny)
        # return bicubic_interpolate(u, xq, yq, dx, dy, Nx, Ny)

    # RK4 stages
    k1x = interp(a, X, Y)
    k1y = interp(b, X, Y)

    X2 = X - 0.5 * dt * k1x
    Y2 = Y - 0.5 * dt * k1y
    k2x = interp(a, X2, Y2)
    k2y = interp(b, X2, Y2)

    X3 = X - 0.5 * dt * k2x
    Y3 = Y - 0.5 * dt * k2y
    k3x = interp(a, X3, Y3)
    k3y = interp(b, X3, Y3)

    X4 = X - dt * k3x
    Y4 = Y - dt * k3y
    k4x = interp(a, X4, Y4)
    k4y = interp(b, X4, Y4)

    X_back = X - (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
    Y_back = Y - (dt / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)

    # Interpolate q at backtracked positions
    q_new = interp(q, X_back, Y_back)

    return q_new


# ── WENO5 + SSP-RK3 Eulerian advection ───────────────────────────────────────

@njit
def _weno5_left(vm2, vm1, v0, vp1, vp2):
    """Left-biased WENO5 reconstruction at i+1/2 (upwind for u > 0).
    Uses stencil {q_{i-2}, q_{i-1}, q_i, q_{i+1}, q_{i+2}}.
    Returns the reconstructed value at i+1/2."""
    eps = 1.0e-6

    # Three candidate polynomials
    r0 = (2.0*vm2 - 7.0*vm1 + 11.0*v0) / 6.0
    r1 = (-vm1 + 5.0*v0 + 2.0*vp1) / 6.0
    r2 = (2.0*v0 + 5.0*vp1 - vp2) / 6.0

    # Smoothness indicators (Jiang-Shu 1996)
    b0 = (13.0/12.0)*(vm2 - 2.0*vm1 + v0)**2 + (1.0/4.0)*(vm2 - 4.0*vm1 + 3.0*v0)**2
    b1 = (13.0/12.0)*(vm1 - 2.0*v0 + vp1)**2 + (1.0/4.0)*(vm1 - vp1)**2
    b2 = (13.0/12.0)*(v0 - 2.0*vp1 + vp2)**2 + (1.0/4.0)*(3.0*v0 - 4.0*vp1 + vp2)**2

    # Ideal weights
    d0, d1, d2 = 0.1, 0.6, 0.3

    # Non-linear weights
    a0 = d0 / (eps + b0)**2
    a1 = d1 / (eps + b1)**2
    a2 = d2 / (eps + b2)**2
    a_sum = a0 + a1 + a2

    w0 = a0 / a_sum
    w1 = a1 / a_sum
    w2 = a2 / a_sum

    return w0*r0 + w1*r1 + w2*r2


@njit
def _weno5_right(vm1, v0, vp1, vp2, vp3):
    """Right-biased WENO5 reconstruction at i+1/2 (upwind for u < 0).
    Uses stencil {q_{i-1}, q_i, q_{i+1}, q_{i+2}, q_{i+3}}.
    Returns the reconstructed value at i+1/2."""
    eps = 1.0e-6

    # Three candidate polynomials (mirror of left-biased)
    r0 = (2.0*vp3 - 7.0*vp2 + 11.0*vp1) / 6.0
    r1 = (-vp2 + 5.0*vp1 + 2.0*v0) / 6.0
    r2 = (2.0*vp1 + 5.0*v0 - vm1) / 6.0

    # Smoothness indicators (mirrored)
    b0 = (13.0/12.0)*(vp3 - 2.0*vp2 + vp1)**2 + (1.0/4.0)*(3.0*vp1 - 4.0*vp2 + vp3)**2
    b1 = (13.0/12.0)*(vp2 - 2.0*vp1 + v0)**2 + (1.0/4.0)*(vp2 - v0)**2
    b2 = (13.0/12.0)*(vp1 - 2.0*v0 + vm1)**2 + (1.0/4.0)*(vp1 - 4.0*v0 + 3.0*vm1)**2

    # Ideal weights (mirrored)
    d0, d1, d2 = 0.1, 0.6, 0.3

    a0 = d0 / (eps + b0)**2
    a1 = d1 / (eps + b1)**2
    a2 = d2 / (eps + b2)**2
    a_sum = a0 + a1 + a2

    w0 = a0 / a_sum
    w1 = a1 / a_sum
    w2 = a2 / a_sum

    return w0*r0 + w1*r1 + w2*r2


@njit(parallel=True)
def _weno5_rhs(q, a, b, dx, dy, phi, w_cut):
    """Compute RHS = -(u ∂q/∂x + v ∂q/∂y) using upwind WENO5.

    Only computes the RHS for cells where phi[j,i] <= w_cut (i.e. inside the
    solid + a thin buffer).  Fluid cells beyond the extrapolation band are
    left at zero so the scheme never sees the artificial zero-padding that
    results from the `* solid_mask` operation in the caller.

    The stencil from any solid cell (phi ≤ 0) extends at most 2 cells into
    the fluid.  As long as the caller extrapolates at least 2 layers beyond
    the interface, those stencil values are smooth continuations of the
    reference map and WENO5 operates correctly.
    """
    Ny, Nx = q.shape
    rhs = np.zeros_like(q)

    for j in prange(2, Ny - 2):
        for i in range(2, Nx - 2):
            # Only advect inside the solid (+ optional thin buffer)
            if phi[j, i] > w_cut:
                continue

            u_ij = a[j, i]
            v_ij = b[j, i]

            # ── x-direction ──────────────────────────────────────────────────
            # Face i+1/2
            if u_ij >= 0.0:
                qx_p = _weno5_left(q[j, i-2], q[j, i-1], q[j, i], q[j, i+1], q[j, i+2])
            else:
                if i + 3 < Nx:
                    qx_p = _weno5_right(q[j, i-1], q[j, i], q[j, i+1], q[j, i+2], q[j, i+3])
                else:
                    qx_p = _weno5_left(q[j, i-2], q[j, i-1], q[j, i], q[j, i+1], q[j, i+2])

            # Face i-1/2
            if u_ij >= 0.0:
                if i >= 3:
                    qx_m = _weno5_left(q[j, i-3], q[j, i-2], q[j, i-1], q[j, i], q[j, i+1])
                else:
                    qx_m = _weno5_left(q[j, i-2], q[j, i-1], q[j, i], q[j, i+1], q[j, i+2])
            else:
                qx_m = _weno5_right(q[j, i-1], q[j, i], q[j, i+1], q[j, i+2],
                                     q[j, i+3] if i+3 < Nx else q[j, Nx-1])

            dqdx = (qx_p - qx_m) / dx

            # ── y-direction ──────────────────────────────────────────────────
            # Face j+1/2
            if v_ij >= 0.0:
                qy_p = _weno5_left(q[j-2, i], q[j-1, i], q[j, i], q[j+1, i], q[j+2, i])
            else:
                if j + 3 < Ny:
                    qy_p = _weno5_right(q[j-1, i], q[j, i], q[j+1, i], q[j+2, i], q[j+3, i])
                else:
                    qy_p = _weno5_left(q[j-2, i], q[j-1, i], q[j, i], q[j+1, i], q[j+2, i])

            # Face j-1/2
            if v_ij >= 0.0:
                if j >= 3:
                    qy_m = _weno5_left(q[j-3, i], q[j-2, i], q[j-1, i], q[j, i], q[j+1, i])
                else:
                    qy_m = _weno5_left(q[j-2, i], q[j-1, i], q[j, i], q[j+1, i], q[j+2, i])
            else:
                qy_m = _weno5_right(q[j-1, i], q[j, i], q[j+1, i], q[j+2, i],
                                     q[j+3, i] if j+3 < Ny else q[Ny-1, i])

            dqdy = (qy_p - qy_m) / dy

            rhs[j, i] = -(u_ij * dqdx + v_ij * dqdy)

    return rhs


def advect_weno5_rk3(q, a, b, dx, dy, dt, phi, w_cut=0.0):
    """Advect scalar field q using WENO5 spatial discretisation
    and SSP-RK3 (Shu-Osher) time integration.

    phi   : level-set field (same shape as q); only cells with phi <= w_cut
            are advected.  w_cut=0.0 restricts advection to the solid interior.
    w_cut : cutoff phi value for the active advection region.  The stencil
            from any active cell extends at most 2 cells into the fluid, so
            the extrapolation buffer must be at least 2 layers wide.
    """
    # Stage 1
    q1 = q + dt * _weno5_rhs(q, a, b, dx, dy, phi, w_cut)

    # Stage 2
    q2 = 0.75 * q + 0.25 * (q1 + dt * _weno5_rhs(q1, a, b, dx, dy, phi, w_cut))

    # Stage 3
    q_new = (1.0/3.0) * q + (2.0/3.0) * (q2 + dt * _weno5_rhs(q2, a, b, dx, dy, phi, w_cut))

    return q_new


# ── 2nd-order central difference + SSP-RK3 advection ─────────────────────────

@njit(parallel=True)
def _central2_rhs(q, a, b, dx, dy, phi, w_cut):
    """Compute RHS = -(u ∂q/∂x + v ∂q/∂y) using 2nd-order central differences.

    Only evaluates at cells where phi[j,i] <= w_cut.  Stencil width is ±1,
    so a single extrapolation layer beyond the interface is sufficient.
    """
    Ny, Nx = q.shape
    rhs = np.zeros_like(q)
    inv_2dx = 0.5 / dx
    inv_2dy = 0.5 / dy

    for j in prange(1, Ny - 1):
        for i in range(1, Nx - 1):
            if phi[j, i] > w_cut:
                continue
            dqdx = (q[j, i+1] - q[j, i-1]) * inv_2dx
            dqdy = (q[j+1, i] - q[j-1, i]) * inv_2dy
            rhs[j, i] = -(a[j, i] * dqdx + b[j, i] * dqdy)

    return rhs


def advect_central2_rk3(q, a, b, dx, dy, dt, phi, w_cut=0.0):
    """Advect scalar field q using 2nd-order central differences in space
    and SSP-RK3 (Shu-Osher) in time.

    phi   : level-set field; only cells with phi <= w_cut are advected.
    w_cut : 0.0 restricts advection to the solid interior.
    """
    # Stage 1
    q1 = q + dt * _central2_rhs(q, a, b, dx, dy, phi, w_cut)

    # Stage 2
    q2 = 0.75 * q + 0.25 * (q1 + dt * _central2_rhs(q1, a, b, dx, dy, phi, w_cut))

    # Stage 3
    q_new = (1.0/3.0) * q + (2.0/3.0) * (q2 + dt * _central2_rhs(q2, a, b, dx, dy, phi, w_cut))

    return q_new


# ── Switchable reference-map advection dispatcher ────────────────────────────

def advect_reference_map(q, a, b, X, Y, dt, dx, dy, phi,
                         scheme='semilagrangian', w_cut=0.0):
    """Advect a reference-map component ``q`` with a selectable scheme.

    Parameters
    ----------
    scheme : {'semilagrangian', 'central2', 'weno5'}
        'semilagrangian' : RK4 semi-Lagrangian backtrace (bilinear interp).
                           Robust and keeps the reconstructed level set
                           well-behaved; this is the DEFAULT.  The caller is
                           expected to mask/extrapolate the fluid side.
        'central2'       : 2nd-order central + SSP-RK3, Eulerian, only updates
                           cells with phi <= w_cut (solid + thin buffer).
        'weno5'          : WENO5 + SSP-RK3, Eulerian, only updates cells with
                           phi <= w_cut.
    w_cut : float
        Cutoff level-set value for the Eulerian schemes (ignored by the
        semi-Lagrangian branch).
    """
    if scheme == 'semilagrangian':
        return advect_semi_lagrangian_rk4(q, a, b, X, Y, dt, dx, dy)
    elif scheme == 'central2':
        return advect_central2_rk3(q, a, b, dx, dy, dt, phi, w_cut)
    elif scheme == 'weno5':
        return advect_weno5_rk3(q, a, b, dx, dy, dt, phi, w_cut)
    else:
        raise ValueError(
            "Unknown advection scheme %r (expected 'semilagrangian', "
            "'central2' or 'weno5')" % (scheme,)
        )


@njit(parallel=True)
def compute_solid_stress(X1, X2, dx, dy, mu_s, kappa, phi):
    Ny, Nx = X1.shape
    sxx = np.zeros((Ny, Nx))
    sxy = np.zeros((Ny, Nx))
    syy = np.zeros((Ny, Nx))
    J = np.ones((Ny, Nx))

    inv_2dx = 1.0 / (2.0 * dx)
    inv_2dy = 1.0 / (2.0 * dy)

    for j in prange(1, Ny - 1):
        for i in range(1, Nx - 1):
            if phi[j, i] <= 0:
                # 1. Gradients of Reference Map X (G = grad_x X)
                # Use one-sided stencil when a neighbor is in the fluid (phi > 0),
                # to avoid mixing solid and extrapolated fluid-side values.
                # x-direction (g11, g21)
                left_fluid  = phi[j, i-1] > 0.0
                right_fluid = phi[j, i+1] > 0.0
                if left_fluid and not right_fluid:
                    # forward difference
                    g11 = (X1[j, i+1] - X1[j, i]) / dx
                    g21 = (X2[j, i+1] - X2[j, i]) / dx
                elif right_fluid and not left_fluid:
                    # backward difference
                    g11 = (X1[j, i] - X1[j, i-1]) / dx
                    g21 = (X2[j, i] - X2[j, i-1]) / dx
                else:
                    # central (both solid, or both fluid — latter is rare boundary case)
                    g11 = (X1[j, i+1] - X1[j, i-1]) * inv_2dx
                    g21 = (X2[j, i+1] - X2[j, i-1]) * inv_2dx

                # y-direction (g12, g22)
                bot_fluid = phi[j-1, i] > 0.0
                top_fluid = phi[j+1, i] > 0.0
                if bot_fluid and not top_fluid:
                    # forward difference
                    g12 = (X1[j+1, i] - X1[j, i]) / dy
                    g22 = (X2[j+1, i] - X2[j, i]) / dy
                elif top_fluid and not bot_fluid:
                    # backward difference
                    g12 = (X1[j, i] - X1[j-1, i]) / dy
                    g22 = (X2[j, i] - X2[j-1, i]) / dy
                else:
                    g12 = (X1[j+1, i] - X1[j-1, i]) * inv_2dy
                    g22 = (X2[j+1, i] - X2[j-1, i]) * inv_2dy

                # 2. Deformation Gradient F = inv(G)
                detG = g11 * g22 - g12 * g21
                if abs(detG) < 1e-10:
                    continue
                # Clamp detG: prevents J from blowing up when the reference map
                # folds. Bounds J to [1/3, 3]. Normal operation stays in ~[0.6, 1.1].
                # if detG < 0.33:
                #     detG = 0.33
                # elif detG > 3.0:
                #     detG = 3.0

                # F components
                f11, f12 =  g22 / detG, -g12 / detG
                f21, f22 = -g21 / detG,  g11 / detG

                # 3. Left Cauchy-Green B = F * F.T
                b11 = f11*f11 + f12*f12
                b12 = f11*f21 + f12*f22
                b22 = f21*f21 + f22*f22

                # 4. Jacobian J = det(F) = 1/det(G)
                j_val = 1.0 / detG
                J[j, i] = j_val

                # 5. Neo-Hookean Stress (Kamrin form): sigma = mu*B + kappa*(J-1)*I
                vol_term = kappa * (j_val - 1.0)

                s_xx = mu_s * b11 + vol_term
                s_xy = mu_s * b12
                s_yy = mu_s * b22 + vol_term

                # Clamp stress magnitude: cap at 50*mu_s to prevent blow-up
                # from corrupt reference map without zeroing the stress direction.
                # stress_mag = (s_xx*s_xx + 2.0*s_xy*s_xy + s_yy*s_yy) ** 0.5
                # max_stress = 50.0 * mu_s
                # if stress_mag > max_stress:
                #     scale = max_stress / stress_mag
                #     s_xx *= scale
                #     s_xy *= scale
                #     s_yy *= scale

                sxx[j, i] = s_xx
                sxy[j, i] = s_xy
                syy[j, i] = s_yy

    # apply smoothing
    # sxx = gaussian_filter(sxx, sigma=0.5)
    # sxy = gaussian_filter(sxy, sigma=0.5)
    # syy = gaussian_filter(syy, sigma=0.5)

    return sxx, sxy, syy, J

def heaviside_smooth_alt(x, w_t):
    inv_wt = 1.0 / w_t
    inv_pi = 1.0 / np.pi

    # Compute transition formula for all points (will be masked later)
    H = 0.5 * (1.0 + x * inv_wt + inv_pi * np.sin(np.pi * x * inv_wt))

    # Apply masks using np.where (more efficient than boolean indexing)
    H = np.where(x > w_t, 1.0, H)
    H = np.where(x < -w_t, 0.0, H)

    return H

def velocity_RK4(u, v, p, X1, X2, velocity_bc, mu_s, kappa, eta_s , dx, dy, dt, rho_s, rho_f, phi, mu_f, w_t, gamma=0.0):
    """
    RK4 integration using stress divergence from blended stress field.
    """

    # PRE-COMPUTE ELASTIC STRESS (independent of velocity during RK4 stages)
    # Only the reference maps (X1, X2) and phi matter for elastic stress
    sigma_sxx_elastic, sigma_sxy_elastic, sigma_syy_elastic, J = compute_solid_stress(X1, X2, dx, dy, mu_s, kappa, phi)

    # PRE-COMPUTE HEAVISIDE AND GRADIENTS (constant during RK4)
    H = heaviside_smooth_alt(phi, w_t)
    dH_dx = grad_central_x_2nd(H, dx)
    dH_dy = grad_central_y_2nd(H, dy)
    rho_local = (1 - H) * rho_s + H * rho_f

    # PRE-COMPUTE SURFACE TENSION FORCE (if needed)
    if gamma > 1e-12:
        kappa_curv = compute_curvature(phi, dx, dy)
        st_force_x = -gamma * kappa_curv * dH_dx
        st_force_y = -gamma * kappa_curv * dH_dy
    else:
        st_force_x = 0.0
        st_force_y = 0.0

    solid_mask = (phi <= 0.0)

    def rhs(u_stage, v_stage):
        # Apply BCs to intermediate stage velocities so that Laplacian,
        # upwind, and gradient operators see correct boundary values
        u_stage, v_stage = apply_velocity_BCs(velocity_bc, u_stage, v_stage)

        # Start with elastic stress
        if eta_s > 0.0 and np.any(solid_mask):
            # Add viscous damping contribution (Kelvin-Voigt)
            sigma_sxx = sigma_sxx_elastic.copy()
            sigma_sxy = sigma_sxy_elastic.copy()
            sigma_syy = sigma_syy_elastic.copy()

            du_dx = grad_central_x_2nd(u_stage, dx)
            dv_dy = grad_central_y_2nd(v_stage, dy)
            du_dy = grad_central_y_2nd(u_stage, dy)
            dv_dx = grad_central_x_2nd(v_stage, dx)

            sigma_sxx[solid_mask] += eta_s * du_dx[solid_mask]
            sigma_syy[solid_mask] += eta_s * dv_dy[solid_mask]
            sigma_sxy[solid_mask] += eta_s * 0.5 * (du_dy[solid_mask] + dv_dx[solid_mask])
        else:
            # No viscous damping, use pre-computed elastic stress
            sigma_sxx = sigma_sxx_elastic
            sigma_sxy = sigma_sxy_elastic
            sigma_syy = sigma_syy_elastic

        return velocity_rhs_blended_optimized(
            u_stage, v_stage, p, sigma_sxx, sigma_sxy, sigma_syy,
            dx, dy, phi, mu_f, H, dH_dx, dH_dy, rho_local,
            st_force_x, st_force_y
        )

    k1u, k1v = rhs(u, v)

    u1 = u + 0.5 * dt * k1u
    v1 = v + 0.5 * dt * k1v
    k2u, k2v = rhs(u1, v1)

    u2 = u + 0.5 * dt * k2u
    v2 = v + 0.5 * dt * k2v
    k3u, k3v = rhs(u2, v2)

    u3 = u + dt * k3u
    v3 = v + dt * k3v
    k4u, k4v = rhs(u3, v3)

    u_new = u + (dt / 6.0) * (k1u + 2*k2u + 2*k3u + k4u)
    v_new = v + (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)

    u_new, v_new = apply_velocity_BCs(velocity_bc, u_new, v_new)

    return u_new, v_new, sigma_sxx_elastic, sigma_sxy_elastic, sigma_syy_elastic, J


@njit
def compute_curvature(phi, dx, dy) -> np.ndarray:
    """
    Computes curvature kappa = div(grad(phi) / |grad(phi)|)
    """
    # 1. Compute gradients of Phi
    phi_x = grad_central_x_2nd(phi, dx)
    phi_y = grad_central_y_2nd(phi, dy)
    
    # 2. Compute Normal Vector n = grad(phi) / |grad(phi)|
    grad_mag = np.sqrt(phi_x**2 + phi_y**2) + 1e-12 # Prevent div by zero
    
    nx = phi_x / grad_mag
    ny = phi_y / grad_mag
    
    # 3. Compute Curvature kappa = div(n)
    # Using 2nd order central difference is usually sufficient/stable for normals
    # but since we have 4th order available, we can use it, 
    # provided phi is smooth (reinitialized).
    dnx_dx = grad_central_x_2nd(nx, dx)
    dny_dy = grad_central_y_2nd(ny, dy)
    
    kappa = dnx_dx + dny_dy
    
    return kappa

def velocity_rhs_blended_optimized(u, v, p,
                                   sigma_sxx_s_elastic, sigma_sxy_s_elastic, sigma_syy_s_elastic,
                                   dx, dy, phi, mu_f,
                                   H, dH_dx, dH_dy, rho_local,
                                   st_force_x, st_force_y):
    """
    CONSERVATIVE BLENDING: Blends stress tensors before taking divergence.
    Reference: Jain et al. (2019), Section 4.3
    """
    # 1. Compute Fluid Viscous Stress components (excluding pressure)
    du_dx = grad_central_x_2nd(u, dx)
    dv_dy = grad_central_y_2nd(v, dy)
    du_dy = grad_central_y_2nd(u, dy)
    dv_dx = grad_central_x_2nd(v, dx)

    sigma_fxx = 2 * mu_f * du_dx
    sigma_fyy = 2 * mu_f * dv_dy
    sigma_fxy = mu_f * (du_dy + dv_dx)

    # 2. Blend the Stress Tensors (Global Cauchy Stress)
    # sigma_global = H * sigma_f + (1 - H) * sigma_s
    # Note: Pressure is handled separately as -grad(P) in the RHS
    sig_xx = H * sigma_fxx + (1 - H) * sigma_sxx_s_elastic
    sig_yy = H * sigma_fyy + (1 - H) * sigma_syy_s_elastic
    sig_xy = H * sigma_fxy + (1 - H) * sigma_sxy_s_elastic

    # 3. Compute Divergence of the BLENDED stress
    dsxx_dx = grad_central_x_2nd(sig_xx, dx)
    dsxy_dy = grad_central_y_2nd(sig_xy, dy)
    dsxy_dx = grad_central_x_2nd(sig_xy, dx)
    dsyy_dy = grad_central_y_2nd(sig_yy, dy)

    div_sigma_x = dsxx_dx + dsxy_dy
    div_sigma_y = dsxy_dx + dsyy_dy

    # 4. Advection (3rd order upwind)
    u_adv = -u * diff_upwind_3rd(u, u, dx, 1) - v * diff_upwind_3rd(u, v, dy, 0)
    v_adv = -u * diff_upwind_3rd(v, u, dx, 1) - v * diff_upwind_3rd(v, v, dy, 0)

    # 5. Pressure Gradient
    dp_dx = grad_central_x_2nd(p, dx)
    dp_dy = grad_central_y_2nd(p, dy)

    # 6. Final RHS using the conservative divergence
    rhs_u = u_adv + (div_sigma_x + st_force_x - dp_dx) / (rho_local + 1e-12)
    rhs_v = v_adv + (div_sigma_y + st_force_y - dp_dy) / (rho_local + 1e-12)

    return rhs_u, rhs_v

def apply_velocity_BCs(bc, u, v):
    return bc(u, v)

def build_poisson_matrix(Nx, Ny, dx, dy):
    N = Nx * Ny
    A = lil_matrix((N, N))
    
    # Pre-compute coefficients
    cx = 1.0 / dx**2
    cy = 1.0 / dy**2

    def idx(i, j):
        return i + j * Nx

    for j in range(Ny):
        for i in range(Nx):
            k = idx(i, j)

            # 1. Initialize diagonal with standard interior stencil value
            #    (This remains constant: -2/dx^2 - 2/dy^2)
            A[k, k] = -2 * cx - 2 * cy

            # 2. X-Direction Neighbors
            # ------------------------
            # Check West (i-1)
            if i > 0:
                A[k, idx(i - 1, j)] += cx
            else:
                # Boundary! Ghost point p[-1] == p[1].
                # We add the missing contribution to the EXISTING neighbor (p[1]).
                A[k, idx(i + 1, j)] += cx

            # Check East (i+1)
            if i < Nx - 1:
                A[k, idx(i + 1, j)] += cx
            else:
                # Boundary! Ghost point p[N] == p[N-2].
                A[k, idx(i - 1, j)] += cx

            # 3. Y-Direction Neighbors
            # ------------------------
            # Check South (j-1)
            if j > 0:
                A[k, idx(i, j - 1)] += cy
            else:
                # Boundary! Ghost point p[-1] == p[1].
                A[k, idx(i, j + 1)] += cy

            # Check North (j+1)
            if j < Ny - 1:
                A[k, idx(i, j + 1)] += cy
            else:
                # Boundary! Ghost point p[N] == p[N-2].
                A[k, idx(i, j - 1)] += cy

    # Note: Matrix A is singular (all Neumann). 
    # You must pin a node or enforce mean(p)=0 during the solve.
    return A.tocsr()

def _compute_divergence(a_star, b_star, dx, dy):
    """Compute velocity divergence using 2nd-order central differences.
    Boundary cells are zero — consistent with Neumann BC (dp/dn=0) in the Poisson solve.
    """
    divU = np.zeros_like(a_star)
    divU[1:-1, 1:-1] = (
        (a_star[1:-1, 2:] - a_star[1:-1, :-2]) / (2 * dx) +
        (b_star[2:,   1:-1] - b_star[:-2, 1:-1]) / (2 * dy)
    )
    return divU

def _compute_divergence_rc(a_star, b_star, p_prev, dt, rho, dx, dy):
    """Compute velocity divergence using Rhie-Chow interpolated face velocities.

    Eliminates odd-even pressure decoupling on collocated grids by replacing
    the naive wide central-difference divergence with face-velocity divergence
    that includes a pressure correction term.
    """
    Ny, Nx = a_star.shape
    divU = np.zeros((Ny, Nx))

    is_variable_rho = isinstance(rho, np.ndarray) and rho.ndim == 2 and np.ptp(rho) > 1e-10

    # Cell-center pressure gradients (same stencil as grad_central_x_2nd / grad_central_y_2nd)
    dpdx_cc = np.zeros((Ny, Nx))
    dpdy_cc = np.zeros((Ny, Nx))

    dpdx_cc[:, 1:-1] = (p_prev[:, 2:] - p_prev[:, :-2]) / (2.0 * dx)
    dpdx_cc[:, 0]  = (-3.0*p_prev[:, 0]  + 4.0*p_prev[:, 1]  - p_prev[:, 2])  / (2.0*dx)
    dpdx_cc[:, -1] = ( 3.0*p_prev[:, -1] - 4.0*p_prev[:, -2] + p_prev[:, -3]) / (2.0*dx)

    dpdy_cc[1:-1, :] = (p_prev[2:, :] - p_prev[:-2, :]) / (2.0 * dy)
    dpdy_cc[0, :]  = (-3.0*p_prev[0, :]  + 4.0*p_prev[1, :]  - p_prev[2, :])  / (2.0*dy)
    dpdy_cc[-1, :] = ( 3.0*p_prev[-1, :] - 4.0*p_prev[-2, :] + p_prev[-3, :]) / (2.0*dy)

    # --- X-direction: face velocities at (j, i+1/2) for i=0..Nx-2 ---
    u_face = 0.5 * (a_star[:, :-1] + a_star[:, 1:])                     # (Ny, Nx-1)
    face_dpdx = (p_prev[:, 1:] - p_prev[:, :-1]) / dx                   # compact
    avg_dpdx  = 0.5 * (dpdx_cc[:, :-1] + dpdx_cc[:, 1:])               # interpolated wide

    if is_variable_rho:
        inv_rho = 1.0 / rho
        d_f_x = dt * 0.5 * (inv_rho[:, :-1] + inv_rho[:, 1:])
    else:
        d_f_x = dt / float(np.mean(rho))   # scalar: rho may be a constant 2D array

    u_face_rc = u_face - d_f_x * (face_dpdx - avg_dpdx)

    # --- Y-direction: face velocities at (j+1/2, i) for j=0..Ny-2 ---
    v_face = 0.5 * (b_star[:-1, :] + b_star[1:, :])                     # (Ny-1, Nx)
    face_dpdy = (p_prev[1:, :] - p_prev[:-1, :]) / dy                   # compact
    avg_dpdy  = 0.5 * (dpdy_cc[:-1, :] + dpdy_cc[1:, :])               # interpolated wide

    if is_variable_rho:
        d_f_y = dt * 0.5 * (inv_rho[:-1, :] + inv_rho[1:, :])
    else:
        d_f_y = dt / float(np.mean(rho))   # scalar: rho may be a constant 2D array

    v_face_rc = v_face - d_f_y * (face_dpdy - avg_dpdy)

    # --- Compact divergence from face velocities (interior cells only) ---
    divU[1:-1, 1:-1] = (
        (u_face_rc[1:-1, 1:] - u_face_rc[1:-1, :-1]) / dx +
        (v_face_rc[1:, 1:-1] - v_face_rc[:-1, 1:-1]) / dy
    )

    return divU

def _compute_pressure_gradient(p, dx, dy):
    """Compute pressure gradient with 2nd-order central (interior) and one-sided (boundary) stencils."""
    dpdx = np.zeros_like(p)
    dpdy = np.zeros_like(p)

    dpdx[1:-1, 1:-1] = (p[1:-1, 2:]   - p[1:-1, :-2]) / (2 * dx)
    dpdy[1:-1, 1:-1] = (p[2:,   1:-1] - p[:-2, 1:-1]) / (2 * dy)

    # left/right boundaries in x
    dpdx[:, 0]  = (-3.0*p[:, 0]  + 4.0*p[:, 1]  - p[:, 2])   / (2.0 * dx)
    dpdx[:, -1] = ( 3.0*p[:, -1] - 4.0*p[:, -2] + p[:, -3]) / (2.0 * dx)

    # bottom/top boundaries in y
    dpdy[0, :]  = (-3.0*p[0, :]  + 4.0*p[1, :]  - p[2, :])   / (2.0 * dy)
    dpdy[-1, :] = ( 3.0*p[-1, :] - 4.0*p[-2, :] + p[-3, :]) / (2.0 * dy)

    return dpdx, dpdy

def _precompute_poisson_eigenvalues(Nx, Ny, dx, dy):
    """
    Precompute eigenvalues of the discrete Laplacian with Neumann BCs.
    The matrix uses ghost-point mirroring p[-1]=p[1], p[N]=p[N-2], which
    corresponds to DCT-I: λ = -2(1-cos(πk/(N-1)))/h².
    """
    kx = np.arange(Nx)
    ky = np.arange(Ny)
    lam_x = -2.0 * (1.0 - np.cos(np.pi * kx / (Nx - 1))) / dx**2
    lam_y = -2.0 * (1.0 - np.cos(np.pi * ky / (Ny - 1))) / dy**2
    eigenvalues = lam_x[np.newaxis, :] + lam_y[:, np.newaxis]
    # Pin the (0,0) mode to avoid division by zero (mean is removed separately)
    eigenvalues[0, 0] = 1.0
    return eigenvalues


def _solve_poisson_dct(rhs_2d, eigenvalues):
    """
    Direct Poisson solve using DCT-I (type=1, unnormalized).
    O(N log N) — no iteration needed.
    Matches build_poisson_matrix ghost-cell convention: p[-1]=p[1], p[N]=p[N-2].
    DCT-I diagonalizes the asymmetric Neumann Poisson matrix exactly.
    Do NOT use norm='ortho' — it changes the transform matrix and breaks diagonalization.
    """
    rhs_hat = dctn(rhs_2d, type=1)
    p_hat = rhs_hat / eigenvalues
    p = idctn(p_hat, type=1)
    p -= np.mean(p)
    return p


def _apply_variable_poisson(p_flat, Nx, Ny, dx, dy, inv_rho):
    """
    Matrix-free application of ∇·((1/ρ)∇p) using face-averaged 1/ρ.
    Avoids building an explicit sparse matrix every step.
    """
    p = p_flat.reshape((Ny, Nx))
    result = np.zeros_like(p)
    cx = 1.0 / dx**2
    cy = 1.0 / dy**2

    # Interior + boundary via padded arrays with Neumann ghost points
    # Pad p with ghost values: p[-1]=p[1], p[N]=p[N-2]
    p_padx = np.empty((Ny, Nx + 2))
    p_padx[:, 1:-1] = p
    p_padx[:, 0] = p[:, 1]      # ghost left = mirror of col 1
    p_padx[:, -1] = p[:, -2]    # ghost right = mirror of col N-2

    p_pady = np.empty((Ny + 2, Nx))
    p_pady[1:-1, :] = p
    p_pady[0, :] = p[1, :]      # ghost bottom = mirror of row 1
    p_pady[-1, :] = p[-2, :]    # ghost top = mirror of row N-2

    # Face-averaged inv_rho in x-direction
    inv_rho_padx = np.empty((Ny, Nx + 2))
    inv_rho_padx[:, 1:-1] = inv_rho
    inv_rho_padx[:, 0] = inv_rho[:, 1]
    inv_rho_padx[:, -1] = inv_rho[:, -2]

    # East face: beta_{i+1/2} = 0.5*(inv_rho[i] + inv_rho[i+1])
    beta_e = 0.5 * (inv_rho_padx[:, 1:-1] + inv_rho_padx[:, 2:])
    # West face: beta_{i-1/2} = 0.5*(inv_rho[i-1] + inv_rho[i])
    beta_w = 0.5 * (inv_rho_padx[:, 0:-2] + inv_rho_padx[:, 1:-1])

    result += cx * (beta_e * (p_padx[:, 2:] - p) - beta_w * (p - p_padx[:, :-2]))

    # Face-averaged inv_rho in y-direction
    inv_rho_pady = np.empty((Ny + 2, Nx))
    inv_rho_pady[1:-1, :] = inv_rho
    inv_rho_pady[0, :] = inv_rho[1, :]
    inv_rho_pady[-1, :] = inv_rho[-2, :]

    beta_n = 0.5 * (inv_rho_pady[1:-1, :] + inv_rho_pady[2:, :])
    beta_s = 0.5 * (inv_rho_pady[0:-2, :] + inv_rho_pady[1:-1, :])

    result += cy * (beta_n * (p_pady[2:, :] - p) - beta_s * (p - p_pady[:-2, :]))

    return result.ravel()


# ── Periodic (FFT) Poisson solver ─────────────────────────────────────────────
# For doubly-periodic domains (e.g. Taylor-Green) the pressure must satisfy
# periodic BCs, NOT the homogeneous-Neumann BCs assumed by the DCT solver.  The
# grid carries an overlap column/row (x[-1] == x[0] physically), so the unique
# periodic field lives on the reduced (Ny-1, Nx-1) sub-grid with period L.

def _precompute_poisson_eigenvalues_periodic(Nx, Ny, dx, dy):
    """Fourier symbol of div(grad(.)) for the WIDE 2nd-order central operators
    used here (``(f[i+1]-f[i-1])/2h`` applied twice) on the reduced
    (Ny-1) x (Nx-1) periodic sub-grid.

    The symbol is ``-sin(2*pi*k/m)**2 / h**2`` per direction, which is the exact
    eigenvalue of div∘grad for the wide central stencil.  Matching the solver's
    eigenvalues to the *actual* discrete operators (rather than the compact
    Laplacian) makes the projection exact for all resolved modes.

    Two families of modes have a (near-)zero eigenvalue and are unprojectable:
    the constant mode (k=0) and the checkerboard / Nyquist mode (k=m/2, where
    sin=0).  Both are pinned to 1.0 here and their corrections are zeroed in
    `_solve_poisson_fft`; a `mask` of the pinned entries is returned alongside.
    """
    mx = Nx - 1
    my = Ny - 1
    kx = np.arange(mx)
    ky = np.arange(my)
    lam_x = -(np.sin(2.0 * np.pi * kx / mx) / dx) ** 2
    lam_y = -(np.sin(2.0 * np.pi * ky / my) / dy) ** 2
    eig = lam_x[np.newaxis, :] + lam_y[:, np.newaxis]
    null = np.abs(eig) < 1e-12       # constant + Nyquist/checkerboard modes
    eig = eig.copy()
    eig[null] = 1.0
    return eig, null


def _tile_overlap(field_reduced, Ny, Nx):
    """Pad a reduced (Ny-1, Nx-1) periodic field back to the full (Ny, Nx)
    overlap grid by wrapping row/col 0 onto the last row/col."""
    out = np.empty((Ny, Nx))
    out[:-1, :-1] = field_reduced
    out[-1, :-1] = field_reduced[0, :]
    out[:-1, -1] = field_reduced[:, 0]
    out[-1, -1] = field_reduced[0, 0]
    return out


def _solve_poisson_fft(rhs_full, eigenvalues_periodic):
    """Direct periodic Poisson solve via 2D FFT on the reduced sub-grid.

    `eigenvalues_periodic` is the (eig, null) pair from
    `_precompute_poisson_eigenvalues_periodic`; corrections at the pinned null
    modes (constant + checkerboard) are set to zero.
    """
    eig, null = eigenvalues_periodic
    Ny, Nx = rhs_full.shape
    r = rhs_full[:-1, :-1].copy()
    r -= np.mean(r)
    rhat = np.fft.fft2(r)
    phat = rhat / eig
    phat[null] = 0.0
    p_reduced = np.real(np.fft.ifft2(phat))
    p = _tile_overlap(p_reduced, Ny, Nx)
    p -= np.mean(p)
    return p


def _compute_divergence_periodic(a_star, b_star, dx, dy):
    """2nd-order central velocity divergence with periodic wrap."""
    Ny, Nx = a_star.shape
    au = a_star[:-1, :-1]
    bv = b_star[:-1, :-1]
    dudx = (np.roll(au, -1, axis=1) - np.roll(au, 1, axis=1)) / (2.0 * dx)
    dvdy = (np.roll(bv, -1, axis=0) - np.roll(bv, 1, axis=0)) / (2.0 * dy)
    return _tile_overlap(dudx + dvdy, Ny, Nx)


def _compute_pressure_gradient_periodic(p, dx, dy):
    """2nd-order central pressure gradient with periodic wrap."""
    Ny, Nx = p.shape
    pr = p[:-1, :-1]
    dpdx_r = (np.roll(pr, -1, axis=1) - np.roll(pr, 1, axis=1)) / (2.0 * dx)
    dpdy_r = (np.roll(pr, -1, axis=0) - np.roll(pr, 1, axis=0)) / (2.0 * dy)
    return _tile_overlap(dpdx_r, Ny, Nx), _tile_overlap(dpdy_r, Ny, Nx)


def pressure_projection_amg(a_star, b_star, dx, dy, dt, rho, velocity_bc, A=None, ml=None,
                            p_prev=None, eigenvalues=None, bc_type='neumann'):
    """
    Pressure projection step. Handles both uniform and variable density and
    keeps the pressure BC consistent with the velocity BC.

    bc_type='neumann'  : walls (no-slip / free-slip).  Constant density uses a
                         DCT-I direct solve — O(N log N); variable density uses
                         matrix-free CG with a DCT/AMG preconditioner.
    bc_type='periodic' : doubly-periodic domain (e.g. Taylor-Green).  Uses a
                         direct FFT solve.  `eigenvalues` must come from
                         `_precompute_poisson_eigenvalues_periodic`.  Density is
                         treated as (near-)constant in the Poisson operator; the
                         velocity correction still uses the local rho.

    Pass `eigenvalues` (matching `bc_type`) to enable the direct solver; if not
    provided, the Neumann path falls back to an AMG iterative solve.
    """
    Ny, Nx = a_star.shape
    N = Nx * Ny

    # ── Periodic branch (consistent with periodic velocity BCs) ──────────────
    if bc_type == 'periodic':
        if eigenvalues is None:
            eigenvalues = _precompute_poisson_eigenvalues_periodic(Nx, Ny, dx, dy)
        divU = _compute_divergence_periodic(a_star, b_star, dx, dy)
        rho_bar = float(np.mean(rho)) if isinstance(rho, np.ndarray) else float(rho)
        rhs_2d = rho_bar * divU / dt
        p_correction = _solve_poisson_fft(rhs_2d, eigenvalues)
        dpdx, dpdy = _compute_pressure_gradient_periodic(p_correction, dx, dy)
        a = a_star - (dt / rho) * dpdx
        b = b_star - (dt / rho) * dpdy
        a, b = apply_velocity_BCs(velocity_bc, a, b)
        p = (p_prev + p_correction) if p_prev is not None else p_correction
        p -= np.mean(p)
        return a, b, p, A, ml

    if p_prev is not None:
        divU = _compute_divergence_rc(a_star, b_star, p_prev, dt, rho, dx, dy)
    else:
        divU = _compute_divergence(a_star, b_star, dx, dy)

    # Determine if density is variable
    is_variable_rho = isinstance(rho, np.ndarray) and rho.ndim == 2 and np.ptp(rho) > 1e-10

    if is_variable_rho:
        # Variable-density projection: ∇·((1/ρ)∇φ) = (1/dt) ∇·u*
        # φ is the pressure correction (Poisson solution)
        rhs = (divU / dt).ravel()
        rhs -= np.mean(rhs)

        inv_rho = 1.0 / rho

        # Matrix-free operator (avoids rebuilding sparse matrix every step)
        matvec = lambda v: _apply_variable_poisson(v, Nx, Ny, dx, dy, inv_rho)
        A_op = LinearOperator((N, N), matvec=matvec)

        # Preconditioner: DCT (fast) if eigenvalues available, else AMG
        if eigenvalues is not None:
            def dct_precond_matvec(r):
                return _solve_poisson_dct(r.reshape((Ny, Nx)), eigenvalues).ravel()
            precond = LinearOperator((N, N), matvec=dct_precond_matvec)
        else:
            if A is None:
                A = build_poisson_matrix(Nx, Ny, dx, dy)
            if ml is None:
                ml = pyamg.ruge_stuben_solver(A)
            precond = ml.aspreconditioner()

        x0 = np.zeros(N)
        p_flat, info = cg(A_op, rhs, x0=x0, tol=1e-6, maxiter=200, M=precond)

        p_correction = p_flat.reshape((Ny, Nx))
        p_correction -= np.mean(p_correction)
    else:
        # Constant-density projection: ∇²φ = (ρ/dt) ∇·u*
        rhs_2d = rho * divU / dt

        if eigenvalues is not None:
            # DCT direct solve — O(N log N), exact to machine precision
            p_correction = _solve_poisson_dct(rhs_2d, eigenvalues)
        else:
            # Fallback: AMG iterative solve
            if A is None:
                A = build_poisson_matrix(Nx, Ny, dx, dy)
            if ml is None:
                ml = pyamg.ruge_stuben_solver(A)

            rhs = rhs_2d.ravel()
            rhs -= np.mean(rhs)
            p_flat = ml.solve(rhs, tol=1e-6, maxiter=100, x0=None)
            p_correction = p_flat.reshape((Ny, Nx))
            p_correction -= np.mean(p_correction)

    # Velocity correction uses gradient of the CORRECTION only
    dpdx, dpdy = _compute_pressure_gradient(p_correction, dx, dy)

    a = a_star - (dt / rho) * dpdx
    b = b_star - (dt / rho) * dpdy

    a, b = apply_velocity_BCs(velocity_bc, a, b)

    # Accumulate pressure: p_new = p_prev + correction (incremental projection)
    if p_prev is not None:
        p = p_prev + p_correction
    else:
        p = p_correction
    p -= np.mean(p)

    return a, b, p, A, ml

def rebuild_phi_from_reference_map(X1, X2, phi_init_func):
    return phi_init_func(X1, X2)

def reinitialize_phi_PDE(phi_in, dx, dy, num_iters, apply_phi_BCs_func, dt_reinit_factor=0.5):
    phi = phi_in.copy()
    phi_initial_sign = phi_in / np.sqrt(phi_in**2 + dx**2)  # Avoid division by zero, ensure no NaNs.
    dt_reinit = dt_reinit_factor * min(dx, dy)

    for _ in range(num_iters):
        phi_padded = np.pad(phi, 1, mode='edge') 
        
        # Calculate forward and backward differences from the padded array.       
        Dx_m = (phi_padded[1:-1, 1:-1] - phi_padded[1:-1, 0:-2]) / dx # (Backward difference in x at point i,j)
        Dx_p = (phi_padded[1:-1, 2:]   - phi_padded[1:-1, 1:-1]) / dx # (Forward difference in x at point i,j)
        Dy_m = (phi_padded[1:-1, 1:-1] - phi_padded[0:-2, 1:-1]) / dy # (Backward difference in y at point i,j)
        Dy_p = (phi_padded[2:,   1:-1] - phi_padded[1:-1, 1:-1]) / dy # (Forward difference in y at point i,j)

        # Calculate upwinded squared gradients based on S(phi_initial)
        # These are (phi_x)^2 and (phi_y)^2 in the PDE, chosen with upwinding
        # to ensure information propagates from the interface.
        grad_phi_x_sq = np.zeros_like(phi)
        grad_phi_y_sq = np.zeros_like(phi)

        # --- Upwind scheme based on Sussman, Smereka, Osher (1994) ---
        # Mask for S(phi_initial) > 0 (points outside the solid, phi should be positive)
        mask_pos = (phi_initial_sign > 0)
        grad_phi_x_sq[mask_pos] = np.maximum( np.maximum(Dx_m[mask_pos], 0.)**2,  np.minimum(Dx_p[mask_pos], 0.)**2 )
        grad_phi_y_sq[mask_pos] = np.maximum( np.maximum(Dy_m[mask_pos], 0.)**2,  np.minimum(Dy_p[mask_pos], 0.)**2 )

        # Mask for S(phi_initial) < 0 (points inside the solid, phi should be negative)
        mask_neg = (phi_initial_sign < 0)
        grad_phi_x_sq[mask_neg] = np.maximum( np.minimum(Dx_m[mask_neg], 0.)**2,  np.maximum(Dx_p[mask_neg], 0.)**2 )
        grad_phi_y_sq[mask_neg] = np.maximum( np.minimum(Dy_m[mask_neg], 0.)**2,  np.maximum(Dy_p[mask_neg], 0.)**2 )
        
        grad_phi_mag = np.sqrt(grad_phi_x_sq + grad_phi_y_sq)
        
        # PDE term: S(phi_initial) * ( |grad(phi)| - 1 )
        dphi_dtau = phi_initial_sign * (grad_phi_mag - 1.0)
        
        # Update phi using Forward Euler in pseudo-time tau
        phi = phi - dt_reinit * dphi_dtau

        if apply_phi_BCs_func is not None:
            phi = apply_phi_BCs_func(phi)

    return phi
