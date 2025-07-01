# pyRMT.py
#
# Copyright (c) 2025 Saman Seifi, PhD
#
# This code contains all the functionalities needed to deveklop simulations 
# using Reference Map Technique (RMT).
#

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import lil_matrix
import pyamg
from numba import njit, prange

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

@njit(parallel=True)
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

        for j in prange(1, Ny - 1):
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
                            coeffs1 = np.linalg.solve(Aw, Bw1)
                            coeffs2 = np.linalg.solve(Aw, Bw2)

                            X1_ext[j, i] = coeffs1[0] + coeffs1[1] * x0 + coeffs1[2] * y0
                            X2_ext[j, i] = coeffs2[0] + coeffs2[1] * x0 + coeffs2[2] * y0
                            known_flag[j, i] = True

    return X1_ext, X2_ext

def compute_timestep(a, b, dx, dy, CFL, dt_min_cap, mu_s, rho_s):
    """
    Compute stable timestep based on time scales of solid (elastic) and fluid (advective) components.

    Time scales:
      – Solid elastic wave time scale (dt_solid):  
        Based on the speed of shear waves in the solid, cs_solid = sqrt(μ_s / ρ_s).  
        dt_solid ≈ CFL * Δx / cs_solid
      – Fluid advection time scale (dt_fluid):  
        Based on the maximum local fluid velocity, u_max = max(|u|).  
        dt_fluid ≈ CFL * Δx / u_max
      – dt_min_cap:  
        User‐provided absolute minimum cap on the timestep.

    The actual timestep is the minimum of these three.
    """
    cs_solid = np.sqrt(mu_s / (rho_s + 1e-12))
    dt_solid = CFL * dx / (cs_solid + 1e-14)
    u_max = np.max(np.sqrt(a**2 + b**2))
    dt_fluid = CFL * dx / (u_max + 1e-6)
    dt = min(dt_solid, dt_fluid, dt_min_cap)
    
    return dt

@njit
def advect_semi_lagrangian_rk4(q, a, b, X, Y, dt, dx, dy):
    Ny, Nx = q.shape

    def interp(u, xq, yq):
        return bilinear_interpolate(u, xq, yq, dx, dy, Nx, Ny)

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

@njit(parallel=True)
def bilinear_interpolate(u, xq, yq, dx, dy, Nx, Ny):
    """
    Fast bilinear interpolator for 2D grid data.

    Parameters:
        u  : 2D array of shape (Ny, Nx)
        xq : query x positions (same shape as u)
        yq : query y positions (same shape as u)
        dx, dy : grid spacing
        Nx, Ny : number of grid points in x and y

    Returns:
        Interpolated values at query points
    """
    out = np.zeros_like(xq)
    for j in prange(xq.shape[0]):
        for i in range(xq.shape[1]):
            x = xq[j, i] / dx
            y = yq[j, i] / dy

            ix = int(np.floor(x))
            iy = int(np.floor(y))

            fx = x - ix
            fy = y - iy

            # Clamp indices
            if ix < 0 or iy < 0 or ix >= Nx - 1 or iy >= Ny - 1:
                out[j, i] = 0.0
                continue

            v00 = u[iy,   ix  ]
            v10 = u[iy,   ix+1]
            v01 = u[iy+1, ix  ]
            v11 = u[iy+1, ix+1]

            out[j, i] = (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v10 + \
                        (1 - fx) * fy * v01 + fx * fy * v11

    return out

def compute_solid_stress(X1, X2, dx, dy, mu_s, kappa, phi, a, b, p, eta_s=0.0):
    """
    Compute solid stress tensor for an incompressible Neo-Hookean solid with a separate
    pressure field passed in (p), used as a Lagrange multiplier for incompressibility.

    σ = -p_s * I + μ (F F^T - I)

    Parameters:
    - X1, X2 : reference maps
    - dx, dy : grid spacing
    - mu_s   : shear modulus of solid
    - phi    : level set function (solid region if phi <= 0)
    - a, b   : velocity components (for damping)
    - p      : pressure field (Lagrange multiplier for incompressibility)
    - eta_s  : optional damping viscosity

    Returns:
    - sxx, sxy, syy : components of Cauchy stress tensor
    - J             : determinant of deformation gradient (for diagnostics only)
    """

    pad_width = 3

    X1_padded = np.pad(X1, pad_width, mode='edge')
    X2_padded = np.pad(X2, pad_width, mode='edge')

    dX1_dx = grad_x_4th(X1_padded, dx)
    dX1_dy = grad_y_4th(X1_padded, dy)
    dX2_dx = grad_x_4th(X2_padded, dx)
    dX2_dy = grad_y_4th(X2_padded, dy)

    dX1_dx = dX1_dx[pad_width:-pad_width, pad_width:-pad_width]
    dX1_dy = dX1_dy[pad_width:-pad_width, pad_width:-pad_width]
    dX2_dx = dX2_dx[pad_width:-pad_width, pad_width:-pad_width]
    dX2_dy = dX2_dy[pad_width:-pad_width, pad_width:-pad_width]

    Ny, Nx = X1.shape
    sxx = np.zeros((Ny, Nx))
    sxy = np.zeros((Ny, Nx))
    syy = np.zeros((Ny, Nx))
    J = np.ones((Ny, Nx))

    solid_mask = phi <= 0
    idxs = np.where(solid_mask)

    G = np.stack([
        np.stack([dX1_dx[idxs], dX1_dy[idxs]], axis=-1),
        np.stack([dX2_dx[idxs], dX2_dy[idxs]], axis=-1)
    ], axis=-2)

    detG = G[:, 0, 0] * G[:, 1, 1] - G[:, 0, 1] * G[:, 1, 0]
    good = np.abs(detG) > 1e-10
    Ginv = np.zeros_like(G)
    Ginv[good, 0, 0] =  G[good, 1, 1] / detG[good]
    Ginv[good, 0, 1] = -G[good, 0, 1] / detG[good]
    Ginv[good, 1, 0] = -G[good, 1, 0] / detG[good]
    Ginv[good, 1, 1] =  G[good, 0, 0] / detG[good]

    F = Ginv
    Ft = np.transpose(F, axes=(0, 2, 1))
    FFt = np.einsum('nij,njk->nik', F, Ft)
    
    # calculate trace of FFt
    trace_FFt = FFt[:, 0, 0] + FFt[:, 1, 1]
    
    I = np.eye(2)
    I_expand = np.broadcast_to(I, FFt.shape)

    # Use pressure at the solid points as incompressibility Lagrange multiplier
    p_solid = p[idxs]

    J_temp = np.linalg.det(F)
    J[idxs] = J_temp

    # sigma = -p_solid[:, None, None] * I_expand + mu_s * (FFt)
    # sigma = mu_s * (FFt - (1/3)*(trace_FFt + 1)[:, None, None] * I_expand)
    sigma = (mu_s / J_temp)[:, None, None] * (FFt) + kappa * (np.log(J_temp))[:, None, None] * I_expand

    sxx[idxs] = sigma[:, 0, 0]
    sxy[idxs] = sigma[:, 0, 1]  # Note: correct ordering for symmetric stress
    syy[idxs] = sigma[:, 1, 1]

    if eta_s > 0:
        du_dx = grad_x_4th(a, dx)
        dv_dy = grad_y_4th(b, dy)
        du_dy = grad_y_4th(a, dy)
        dv_dx = grad_x_4th(b, dx)

        strain_rate_xx = du_dx
        strain_rate_yy = dv_dy
        strain_rate_xy = 0.5 * (du_dy + dv_dx)

        sxx[idxs] += eta_s * strain_rate_xx[idxs]
        syy[idxs] += eta_s * strain_rate_yy[idxs]
        sxy[idxs] += eta_s * strain_rate_xy[idxs]

    sxx = gaussian_filter(sxx * solid_mask, 0.5)
    sxy = gaussian_filter(sxy * solid_mask, 0.5)
    syy = gaussian_filter(syy * solid_mask, 0.5)

    return sxx, sxy, syy, J

def heaviside_smooth_alt(x, w_t):
    """
    Smooth Heaviside function with a transition width of w_t.
    This function is used to blend solid and fluid stresses and densities.
    
    The Heaviside function is defined as:
    
        H(x) = 0 for x < -w_t
        H(x) = 0.5 + (x/w_t)/2 + (1/pi) * sin(pi*x/w_t)/2 for -w_t <= x <= w_t
        H(x) = 1 for x > w_t
        
    where w_t is the transition width.
  
    """
    H = np.zeros_like(x)
    inside_band = np.abs(x) <= w_t
    H[x > w_t] = 1.0
    H[inside_band] = 0.5 * (1 + x[inside_band]/w_t + (1/np.pi) * np.sin(np.pi * x[inside_band]/w_t))
    
    return H

def velocity_RK4(u, v, p, X1, X2, velocity_bc, mu_s, kappa, eta_s , dx, dy, dt, rho_s, rho_f, phi, mu_f, w_t):
    """
    RK4 integration using stress divergence from blended stress field.
    """
    def rhs(u_stage, v_stage, p):
        sigma_sxx, sigma_sxy, sigma_syy, J = compute_solid_stress(X1, X2, dx, dy, mu_s, 
                                                                  kappa, phi, u_stage, 
                                                                  v_stage, p,  eta_s)   
        return velocity_rhs_blended(u, v, p, sigma_sxx, sigma_sxy, sigma_syy,
                                    dx, dy, phi,mu_f, rho_s, rho_f, w_t)

    k1u, k1v = rhs(u, v, p)

    u1 = u + 0.5 * dt * k1u
    v1 = v + 0.5 * dt * k1v
    k2u, k2v = rhs(u1, v1, p)

    u2 = u + 0.5 * dt * k2u
    v2 = v + 0.5 * dt * k2v
    k3u, k3v = rhs(u2, v2, p)

    u3 = u + dt * k3u
    v3 = v + dt * k3v
    k4u, k4v = rhs(u3, v3, p)

    u_new = u + (dt / 6.0) * (k1u + 2*k2u + 2*k3u + k4u)
    v_new = v + (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)
    
    u_new, v_new = apply_velocity_BCs(velocity_bc, u_new, v_new)

    return u_new, v_new

@njit
def grad_x_4th(f, dx):
    df_dx = np.zeros_like(f)

    # Interior: 4th-order central difference
    df_dx[:, 2:-2] = (-f[:, 4:] + 8*f[:, 3:-1] - 8*f[:, 1:-3] + f[:, 0:-4]) / (12 * dx)

    # Left boundary (x = 0, 1)
    df_dx[:, 0] = (-25*f[:, 0] + 48*f[:, 1] - 36*f[:, 2] + 16*f[:, 3] - 3*f[:, 4]) / (12 * dx)
    df_dx[:, 1] = (-3*f[:, 0] - 10*f[:, 1] + 18*f[:, 2] - 6*f[:, 3] + f[:, 4]) / (12 * dx)

    # Right boundary (x = -2, -1)
    df_dx[:, -2] = (3*f[:, -1] + 10*f[:, -2] - 18*f[:, -3] + 6*f[:, -4] - f[:, -5]) / (12 * dx)
    df_dx[:, -1] = (25*f[:, -1] - 48*f[:, -2] + 36*f[:, -3] - 16*f[:, -4] + 3*f[:, -5]) / (12 * dx)

    return df_dx

@njit
def grad_y_4th(f, dy):
    df_dy = np.zeros_like(f)

    # Interior
    df_dy[2:-2, :] = (-f[4:, :] + 8*f[3:-1, :] - 8*f[1:-3, :] + f[0:-4, :]) / (12 * dy)

    # Bottom boundary (y = 0, 1)
    df_dy[0, :] = (-25*f[0, :] + 48*f[1, :] - 36*f[2, :] + 16*f[3, :] - 3*f[4, :]) / (12 * dy)
    df_dy[1, :] = (-3*f[0, :] - 10*f[1, :] + 18*f[2, :] - 6*f[3, :] + f[4, :]) / (12 * dy)

    # Top boundary (y = -2, -1)
    df_dy[-2, :] = (3*f[-1, :] + 10*f[-2, :] - 18*f[-3, :] + 6*f[-4, :] - f[-5, :]) / (12 * dy)
    df_dy[-1, :] = (25*f[-1, :] - 48*f[-2, :] + 36*f[-3, :] - 16*f[-4, :] + 3*f[-5, :]) / (12 * dy)

    return df_dy

@njit
def lap_4th(f, dx, dy):
    """
    4th-order accurate Laplacian for 2D array f, with 2nd-order one-sided stencils at boundaries.
    """
    lap = np.zeros_like(f)

    # 4th-order central differences for interior
    d2f_dx2 = np.zeros_like(f)
    d2f_dy2 = np.zeros_like(f)

    # X second derivative (central, 4th order)
    d2f_dx2[:, 2:-2] = (
        -f[:, 4:] + 16*f[:, 3:-1] - 30*f[:, 2:-2] + 16*f[:, 1:-3] - f[:, 0:-4]
    ) / (12 * dx**2)
    # Y second derivative (central, 4th order)
    d2f_dy2[2:-2, :] = (
        -f[4:, :] + 16*f[3:-1, :] - 30*f[2:-2, :] + 16*f[1:-3, :] - f[0:-4, :]
    ) / (12 * dy**2)

    # 4th-order one-sided stencils for boundaries (x-direction)
    # Left boundary (i=0)
    d2f_dx2[:, 0] = (45*f[:, 0] - 154*f[:, 1] + 214*f[:, 2] - 156*f[:, 3] + 61*f[:, 4] - 10*f[:, 5]) / (12 * dx**2)
    # i=1
    d2f_dx2[:, 1] = (10*f[:, 0] - 15*f[:, 1] - 4*f[:, 2] + 14*f[:, 3] - 6*f[:, 4] + f[:, 5]) / (12 * dx**2)
    # Right boundary (i=-1)
    d2f_dx2[:, -1] = (45*f[:, -1] - 154*f[:, -2] + 214*f[:, -3] - 156*f[:, -4] + 61*f[:, -5] - 10*f[:, -6]) / (12 * dx**2)
    # i=-2
    d2f_dx2[:, -2] = (10*f[:, -1] - 15*f[:, -2] - 4*f[:, -3] + 14*f[:, -4] - 6*f[:, -5] + f[:, -6]) / (12 * dx**2)

    # 4th-order one-sided stencils for boundaries (y-direction)
    # Bottom boundary (j=0)
    d2f_dy2[0, :] = (45*f[0, :] - 154*f[1, :] + 214*f[2, :] - 156*f[3, :] + 61*f[4, :] - 10*f[5, :]) / (12 * dy**2)
    # j=1
    d2f_dy2[1, :] = (10*f[0, :] - 15*f[1, :] - 4*f[2, :] + 14*f[3, :] - 6*f[4, :] + f[5, :]) / (12 * dy**2)
    # Top boundary (j=-1)
    d2f_dy2[-1, :] = (45*f[-1, :] - 154*f[-2, :] + 214*f[-3, :] - 156*f[-4, :] + 61*f[-5, :] - 10*f[-6, :]) / (12 * dy**2)
    # j=-2
    d2f_dy2[-2, :] = (10*f[-1, :] - 15*f[-2, :] - 4*f[-3, :] + 14*f[-4, :] - 6*f[-5, :] + f[-6, :]) / (12 * dy**2)

    lap = d2f_dx2 + d2f_dy2
    return lap

def velocity_rhs_blended(u, v, p,
                         sigma_sxx_s, sigma_sxy_s, sigma_syy_s,
                         dx, dy, phi,
                         mu_f, rho_s, rho_f, w_t):
    """
    Compute the right-hand side (RHS) of the velocity evolution equation
    in a one-fluid, Eulerian fluid–structure interaction (FSI) framework
    using a blended velocity formulation.

    This routine handles:
    1. Advection of velocity (central differences)
    2. Fluid stress divergence using Laplacian (Newtonian stress)
    3. Solid stress divergence using the full tensor divergence of the Neo-Hookean Cauchy stress
    4. Interfacial jump term from the difference between fluid and solid stress tensors,
       applied via the gradient of a smooth Heaviside function
    5. Enforcing incompressibility via a pressure gradient term

    ----------
    Mathematical formulation:

    The evolution of velocity u is governed by:
        ∂u/∂t + (u · ∇)u = (1/ρ) ∇·σ

    Where the total stress is:
        σ = (1 - H) σ_solid + H σ_fluid

    Deviatoric Fluid stress:
        σ_fluid = 2μ_f D(u) = 2μ_f sym(∇u)
        → ∇·σ_fluid ≈ μ_f ∇²u       (via Laplacian approximation 
                                     assuming incompressibility 
                                     and Newtonian fluid)

    Solid stress divergence is computed as:
        ∇·σ_solid = 
            [∂x σ_xx^s + ∂y σ_xy^s,
             ∂x σ_xy^s + ∂y σ_yy^s]

    therefore:
        ∇·σ = (1 - H) ∇·σ_solid + H ∇·σ_fluid + f_jump
    
    where f_jump is the jump term that accounts for the difference in stress across the interface:
        f_jump = (σ_fluid - σ_solid) · ∇H

    where:
        - H is a smooth Heaviside function from level set φ
        - ∇H is the interfacial gradient used to localize interfacial forces
        
    and - ∇p is the pressure gradient applied to the fluid and the solid enforcing global incompressibility.

    ----------
    Parameters:
    - u, v              : velocity components
    - sigma_sxx_s, ...  : solid stress tensor components
    - dx, dy            : grid spacing
    - phi               : level set function (φ = 0 interface)
    - mu_f              : dynamic viscosity of the fluid
    - rho_s, rho_f      : densities of solid and fluid
    - w_t               : smoothing width for the Heaviside function
    - p                 : pressure field

    Returns:
    - rhs_u, rhs_v      : right-hand side of the velocity update equations
    """

    # --- Advection (central difference) ---
    # u_adv = - u * (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx) \
    #         - v * (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dy)

    # v_adv = - u * (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dx) \
    #         - v * (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dy)

    u_adv = -u * grad_x_4th(u, dx) - v * grad_y_4th(u, dy)
    v_adv = -u * grad_x_4th(v, dx) - v * grad_y_4th(v, dy)
    # u_adv = -advection_rhs_velocity(u, u, v, dx, dy)
    # v_adv = -advection_rhs_velocity(v, u, v, dx, dy)
    
    # --- Heaviside and ∇H ---
    H = heaviside_smooth_alt(phi, w_t)
    dH_dx = grad_x_4th(H, dx)
    dH_dy = grad_y_4th(H, dy)

    # --- Local density ---
    rho_local = (1 - H) * rho_s + H * rho_f

    u_lap = mu_f * lap_4th(u, dx, dy)
    v_lap = mu_f * lap_4th(v, dx, dy)

    # --- Solid stress divergence ---
    dsxx_dx = grad_x_4th(sigma_sxx_s, dx)
    dsxy_dy = grad_y_4th(sigma_sxy_s, dy)

    dsxy_dx = grad_x_4th(sigma_sxy_s, dx)
    dsyy_dy = grad_y_4th(sigma_syy_s, dy)

    div_solid_x = dsxx_dx + dsxy_dy
    div_solid_y = dsxy_dx + dsyy_dy

    # --- Fluid stress components (only for jump term) ---
    # 4th-order central differences for derivatives
    du_dx = grad_x_4th(u, dx)
    dv_dy = grad_y_4th(v, dy)
    du_dy = grad_y_4th(u, dy)
    dv_dx = grad_x_4th(v, dx)

    dp_dx = grad_x_4th(p, dx)
    dp_dy = grad_y_4th(p, dy)

    sigma_sxx_f = 2 * mu_f * du_dx
    sigma_syy_f = 2 * mu_f * dv_dy
    sigma_sxy_f = mu_f * (du_dy + dv_dx)

    # --- Jump term (sigma_f - sigma_s) · ∇H ---
    jump_x = (sigma_sxx_f - sigma_sxx_s) * dH_dx + (sigma_sxy_f - sigma_sxy_s) * dH_dy
    jump_y = (sigma_sxy_f - sigma_sxy_s) * dH_dx + (sigma_syy_f - sigma_syy_s) * dH_dy

    # --- Final RHS ---
    rhs_u = u_adv + ((1 - H) * div_solid_x + H *(u_lap) + jump_x) / (rho_local + 1e-12)
    rhs_v = v_adv + ((1 - H) * div_solid_y + H *(v_lap) + jump_y) / (rho_local + 1e-12)
                             
    return rhs_u, rhs_v

def apply_velocity_BCs(bc, u, v):
    return bc(u, v)

def build_poisson_matrix(Nx, Ny, dx, dy):
    N = Nx * Ny
    A = lil_matrix((N, N))

    def idx(i, j):
        return i + j * Nx

    for j in range(Ny):
        for i in range(Nx):
            k = idx(i, j)

            if 0 < i < Nx - 1 and 0 < j < Ny - 1:
                # Interior 5-point stencil
                A[k, idx(i, j)]     = -2 / dx**2 - 2 / dy**2
                A[k, idx(i + 1, j)] = 1 / dx**2
                A[k, idx(i - 1, j)] = 1 / dx**2
                A[k, idx(i, j + 1)] = 1 / dy**2
                A[k, idx(i, j - 1)] = 1 / dy**2

            else:
                # Neumann BCs using one-sided second-order approximations
                count = 0
                A[k, k] = 0

                # x-direction neighbors
                if i > 0:
                    A[k, idx(i - 1, j)] = 1 / dx**2
                    A[k, k]            -= 1 / dx**2
                    count += 1
                if i < Nx - 1:
                    A[k, idx(i + 1, j)] = 1 / dx**2
                    A[k, k]            -= 1 / dx**2
                    count += 1

                # y-direction neighbors
                if j > 0:
                    A[k, idx(i, j - 1)] = 1 / dy**2
                    A[k, k]            -= 1 / dy**2
                    count += 1
                if j < Ny - 1:
                    A[k, idx(i, j + 1)] = 1 / dy**2
                    A[k, k]            -= 1 / dy**2
                    count += 1

                # If completely isolated (corner), anchor it
                if count == 0:
                    A[k, k] = 1.0

    return A.tocsr()

def pressure_projection_amg(a_star, b_star, dx, dy, dt, rho, velocity_bc, A=None, ml=None, p_prev=None):
    """
    Enforce incompressibility by solving ∇²p = (ρ/Δt) ∇·U* with Conjugate Gradient.
    
        1) Compute divergence of tentative velocity ∇·U*:
                (∂u*/∂x + ∂v*/∂y)_{j,i} = (u*_{j,i+1} - u*_{j,i-1})/(2 dx)
                                        + (v*_{j+1,i} - v*_{j-1,i})/(2 dy)

        2) Build RHS for Poisson: rhs = (ρ/Δt) ∇·U* , then anchor ⟨rhs⟩=0
        3) Assemble Poisson matrix A discretizing ∇²:
                ∇²p ≈ (p_{i+1,j} + p_{i-1,j} - 2p_{i,j})/dx²
                    + (p_{i,j+1} + p_{i,j-1} - 2p_{i,j})/dy²

        4) Solve A p = rhs with Conjugate Gradient
        5) Reshape back to 2D p_{j,i} and enforce ⟨p⟩=0
        6) Compute pressure gradients ∇p:
                ∂p/∂x ≈ (p_{j,i+1} - p_{j,i-1})/(2 dx)
                ∂p/∂y ≈ (p_{j+1,i} - p_{j-1,i})/(2 dy)
        7) Correct velocities: U = U* - (Δt/ρ) ∇p
                u_{j,i} = u*_{j,i} - (Δt/ρ) ∂p/∂x
                v_{j,i} = v*_{j,i} - (Δt/ρ) ∂p/∂y

        8) Reapply boundary conditions (e.g., moving lid, no-slip)
    """

    Ny, Nx = a_star.shape
    N = Nx * Ny

    divU = np.zeros_like(a_star)
    # 4th-order central difference for divergence
    divU[2:-2, 2:-2] = (
        (-a_star[2:-2, 4:] + 8*a_star[2:-2, 3:-1] - 8*a_star[2:-2, 1:-3] + a_star[2:-2, 0:-4]) / (12 * dx) +
        (-b_star[4:, 2:-2] + 8*b_star[3:-1, 2:-2] - 8*b_star[1:-3, 2:-2] + b_star[0:-4, 2:-2]) / (12 * dy)
    )
    # 2nd-order central difference at boundaries (fallback)
    divU[1:-1, 1:-1] = (
        (a_star[1:-1, 2:] - a_star[1:-1, :-2]) / (2 * dx) +
        (b_star[2:, 1:-1] - b_star[:-2, 1:-1]) / (2 * dy)
    )

    rhs = (rho * divU / dt).ravel()
    rhs -= np.mean(rhs)

    if A is None:
        A = build_poisson_matrix(Nx, Ny, dx, dy)

    if ml is None:
        ml = pyamg.ruge_stuben_solver(A)  # Precompute AMG hierarchy

    p_flat = ml.solve(rhs, tol=1e-8, maxiter=50, x0=p_prev.ravel() if p_prev is not None else None)
    p = p_flat.reshape((Ny, Nx))
    p -= np.mean(p)

    dpdx = np.zeros_like(p)
    dpdy = np.zeros_like(p)
    # 4th-order central differences for pressure gradients (interior)
    dpdx[2:-2, 2:-2] = (-p[2:-2, 4:] + 8*p[2:-2, 3:-1] - 8*p[2:-2, 1:-3] + p[2:-2, 0:-4]) / (12 * dx)
    dpdy[2:-2, 2:-2] = (-p[4:, 2:-2] + 8*p[3:-1, 2:-2] - 8*p[1:-3, 2:-2] + p[0:-4, 2:-2]) / (12 * dy)
    # 2nd-order central differences at boundaries (fallback)
    dpdx[1:-1, 1:-1] = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
    dpdy[1:-1, 1:-1] = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)

    a = a_star - (dt / rho) * dpdx
    b = b_star - (dt / rho) * dpdy

    a, b = apply_velocity_BCs(velocity_bc, a, b)
    
    return a, b, p, A, ml

def rebuild_phi_from_reference_map(X1, X2, phi_init_func):
    """
    Rebuilds the level set function φ by evaluating the initializer at the reference map positions (X1, X2).
    
    Parameters:
    - X1, X2: reference maps (advected X, Y coordinates)
    - phi_init_func: a function like `initialize_disc(X, Y, ...)` that returns the initial level set
    
    Returns:
    - φ rebuilt on current grid using reference coordinates
    """
    return phi_init_func(X1, X2)

def reinitialize_phi_PDE(phi_in, dx, dy, num_iters, apply_phi_BCs_func, dt_reinit_factor=0.5):
    """
    Reinitializes ϕ to be a signed distance function using a PDE-based method.
    Solves: 
                ϕ_τ + S(ϕ₀) * ( |∇ϕ| - 1 ) = 0

    Args:
        phi_in (np.ndarray): The level set function to reinitialize. This phi is assumed
                             to correctly define the interface (phi_in=0), but may not be
                             an SDF away from it.
        dx (float): Grid spacing in x.
        dy (float): Grid spacing in y.
        num_iters (int): Number of pseudo-time iterations for the reinitialization PDE.
                         Typically 5-10 iterations are sufficient.
        apply_phi_BCs_func (callable): Function to apply boundary conditions to phi.
                                       This function will be called after each iteration.
                                       Example: `apply_phi_BCs` from your existing code.
        dt_reinit_factor (float): Factor to determine the pseudo-timestep for reinitialization.
                                  dt_reinit = dt_reinit_factor * min(dx, dy).
                                  For stability with this explicit upwind scheme, this
                                  factor should generally be <= 1.0. A value of 0.5 is common.

    Returns:
        np.ndarray: The reinitialized level set function phi.
    """
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
            
    return phi

@njit(parallel=True)
def compute_rhie_chow_face_velocities(u_star, v_star, p, dx, dy, rho, dt, A_coeffs_u, A_coeffs_v):
    Ny, Nx = u_star.shape
    u_face_east = np.zeros_like(u_star) # u at east faces (i+1/2, j)
    u_face_west = np.zeros_like(u_star) # u at west faces (i-1/2, j)
    v_face_north = np.zeros_like(v_star) # v at north faces (i, j+1/2)
    v_face_south = np.zeros_like(v_star) # v at south faces (i, j-1/2)

    D_u_cells = 1.0 / (A_coeffs_u + 1e-12) # Inverse of diagonal coefficient
    D_v_cells = 1.0 / (A_coeffs_v + 1e-12)

    # Loop over internal cells for face velocities
    for j in prange(Ny):
        for i in range(Nx):
            # East Face (u_face_east[i,j] is velocity at (i+1/2, j))
            if i < Nx - 1:
                # Linear interpolation of provisional velocities
                u_avg = 0.5 * (u_star[j, i] + u_star[j, i+1])
                D_u_avg = 0.5 * (D_u_cells[j, i] + D_u_cells[j, i+1])

                # Pressure gradient at face
                dpdx_face = (p[j, i+1] - p[j, i]) / dx

                # Averaged cell-centered pressure gradient (for the subtraction term)
                # This needs proper definition. Let's use central differences on cell centers
                dpdx_P = (p[j, min(Nx-1, i+1)] - p[j, max(0, i-1)]) / (2 * dx) if (i > 0 and i < Nx-1) else 0.0 # Approximation for boundary
                dpdx_E = (p[j, min(Nx-1, i+2)] - p[j, i]) / (2 * dx) if (i+1 > 0 and i+1 < Nx-1) else 0.0 # Approximation for boundary
                
                dpdx_avg_cell = 0.5 * (dpdx_P + dpdx_E)

                u_face_east[j, i] = u_avg - D_u_avg * (dpdx_face - dpdx_avg_cell)
            else: # Boundary: assuming velocity is given by BC or extrapolation
                u_face_east[j, i] = u_star[j, i] # Or extrapolate u_star as boundary face velocity

            # North Face (v_face_north[i,j] is velocity at (i, j+1/2))
            if j < Ny - 1:
                v_avg = 0.5 * (v_star[j, i] + v_star[j+1, i])
                D_v_avg = 0.5 * (D_v_cells[j, i] + D_v_cells[j+1, i])

                dpdy_face = (p[j+1, i] - p[j, i]) / dy

                dpdy_P = (p[min(Ny-1, j+1), i] - p[max(0, j-1), i]) / (2 * dy) if (j > 0 and j < Ny-1) else 0.0
                dpdy_N = (p[min(Ny-1, j+2), i] - p[j, i]) / (2 * dy) if (j+1 > 0 and j+1 < Ny-1) else 0.0
                
                dpdy_avg_cell = 0.5 * (dpdy_P + dpdy_N)

                v_face_north[j, i] = v_avg - D_v_avg * (dpdy_face - dpdy_avg_cell)
            else: # Boundary
                v_face_north[j, i] = v_star[j, i] # Or extrapolate v_star as boundary face velocity
    
    # For the west and south faces for divergence calculation at (i,j) cell center
    # u_face_west[j, i] would be u_face_east[j, i-1]
    # v_face_south[j, i] would be v_face_north[j-1, i]

    return u_face_east, v_face_north

# Modify pressure_projection_amg
def pressure_projection_amg_RC(a_star, b_star, dx, dy, dt, rho, mu_f, velocity_bc, A=None, ml=None, p_prev=None):
    Ny, Nx = a_star.shape
    N = Nx * Ny

    rho_local = rho # Assuming rho is correct now
    A_coeffs_u_approx = rho_local / dt + 2 * mu_f / (dx*dx) + 2 * mu_f / (dy*dy)
    A_coeffs_v_approx = rho_local / dt + 2 * mu_f / (dx*dx) + 2 * mu_f / (dy*dy)
    
    # Ensure A_coeffs_u_approx and A_coeffs_v_approx are 2D arrays, same shape as u_star
    A_coeffs_u_arr = np.full_like(a_star, A_coeffs_u_approx)
    A_coeffs_v_arr = np.full_like(b_star, A_coeffs_v_approx)

    u_face_east, v_face_north = compute_rhie_chow_face_velocities(a_star, b_star, p_prev, dx, dy, rho_local, dt, A_coeffs_u_arr, A_coeffs_v_arr) # Pass p_prev for initial pressure gradients

    divU = np.zeros_like(a_star)
    # Compute divergence using Rhie-Chow interpolated face velocities
    # u_i+1/2 - u_i-1/2. u_i-1/2 for cell (i,j) is u_face_east for cell (i-1, j)
    # v_j+1/2 - v_j-1/2. v_j-1/2 for cell (i,j) is v_face_north for cell (i, j-1)
    
    # Interior divergence (using 0 to Nx-2, 0 to Ny-2 for the east/north face arrays)
    divU[1:-1, 1:-1] = (
        (u_face_east[1:-1, 1:-1] - u_face_east[1:-1, 0:-2]) / dx + # u_face_east[i-1] is effectively u_face_west[i]
        (v_face_north[1:-1, 1:-1] - v_face_north[0:-2, 1:-1]) / dy  # v_face_north[j-1] is effectively v_face_south[j]
    )

    rhs = (rho_local * divU / dt).ravel() # Use rho_local here
    rhs -= np.mean(rhs)

    if A is None:
        A = build_poisson_matrix(Nx, Ny, dx, dy)

    if ml is None:
        ml = pyamg.ruge_stuben_solver(A)

    p_flat = ml.solve(rhs, tol=1e-8, maxiter=50, x0=p_prev.ravel() if p_prev is not None else None)
    p = p_flat.reshape((Ny, Nx))
    p -= np.mean(p) # Ensure zero mean pressure

    # Standard pressure gradient calculation (usually fine with central differences for dp/dx, dp/dy)
    dpdx = np.zeros_like(p)
    dpdy = np.zeros_like(p)
    dpdx[1:-1, 1:-1] = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
    dpdy[1:-1, 1:-1] = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)

    a = a_star - (dt / rho_local) * dpdx # Use rho_local here
    b = b_star - (dt / rho_local) * dpdy # Use rho_local here

    a, b = apply_velocity_BCs(velocity_bc, a, b)
    
    return a, b, p, A, ml
