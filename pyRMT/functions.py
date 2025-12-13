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

def compute_timestep(a, b, dx, dy, CFL, dt_min_cap, mu_s, rho_s, gamma, rho_f):
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
    # 1. Solid Wave Speed
    cs_solid = np.sqrt(mu_s / (rho_s + 1e-12))
    dt_solid = CFL * dx / (cs_solid + 1e-14)
    
    # 2. Fluid Advection Speed
    u_max = np.max(np.sqrt(a**2 + b**2))
    dt_fluid = CFL * dx / (u_max + 1e-6)

    # 3. Surface Tension Capillary Wave Speed (New)
    dt_st = 1.0
    if gamma > 1e-12:
        # Approximate capillary timestep constraint (Brackbill)
        # dt < sqrt( (rho * dx^3) / (2 * pi * gamma) )
        rho_avg = 0.5 * (rho_s + rho_f)
        dt_st = np.sqrt( (rho_avg * dx**3) / (2 * np.pi * gamma) ) * 0.5 # Safety factor 0.5

    dt = min(dt_solid, dt_fluid, dt_st, dt_min_cap)
    
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

@njit
def advect_semi_lagrangian_maccormack(q, u, v, X, Y, dt, dx, dy):
    """
    MacCormack Predictor-Corrector for Semi-Lagrangian Advection.
    Reduces diffusion by subtracting the error of a forward-backward round trip.
    """
    # 1. Forward Step (Standard SL)
    q_star = advect_semi_lagrangian_rk4(q, u, v, X, Y, dt, dx, dy)
    
    # 2. Backward Step (Reverse Advection)
    # Advect q_star backwards by -dt
    q_rev = advect_semi_lagrangian_rk4(q_star, u, v, X, Y, -dt, dx, dy)
    
    # 3. Error Correction
    # The difference (q - q_rev) is the diffusion error accumulated in one step.
    # We add half this error back to the result.
    q_new = q_star + 0.5 * (q - q_rev)
    
    return q_new


@njit
def cubic_convolution(v0, v1, v2, v3, x):
    """
    Catmull-Rom Cubic Spline interpolation.
    v0, v1, v2, v3 are values at x=-1, 0, 1, 2.
    x is the fractional distance between v1 and v2 (0 <= x <= 1).
    """
    a0 = -0.5*v0 + 1.5*v1 - 1.5*v2 + 0.5*v3
    a1 = v0 - 2.5*v1 + 2.0*v2 - 0.5*v3
    a2 = -0.5*v0 + 0.5*v2
    a3 = v1
    return a0*x**3 + a1*x**2 + a2*x + a3

@njit(parallel=True)
def bicubic_interpolate(u, xq, yq, dx, dy, Nx, Ny):
    """
    Bicubic interpolation for Semi-Lagrangian Advection.
    Reduces numerical diffusion significantly compared to bilinear.
    """
    out = np.zeros_like(xq)
    
    for j in prange(xq.shape[0]):
        for i in range(xq.shape[1]):
            # Normalized coordinates
            x_idx = xq[j, i] / dx
            y_idx = yq[j, i] / dy
            
            # Integer part (bottom-left corner of the central cell)
            ix = int(np.floor(x_idx))
            iy = int(np.floor(y_idx))
            
            # Fractional part
            fx = x_idx - ix
            fy = y_idx - iy
            
            # We need a 4x4 stencil: indices from ix-1 to ix+2
            # Handle boundaries by clamping
            row_vals = np.zeros(4)
            
            for m in range(4): # Loop over y-rows (local index 0..3)
                # Global y index: iy - 1 + m
                y_global = iy - 1 + m
                
                # Clamp y
                if y_global < 0: y_global = 0
                if y_global >= Ny: y_global = Ny - 1
                
                # Gather the 4 x-values for this row
                col_vals = np.zeros(4)
                for n in range(4): # Loop over x-cols (local index 0..3)
                    x_global = ix - 1 + n
                    
                    # Clamp x
                    if x_global < 0: x_global = 0
                    if x_global >= Nx: x_global = Nx - 1
                    
                    col_vals[n] = u[y_global, x_global]
                
                # Interpolate this row in x-direction
                row_vals[m] = cubic_convolution(col_vals[0], col_vals[1], col_vals[2], col_vals[3], fx)
            
            # Final interpolate in y-direction
            out[j, i] = cubic_convolution(row_vals[0], row_vals[1], row_vals[2], row_vals[3], fy)
            
    return out

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
    Quasi-incompressible Neo-Hookean solid stress in RMT form.

    We use the reference map X(x):
        G = ∂X/∂x  (current -> reference)
        F = G^{-1} = ∂x/∂X  (deformation gradient)
        B = F F^T  (left Cauchy-Green tensor)
        J = det(F)

    Energy (typical penalty form):
        W = (mu_s/2) (I1_bar - 2) + (kappa/2) (ln J)^2

    Cauchy stress (2D, penalty form):
        σ = (mu_s / J) (B - I) + (kappa ln J / J) I

    Here we implement a simplified version:
        σ = (mu_s / J) (B - I) + (kappa ln J / J) I

    Parameters
    ----------
    X1, X2 : ndarray
        Reference map components X(x) on the Eulerian grid.
    dx, dy : float
        Grid spacings.
    mu_s   : float
        Solid shear modulus.
    kappa  : float
        Bulk modulus penalty (controls J ≈ 1).
    phi    : ndarray
        Level set; solid region is phi <= 0.
    a, b   : ndarray
        Velocity components (for viscous damping).
    p      : ndarray
        Pressure field (not used here; incompressibility handled via kappa + projection).
    eta_s  : float, optional
        Solid viscosity (Kelvin–Voigt damping).

    Returns
    -------
    sxx, sxy, syy : ndarray
        Cauchy stress components in the solid (zero in fluid).
    J             : ndarray
        Determinant of F (for diagnostics).
    """

    pad_width = 4

    # Pad reference maps for high-order gradients
    X1_padded = np.pad(X1, pad_width, mode='edge')
    X2_padded = np.pad(X2, pad_width, mode='edge')

    dX1_dx = grad_central_x_2nd(X1_padded, dx)
    dX1_dy = grad_central_y_2nd(X1_padded, dy)
    dX2_dx = grad_central_x_2nd(X2_padded, dx)
    dX2_dy = grad_central_y_2nd(X2_padded, dy)

    # Remove padding
    dX1_dx = dX1_dx[pad_width:-pad_width, pad_width:-pad_width]
    dX1_dy = dX1_dy[pad_width:-pad_width, pad_width:-pad_width]
    dX2_dx = dX2_dx[pad_width:-pad_width, pad_width:-pad_width]
    dX2_dy = dX2_dy[pad_width:-pad_width, pad_width:-pad_width]

    Ny, Nx = X1.shape
    sxx = np.zeros((Ny, Nx))
    sxy = np.zeros((Ny, Nx))
    syy = np.zeros((Ny, Nx))
    J   = np.ones((Ny, Nx))

    # Solid region indices
    solid_mask = (phi <= 0.0)
    idxs = np.where(solid_mask)

    if idxs[0].size == 0:
        # No solid; early exit
        return sxx, sxy, syy, J

    # --- Build G = ∂X/∂x ---
    G11 = dX1_dx[idxs]
    G12 = dX1_dy[idxs]
    G21 = dX2_dx[idxs]
    G22 = dX2_dy[idxs]

    # det(G) and inverse
    detG = G11 * G22 - G12 * G21
    good = np.abs(detG) > 1e-10

    # Initialize F entries
    F11 = np.zeros_like(G11)
    F12 = np.zeros_like(G12)
    F21 = np.zeros_like(G21)
    F22 = np.zeros_like(G22)

    # Only invert where detG is reasonable
    F11[good] =  G22[good] / detG[good]
    F12[good] = -G12[good] / detG[good]
    F21[good] = -G21[good] / detG[good]
    F22[good] =  G11[good] / detG[good]

    # J = det(F) = 1 / det(G)
    J_temp = np.ones_like(G11)
    J_temp[good] = 1.0 / detG[good]

    J[idxs] = J_temp
    invJ = 1.0 / J_temp
    lnJ = np.log(J_temp)

    # --- B = F F^T ---
    B11 = F11 * F11 + F12 * F12
    B22 = F21 * F21 + F22 * F22
    B12 = F11 * F21 + F12 * F22   # = B21

    # Cauchy stress: σ = (mu_s / J)(B - I) + (kappa ln J / J) I
    # Off-diagonal has no volumetric part.
    sigma11 = mu_s * invJ * (B11 - 1.0) + kappa * lnJ * invJ
    sigma22 = mu_s * invJ * (B22 - 1.0) + kappa * lnJ * invJ
    sigma12 = mu_s * invJ * B12

    # Place into full arrays
    sxx[idxs] = sigma11
    sxy[idxs] = sigma12
    syy[idxs] = sigma22

    # --- Optional Kelvin–Voigt damping: η_s D(u) ---
    if eta_s > 0.0:
        du_dx = grad_central_x_2nd(a, dx)
        dv_dy = grad_central_y_2nd(b, dy)
        du_dy = grad_central_y_2nd(a, dy)
        dv_dx = grad_central_x_2nd(b, dx)

        strain_rate_xx = du_dx
        strain_rate_yy = dv_dy
        strain_rate_xy = 0.5 * (du_dy + dv_dx)

        sxx[idxs] += eta_s * strain_rate_xx[idxs]
        syy[idxs] += eta_s * strain_rate_yy[idxs]
        sxy[idxs] += eta_s * strain_rate_xy[idxs]

    # No Gaussian filtering here; smoothness is controlled physically
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

def velocity_RK4(u, v, p, X1, X2, velocity_bc, mu_s, kappa, eta_s , dx, dy, dt, rho_s, rho_f, phi, mu_f, w_t, gamma=0.0):
    """
    RK4 integration using stress divergence from blended stress field.
    """
    def rhs(u_stage, v_stage, p):
        sigma_sxx, sigma_sxy, sigma_syy, J = compute_solid_stress(X1, X2, dx, dy, mu_s, 
                                                                  kappa, phi, u_stage, 
                                                                  v_stage, p,  eta_s)   
        return velocity_rhs_blended(u, v, p, sigma_sxx, sigma_sxy, sigma_syy,
                                    dx, dy, phi,mu_f, rho_s, rho_f, w_t, gamma)

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
def grad_central_x_2nd(f, dx):
    df_dx = np.zeros_like(f)
    # Interior 2nd order central
    df_dx[:, 1:-1] = (f[:, 2:] - f[:, 0:-2]) / (2 * dx)

    # Left boundary (i=0)
    df_dx[:, 0] = (-3*f[:, 0] + 4*f[:, 1] - f[:, 2]) / (2*dx)
    # Right boundary (i=Nx-1)
    df_dx[:, -1] = (3*f[:, -1] - 4*f[:, -2] + f[:, -3]) / (2*dx)
    return df_dx

@njit
def grad_central_y_2nd(f, dy):
    df_dy = np.zeros_like(f)
    df_dy[1:-1, :] = (f[2:, :] - f[0:-2, :]) / (2 * dy)

    # Bottom boundary (j=0)
    df_dy[0, :] = (-3*f[0, :] + 4*f[1, :] - f[2, :]) / (2*dy)
    # Top boundary (j=Ny-1)
    df_dy[-1, :] = (3*f[-1, :] - 4*f[-2, :] + f[-3, :]) / (2*dy)
    return df_dy

@njit
def diff_upwind_3rd(f, u, dx, axis):
    """
    3rd Order Upwind Biased Finite Difference
    axis=0 for y, axis=1 for x
    """
    df = np.zeros_like(f)
    Ny, Nx = f.shape
    
    if axis == 1: # X-deriv
        for j in range(Ny):
            for i in range(2, Nx-2):
                vel = u[j, i]
                if vel > 0:
                    # Backward biased
                    df[j, i] = (2*f[j, i+1] + 3*f[j, i] - 6*f[j, i-1] + f[j, i-2]) / (6*dx)
                else:
                    # Forward biased
                    df[j, i] = (-f[j, i+2] + 6*f[j, i+1] - 3*f[j, i] - 2*f[j, i-1]) / (6*dx)
    else: # Y-deriv
        for i in range(Nx):
            for j in range(2, Ny-2):
                vel = u[j, i]
                if vel > 0:
                    df[j, i] = (2*f[j+1, i] + 3*f[j, i] - 6*f[j-1, i] + f[j-2, i]) / (6*dx)
                else:
                    df[j, i] = (-f[j+2, i] + 6*f[j+1, i] - 3*f[j, i] - 2*f[j-1, i]) / (6*dx)
    return df

@njit
def lap_2nd(f, dx, dy):
    lap = np.zeros_like(f)
    # Second derivative in x
    lap[:, 1:-1] += (f[:, 2:] - 2*f[:, 1:-1] + f[:, 0:-2]) / dx**2
    # Boundaries (second-order one-sided)
    # lap[:, 0] += (2*f[:, 1] - 5*f[:, 0] + 4*f[:, 0] - f[:, 2]) / dx**2
    # lap[:, -1] += (2*f[:, -2] - 5*f[:, -1] + 4*f[:, -1] - f[:, -3]) / dx**2

    # Second derivative in y
    lap[1:-1, :] += (f[2:, :] - 2*f[1:-1, :] + f[0:-2, :]) / dy**2
    # Boundaries (second-order one-sided)
    # lap[0, :] += (2*f[1, :] - 5*f[0, :] + 4*f[0, :] - f[2, :]) / dy**2
    # lap[-1, :] += (2*f[-2, :] - 5*f[-1, :] + 4*f[-1, :] - f[-3, :]) / dy**2

    return lap

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


def velocity_rhs_blended(u, v, p,
                         sigma_sxx_s, sigma_sxy_s, sigma_syy_s,
                         dx, dy, phi,
                         mu_f, rho_s, rho_f, w_t, gamma):
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

    # u_adv = -u * grad_central_x_2nd(u, dx) - v * grad_central_y_2nd(u, dy)
    # v_adv = -u * grad_central_x_2nd(v, dx) - v * grad_central_y_2nd(v, dy)
    # Use 3rd order upwind for advection stability at Re=1000
    u_adv = -u * diff_upwind_3rd(u, u, dx, 1) - v * diff_upwind_3rd(u, v, dy, 0)
    v_adv = -u * diff_upwind_3rd(v, u, dx, 1) - v * diff_upwind_3rd(v, v, dy, 0)
        
    # --- Heaviside and ∇H ---
    H = heaviside_smooth_alt(phi, w_t)
    dH_dx = grad_central_x_2nd(H, dx)
    dH_dy = grad_central_y_2nd(H, dy)

    # --- Local density ---
    rho_local = (1 - H) * rho_s + H * rho_f

    u_lap = lap_2nd(u, dx, dy)
    v_lap = lap_2nd(v, dx, dy)

    # --- Solid stress divergence ---
    dsxx_dx = grad_central_x_2nd(sigma_sxx_s, dx)
    dsxy_dy = grad_central_y_2nd(sigma_sxy_s, dy)

    dsxy_dx = grad_central_x_2nd(sigma_sxy_s, dx)
    dsyy_dy = grad_central_y_2nd(sigma_syy_s, dy)
    
    div_solid_x = dsxx_dx + dsxy_dy
    div_solid_y = dsxy_dx + dsyy_dy

    # --- Fluid stress components (only for jump term) ---
    # 4th-order central differences for derivatives
    du_dx = grad_central_x_2nd(u, dx)
    dv_dy = grad_central_y_2nd(v, dy)
    du_dy = grad_central_y_2nd(u, dy)
    dv_dx = grad_central_x_2nd(v, dx)

    div_vel = du_dx + dv_dy  # Should be ~0 for incompressible
    grad_div_vel_x = grad_central_x_2nd(div_vel, dx)
    grad_div_vel_y = grad_central_y_2nd(div_vel, dy)

    dp_dx = grad_central_x_2nd(p, dx)
    dp_dy = grad_central_y_2nd(p, dy)
    
    sigma_sxx_f = 2 * mu_f * du_dx
    sigma_syy_f = 2 * mu_f * dv_dy
    sigma_sxy_f = mu_f * (du_dy + dv_dx)

    # --- Jump term (sigma_f - sigma_s) · ∇H ---
    jump_x = (sigma_sxx_f - sigma_sxx_s) * dH_dx + (sigma_sxy_f - sigma_sxy_s) * dH_dy
    jump_y = (sigma_sxy_f - sigma_sxy_s) * dH_dx + (sigma_syy_f - sigma_syy_s) * dH_dy
    
        # --- NEW: Surface Tension Force ---
    st_force_x = 0.0
    st_force_y = 0.0
    
    if gamma > 1e-12:
        kappa = compute_curvature(phi, dx, dy)
        # Force = - gamma * kappa * grad(H)
        # Negative sign creates tension (minimizes area)
        st_force_x = -gamma * kappa * dH_dx
        st_force_y = -gamma * kappa * dH_dy

    # --- Final RHS ---
    rhs_u = u_adv + ((1 - H) * div_solid_x + H *(mu_f * (u_lap + grad_div_vel_x)) + jump_x + st_force_x - dp_dx) / (rho_local + 1e-12)
    rhs_v = v_adv + ((1 - H) * div_solid_y + H *(mu_f * (v_lap + grad_div_vel_y)) + jump_y + st_force_y - dp_dy) / (rho_local + 1e-12)
                             
    return rhs_u, rhs_v

def apply_velocity_BCs(bc, u, v):
    return bc(u, v)

# from scipy.sparse import lil_matrix, csr_matrix

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

    # 2nd-order central everywhere except outermost ring
    divU[1:-1, 1:-1] = (
        (a_star[1:-1, 2:] - a_star[1:-1, :-2]) / (2 * dx) +
        (b_star[2:,   1:-1] - b_star[:-2, 1:-1]) / (2 * dy)
    )

    rhs = (rho * divU / dt).ravel()
    rhs -= np.mean(rhs)

    if A is None:
        A = build_poisson_matrix(Nx, Ny, dx, dy)

    if ml is None:
        ml = pyamg.ruge_stuben_solver(A)  # Precompute AMG hierarchy

    p_flat = ml.solve(rhs, tol=1e-6, maxiter=100, x0=p_prev.ravel() if p_prev is not None else None)
    p = p_flat.reshape((Ny, Nx))
    p -= np.mean(p)

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