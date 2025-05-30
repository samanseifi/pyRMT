# main.py
import numpy as np

from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg, spsolve

from numba import njit, prange
import pyamg

import h5py

import cProfile
import pstats

def create_grid(Nx, Ny, Lx, Ly):
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    return X, Y, dx, dy

def initialize_level_set(X, Y, x0, y0, R):
    """ 
    Initialize a disc-shaped level set function phi centered at (x0, y0) with radius R.
    """
    r = np.sqrt((X - x0)**2 + (Y - y0)**2)
    phi = r - R
    return phi

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

def extrapolate_transverse_layers_slow(X, phi, dx, dy, initial_band_width, max_layers):
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
    Ny, Nx = X.shape
    X_ext = X.copy()

    solid_flag = (phi < 0)
    known_flag = solid_flag.copy()

    Xgrid, Ygrid = np.meshgrid(dx * np.arange(Nx), dy * np.arange(Ny))

    stencil_radius = 4 * np.sqrt(dx**2 + dy**2)
    max_iter = 20

    for layer in range(max_layers):
        # Find targets: cells adjacent to known region
        target_flag = np.zeros_like(X, dtype=bool)
        for j in range(1, Ny-1):
            for i in range(1, Nx-1):
                if not known_flag[j, i] and np.any(known_flag[j-1:j+2, i-1:i+2]):
                    target_flag[j, i] = True

        if not np.any(target_flag):
            break

        # Extrapolate
        for iter in range(max_iter):
            progress = False
            for j in range(1, Ny-1):
                for i in range(1, Nx-1):
                    if target_flag[j, i]:
                        # Build stencil
                        x_pts, y_pts, x_vals, wts = [], [], [], []
                        for jj in range(max(0, j-4), min(Ny, j+5)):
                            for ii in range(max(0, i-4), min(Nx, i+5)):
                                if known_flag[jj, ii]:
                                    dist = np.sqrt((Xgrid[jj, ii] - Xgrid[j, i])**2 + (Ygrid[jj, ii] - Ygrid[j, i])**2)
                                    if dist <= stencil_radius:
                                        wt = np.exp(-(dist / stencil_radius)**2)
                                        x_pts.append(Xgrid[jj, ii])
                                        y_pts.append(Ygrid[jj, ii])
                                        x_vals.append(X_ext[jj, ii])
                                        wts.append(wt)
                        if len(x_vals) >= 3:
                            A = np.vstack([np.ones(len(x_pts)), x_pts, y_pts]).T
                            W = np.diag(wts)
                            coeffs = np.linalg.lstsq(W @ A, W @ np.array(x_vals), rcond=None)[0]
                            X_ext[j, i] = coeffs[0] + coeffs[1] * Xgrid[j, i] + coeffs[2] * Ygrid[j, i]
                            known_flag[j, i] = True
                            target_flag[j, i] = False
                            progress = True
            if not progress:
                break

    return X_ext

@njit(parallel=True)
def extrapolate_transverse_layers(X, phi, dx, dy, band_width, max_layers):
    Ny, Nx = X.shape
    X_ext = X.copy()
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
                    b = np.zeros(100)
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
                                    b[count] = X_ext[jj, ii]
                                    w[count] = np.exp(-dist_sq / stencil_radius_sq)
                                    count += 1

                    if count >= 3:
                        Aw = np.zeros((3, 3))
                        Bw = np.zeros(3)

                        for n in range(count):
                            a0 = A[n, 0]
                            a1 = A[n, 1]
                            a2 = A[n, 2]
                            bw = b[n]
                            ww = w[n]

                            Aw[0, 0] += ww * a0 * a0
                            Aw[0, 1] += ww * a0 * a1
                            Aw[0, 2] += ww * a0 * a2
                            Aw[1, 1] += ww * a1 * a1
                            Aw[1, 2] += ww * a1 * a2
                            Aw[2, 2] += ww * a2 * a2

                            Bw[0] += ww * a0 * bw
                            Bw[1] += ww * a1 * bw
                            Bw[2] += ww * a2 * bw

                        # Symmetrize Aw
                        Aw[1, 0] = Aw[0, 1]
                        Aw[2, 0] = Aw[0, 2]
                        Aw[2, 1] = Aw[1, 2]

                        # Solve Aw @ coeffs = Bw
                        det = (Aw[0,0]*(Aw[1,1]*Aw[2,2] - Aw[1,2]*Aw[2,1])
                            - Aw[0,1]*(Aw[1,0]*Aw[2,2] - Aw[1,2]*Aw[2,0])
                            + Aw[0,2]*(Aw[1,0]*Aw[2,1] - Aw[1,1]*Aw[2,0]))

                        if np.abs(det) > 1e-10:
                            coeffs = np.linalg.solve(Aw, Bw)
                            X_ext[j, i] = coeffs[0] + coeffs[1] * x0 + coeffs[2] * y0
                            known_flag[j, i] = True

    return X_ext

@njit(parallel=True)
def extrapolate_transverse_layers_2field(X1, X2, phi, dx, dy, band_width, max_layers):
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

def advect_semi_lagrangian_rk4_slow(q, a, b, X, Y, dt):
    """
    Semi-Lagrangian advection using RK4 backtracking for high accuracy.

    Parameters:
        q: 2D array of the field to advect.
        a, b: 2D arrays for x- and y-components of velocity.
        X, Y: 2D arrays of grid coordinates.
        dt: time step.

    Returns:
        q_new: advected field.
    """
    def interp(u, x, y):
        return interpolate_velocity(u, x, y, X, Y)

    # Stage 1
    k1x = interp(a, X, Y)
    k1y = interp(b, X, Y)

    # Stage 2
    X2 = X - 0.5 * dt * k1x
    Y2 = Y - 0.5 * dt * k1y
    k2x = interp(a, X2, Y2)
    k2y = interp(b, X2, Y2)

    # Stage 3
    X3 = X - 0.5 * dt * k2x
    Y3 = Y - 0.5 * dt * k2y
    k3x = interp(a, X3, Y3)
    k3y = interp(b, X3, Y3)

    # Stage 4
    X4 = X - dt * k3x
    Y4 = Y - dt * k3y
    k4x = interp(a, X4, Y4)
    k4y = interp(b, X4, Y4)

    # Combine stages to compute final departure points
    X_back = X - (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
    Y_back = Y - (dt / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)

    # Interpolate the field q at the backtracked locations
    interpolator = RegularGridInterpolator(
        (Y[:, 0], X[0, :]), q,
        method='linear', bounds_error=False, fill_value=None
    )
    points = np.stack([Y_back.ravel(), X_back.ravel()], axis=-1)
    q_new = interpolator(points).reshape(q.shape)

    # Optional: fallback to original value if NaNs occur
    q_new[np.isnan(q_new)] = q[np.isnan(q_new)]

    return q_new

def interpolate_velocity(u, Xq, Yq, X_grid, Y_grid):
    """
    Interpolate velocity field `u` at query points (Xq, Yq).
    """
    interpolator = RegularGridInterpolator(
        (Y_grid[:, 0], X_grid[0, :]), u,
        method='linear', bounds_error=False, fill_value=0.0
    )
    points = np.stack([Yq.ravel(), Xq.ravel()], axis=-1)
    return interpolator(points).reshape(u.shape)

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

def compute_solid_stress(X1, X2, dx, dy, mu_s, kappa, phi, a, b, eta_s=0.0):
    """
    Compute solid stress tensor based on reference maps X1, X2 
    (with extrapolation into fluid region). The stress tensor is 
    computed using Neo-Hookean Cauchy stress formula:
    
        σ = μ_s / J (F F^T - I) + kappa (J-1) I
    
    where F is the deformation gradient computed within extrapolated region,
    J is the determinant of F, and I is the identity matrix.
   
    The solid mask is applied to ensure that the stress is only computed
    in the solid region (phi <= 0).
    The function returns the stress components sxx, sxy, syy and the Jacobian J.
    
    Incorporating Maxwell Stresses from electrohydrodynamics (EHD)
    is not included in this function.
    """
    
    # Gradients
    dX1_dy, dX1_dx = np.gradient(X1, dy, dx, edge_order=2)
    dX2_dy, dX2_dx = np.gradient(X2, dy, dx, edge_order=2)

    # Preallocate output
    Ny, Nx = X1.shape
    sxx = np.zeros((Ny, Nx))
    sxy = np.zeros((Ny, Nx))
    syy = np.zeros((Ny, Nx))
    J = np.ones((Ny, Nx))

    # Solid with a narrow band mask for smoother divergence computation later
    solid_mask = phi <= 0

    # Flatten indices for masked region
    idxs = np.where(solid_mask)
    
    # Construct G matrix: shape (n, 2, 2)
    G = np.stack([
        np.stack([dX1_dx[idxs], dX1_dy[idxs]], axis=-1),
        np.stack([dX2_dx[idxs], dX2_dy[idxs]], axis=-1)
    ], axis=-2)  # shape: (n, 2, 2)

    # Invert G safely (skip points where inversion fails)
    detG = G[:, 0, 0] * G[:, 1, 1] - G[:, 0, 1] * G[:, 1, 0]
    good = np.abs(detG) > 1e-10
    Ginv = np.zeros_like(G)
    Ginv[good, 0, 0] =  G[good, 1, 1] / detG[good]
    Ginv[good, 0, 1] = -G[good, 0, 1] / detG[good]
    Ginv[good, 1, 0] = -G[good, 1, 0] / detG[good]
    Ginv[good, 1, 1] =  G[good, 0, 0] / detG[good]

    # Compute F = inv(G)
    F = Ginv
    Ft = np.transpose(F, axes=(0, 2, 1))
    FFt = np.einsum('nij,njk->nik', F, Ft)
    I = np.eye(2)
    I_expand = np.broadcast_to(I, FFt.shape)
    J_temp = np.linalg.det(F)
    J_temp = gaussian_filter(J_temp, 1.0)
    # clamp to a physically reasonable window
    J_temp = np.clip(J_temp, 0.5, 2.0)

    J[idxs] = J_temp
    sigma = (mu_s / J_temp)[:, None, None] * (FFt - I_expand) + \
            kappa * (J_temp - 1)[:, None, None] * I_expand

    # Write back to sxx, sxy, syy
    sxx[idxs] = sigma[:, 0, 0]
    sxy[idxs] = sigma[:, 1, 0]
    syy[idxs] = sigma[:, 1, 1]
    
    # Apply damping to solid stress
    if eta_s > 0:
        du_dx = (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1)) / (2 * dx)
        dv_dy = (np.roll(b, -1, axis=0) - np.roll(b, 1, axis=0)) / (2 * dy)
        du_dy = (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0)) / (2 * dy)
        dv_dx = (np.roll(b, -1, axis=1) - np.roll(b, 1, axis=1)) / (2 * dx)

        strain_rate_xx = du_dx
        strain_rate_yy = dv_dy
        strain_rate_xy = 0.5 * (du_dy + dv_dx)

        # Apply damping only inside solid
        sxx[idxs] += eta_s * strain_rate_xx[idxs]
        syy[idxs] += eta_s * strain_rate_yy[idxs]
        sxy[idxs] += eta_s * strain_rate_xy[idxs]
    
    # Apply Gaussian smoothing to the stress components
    sxx = gaussian_filter(sxx, 1.0)*solid_mask
    sxy = gaussian_filter(sxy, 1.0)*solid_mask
    syy = gaussian_filter(syy, 1.0)*solid_mask

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

def velocity_RK4(u, v, p, X1, X2, mu_s, kappa, eta_s , dx, dy, dt, rho_s, rho_f, phi, mu_f, w_t):
    """
    RK4 integration using stress divergence from blended stress field.
    """
    def rhs(u_stage, v_stage, p):
        sigma_sxx, sigma_sxy, sigma_syy, J = compute_solid_stress(X1, X2, dx, dy, mu_s, 
                                                                  kappa, phi, u_stage, 
                                                                  v_stage, eta_s)   
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

    return u_new, v_new

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
        ∇·σ = (1 - H) ∇·σ_solid + H ∇·σ_fluid - ∇p + f_jump
    
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
    u_adv = - u * (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx) \
            - v * (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dy)

    v_adv = - u * (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dx) \
            - v * (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dy)

    # --- Heaviside and ∇H ---
    H = heaviside_smooth_alt(phi, w_t)
    dH_dx = (np.roll(H, -1, axis=1) - np.roll(H, 1, axis=1)) / (2 * dx)
    dH_dy = (np.roll(H, -1, axis=0) - np.roll(H, 1, axis=0)) / (2 * dy)

    # --- Local density ---
    rho_local = (1 - H) * rho_s + H * rho_f

    # --- Fluid stress divergence via Laplacian ---
    def lap(f):
        return (np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1)) / dx**2 + \
               (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0)) / dy**2

    u_lap = mu_f * lap(u)
    v_lap = mu_f * lap(v)

    # --- Solid stress divergence ---
    dsxx_dx = (np.roll(sigma_sxx_s, -1, axis=1) - np.roll(sigma_sxx_s, 1, axis=1)) / (2 * dx)
    dsxy_dy = (np.roll(sigma_sxy_s, -1, axis=0) - np.roll(sigma_sxy_s, 1, axis=0)) / (2 * dy)

    dsxy_dx = (np.roll(sigma_sxy_s, -1, axis=1) - np.roll(sigma_sxy_s, 1, axis=1)) / (2 * dx)
    dsyy_dy = (np.roll(sigma_syy_s, -1, axis=0) - np.roll(sigma_syy_s, 1, axis=0)) / (2 * dy)

    div_solid_x = dsxx_dx + dsxy_dy
    div_solid_y = dsxy_dx + dsyy_dy
    
    # --- Apply gradient of pressure gradient ---
    dpdx = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2 * dx)
    dpdy = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2 * dy)

    # --- Fluid stress components (only for jump term) ---
    du_dx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
    dv_dy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dy)
    du_dy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dy)
    dv_dx = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dx)

    sigma_sxx_f = 2 * mu_f * du_dx
    sigma_syy_f = 2 * mu_f * dv_dy
    sigma_sxy_f = mu_f * (du_dy + dv_dx)

    # --- Jump term (sigma_f - sigma_s) · ∇H ---
    jump_x = (sigma_sxx_f - sigma_sxx_s) * dH_dx + (sigma_sxy_f - sigma_sxy_s) * dH_dy
    jump_y = (sigma_sxy_f - sigma_sxy_s) * dH_dx + (sigma_syy_f - sigma_syy_s) * dH_dy

    # --- Final RHS ---
    rhs_u = u_adv + ((1 - H) * div_solid_x + H * u_lap - dpdx + jump_x) / (rho_local + 1e-12)
    rhs_v = v_adv + ((1 - H) * div_solid_y + H * v_lap - dpdy + jump_y) / (rho_local + 1e-12)

    return rhs_u, rhs_v

def lid_bc(u, v):
    u_bc = u.copy()
    v_bc = v.copy()

    # Right boundary (no-slip)
    u_bc[:, -1] = 0.0
    v_bc[:, -1] = 0.0

    # Left boundary (no-slip)
    u_bc[:, 0] = 0.0
    v_bc[:, 0] = 0.0

    # Bottom wall (no-slip)
    u_bc[0, :] = 0.0
    v_bc[0, :] = 0.0

    # Top lid (moving right)
    u_bc[-1, :] = 1.0
    v_bc[-1, :] = 0.0

    # Corner adjustments
    u_bc[0, 0] = 0.0; v_bc[0, 0] = 0.0
    u_bc[0, -1] = 0.0; v_bc[0, -1] = 0.0
    u_bc[-1, 0] = 0.0; v_bc[-1, 0] = 0.0
    u_bc[-1, -1] = 0.0; v_bc[-1, -1] = 0.0

    return u_bc, v_bc

def divergence_2d(u, v, dx, dy):
    """
    Central-difference approximation of divergence of (u, v).
    """
    divU = np.zeros_like(u)
    divU[1:-1,1:-1] = (
        (u[1:-1,2:] - u[1:-1,:-2]) / (2*dx) +
        (v[2:,1:-1] - v[:-2,1:-1]) / (2*dy)
    )
    return divU

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


def pressure_projection_CG(a_star, b_star, dx, dy, dt, rho, A):
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
    divU[1:-1, 1:-1] = (
        (a_star[1:-1, 2:] - a_star[1:-1, :-2]) / (2 * dx) +
        (b_star[2:, 1:-1] - b_star[:-2, 1:-1]) / (2 * dy)
    )

    rhs = (rho * divU / dt).ravel()
    rhs -= np.mean(rhs)  # anchor pressure

    # A = build_poisson_matrix(Nx, Ny, dx, dy)

    p_flat, info = cg(A, rhs, x0=np.zeros_like(rhs), rtol=1e-8, maxiter=1000)
    if info != 0:
        print(f"[CG Warning] Pressure solve did not converge. Info: {info}")

    p = p_flat.reshape((Ny, Nx))
    p -= np.mean(p)  # anchor again

    dpdx = np.zeros_like(p)
    dpdy = np.zeros_like(p)
    dpdx[1:-1, 1:-1] = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
    dpdy[1:-1, 1:-1] = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)

    a = a_star - (dt / rho) * dpdx
    b = b_star - (dt / rho) * dpdy

    a, b = lid_bc(a, b)
    return a, b, p

def pressure_projection_amg(a_star, b_star, dx, dy, dt, rho, A=None, ml=None, p_prev=None):
    Ny, Nx = a_star.shape
    N = Nx * Ny

    divU = np.zeros_like(a_star)
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
    dpdx[1:-1, 1:-1] = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
    dpdy[1:-1, 1:-1] = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)

    a = a_star - (dt / rho) * dpdx
    b = b_star - (dt / rho) * dpdy

    a, b = lid_bc(a, b)
    return a, b, p, A, ml

def rebuild_phi_from_reference_map(X1, X2, X, Y, x0, y0, R):
    """
    Rebuild the level set function phi from the reference maps X1 and X2.
    This function is used to update the level set function after advection.
    """
    r = np.sqrt((X1 - x0)**2 + (X2 - y0)**2)
    phi = r - R
    return phi

def reinitialize_phi_PDE(phi_in, dx, dy, num_iters, apply_phi_BCs_func, dt_reinit_factor=0.5):
    """
    Reinitializes phi to be a signed distance function using a PDE-based method.
    Solves: phi_tau + S(phi_initial) * ( |grad(phi)| - 1 ) = 0
    
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
        np.ndarray: The reinitialized level set function.
    """
    phi = phi_in.copy()
    
    # S(phi_initial): Determine the sign of the initial phi field.
    # np.sign(0) is 0, which correctly identifies interface points.
    phi_initial_sign = phi_in / np.sqrt(phi_in**2 + dx**2)  # Avoid division by zero, ensure no NaNs.

    # Pseudo-timestep for the reinitialization PDE
    dt_reinit = dt_reinit_factor * min(dx, dy)

    for _ in range(num_iters):
        # Pad phi to handle boundaries when computing finite differences.
        # 'edge' mode replicates border values, a common approach if specific ghost cell
        # values based on complex BCs are not set up prior to gradient calculation.
        phi_padded = np.pad(phi, 1, mode='edge') 
        
        # Calculate forward and backward differences from the padded array.
        # These correspond to differences on the original grid dimensions.        
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
        
        # For S(phi_initial) == 0 (interface points themselves):
        # grad_phi_x_sq and grad_phi_y_sq remain 0 at these points.
        # This means grad_phi_mag will be 0 for these points.
        # The update term S(phi_initial) * (grad_phi_mag - 1) becomes 0 * (0 - 1) = 0.
        # Thus, interface points (where phi_initial_sign == 0) are kept fixed during reinitialization.
        # This is a desired behavior.

        grad_phi_mag = np.sqrt(grad_phi_x_sq + grad_phi_y_sq)
        
        # PDE term: S(phi_initial) * ( |grad(phi)| - 1 )
        dphi_dtau = phi_initial_sign * (grad_phi_mag - 1.0)
        
        # Update phi using Forward Euler in pseudo-time tau
        phi = phi - dt_reinit * dphi_dtau
            
    return phi

def semi_implicit_advect_vector(X1, X2, u, v, dx, dy, dt):
    """
    Semi-implicit advection for 2D vector field (X1, X2).
    Solves: (X^{n+1} - X^n)/dt + u · ∇X^{n+1} = 0 using backward Euler.
    """
    Ny, Nx = X1.shape
    N = Nx * Ny

    def idx(i, j):
        return i + j * Nx

    # Build a single advection matrix A (same for both components)
    A = lil_matrix((N, N))
    for j in range(Ny):
        for i in range(Nx):
            k = idx(i, j)
            u_ij = u[j, i]
            v_ij = v[j, i]
            A[k, k] = 1.0
            A[k, k] += 1e-8  # Regularize diagonal

            if i > 0:
                A[k, idx(i-1, j)] -= 0.5 * dt * u_ij / dx
                A[k, k]           += 0.5 * dt * u_ij / dx
            if i < Nx - 1:
                A[k, idx(i+1, j)] += 0.5 * dt * u_ij / dx
                A[k, k]           -= 0.5 * dt * u_ij / dx
            if j > 0:
                A[k, idx(i, j-1)] -= 0.5 * dt * v_ij / dy
                A[k, k]           += 0.5 * dt * v_ij / dy
            if j < Ny - 1:
                A[k, idx(i, j+1)] += 0.5 * dt * v_ij / dy
                A[k, k]           -= 0.5 * dt * v_ij / dy

    A = A.tocsr()

    # Solve for X1
    X1_flat = X1.ravel()
    X1_new_flat = spsolve(A, X1_flat)
    X1_new = X1_new_flat.reshape((Ny, Nx))

    # Solve for X2
    X2_flat = X2.ravel()
    X2_new_flat = spsolve(A, X2_flat)
    X2_new = X2_new_flat.reshape((Ny, Nx))

    return X1_new, X2_new

if __name__ == "__main__":

    profiler = cProfile.Profile()
    profiler.enable()

    # --------------------------
    # Grid Setup
    # --------------------------
    Nx, Ny = 256, 256
    Lx, Ly = 1.0, 1.0
    X, Y, dx, dy = create_grid(Nx, Ny, Lx, Ly)

    # --------------------------
    # Initial Fields
    # --------------------------
    phi = initialize_level_set(X, Y, x0=0.6, y0=0.5, R=0.2)
    phi = apply_phi_BCs(phi)
    phi0 = phi.copy()
    solid_mask = (phi <= 0).astype(float)

    X1 = X.copy()
    X2 = Y.copy()
    X1 = X * solid_mask
    X2 = Y * solid_mask

    # X1 = extrapolate_transverse_layers(X1, phi, dx, dy, 3 * dx, 2)
    # X2 = extrapolate_transverse_layers(X2, phi, dx, dy, 3 * dx, 2)
    X1, X2 = extrapolate_transverse_layers_2field(X1, X2, phi, dx, dy, 3 * dx, 3)


    # Physical Parameters
    mu_s, kappa, rho_s, eta_s = 0.1, 10.0, 1.0, 0.01
    mu_f, rho_f = 0.01, 1.0
    w_t =4 * dx

    # Velocity and Pressure
    a = np.zeros((Nx, Ny))
    b = np.zeros((Nx, Ny))
    p = np.zeros((Nx, Ny))

    A = build_poisson_matrix(Nx, Ny, dx, dy)

    CFL = 0.2
    dt_min_cap = 1e-3
    max_steps = 82000*3
    
    # with h5py.File("frames/data_164000.h5", "r") as f:
    #     phi = f["phi"][:]
    #     sigma_xx = f["sigma_xx"][:]
    #     sigma_xy = f["sigma_xy"][:]
    #     sigma_yy = f["sigma_yy"][:]
    #     X1 = f["X1"][:]
    #     X2 = f["X2"][:]
    #     J = f["J"][:]
    #     a = f["a"][:]
    #     b = f["b"][:]
    #     p = f["p"][:]
    #     div_vel = f["div_vel"][:]
        
    # --------------------------
    # Time Loop
    # --------------------------
    for step in range(1, max_steps + 1):
        dt = compute_timestep(a, b, dx, dy, CFL, dt_min_cap, mu_s, rho_s)
        
        dt *= 0.1

        phi = rebuild_phi_from_reference_map(X1, X2, X, Y, x0=0.6, y0=0.5, R=0.2)
        
        # if step % 100 == 0:
        phi = reinitialize_phi_PDE(phi, dx, dy, num_iters=100, apply_phi_BCs_func=None, dt_reinit_factor=0.1)

        solid_mask = (phi <= 0).astype(float)
        
        X1 = advect_semi_lagrangian_rk4(X1, a, b, X, Y, dt, dx, dy)
        X2 = advect_semi_lagrangian_rk4(X2, a, b, X, Y, dt, dx, dy)
        # X1, X2 = semi_implicit_advect_vector(X1, X2, a, b, dx, dy, dt)
        
        X1 *= solid_mask
        X2 *= solid_mask

        # X1 = extrapolate_transverse_layers(X1, phi, dx, dy, 3 * dx, 3)
        # X2 = extrapolate_transverse_layers(X2, phi, dx, dy, 3 * dx, 3)

        X1, X2 = extrapolate_transverse_layers_2field(X1, X2, phi, dx, dy, 3 * dx, 3)
        # X1, X2 = apply_reference_map_BCs(X1, X2, a, b, dt)
        
        a_star, b_star = velocity_RK4(a, b, p, X1, X2, mu_s, kappa, eta_s , dx, dy, dt, rho_s, rho_f, phi, mu_f, w_t)
        a_star, b_star = lid_bc(a_star, b_star)
        
        # Just for outputting stresses
        sigma_sxx, sigma_sxy, sigma_syy, J = compute_solid_stress(X1, X2, dx, dy, mu_s, kappa, phi, a, b, eta_s)

        H = heaviside_smooth_alt(phi, w_t)  
        rho_local = (1 - H) * rho_s + H * rho_f
        
        # Apply global pressure projection
        a, b, p, A, ml = pressure_projection_amg(a_star, b_star, dx, dy, dt, rho_local, A=A, ml=None, p_prev=p)
        a, b = lid_bc(a, b)
        
        if step % 100 == 0 or step == 1:
            vmag = np.sqrt(a**2 + b**2)
            div = divergence_2d(a, b, dx, dy)
            
            # Compute the area of the solid region
            solid_area = np.sum(solid_mask) * dx * dy
            print(
                    f"[Step {step:05d}] dt={dt:.2e}, "
                    f"max|v|={np.max(vmag):.3f}, "
                    f"min(J)={np.min(J):.3f}, "
                    f"mean(J)={np.mean(J):.3f}, "
                    f"max|σ_solid|={np.max(np.abs(sigma_sxx)):.2f}, "
                    f"max divergence = {np.max(np.abs(div)):.2e}, "
                    f"solid area = {solid_area:.4f}"
                    )

        # Visualization
        if step == 1 or step % 100 == 0:
            div = divergence_2d(a, b, dx, dy)

            with h5py.File(f"frames/data_{step:05d}.h5", "w") as f:
                f.create_dataset("phi", data=phi)
                f.create_dataset("X1", data=X1)
                f.create_dataset("X2", data=X2)
                f.create_dataset("J", data=J)
                f.create_dataset("a", data=a)
                f.create_dataset("b", data=b)
                f.create_dataset("p", data=p)
                f.create_dataset("sigma_xx", data=sigma_sxx)
                f.create_dataset("sigma_yy", data=sigma_syy)
                f.create_dataset("sigma_xy", data=sigma_sxy)
                f.create_dataset("div_vel", data=div)
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(30)  # Top 30 most expensive calls
