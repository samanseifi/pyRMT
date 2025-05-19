# main.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator


import glob
import imageio


import h5py

def create_grid(Nx, Ny, Lx, Ly):
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    return X, Y, dx, dy

def initialize_level_set(X, Y, x0, y0, R):
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
    return phi

def extrapolate_transverse_layers(X, phi, dx, dy, initial_band_width, max_layers):
    """
    Extrapolate solid reference maps (X1, X2) into fluid region.
    Iteratively extrapolates layer-by-layer near the interface.
    """
    Ny, Nx = X.shape
    X_ext = X.copy()

    solid_flag = (phi <= 0)
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

def compute_timestep(a, b, dx, dy, CFL, dt_min_cap, mu_s, rho_s):
    """
    Compute stable timestep based on fluid and solid speeds.
    """
    cs_solid = np.sqrt(mu_s / (rho_s + 1e-12))
    dt_solid = CFL * dx / (cs_solid + 1e-14)
    u_max = np.max(np.sqrt(a**2 + b**2))
    dt_fluid = CFL * dx / (u_max + 1e-6)
    dt = min(dt_solid, dt_fluid, dt_min_cap)
    return dt

def advect_phi(phi, a, b, X, Y, dt):
    """
    Semi-Lagrangian advection of phi field using bilinear interpolation.
    """
    X_back = X - dt * a
    Y_back = Y - dt * b

    interpolator = RegularGridInterpolator(
        (Y[:,0], X[0,:]), phi,
        method='linear', bounds_error=False, fill_value=None
    )
    points = np.stack([Y_back.ravel(), X_back.ravel()], axis=-1)
    phi_new = interpolator(points).reshape(phi.shape)

    # fallback if NaN (optional: fallback to old phi)
    phi_new[np.isnan(phi_new)] = phi[np.isnan(phi_new)]

    return phi_new

def compute_solid_stress(X1, X2, dx, dy, mu_s, kappa, phi):
    # Gradients
    dX1_dy, dX1_dx = np.gradient(X1, dy, dx, edge_order=2)
    dX2_dy, dX2_dx = np.gradient(X2, dy, dx, edge_order=2)

    # Preallocate output
    Ny, Nx = X1.shape
    sxx = np.zeros((Ny, Nx))
    sxy = np.zeros((Ny, Nx))
    syy = np.zeros((Ny, Nx))
    J = np.ones((Ny, Nx))

    # Solid masks
    solid_mask = phi <= 2*dx

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

    J_temp = np.clip(np.linalg.det(F), 1e-3, 1e3)
    J_temp = gaussian_filter(J_temp, sigma=0.5)
    J[idxs] = J_temp

    # sigma = mu/J (F F^T - I) + kappa (J-1) I
    sigma = (mu_s / J_temp)[:, None, None] * (FFt - I_expand) + \
            kappa * (J_temp - 1)[:, None, None] * I_expand


    # Write back to sxx, sxy, syy
    sxx[idxs] = sigma[:, 0, 0]
    sxy[idxs] = sigma[:, 1, 0]
    syy[idxs] = sigma[:, 1, 1]
    
    # Optional damping for soft solids (pseudo-viscous)
    strain_rate_xx = dX1_dx * 0.0  # Placeholder if you compute it explicitly
    strain_rate_yy = dX2_dy * 0.0
    strain_rate_xy = 0.5 * (dX1_dy + dX2_dx)

    eta_s = 0.1  # solid damping coefficient
    sxx[idxs] += eta_s * strain_rate_xx[idxs]
    syy[idxs] += eta_s * strain_rate_yy[idxs]
    sxy[idxs] += eta_s * strain_rate_xy[idxs]
    
    # Apply smoothing to stress fields
    sxx = gaussian_filter(sxx, sigma=0.5)
    sxy = gaussian_filter(sxy, sigma=0.5)
    syy = gaussian_filter(syy, sigma=0.5)
    
    # Apply solid mask
    solid_mask = (phi <= 0).astype(float)
    sxx = solid_mask * sxx
    sxy = solid_mask * sxy
    syy = solid_mask * syy

    return sxx, sxy, syy, J

def compute_fluid_stress(a, b, mu_f, dx, dy, phi):
    # Compute velocity gradients
    da_dy, da_dx = np.gradient(a, dy, dx, edge_order=2)
    db_dy, db_dx = np.gradient(b, dy, dx, edge_order=2)

    # Compute viscous stress components
    sigma_xx = 2 * mu_f * da_dx
    sigma_yy = 2 * mu_f * db_dy
    sigma_xy = mu_f * (da_dy + db_dx)

    # Apply fluid-only mask
    if phi is not None:
        fluid_mask = phi >= 0
    else:
        fluid_mask = np.ones_like(a, dtype=bool)

    # Initialize output arrays
    sxx = np.zeros_like(a)
    sxy = np.zeros_like(a)
    syy = np.zeros_like(a)

    # Assign stress only in fluid region
    sxx[fluid_mask] = sigma_xx[fluid_mask]
    sxy[fluid_mask] = sigma_xy[fluid_mask]
    syy[fluid_mask] = sigma_yy[fluid_mask]

    return sxx, sxy, syy

def heaviside_smooth_alt(x, w_t):
    """
    Smooth Heaviside function based on transition width w_t.
    """
    H = np.zeros_like(x)
    inside_band = np.abs(x) <= w_t
    H[x > w_t] = 1.0
    H[inside_band] = 0.5 * (1 + x[inside_band]/w_t + (1/np.pi) * np.sin(np.pi * x[inside_band]/w_t))
    return H

def velocity_RK3(a, b, sxx, sxy, syy, rho_local, dx, dy, dt, phi, nu_s):
    rhs_a1, rhs_b1 = velocity_rhs(a, b, sxx, sxy, syy, rho_local, dx, dy, phi, nu_s)
    a1 = a + dt * rhs_a1
    b1 = b + dt * rhs_b1
    a1, b1 = lid_bc(a1, b1)

    rhs_a2, rhs_b2 = velocity_rhs(a1, b1, sxx, sxy, syy, rho_local, dx, dy, phi, nu_s)
    a2 = 0.75 * a + 0.25 * (a1 + dt * rhs_a2)
    b2 = 0.75 * b + 0.25 * (b1 + dt * rhs_b2)
    a2, b2 = lid_bc(a1, b1)


    rhs_a3, rhs_b3 = velocity_rhs(a2, b2, sxx, sxy, syy, rho_local, dx, dy, phi, nu_s)
    a_new = (1/3) * a + (2/3) * (a2 + dt * rhs_a3)
    b_new = (1/3) * b + (2/3) * (b2 + dt * rhs_b3)
    a_new, b_new = lid_bc(a_new, b_new)

    return a_new, b_new

def velocity_rhs(a, b, sxx, sxy, syy, rho_local, dx, dy, phi, nu_s):
    # Advection
    adv_a = -advection_rhs_velocity(a, a, b, dx, dy)
    adv_b = -advection_rhs_velocity(b, a, b, dx, dy)

    # Stress divergence
    dsxx_dy, dsxx_dx = np.gradient(sxx, dy, dx, edge_order=2)
    dsxy_dy, dsxy_dx = np.gradient(sxy, dy, dx, edge_order=2)
    dsyy_dy, dsyy_dx = np.gradient(syy, dy, dx, edge_order=2)

    div_tau_x = dsxx_dx + dsxy_dy
    div_tau_y = dsxy_dx + dsyy_dy

    stress_a = div_tau_x / (rho_local + 1e-12)
    stress_b = div_tau_y / (rho_local + 1e-12)

    # Viscous penalty (solid only)
    if phi is None:
        solid_mask = np.ones_like(a)
        visc_a = np.ones_like(a)
        visc_b = np.ones_like(b)
    else:
        solid_mask = (phi <= 0).astype(float)
        lap_a = laplacian_2D(a, dx, dy)
        lap_b = laplacian_2D(b, dx, dy)

        visc_a = nu_s * lap_a * solid_mask / (rho_local + 1e-12)
        visc_b = nu_s * lap_b * solid_mask / (rho_local + 1e-12)

    # Combine all
    rhs_a = adv_a + stress_a + visc_a
    rhs_b = adv_b + stress_b + visc_b

    return rhs_a, rhs_b

def lid_bc(u, v):
    """
    Apply lid-driven cavity boundary conditions:
    - Top lid moves rightward at u = 1
    - Other walls are no-slip
    """
    u_bc = u.copy()
    v_bc = v.copy()

    Ny, Nx = u.shape

    # Right boundary (outflow)
    u_bc[:, -1] = 0.0
    v_bc[:, -1] = 0.0

    # Left boundary (inflow)
    u_bc[:, 0] = 0.0
    v_bc[:, 0] = 0.0

    # Bottom wall (no-slip)
    u_bc[0, :] = 0.0
    v_bc[0, :] = 0.0

    # Top lid (moving right)
    u_bc[-1, :] = 1 # 1 for lid-driven
    v_bc[-1, :] = 0.0

    return u_bc, v_bc

def advection_rhs_velocity(q, a, b, dx, dy):
    """
    Advection of velocity components: d/dx(a*q) + d/dy(b*q)
    """
    Ny, Nx = q.shape
    fx = np.zeros((Ny, Nx))
    fy = np.zeros((Ny, Nx))

    for j in range(Ny):
        flux_x = a[j, :] * q[j, :]
        fx[j, :] = weno_flux_z(flux_x, a[j, :], dx)

    for i in range(Nx):
        flux_y = b[:, i] * q[:, i]
        fy[:, i] = weno_flux_z(flux_y, b[:, i], dy)

    return fx + fy

def weno_flux_z(U, vel, dx):
    """
    WENO-Z reconstruction for advection flux splitting.
    """
    N = len(U)
    f = vel * U
    alpha = np.max(np.abs(vel))

    f_plus = 0.5 * (f + alpha * U)
    f_minus = 0.5 * (f - alpha * U)

    # Periodic extension
    fpe = np.concatenate([f_plus[-3:], f_plus, f_plus[:3]])
    fme = np.concatenate([f_minus[-3:], f_minus, f_minus[:3]])

    fhat_plus = np.zeros(N + 1)
    fhat_minus = np.zeros(N + 1)

    for i in range(N + 1):
        fhat_plus[i] = weno_z_reconstruct(fpe[i:i+5])
        fhat_minus[i] = weno_z_reconstruct(fme[i:i+5][::-1])

    dudx = (fhat_plus[1:] - fhat_plus[:-1] + fhat_minus[1:] - fhat_minus[:-1]) / dx

    return dudx

def weno_z_reconstruct(f):
    """
    5-point WENO-Z reconstruction at an interface.
    """
    eps = 1e-6
    IS0 = (13/12)*(f[0]-2*f[1]+f[2])**2 + (1/4)*(f[0]-4*f[1]+3*f[2])**2
    IS1 = (13/12)*(f[1]-2*f[2]+f[3])**2 + (1/4)*(f[1]-f[3])**2
    IS2 = (13/12)*(f[2]-2*f[3]+f[4])**2 + (1/4)*(3*f[2]-4*f[3]+f[4])**2

    tau5 = np.abs(IS0 - IS2)

    alpha0 = 0.1 * (1 + (tau5 / (IS0 + eps))**2)
    alpha1 = 0.6 * (1 + (tau5 / (IS1 + eps))**2)
    alpha2 = 0.3 * (1 + (tau5 / (IS2 + eps))**2)

    alpha_sum = alpha0 + alpha1 + alpha2

    w0 = alpha0 / alpha_sum
    w1 = alpha1 / alpha_sum
    w2 = alpha2 / alpha_sum

    q0 = (2*f[0] - 7*f[1] + 11*f[2]) / 6
    q1 = (-f[1] + 5*f[2] + 2*f[3]) / 6
    q2 = (2*f[2] + 5*f[3] - f[4]) / 6

    return w0*q0 + w1*q1 + w2*q2

def pressure_poisson_varrho(u_star, v_star, p_old,
                            dx, dy, dt, rho,
                            max_iter=500, tol=1e-6, omega=1.7):
    """
    Variable‑density Poisson:  div( (dt/rho) grad p ) = div(u*)
    Gauss–Seidel‑SOR with one reference pressure (pinned at [0,0]).
    """
    Ny, Nx = u_star.shape
    beta = dt / (rho + 1e-12)

    inv_dx2 = 1.0 / dx**2
    inv_dy2 = 1.0 / dy**2
    rhs     = divergence_2d(u_star, v_star, dx, dy)

    p = p_old.copy()

    for _ in range(max_iter):
        p_prev = p.copy()

        for j in range(1, Ny-1):
            for i in range(1, Nx-1):
                bxW = 2*beta[j,i]*beta[j,i-1] / (beta[j,i] + beta[j,i-1] + 1e-14)
                bxE = 2*beta[j,i]*beta[j,i+1] / (beta[j,i] + beta[j,i+1] + 1e-14)
                byS = 2*beta[j,i]*beta[j-1,i] / (beta[j,i] + beta[j-1,i] + 1e-14)
                byN = 2*beta[j,i]*beta[j+1,i] / (beta[j,i] + beta[j+1,i] + 1e-14)

                denom = (bxW + bxE)*inv_dx2 + (byS + byN)*inv_dy2

                p_new = (
                    bxE*inv_dx2 * p[j, i+1] + bxW*inv_dx2 * p[j, i-1] +
                    byN*inv_dy2 * p[j+1, i] + byS*inv_dy2 * p[j-1, i] -
                    rhs[j, i]
                ) / denom

                # SOR update
                p[j, i] = (1-omega)*p[j, i] + omega*p_new

        # homogeneous Neumann (can be mixed with periodic if desired)
        p[:,  0] = p[:, 1]
        p[:, -1] = p[:, -2]
        p[ 0, :] = p[1, :]
        p[-1, :] = p[-2, :]

        p[0,0] = 0.0                 # reference pressure

        if np.max(np.abs(p - p_prev)) < tol:
            break

    return p


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

def laplacian_2D(U, dx, dy):
    """
    5-point Laplacian of scalar field U.
    """
    lapU = np.zeros_like(U)
    lapU[1:-1,1:-1] = (
        (U[1:-1,2:] - 2*U[1:-1,1:-1] + U[1:-1,:-2]) / dx**2 +
        (U[2:,1:-1] - 2*U[1:-1,1:-1] + U[:-2,1:-1]) / dy**2
    )
    return lapU


import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

def reconstruct_phi_from_reference_map(X1, X2, phi0, X0, Y0):
    """
    Reconstruct level set φ(x, t) = φ(ξ(x, t), 0)
    where X1, X2 are ξ components, and phi0 is φ(ξ, 0).
    """
    interpolator = RegularGridInterpolator((Y0[:, 0], X0[0, :]), phi0, method='linear', bounds_error=False, fill_value=np.nan)
    pts = np.stack([X2.ravel(), X1.ravel()], axis=-1)
    phi = interpolator(pts).reshape(X1.shape)
    phi[np.isnan(phi)] = 1.0  # default to fluid if undefined
    return phi

def rhie_chow_projection(u_star, v_star, p, dx, dy, dt, rho, sigma_xx, sigma_xy, sigma_yy):
    """
    Rhie-Chow-like projection with corrected pressure gradient and body force from divergence of stress.
    """
    epsilon = 1e-12
    rho_safe = rho + epsilon

    # Pressure gradients (central)
    dpdx = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2 * dx)
    dpdy = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2 * dy)

    # Body force (divergence of stress)
    fx = (np.roll(sigma_xx, -1, axis=1) - np.roll(sigma_xx, 1, axis=1)) / (2 * dx) + \
         (np.roll(sigma_xy, -1, axis=0) - np.roll(sigma_xy, 1, axis=0)) / (2 * dy)
    fy = (np.roll(sigma_xy, -1, axis=1) - np.roll(sigma_xy, 1, axis=1)) / (2 * dx) + \
         (np.roll(sigma_yy, -1, axis=0) - np.roll(sigma_yy, 1, axis=0)) / (2 * dy)

    fx /= rho_safe
    fy /= rho_safe

    # Corrected velocities
    u_corr = u_star - dt * (dpdx / rho_safe - fx)
    v_corr = v_star - dt * (dpdy / rho_safe - fy)

    return u_corr, v_corr

def filter_velocity_in_fluid(a, b, phi, sigma=0.5):
    """
    Apply Gaussian smoothing only in the fluid region (phi > 0).
    """
    fluid_mask = (phi > 0).astype(float)
    a_smoothed = fluid_mask * gaussian_filter(a, sigma=sigma) + (1 - fluid_mask) * a
    b_smoothed = fluid_mask * gaussian_filter(b, sigma=sigma) + (1 - fluid_mask) * b
    return a_smoothed, b_smoothed


def velocity_projection_varrho_with_rhie_chow(u_star, v_star, p, dx, dy, dt, rho, sigma_xx, sigma_xy, sigma_yy):
    """
    Rhie-Chow-like velocity correction with body force inclusion to suppress checkerboard artifacts.
    """
    epsilon = 1e-12
    rho_safe = rho + epsilon
    Nx, Ny = p.shape

    # Interpolate face-centered pressure gradients
    dpdx_face = np.zeros_like(p)
    dpdy_face = np.zeros_like(p)

    dpdx_face[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * dx)
    dpdy_face[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2 * dy)

    # Cell-centered gradients
    dpdx_cell = np.zeros_like(p)
    dpdy_cell = np.zeros_like(p)
    dpdx_cell[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * dx)
    dpdy_cell[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2 * dy)

    # Rhie-Chow interpolation: mix face and cell-centered gradients
    dpdx_rc = 0.5 * (dpdx_face + dpdx_cell)
    dpdy_rc = 0.5 * (dpdy_face + dpdy_cell)

    # Body force term
    fx, fy = compute_body_force(sigma_xx, sigma_xy, sigma_yy, dx, dy, rho)

    u_corr = u_star - dt * (dpdx_rc - fx)
    v_corr = v_star - dt * (dpdy_rc - fy)

    return u_corr, v_corr

def compute_body_force(sxx, sxy, syy, dx, dy, rho):
    """
    Compute the body force per unit mass from the divergence of the stress tensor.
    """
    fx = np.zeros_like(sxx)
    fy = np.zeros_like(syy)

    fx[:, 1:-1] = (sxx[:, 2:] - sxx[:, :-2]) / (2 * dx)
    fx[1:-1, :] += (sxy[2:, :] - sxy[:-2, :]) / (2 * dy)

    fy[:, 1:-1] = (sxy[:, 2:] - sxy[:, :-2]) / (2 * dx)
    fy[1:-1, :] += (syy[2:, :] - syy[:-2, :]) / (2 * dy)

    fx /= (rho + 1e-12)
    fy /= (rho + 1e-12)

    return fx, fy

# --------------------------
# Grid Setup
# --------------------------
Nx, Ny = 50, 50
Lx, Ly = 1.0, 1.0
X, Y, dx, dy = create_grid(Nx, Ny, Lx, Ly)

# --------------------------
# Initial Fields
# --------------------------
phi = initialize_level_set(X, Y, x0=0.6, y0=0.5, R=0.2)
phi = apply_phi_BCs(phi)
solid_mask = (phi <= 0).astype(float)

X1 = X * solid_mask
X2 = Y * solid_mask

X1 = extrapolate_transverse_layers(X1, phi, dx, dy, 3 * dx, 3)
X2 = extrapolate_transverse_layers(X2, phi, dx, dy, 3 * dx, 3)

# Physical Parameters
mu_s, kappa, rho_s, nu_s = 0.1, 10.0, 1.0, 0.01
mu_f, rho_f = 0.01, 1.0
w_t = 1.5 * dx

# Velocity and Pressure
a = np.zeros((Nx, Ny))
b = np.zeros((Nx, Ny))
p = np.zeros((Nx, Ny))

CFL = 0.2
dt_min_cap = 1e-3
max_steps = 3000000

# --------------------------
# Time Loop
# --------------------------
for step in range(1, max_steps + 1):
    dt = compute_timestep(a, b, dx, dy, CFL, dt_min_cap, mu_s, rho_s)
    
    total_time = step * dt
    # only solve for fluid until total_time ~ 10
    if total_time > 2:
        # Advection of phi
        dt *= 0.1
        
        phi = advect_phi(phi, a, b, X, Y, dt)
        phi = apply_phi_BCs(phi)
        solid_mask = (phi <= 0).astype(float)

        # Advect Reference Maps
        X1 = advect_phi(X1, a, b, X, Y, dt)
        X2 = advect_phi(X2, a, b, X, Y, dt)

        X1 *= solid_mask
        X2 *= solid_mask

        X1 = extrapolate_transverse_layers(X1, phi, dx, dy, 3 * dx, 5)
        X2 = extrapolate_transverse_layers(X2, phi, dx, dy, 3 * dx, 5)
        

        # Compute Stress Fields
        sigma_sxx, sigma_sxy, sigma_syy, J = compute_solid_stress(X1, X2, dx, dy, mu_s, kappa, phi)
        sigma_fxx, sigma_fxy, sigma_fyy = compute_fluid_stress(a, b, mu_f, dx, dy, phi)

        H = heaviside_smooth_alt(phi, w_t)
        sigma_xx = (1 - H) * sigma_sxx + H * sigma_fxx
        sigma_xy = (1 - H) * sigma_sxy + H * sigma_fxy
        sigma_yy = (1 - H) * sigma_syy + H * sigma_fyy    
    
        rho_local = (1 - H) * rho_s + H * rho_f
        
    else:
        # We only solve for fluid stress
        sigma_fxx, sigma_fxy, sigma_fyy = compute_fluid_stress(a, b, mu_f, dx, dy, None)
        sigma_xx = sigma_fxx
        sigma_xy = sigma_fxy
        sigma_yy = sigma_fyy
        
        rho_local = rho_f*np.ones_like(a)

    # RK3 Update
    a_star, b_star = velocity_RK3(a, b, sigma_xx, sigma_xy, sigma_yy, rho_local, dx, dy, dt, phi, nu_s)
    a_star, b_star = lid_bc(a_star, b_star)

    # Pressure Solve
    p = pressure_poisson_varrho(a_star, b_star, p, dx, dy, dt, rho_local, 100, 1e-5, 1.5)

    # Use Rhie-Chow projection with body force
    a_new, b_new = rhie_chow_projection(
        a_star, b_star, p, dx, dy, dt, rho_local, sigma_xx, sigma_xy, sigma_yy
    )
    a = gaussian_filter(a_new, sigma=0.5)
    b = gaussian_filter(b_new, sigma=0.5)
    
    # smooth velocity fields
    # a, b = filter_velocity_in_fluid(a_new, b_new, phi, sigma=0.5)
    a, b = lid_bc(a, b)
    
    if step % 50 == 0 or step == 1:
        vmag = np.sqrt(a**2 + b**2)
        if total_time > 2:
            solid_J = J[phi <= 0]
            print(
            f"[Step {step:05d}] dt={dt:.2e}, "
            f"max|v|={np.max(vmag):.3f}, "
            f"min(J)={np.min(solid_J):.3f}, "
            f"mean(J)={np.mean(solid_J):.3f}, "
            f"max|σ|={np.max(np.abs(sigma_xx)):.2f}"
            )
        else:
            print(
            f"[Step {step:05d}] dt={dt:.2e}, "
            f"max|v|={np.max(vmag):.3f}, "
            f"max|σ|={np.max(np.abs(sigma_xx)):.2f}"
            )

    # Visualization
    if step == 1 or step % 50 == 0:
        # visualize_fields_gif(X, Y, phi, X1, J, a, b, sigma_xx, sigma_yy, sigma_xy, solid_mask, dt, step)
        # visualize_fields(X, Y, phi, X1, J, a, b, sigma_xx, sigma_yy, sigma_xy, solid_mask, dt, step)
        
        with h5py.File(f"frames/data_{step:05d}.h5", "w") as f:
            f.create_dataset("phi", data=phi)
            if total_time > 2:
                f.create_dataset("X1", data=X1)
                f.create_dataset("X2", data=X2)
                f.create_dataset("J", data=J)
            f.create_dataset("a", data=a)
            f.create_dataset("b", data=b)
            f.create_dataset("sigma_xx", data=sigma_xx)
            f.create_dataset("sigma_yy", data=sigma_yy)
            f.create_dataset("sigma_xy", data=sigma_xy)

