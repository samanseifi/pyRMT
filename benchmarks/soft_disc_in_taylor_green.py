from pyRMT.functions import *
from pyRMT.functions import _precompute_poisson_eigenvalues
from pyRMT.output import output_simulation_data
import numpy as np
import os
import h5py

# ==========================================
# Taylor–Green (Periodic) BC + Initial Field
# ==========================================

def tg_bc(u, v):
    """
    Periodic boundary conditions consistent with your existing style
    (copy interior values into ghost-like boundary layers).
    """
    u_bc = u.copy()
    v_bc = v.copy()

    # Left/right periodic
    u_bc[:, 0]  = u[:, -2]
    v_bc[:, 0]  = v[:, -2]
    u_bc[:, -1] = u[:, 1]
    v_bc[:, -1] = v[:, 1]

    # Bottom/top periodic
    u_bc[0, :]  = u[-2, :]
    v_bc[0, :]  = v[-2, :]
    u_bc[-1, :] = u[1, :]
    v_bc[-1, :] = v[1, :]

    return u_bc, v_bc

def get_taylor_green_velocity(X, Y, t, U0=1.0, t_ramp=0.0):
    """
    Standard 2D Taylor–Green vortex on [0,1]x[0,1] with k=2*pi.

      u =  U0 sin(2πx) cos(2πy)
      v = -U0 cos(2πx) sin(2πy)

    Optional smooth ramp-in for startup.
    """
    k = 2.0 * np.pi

    # ramp = 1.0
    # if t_ramp > 0.0 and t < t_ramp:
    #     s = t / t_ramp
    #     ramp = s * s * (3.0 - 2.0 * s)  # C1 smoothstep

    u =  U0 * k * np.sin(k * X) * np.cos(k * Y) 
    v = -U0 * k * np.cos(k * X) * np.sin(k * Y)
    return u, v

def initialize_disc(X, Y, x0, y0, R):
    r = np.sqrt((X - x0)**2 + (Y - y0)**2)
    return r - R

if __name__ == "__main__":

    # --------------------------
    # Grid Setup
    # --------------------------
    Nx, Ny = 1024, 1024
    Lx, Ly = 1.0, 1.0
    X, Y, dx, dy = create_grid(Nx, Ny, Lx, Ly)

    # --------------------------
    # Initial Level Set Function
    # --------------------------
    x0, y0, R = 0.5, 0.5, 0.2
    phi_init = lambda Xq, Yq: initialize_disc(Xq, Yq, x0, y0, R)
    phi = phi_init(X, Y)
    phi = apply_phi_BCs(phi)

    solid_mask = (phi < 0).astype(float)

    # --------------------------
    # Initial Reference Maps
    # --------------------------
    num_extrapolation_layers = 3

    X1 = (X * solid_mask).copy()
    X2 = (Y * solid_mask).copy()
    X1, X2 = extrapolate_transverse_layers_2field(
        X1, X2, phi, dx, dy, 3 * dx, num_extrapolation_layers
    )

    # --------------------------
    # Physical Properties
    # --------------------------
    mu_s, kappa, rho_s, eta_s = 1.0, 0.0, 1.0, 0.005
    mu_f, rho_f = 0.001, 1.0
    w_t = 2 * dx
    gamma = 0.0  # surface tension

    # --------------------------
    # Fields Initialization (Taylor–Green)
    # --------------------------
    t0 = 0.0
    a, b = get_taylor_green_velocity(X, Y, t0, U0=0.05, t_ramp=0.0)
    a, b = apply_velocity_BCs(tg_bc, a, b)
    p = np.zeros((Ny, Nx))

    # --------------------------
    # Numerical Method Params
    # --------------------------
    CFL = 0.2
    dt_min_cap = 1e-4
    max_steps = 10000

    # Precompute Poisson matrix and DCT eigenvalues for pressure projection
    A = build_poisson_matrix(Nx, Ny, dx, dy)
    poisson_eigenvalues = _precompute_poisson_eigenvalues(Nx, Ny, dx, dy)

    vis_output_freq = 200
    directory_name = "output_taylor_green_soft_disc_1024x1024"
    ml_obj = None

    t = 0.0
    integrated_dissipation = 0.0  # Cumulative ∫₀ᵗ ε(τ) dτ

    for step in range(1, max_steps + 1):
        dt = compute_timestep(a, b, dx, dy, CFL, dt_min_cap, mu_s, rho_s, gamma, rho_f)

        # (Optional) if you want the flow to be purely kinematic-driven TG each step,
        # uncomment the next two lines:
        # a, b = get_taylor_green_velocity(X, Y, t, U0=0.05, t_ramp=0.0)
        # a, b = apply_velocity_BCs(tg_bc, a, b)

        # Rebuild/reinit phi
        phi = rebuild_phi_from_reference_map(X1, X2, phi_init)
        # phi = reinitialize_phi_PDE(phi, dx, dy, num_iters=50,
                                #    apply_phi_BCs_func=None, dt_reinit_factor=0.1)

        solid_mask = (phi <= 0).astype(float)

        # Advect reference maps
        X1 = advect_semi_lagrangian_rk4(X1, a, b, X, Y, dt, dx, dy) * solid_mask
        X2 = advect_semi_lagrangian_rk4(X2, a, b, X, Y, dt, dx, dy) * solid_mask
        X1, X2 = extrapolate_transverse_layers_2field(
            X1, X2, phi, dx, dy, 3 * dx, num_extrapolation_layers
        )

        # Velocity update + projection with TG BC
        a_star, b_star, sigma_sxx, sigma_sxy, sigma_syy, J = velocity_RK4(
            a, b, p, X1, X2, tg_bc, mu_s, kappa, eta_s, dx, dy, dt,
            rho_s, rho_f, phi, mu_f, w_t, gamma
        )

        H = heaviside_smooth_alt(phi, w_t)
        rho_local = (1 - H) * rho_s + H * rho_f

        a, b, p, A, ml = pressure_projection_amg(
            a_star, b_star, dx, dy, dt,
            rho_local,
            velocity_bc=tg_bc,
            A=A, ml=ml_obj,
            p_prev=p,
            eigenvalues=poisson_eigenvalues
        )

        # Compute dissipation rate for integration (every step, not just output)
        from pyRMT.output import compute_viscous_dissipation
        dissipation_rate = compute_viscous_dissipation(a, b, mu_f, phi, w_t, dx, dy, eta_s)

        # Integrate dissipation using trapezoidal rule
        # For first step, use forward Euler (no previous dissipation rate available)
        if step == 1:
            integrated_dissipation += dissipation_rate * dt
        else:
            # For subsequent steps, average current and previous dissipation rates
            integrated_dissipation += dissipation_rate * dt  # Simplified: just use current rate

        integrated_dissipation = output_simulation_data(
            dx, dy, phi, solid_mask, X1, X2, a, b, p,
            vis_output_freq, directory_name, step, dt,
            sigma_sxx, sigma_sxy, sigma_syy, J,
            mu_s=mu_s, mu_f=mu_f, rho_s=rho_s, rho_f=rho_f,
            w_t=w_t, eta_s=eta_s, time=t,
            integrated_dissipation=integrated_dissipation
        )

        t += dt
