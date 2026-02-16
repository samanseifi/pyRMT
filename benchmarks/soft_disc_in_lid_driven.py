from pyRMT.functions import *
from pyRMT.functions import _precompute_poisson_eigenvalues
from pyRMT.output import output_simulation_data
import os
import h5py

# Define the boundary conditions for the lid-driven cavity problem
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

def initialize_disc(X, Y, x0, y0, R):
    """
    Initialize a disc-shaped level set function phi centered at (x0, y0) with radius R.
    """
    r = np.sqrt((X - x0)**2 + (Y - y0)**2)
    phi = r - R
    return phi

if __name__ == "__main__":

    # --------------------------
    # Grid Setup
    # --------------------------
    Nx, Ny = 128, 128
    Lx, Ly = 1.0, 1.0
    X, Y, dx, dy = create_grid(Nx, Ny, Lx, Ly)

    # --------------------------
    # Initial Level Set Function
    # --------------------------
    x0, y0, R = 0.6, 0.5, 0.2
    phi_init = lambda X, Y: initialize_disc(X, Y, x0, y0, R)
    phi = phi_init(X, Y)
    phi = apply_phi_BCs(phi)

    phi0 = phi.copy()
    solid_mask = (phi < 0).astype(float)

    # --------------------------
    # Initial Reference Maps
    # --------------------------
    num_extrapolation_layers = 3

    X1 = X.copy()
    X2 = Y.copy()
    X1 = X * solid_mask
    X2 = Y * solid_mask

    X1, X2 = extrapolate_transverse_layers_2field(X1, X2, phi, dx, dy, 3 * dx, num_extrapolation_layers)

    # --------------------------
    # Physical Properties
    # --------------------------
    mu_s, kappa, rho_s, eta_s = 0.1, 1.0, 1.0, 0.01
    mu_f, rho_f = 0.01, 1.0
    w_t = 1.5 * dx

    # --------------------------
    # Fields Initialization
    # --------------------------
    a = np.zeros((Nx, Ny))
    b = np.zeros((Nx, Ny))
    p = np.zeros((Nx, Ny))

    # --------------------------
    # Numerical Method Params
    # --------------------------
    CFL = 0.2
    dt_min_cap = 1e-4
    max_steps = 1000000

    gamma = 0.0  # Surface tension coefficient (try 0.01 to 0.1)

    # Precompute Poisson matrix and DCT eigenvalues for pressure projection
    A = build_poisson_matrix(Nx, Ny, dx, dy)
    poisson_eigenvalues = _precompute_poisson_eigenvalues(Nx, Ny, dx, dy)

    vis_output_freq = 1000

    directory_name = "output_lid_driven_soft_disc_5"
    ml_obj = None

    t = 0.0
    integrated_dissipation = 0.0

    for step in range(1, max_steps + 1):
        dt = compute_timestep(a, b, dx, dy, CFL, dt_min_cap, mu_s, rho_s, gamma, rho_f)

        phi = rebuild_phi_from_reference_map(X1, X2, phi_init)
        phi = reinitialize_phi_PDE(phi, dx, dy, num_iters=50, apply_phi_BCs_func=None, dt_reinit_factor=0.1)

        solid_mask = (phi <= 0).astype(float)

        X1 = advect_semi_lagrangian_rk4(X1, a, b, X, Y, dt, dx, dy) * solid_mask
        X2 = advect_semi_lagrangian_rk4(X2, a, b, X, Y, dt, dx, dy) * solid_mask
        X1, X2 = extrapolate_transverse_layers_2field(X1, X2, phi, dx, dy, 3 * dx, num_extrapolation_layers)

        a_star, b_star, sigma_sxx, sigma_sxy, sigma_syy, J = velocity_RK4(
            a, b, p, X1, X2, lid_bc, mu_s, kappa, eta_s, dx, dy, dt, rho_s, rho_f, phi, mu_f, w_t, gamma
        )

        H = heaviside_smooth_alt(phi, w_t)
        rho_local = (1 - H) * rho_s + H * rho_f

        a, b, p, A, ml_obj = pressure_projection_amg(
            a_star, b_star, dx, dy, dt,
            rho_local,
            velocity_bc=lid_bc,
            A=A, ml=ml_obj,
            p_prev=p,
            eigenvalues=poisson_eigenvalues
        )

        from pyRMT.output import compute_viscous_dissipation
        dissipation_rate = compute_viscous_dissipation(a, b, mu_f, phi, w_t, dx, dy, eta_s)
        integrated_dissipation += dissipation_rate * dt

        integrated_dissipation = output_simulation_data(
            dx, dy, phi, solid_mask, X1, X2, a, b, p,
            vis_output_freq, directory_name, step, dt,
            sigma_sxx, sigma_sxy, sigma_syy, J,
            mu_s=mu_s, mu_f=mu_f, rho_s=rho_s, rho_f=rho_f,
            w_t=w_t, eta_s=eta_s, time=t,
            integrated_dissipation=integrated_dissipation
        )

        t += dt
