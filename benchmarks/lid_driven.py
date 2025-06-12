from pyRMT.functions import *
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

if __name__ == "__main__":

    # --------------------------
    # Grid Setup
    # --------------------------
    Nx, Ny = 128, 128
    Lx, Ly = 1.0, 1.0
    X, Y, dx, dy = create_grid(Nx, Ny, Lx, Ly)
    X1 = X.copy()
    X2 = Y.copy()

    # --------------------------
    # Physical Properties
    # --------------------------
    mu_s, kappa, rho_s, eta_s = 0.0, 0.0, 0.0, 0.0
    mu_f, rho_f = 0.01, 1.0
    w_t = 4 * dx
    
    rho_local = 1.0  # Local density for solid
    phi = np.zeros((Nx, Ny))  # Placeholder for solid mask
    solid_mask = np.zeros((Nx, Ny), dtype=bool)  # Solid mask for visualization
    
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
    dt_min_cap = 1e-3
    max_steps = 100000
    
    # Precompute Poisson matrix for pressure projection
    A = build_poisson_matrix(Nx, Ny, dx, dy)
    
    vis_output_freq = 100
    
    directory_name = "output_lid_driven"

    for step in range(1, max_steps + 1):
        dt = compute_timestep(a, b, dx, dy, CFL, dt_min_cap, mu_s, rho_s)
        dt *= 0.1

        a_star, b_star = velocity_RK4(a, b, p, X1, X2, lid_bc, mu_s, kappa, eta_s, dx, dy, dt, rho_s, rho_f, phi, mu_f, w_t)

        sigma_sxx, sigma_sxy, sigma_syy, J = compute_solid_stress(X1, X2, dx, dy, mu_s, kappa, phi, a, b, eta_s)


        a, b, p, A, ml = pressure_projection_amg(a_star, b_star, dx, dy, dt, rho_local, lid_bc, A=A, ml=None, p_prev=p)

        output_simulation_data(dx, dy, phi, solid_mask, X1, X2, a, b, p, vis_output_freq, directory_name, step, dt, sigma_sxx, sigma_sxy, sigma_syy, J)

