from pyRMT.functions import *

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
    mu_s, kappa, rho_s, eta_s = 0.1, 0.0, 1.0, 0.01
    mu_f, rho_f = 0.01, 1.0
    w_t = 4 * dx
    
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

    for step in range(1, max_steps + 1):
        dt = compute_timestep(a, b, dx, dy, CFL, dt_min_cap, mu_s, rho_s)
        dt *= 0.1

        phi = rebuild_phi_from_reference_map(X1, X2, phi_init)
        phi = reinitialize_phi_PDE(phi, dx, dy, num_iters=200, apply_phi_BCs_func=None, dt_reinit_factor=0.1)

        solid_mask = (phi <= 0).astype(float)

        X1 = advect_semi_lagrangian_rk4(X1, a, b, X, Y, dt, dx, dy) * solid_mask
        X2 = advect_semi_lagrangian_rk4(X2, a, b, X, Y, dt, dx, dy) * solid_mask
        X1, X2 = extrapolate_transverse_layers_2field(X1, X2, phi, dx, dy, 3 * dx, num_extrapolation_layers)

        a_star, b_star = velocity_RK4(a, b, p, X1, X2, lid_bc, mu_s, kappa, eta_s, dx, dy, dt, rho_s, rho_f, phi, mu_f, w_t)

        sigma_sxx, sigma_sxy, sigma_syy, J = compute_solid_stress(X1, X2, dx, dy, mu_s, kappa, phi, a, b, eta_s)

        H = heaviside_smooth_alt(phi, w_t)
        rho_local = (1 - H) * rho_s + H * rho_f

        a, b, p, A, ml = pressure_projection_amg(a_star, b_star, dx, dy, dt, rho_local, lid_bc, A=A, ml=None, p_prev=p)

        if step % vis_output_freq == 0 or step == 1:
            vmag = np.sqrt(a**2 + b**2)
            div = divergence_2d(a, b, dx, dy)
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

            with h5py.File(f"frames_128x128_no_gradP/data_{step:06d}.h5", "w") as f:
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

