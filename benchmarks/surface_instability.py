import numpy as np
import os
from pyRMT.functions import *
from pyRMT.output import output_simulation_data_lite

# --------------------------
# Boundary Conditions
# --------------------------
def slab_piston_bc(u, v):
    """
    Bottom: Fixed (No-slip)
    Sides: Rollers (Free-slip, No-penetration)
    Top: Moving Lid (or change to 0 for static top)
    """
    u_bc = u.copy()
    v_bc = v.copy()

    # Bottom wall (Fixed)
    u_bc[0, :] = 0.0
    v_bc[0, :] = 0.0

    # Left & Right walls (Rollers/Free-slip)
    u_bc[:, 0] = 0.0   # No flow through left wall
    u_bc[:, -1] = 0.0  # No flow through right wall
    # v is left as is (allows sliding up/down)

    # Top lid (Moving lid at U=1)
    u_bc[-1, :] = 0.0
    v_bc[-1, :] = 0.0

    return u_bc, v_bc

def initialize_slab(Y, h):
    """ 
    Level set for a horizontal interface at height h.
    y < h is solid (phi < 0), y > h is fluid (phi > 0).
    """
    return Y - h

if __name__ == "__main__":
    # --------------------------
    # Grid Setup
    # --------------------------
    Nx, Ny = 128, 128
    Lx, Ly = 1.0, 1.0
    X, Y, dx, dy = create_grid(Nx, Ny, Lx, Ly)

    # --------------------------
    # Initial Level Set & Slab
    # --------------------------
    h_slab = 0.3  # Thickness of the slab
    phi = initialize_slab(Y, h_slab)
    phi = apply_phi_BCs(phi)
    
    # phi_init is used for rebuilding phi from X later
    phi_init_func = lambda x_map, y_map: y_map - h_slab

    # --------------------------
    # Initial Reference Maps
    # --------------------------
    # X1 and X2 map current positions back to original positions
    X1 = X.copy()
    X2 = Y.copy()

    # Mask out fluid region so we only extrapolate from the solid
    solid_mask = (phi <= 0).astype(float)
    X1 *= solid_mask
    X2 *= solid_mask

    # Extrapolate 3 layers into the fluid for numerical stability at the interface
    num_extrapolation_layers = 3
    X1, X2 = extrapolate_transverse_layers_2field(X1, X2, phi, dx, dy, num_extrapolation_layers)

    # --------------------------
    # Physical Properties
    # --------------------------
    mu_s, kappa, rho_s, eta_s = 0.5, 10.0, 1.0, 0.05 # Higher kappa for slab stiffness
    mu_f, rho_f = 0.01, 1.0
    w_t = 2 * dx
    gamma = 0.0  # Surface tension
    
    # --------------------------
    # Fields Initialization
    # --------------------------
    a = np.zeros((Ny, Nx)) # u-velocity
    b = np.zeros((Ny, Nx)) # v-velocity
    p = np.zeros((Ny, Nx)) # pressure

    CFL = 0.2
    dt_min_cap = 1e-3
    max_steps = 5000
    vis_output_freq = 50
    directory_name = "output_solid_slab_bottom"

    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    # Precompute Poisson matrix
    A = build_poisson_matrix(Nx, Ny, dx, dy)
    ml_obj = None

    # --------------------------
    # Main Loop
    # --------------------------
    for step in range(1, max_steps + 1):
        # 1. Determine Timestep
        dt = compute_timestep(a, b, dx, dy, CFL, dt_min_cap, mu_s, rho_s, gamma, rho_f)

        # 2. Update Level Set from Reference Map
        # In RMT, the level set is tied to the reference map deformation
        phi = rebuild_phi_from_reference_map(X1, X2, phi_init_func)
        phi = reinitialize_phi_PDE(phi, dx, dy, num_iters=20, apply_phi_BCs_func=None, dt_reinit_factor=0.1)

        # 3. Advect Reference Maps (The "Solid" Memory)
        # We only advect the solid part, then re-extrapolate
        X1 = advect_semi_lagrangian_rk4(X1, a, b, X, Y, dt, dx, dy)
        X2 = advect_semi_lagrangian_rk4(X2, a, b, X, Y, dt, dx, dy)
        
        # Keep solid identity and clean up fluid region
        curr_solid_mask = (phi <= 0).astype(float)
        X1 *= curr_solid_mask
        X2 *= curr_solid_mask
        X1, X2 = extrapolate_transverse_layers_2field(X1, X2, phi, dx, dy, num_extrapolation_layers)

        # 4. Velocity Prediction (RK4)
        a_star, b_star = velocity_RK4(a, b, p, X1, X2, slab_piston_bc, mu_s, kappa, eta_s, dx, dy, dt, rho_s, rho_f, phi, mu_f, w_t, gamma)

        # 5. Pressure Projection (Incompressibility)
        H = heaviside_smooth_alt(phi, w_t)
        rho_local = (1 - H) * rho_s + H * rho_f
        
        a, b, p, A, ml_obj = pressure_projection_amg(
            a_star, b_star, dx, dy, dt,
            rho_local,
            velocity_bc=slab_piston_bc,
            A=A, ml=ml_obj,
            p_prev=p
        )

        # 6. Diagnostics and Output
        if step % vis_output_freq == 0 or step == 1:
            # Compute stress for visualization
            sxx, sxy, syy, J = compute_solid_stress(X1, X2, dx, dy, mu_s, kappa, phi, a, b, p, eta_s)
            
            print(f"Step: {step} | dt: {dt:.5f} | Max V: {np.max(np.sqrt(a**2 + b**2)):.4f}")
            output_simulation_data_lite(
                dx, dy, phi, curr_solid_mask, X1, X2, a, b, p, 
                vis_output_freq, directory_name, step, dt, sxx, sxy, syy, J
            )