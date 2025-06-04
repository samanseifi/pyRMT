from pyRMT.functions import *

import h5py
import argparse
import json

# profiling tools
import cProfile
import pstats

def load_config_from_cli():
    parser = argparse.ArgumentParser(description="Run solid-fluid simulation with config.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    return config

if __name__ == "__main__":

    config = load_config_from_cli()

    profiler = cProfile.Profile()
    profiler.enable()

    # Grid
    Nx, Ny = config["Nx"], config["Ny"]
    Lx, Ly = config["Lx"], config["Ly"]
    X, Y, dx, dy = create_grid(Nx, Ny, Lx, Ly)

    # Initial phi
    if config["level_set_file"]:
        with h5py.File(config["level_set_file"], "r") as f:
            phi = f["phi"][:]
    else:
        shape = config["init_shape"]
        phi = initialize_level_set(X, Y, x0=shape["x0"], y0=shape["y0"], R=shape["R"])

    phi = apply_phi_BCs(phi)
    phi0 = phi.copy()
    solid_mask = (phi <= 0).astype(float)

    X1, X2 = X * solid_mask, Y * solid_mask
    X1, X2 = extrapolate_transverse_layers_2field(
        X1, X2, phi, dx, dy, 3 * dx, config["extrapolation_layer"]
    )

    # Material parameters
    mu_s, kappa, rho_s = config["mu_s"], config["kappa"], config["rho_s"]
    eta_s, mu_f, rho_f = config["eta_s"], config["mu_f"], config["rho_f"]
    CFL, dt_min_cap, max_steps = config["CFL"], config["dt_min_cap"], config["max_steps"]
    reinit_factor, reinit_step = config["reinit_factor"], config["reinit_step"]
    w_t = config["w_t"]
    vis_output_freq = config["vis_output_freq"]

    # Fields
    a = np.zeros((Nx, Ny))
    b = np.zeros((Nx, Ny))
    p = np.zeros((Nx, Ny))
    A = build_poisson_matrix(Nx, Ny, dx, dy)

    for step in range(1, max_steps + 1):
        dt = compute_timestep(a, b, dx, dy, CFL, dt_min_cap, mu_s, rho_s)
        dt *= 0.1

        phi = rebuild_phi_from_reference_map(X1, X2, X, Y, x0=shape["x0"], y0=shape["y0"], R=shape["R"])
        if step % reinit_step == 0:
            phi = reinitialize_phi_PDE(phi, dx, dy, num_iters=200, apply_phi_BCs_func=None, dt_reinit_factor=reinit_factor)

        solid_mask = (phi <= 0).astype(float)

        X1 = advect_semi_lagrangian_rk4(X1, a, b, X, Y, dt, dx, dy) * solid_mask
        X2 = advect_semi_lagrangian_rk4(X2, a, b, X, Y, dt, dx, dy) * solid_mask
        X1, X2 = extrapolate_transverse_layers_2field(X1, X2, phi, dx, dy, 3 * dx, config["extrapolation_layer"])

        a_star, b_star = velocity_RK4(a, b, p, X1, X2, mu_s, kappa, eta_s, dx, dy, dt, rho_s, rho_f, phi, mu_f, w_t)
        a_star, b_star = lid_bc(a_star, b_star)

        sigma_sxx, sigma_sxy, sigma_syy, J = compute_solid_stress(X1, X2, dx, dy, mu_s, kappa, phi, a, b, eta_s)

        H = heaviside_smooth_alt(phi, w_t)
        rho_local = (1 - H) * rho_s + H * rho_f

        a, b, p, A, ml = pressure_projection_amg(a_star, b_star, dx, dy, dt, rho_local, A=A, ml=None, p_prev=p)
        a, b = lid_bc(a, b)

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

            with h5py.File(f"frames/data_{step:06d}.h5", "w") as f:
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
    stats.print_stats(30)

