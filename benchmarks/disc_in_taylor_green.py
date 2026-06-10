"""Soft disc in a Taylor-Green vortex (Jain et al. 2019, Sec. 4.4).

A neo-Hookean disc (R=0.2) is released in an initially imposed Taylor-Green
vortex on a [0,1]^2 box.  The flow stretches the disc; the elastic restoring
stress retracts it, producing a decaying oscillation.  We track kinetic and
strain energy over t in [0, 1].

BC/pressure pairing: the TG streamfunction has zero normal velocity on every
wall, so we use free-slip impermeable walls + Neumann pressure (consistent).

Reference parameters (Jain Sec. 4.4): mu_f=1e-3, rho_s=rho_f=1, paper mu_s=0.5
(== code mu_s=1.0, since the code uses sigma=mu_s*b, i.e. mu_s^code = 2 mu_s^paper).

Usage:
    python benchmarks/disc_in_taylor_green.py [N] [scheme]
        N      : grid points per side (default 128)
        scheme : 'semilagrangian' (default) | 'central2' | 'weno5'
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.functions import (
    create_grid, apply_phi_BCs, extrapolate_transverse_layers_2field,
    compute_timestep, advect_reference_map, rebuild_phi_from_reference_map,
    velocity_RK4, heaviside_smooth_alt, pressure_projection_amg,
    build_poisson_matrix, _precompute_poisson_eigenvalues,
)
from pyRMT.output import (compute_kinetic_energy, compute_strain_energy,
                          compute_viscous_dissipation)
from benchmarks.common import (free_slip_box_bc, initialize_disc,
                               taylor_green_velocity, check_narrow_band,
                               disc_centroid, ensure_dir)


def run(N=128, scheme='semilagrangian', t_end=1.0, out_root="outputs", stress_band=False):
    Lx = Ly = 1.0
    X, Y, dx, dy = create_grid(N, N, Lx, Ly)

    # disc
    x0, y0, R = 0.5, 0.5, 0.2
    phi_init = lambda Xq, Yq: initialize_disc(Xq, Yq, x0, y0, R)
    phi = apply_phi_BCs(phi_init(X, Y))
    solid_mask = (phi <= 0).astype(float)

    # physics (Jain Sec. 4.4)
    mu_s, kappa, rho_s, eta_s = 1.0, 0.0, 1.0, 0.0   # mu_s^code=1.0 == paper 0.5
    mu_f, rho_f = 1.0e-3, 1.0
    w_t = 2.0 * dx
    gamma = 0.0

    num_layers = max(3, check_narrow_band(w_t, dx, 3))  # enforce band >= blend region

    # reference maps
    X1 = X * solid_mask
    X2 = Y * solid_mask
    X1, X2 = extrapolate_transverse_layers_2field(X1, X2, phi, dx, dy, num_layers)

    # initial Taylor-Green field
    a, b = taylor_green_velocity(X, Y, U0=0.05)
    a, b = free_slip_box_bc(a, b)
    p = np.zeros((N, N))

    CFL, dt_min_cap = 0.2, 1e-4
    A = build_poisson_matrix(N, N, dx, dy)
    eig = _precompute_poisson_eigenvalues(N, N, dx, dy)
    ml = None

    out_dir = ensure_dir(os.path.join(out_root, f"disc_tg_N{N}_{scheme}"))
    hist = []
    print(f"[disc-in-TG] N={N}  scheme={scheme}  mu_s={mu_s}  mu_f={mu_f}  layers={num_layers}")

    t = 0.0; step = 0; integ_diss = 0.0
    while t < t_end:
        step += 1
        dt = compute_timestep(a, b, dx, dy, CFL, dt_min_cap, mu_s, rho_s, gamma,
                              rho_f, mu_f=mu_f, eta_s=eta_s, kappa=kappa)
        if t + dt > t_end:
            dt = t_end - t

        # phi/mask from current reference map (compatibility reconstruction)
        phi = rebuild_phi_from_reference_map(X1, X2, phi_init)
        solid_mask = (phi <= 0).astype(float)

        # advect reference map (scheme selectable), reset fluid side, extrapolate
        w_cut = 0.0
        X1 = advect_reference_map(X1, a, b, X, Y, dt, dx, dy, phi, scheme, w_cut) * solid_mask
        X2 = advect_reference_map(X2, a, b, X, Y, dt, dx, dy, phi, scheme, w_cut) * solid_mask
        X1, X2 = extrapolate_transverse_layers_2field(X1, X2, phi, dx, dy, num_layers)

        phi = rebuild_phi_from_reference_map(X1, X2, phi_init)

        a_star, b_star, sxx, sxy, syy, J = velocity_RK4(
            a, b, p, X1, X2, free_slip_box_bc, mu_s, kappa, eta_s, dx, dy, dt,
            rho_s, rho_f, phi, mu_f, w_t, gamma=0.0, stress_band=stress_band)

        H = heaviside_smooth_alt(phi, w_t)
        rho_local = (1 - H) * rho_s + H * rho_f
        a, b, p, A, ml = pressure_projection_amg(
            a_star, b_star, dx, dy, dt, rho_local, velocity_bc=free_slip_box_bc,
            A=A, ml=ml, p_prev=p, eigenvalues=eig, bc_type='neumann')

        ke = compute_kinetic_energy(a, b, rho_f, rho_s, phi, w_t, dx, dy)
        se = compute_strain_energy(X1, X2, phi, mu_s, dx, dy, kappa=kappa)
        diss = compute_viscous_dissipation(a, b, mu_f, phi, w_t, dx, dy, eta_s)
        integ_diss += diss * dt
        cx, cy = disc_centroid(phi, X, Y)
        # disc vertical half-extent (proxy for the stretch oscillation)
        ys = Y[(phi <= 0)]
        ry = 0.5 * (ys.max() - ys.min()) if ys.size else np.nan

        t += dt
        hist.append((t, ke, se, diss, integ_diss, ke + se + integ_diss, ry, J.min()))
        if step % 100 == 0 or t >= t_end:
            print(f"  step {step:5d} t={t:5.3f} KE={ke:.4e} SE={se:.4e} "
                  f"E={ke+se+integ_diss:.4e} ry={ry:.3f} min(J)={J.min():.3f}")

    hist = np.array(hist)
    np.savetxt(os.path.join(out_dir, "energy_history.csv"), hist, delimiter=",",
               header="t,ke,se,dissipation,integrated_dissipation,total_energy,radius_y,minJ",
               comments="")
    E0, E1 = hist[0, 5], hist[-1, 5]
    print(f"[disc-in-TG] total energy drift: {(E1-E0)/max(abs(E0),1e-30)*100:.2f}% over t=[0,{t_end}]")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(hist[:, 0], hist[:, 1], label="KE")
        ax[0].plot(hist[:, 0], hist[:, 2], label="SE")
        ax[0].plot(hist[:, 0], hist[:, 5], "k--", label="KE+SE+∫ε")
        ax[0].set_xlabel("t"); ax[0].set_ylabel("energy"); ax[0].legend()
        ax[0].set_title(f"disc in TG (N={N}, {scheme})")
        ax[1].plot(hist[:, 0], hist[:, 6]); ax[1].set_xlabel("t")
        ax[1].set_ylabel("disc half-height r_y"); ax[1].set_title("stretch oscillation")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "energy_and_oscillation.png"), dpi=130)
        print(f"  saved {out_dir}/energy_and_oscillation.png")
    except Exception as e:
        print(f"  (plot skipped: {e})")
    return hist


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    scheme = sys.argv[2] if len(sys.argv) > 2 else 'semilagrangian'
    run(N=N, scheme=scheme)
