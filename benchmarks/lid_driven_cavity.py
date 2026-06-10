"""Lid-driven cavity — pure fluid solver validation against Ghia et al. (1982).

Runs the incompressible Navier-Stokes solver (no solid) to steady state and
compares the centerline velocity profiles with the Ghia benchmark data in
`data/plot_u_y_Ghia{100,1000}.csv`.

Usage:
    python benchmarks/lid_driven_cavity.py [Re] [N]
        Re : Reynolds number (default 100; data available for 100 and 1000)
        N  : grid points per side (default 129, matches Ghia)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.functions import (
    create_grid, build_poisson_matrix, _precompute_poisson_eigenvalues,
    compute_timestep, momentum_step_rk4, pressure_projection_amg,
)
from benchmarks.common import no_slip_lid_bc, extract_centerlines, ensure_dir


def run(Re=100.0, N=129, max_steps=60000, steady_tol=2e-5, out_root="outputs"):
    Lx = Ly = 1.0
    U_lid = 1.0
    X, Y, dx, dy = create_grid(N, N, Lx, Ly)

    # Pure fluid: no solid fields, density uniform.
    mu_f = rho_f = 1.0
    mu_f = rho_f * U_lid * Lx / Re          # nu = U L / Re  (rho=1)
    rho_f = 1.0
    mu_s = kappa = rho_s = eta_s = 0.0
    w_t = 2.0 * dx
    gamma = 0.0

    # Placeholder solid fields (no solid present).
    phi = np.ones((N, N))                    # phi > 0 everywhere -> all fluid
    solid_mask = np.zeros((N, N), dtype=bool)
    X1, X2 = X.copy(), Y.copy()

    a = np.zeros((N, N)); b = np.zeros((N, N)); p = np.zeros((N, N))
    a, b = no_slip_lid_bc(a, b, U_lid)

    CFL = 0.2
    dt_min_cap = 1e-2

    A = build_poisson_matrix(N, N, dx, dy)
    eig = _precompute_poisson_eigenvalues(N, N, dx, dy)
    ml = None

    bc = lambda u, v: no_slip_lid_bc(u, v, U_lid)

    print(f"[lid-driven] Re={Re:.0f}  N={N}  mu_f={mu_f:.3e}")
    t = 0.0
    for step in range(1, max_steps + 1):
        dt = compute_timestep(a, b, dx, dy, CFL, dt_min_cap, mu_s, rho_s, gamma,
                              rho_f, mu_f=mu_f)
        a_prev = a

        a_star, b_star, *_ = momentum_step_rk4(
            a, b, p, X1, X2, bc, mu_s, kappa, eta_s, dx, dy, dt,
            rho_s, rho_f, phi, mu_f, w_t, gamma)

        a, b, p, A, ml = pressure_projection_amg(
            a_star, b_star, dx, dy, dt, rho_f, velocity_bc=bc,
            A=A, ml=ml, p_prev=p, eigenvalues=eig, bc_type='neumann')

        # steady-state monitor
        if step % 200 == 0 or step == 1:
            res = np.max(np.abs(a - a_prev)) / dt
            t_now = t + dt
            print(f"  step {step:6d}  t={t_now:7.3f}  dt={dt:.2e}  "
                  f"max|v|={np.max(np.hypot(a,b)):.4f}  resid={res:.2e}")
            if step > 1 and res < steady_tol:
                print(f"  -> steady state reached at step {step}")
                break
        t += dt

    # ── compare with Ghia ────────────────────────────────────────────────────
    y, u_line, x, v_line = extract_centerlines(a, b, X, Y)
    out_dir = ensure_dir(os.path.join(out_root, f"lid_driven_Re{int(Re)}"))
    np.savetxt(os.path.join(out_dir, "centerline_u_vs_y.csv"),
               np.column_stack([y, u_line]), delimiter=",", header="y,u", comments="")
    np.savetxt(os.path.join(out_dir, "centerline_v_vs_x.csv"),
               np.column_stack([x, v_line]), delimiter=",", header="x,v", comments="")

    ghia_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", f"plot_u_y_Ghia{int(Re)}.csv")
    err = None
    if os.path.isfile(ghia_path):
        gd = np.loadtxt(ghia_path, delimiter=",", skiprows=1)
        yg, ug = gd[:, 0], gd[:, 1]
        u_interp = np.interp(yg, y, u_line)
        err = np.sqrt(np.mean((u_interp - ug) ** 2))
        print(f"[lid-driven] Re={Re:.0f}  RMS error vs Ghia (u at x=0.5): {err:.4e}")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(5, 5))
            plt.plot(u_line, y, "-", label="pyRMT")
            plt.plot(ug, yg, "o", ms=4, label="Ghia et al. (1982)")
            plt.xlabel("u at x=0.5"); plt.ylabel("y")
            plt.title(f"Lid-driven cavity, Re={int(Re)} (N={N})")
            plt.legend(); plt.tight_layout()
            fig_path = os.path.join(out_dir, f"ghia_compare_Re{int(Re)}.png")
            plt.savefig(fig_path, dpi=130)
            print(f"  saved {fig_path}")
        except Exception as e:
            print(f"  (plot skipped: {e})")
    else:
        print(f"  (no Ghia reference at {ghia_path})")
    return err


if __name__ == "__main__":
    Re = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 129
    run(Re=Re, N=N)
