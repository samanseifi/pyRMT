"""Lid-driven cavity on the staggered (MAC) solver — validation vs Ghia (1982).

Pure-fluid incompressible Navier-Stokes on the MAC grid (exact projection).
Centerline u(y) at x=0.5 is compared to data/plot_u_y_Ghia{Re}.csv.

Usage:
    python benchmarks/mac_lid_driven.py [Re] [N]
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.mac import (mac_grid, momentum_predictor, project,
                       poisson_eigs_neumann, divergence)


def run(Re=100.0, N=128, max_steps=120000, steady_tol=2e-6, out_root="outputs"):
    Lx = Ly = 1.0
    U_lid = 1.0
    dx, dy = mac_grid(N, N, Lx, Ly)
    nu = U_lid * Lx / Re
    rho = 1.0

    u = np.zeros((N, N + 1))      # x-faces
    v = np.zeros((N + 1, N))      # y-faces
    eig = poisson_eigs_neumann(N, N, dx, dy)

    # explicit stability: advection (CFL) + diffusion
    dt = min(0.35 * dx / U_lid, 0.20 * dx * dx / nu)
    print(f"[MAC lid] Re={Re:.0f} N={N} nu={nu:.3e} dt={dt:.2e}")

    t = 0.0
    for step in range(1, max_steps + 1):
        u_prev = u
        ustar, vstar = momentum_predictor(u, v, nu, dx, dy, dt, U_lid)
        u, v, p = project(ustar, vstar, dx, dy, dt, rho, eig)
        t += dt
        if step % 500 == 0 or step == 1:
            res = np.max(np.abs(u - u_prev)) / dt
            dmax = np.max(np.abs(divergence(u, v, dx, dy)))
            print(f"  step {step:6d} t={t:7.3f} resid={res:.2e} max|div|={dmax:.1e} "
                  f"max|u|={np.max(np.abs(u)):.3f}")
            if step > 1 and res < steady_tol:
                print(f"  -> steady at step {step}"); break
            if not np.isfinite(res):
                print("  -> diverged"); break

    # centerline u(y) at x=0.5 (an x-face when N is even)
    i_mid = N // 2
    yc = (np.arange(N) + 0.5) * dy
    u_line = u[:, i_mid]
    out_dir = os.path.join(out_root, f"mac_lid_Re{int(Re)}")
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, "centerline_u_vs_y.csv"),
               np.column_stack([yc, u_line]), delimiter=",", header="y,u", comments="")

    ghia = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", f"plot_u_y_Ghia{int(Re)}.csv")
    if os.path.isfile(ghia):
        g = np.loadtxt(ghia, delimiter=",", skiprows=1)
        ui = np.interp(g[:, 0], yc, u_line)
        rms = np.sqrt(np.mean((ui - g[:, 1])**2))
        print(f"[MAC lid] Re={Re:.0f} RMS vs Ghia = {rms:.4e}")
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(5, 5))
            plt.plot(u_line, yc, "-", label="MAC")
            plt.plot(g[:, 1], g[:, 0], "o", ms=4, label="Ghia (1982)")
            plt.xlabel("u at x=0.5"); plt.ylabel("y")
            plt.title(f"MAC lid-driven, Re={int(Re)} (N={N})"); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"ghia_compare_Re{int(Re)}.png"), dpi=130)
        except Exception as e:
            print(f"  (plot skipped: {e})")
        return rms
    return None


if __name__ == "__main__":
    Re = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    run(Re=Re, N=N)
