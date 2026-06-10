"""Surface tension on the MAC grid — Laplace's law + parasitic currents.

On a staggered grid the CSF force f = -gamma*kappa*grad(H) lands at faces with the
SAME compact gradient as the pressure -> balanced by construction. This should give
much smaller parasitic currents than the collocated solver (which had max|u|~1.3e-2
at N=64 for the same case).

Usage: python benchmarks/mac_surface_tension_drop.py [N] [gamma] [R]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.mac import (mac_grid, momentum_predictor, project, poisson_eigs_neumann,
                       interfacial_force_faces, divergence)
from pyRMT.functions import smoothed_heaviside, compute_curvature


def run(N=64, gamma=0.1, R=0.25, n_steps=2000, out_root="outputs"):
    dx, dy = mac_grid(N, N)
    xc = (np.arange(N) + 0.5) * dx
    Xc, Yc = np.meshgrid(xc, xc)
    phi = np.sqrt((Xc - 0.5) ** 2 + (Yc - 0.5) ** 2) - R    # fixed (static) circle
    rho = 1.0; mu_f = 0.01; nu = mu_f / rho; w_t = 2.0 * dx
    target = gamma / R

    kappa = compute_curvature(phi, dx, dy)
    H = smoothed_heaviside(phi, w_t)
    fu, fv = interfacial_force_faces(kappa, H, gamma, dx, dy)

    u = np.zeros((N, N + 1)); v = np.zeros((N + 1, N));
    eig = poisson_eigs_neumann(N, N, dx, dy)
    dt = 0.5 * np.sqrt(rho * dx ** 3 / (2 * np.pi * gamma))
    band = np.abs(phi) < w_t
    print(f"[MAC ST-drop] N={N} gamma={gamma} R={R} Laplace gamma/R={target:.5f} "
          f"kappa(band) {kappa[band].mean():.3f} (1/R={1/R:.3f}) dt={dt:.2e}")

    hist = []
    for step in range(1, n_steps + 1):
        ustar, vstar = momentum_predictor(u, v, nu, dx, dy, dt, 0.0, fu=fu, fv=fv, rho=rho)
        u, v, p = project(ustar, vstar, dx, dy, dt, rho, eig)
        ins = phi < -2 * w_t; out = phi > 2 * w_t
        dp = p[ins].mean() - p[out].mean()
        umax = max(np.max(np.abs(u)), np.max(np.abs(v)))
        hist.append((step * dt, dp, umax))
        if step % 400 == 0 or step == 1:
            print(f"  step {step:5d} dp={dp:.5f} (target {target:.5f}) max|u|={umax:.3e}")

    hist = np.array(hist)
    dp_f = hist[-50:, 1].mean(); err = abs(dp_f - target) / target
    print(f"[MAC ST-drop] dp={dp_f:.5f} | gamma/R={target:.5f} | rel.err={err*100:.2f}% "
          f"| max spurious |u|={hist[-1,2]:.2e}  (collocated was ~1.3e-2)")
    return dp_f, target, hist[-1, 2]


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    gamma = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    R = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    run(N=N, gamma=gamma, R=R)
