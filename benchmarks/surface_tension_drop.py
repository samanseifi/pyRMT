"""Surface tension validation — Laplace's law for a static drop (CSF model).

A circular interface of radius R with surface tension gamma and no other forces
must develop a pressure jump across the interface of

    Delta p = gamma * kappa = gamma / R      (2D),

and remain (nearly) static — the residual "parasitic"/spurious capillary currents
are the standard quality metric of a continuum-surface-force (CSF) scheme
(f = -gamma * kappa * grad(H)).

This is a STATIC test: the interface is held fixed (analytic level set) and only
the velocity/pressure evolve, so it isolates the surface-tension force + projection
from the reference-map tracking. (Tracking a near-fluid drop through the RMT
reference map is a separate, harder problem — parasitic currents feed back into the
advected interface; see notes in benchmarks/README.md.)

Usage:
    python benchmarks/surface_tension_drop.py [N] [gamma] [R]
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.functions import (
    create_grid, momentum_step_rk4, smoothed_heaviside, pressure_projection_amg,
    build_poisson_matrix, _precompute_poisson_eigenvalues, compute_curvature,
)
from benchmarks.common import free_slip_box_bc, initialize_disc, ensure_dir


def run(N=128, gamma=0.1, R=0.25, n_steps=2000, out_root="outputs"):
    Lx = Ly = 1.0
    X, Y, dx, dy = create_grid(N, N, Lx, Ly)
    phi = initialize_disc(X, Y, 0.5, 0.5, R)        # fixed analytic circle (static)
    X1, X2 = X.copy(), Y.copy()                     # mu_s=0 -> no elastic stress
    mu_s = kappa = rho_s = eta_s = 0.0
    rho_s = 1.0
    mu_f, rho_f = 0.01, 1.0
    w_t = 2.0 * dx
    target = gamma / R

    a = np.zeros((N, N)); b = np.zeros((N, N)); p = np.zeros((N, N))
    A = build_poisson_matrix(N, N, dx, dy)
    eig = _precompute_poisson_eigenvalues(N, N, dx, dy)
    ml = None

    # capillary-limited, fixed time step
    dt = 0.5 * np.sqrt(rho_f * dx ** 3 / (2.0 * np.pi * gamma))

    kap = compute_curvature(phi, dx, dy)
    band = np.abs(phi) < w_t
    out_dir = ensure_dir(os.path.join(out_root, f"surface_tension_drop_N{N}"))
    print(f"[ST-drop] N={N} gamma={gamma} R={R}  Laplace gamma/R={target:.5f}  "
          f"curvature(band) mean={kap[band].mean():.3f} (1/R={1/R:.3f})  dt={dt:.2e}")

    hist = []
    for step in range(1, n_steps + 1):
        a_star, b_star, *_ = momentum_step_rk4(
            a, b, p, X1, X2, free_slip_box_bc, mu_s, kappa, eta_s, dx, dy, dt,
            rho_s, rho_f, phi, mu_f, w_t, gamma=gamma)
        H = smoothed_heaviside(phi, w_t)
        rho_local = (1 - H) * rho_s + H * rho_f
        a, b, p, A, ml = pressure_projection_amg(
            a_star, b_star, dx, dy, dt, rho_local, velocity_bc=free_slip_box_bc,
            A=A, ml=ml, p_prev=p, eigenvalues=eig, bc_type='neumann')

        inside = phi < -2.0 * w_t
        outside = phi > 2.0 * w_t
        dp = p[inside].mean() - p[outside].mean()
        umax = np.max(np.hypot(a, b))
        hist.append((step * dt, dp, umax))
        if step % 400 == 0 or step == 1:
            print(f"  step {step:5d}  dp_in-out={dp:.5f} (target {target:.5f})  "
                  f"max spurious |u|={umax:.3e}")

    hist = np.array(hist)
    np.savetxt(os.path.join(out_dir, "laplace_history.csv"), hist, delimiter=",",
               header="t,delta_p,max_u", comments="")
    dp_final = np.mean(hist[-50:, 1])
    err = abs(dp_final - target) / target
    print(f"[ST-drop] Delta_p={dp_final:.5f} | gamma/R={target:.5f} | rel.err={err*100:.2f}% "
          f"| max spurious |u|={hist[-1,2]:.2e} (Ca={hist[-1,2]*mu_f/gamma:.1e})")
    return dp_final, target, err


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    gamma = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    R = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    run(N=N, gamma=gamma, R=R)
