"""Two soft discs colliding in a Taylor-Green vortex (Jain 2019 Sec. 4.6 regime).

Two neo-Hookean discs are placed above/below the domain centre and an imposed
Taylor-Green vortex drives them toward y=0.5; the repulsive contact force prevents
inter-penetration and they rebound. This uses the *stable* regime of the paper:
EQUAL density (rho_s=rho_f -> constant-density DCT projection, robust) and
sustained vortex forcing rather than a violent head-on initial condition.

Each disc carries its own reference map; the two solid stresses + fluid stress are
combined with the n=2 one-fluid mixture (Jain Eq. 29) in `momentum_step_rk4_2solids`.

Diagnostic: the vertical centre gap decreases (approach), reaches a positive
minimum (contact, no pass-through), then increases (rebound). A divergence is
reported gracefully (no crash) via the advect guard.

Usage:
    python benchmarks/two_disc_tg_collision.py [N] [t_end] [U0] [k_rep]
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.functions import (
    create_grid, apply_phi_BCs, extrapolate_reference_map, compute_timestep,
    advect_reference_map, rebuild_phi_from_reference_map, momentum_step_rk4_2solids,
    smoothed_heaviside, pressure_projection_amg, build_poisson_matrix,
    _precompute_poisson_eigenvalues,
)
from benchmarks.common import (free_slip_box_bc, initialize_disc,
                               taylor_green_velocity, check_narrow_band,
                               disc_centroid, ensure_dir)


def run(N=128, t_end=2.0, U0=0.12, k_rep=3.0, out_root="outputs"):
    Lx = Ly = 1.0
    X, Y, dx, dy = create_grid(N, N, Lx, Ly)
    R = 0.12
    xc = 0.5
    ya0, yb0 = 0.35, 0.65                    # lower / upper disc centres
    pia = lambda Xq, Yq: initialize_disc(Xq, Yq, xc, ya0, R)
    pib = lambda Xq, Yq: initialize_disc(Xq, Yq, xc, yb0, R)
    phi_a = apply_phi_BCs(pia(X, Y))
    phi_b = apply_phi_BCs(pib(X, Y))

    mu_s, kappa, eta_s = 0.5, 0.0, 0.0
    rho_s = rho_f = 1.0                       # EQUAL density -> constant-density DCT
    mu_f = 0.02
    w_t = 2.0 * dx
    w_c = 2.0 * dx
    nl = max(3, check_narrow_band(w_t, dx, 3))

    ma = (phi_a <= 0).astype(float); mb = (phi_b <= 0).astype(float)
    X1a, X2a = extrapolate_reference_map(X * ma, Y * ma, phi_a, dx, dy, nl)
    X1b, X2b = extrapolate_reference_map(X * mb, Y * mb, phi_b, dx, dy, nl)

    # imposed Taylor-Green vortex: u=U0 k sin(kx)cos(ky), v=-U0 k cos(kx)sin(ky).
    # at x=0.5 the lower disc (y<0.5) moves up and the upper disc moves down.
    a, b = taylor_green_velocity(X, Y, U0=U0)
    a, b = free_slip_box_bc(a, b)
    p = np.zeros((N, N))

    CFL, dt_min_cap = 0.2, 1e-3
    A = build_poisson_matrix(N, N, dx, dy)
    eig = _precompute_poisson_eigenvalues(N, N, dx, dy)
    ml = None

    out_dir = ensure_dir(os.path.join(out_root, f"two_disc_tg_N{N}"))
    print(f"[tg-contact] N={N} R={R} U0={U0} k_rep={k_rep} mu_s={mu_s} rho=eq t_end={t_end}")

    t = 0.0; step = 0; hist = []
    diverged = False
    while t < t_end:
        step += 1
        dt = compute_timestep(a, b, dx, dy, CFL, dt_min_cap, mu_s, rho_s, 0.0,
                              rho_f, mu_f=mu_f, kappa=kappa)
        if t + dt > t_end:
            dt = t_end - t
        try:
            phi_a = rebuild_phi_from_reference_map(X1a, X2a, pia)
            phi_b = rebuild_phi_from_reference_map(X1b, X2b, pib)
            ma = (phi_a <= 0).astype(float); mb = (phi_b <= 0).astype(float)
            X1a = advect_reference_map(X1a, a, b, X, Y, dt, dx, dy, phi_a, 'semilagrangian', 0.0) * ma
            X2a = advect_reference_map(X2a, a, b, X, Y, dt, dx, dy, phi_a, 'semilagrangian', 0.0) * ma
            X1b = advect_reference_map(X1b, a, b, X, Y, dt, dx, dy, phi_b, 'semilagrangian', 0.0) * mb
            X2b = advect_reference_map(X2b, a, b, X, Y, dt, dx, dy, phi_b, 'semilagrangian', 0.0) * mb
            X1a, X2a = extrapolate_reference_map(X1a, X2a, phi_a, dx, dy, nl)
            X1b, X2b = extrapolate_reference_map(X1b, X2b, phi_b, dx, dy, nl)
            phi_a = rebuild_phi_from_reference_map(X1a, X2a, pia)
            phi_b = rebuild_phi_from_reference_map(X1b, X2b, pib)

            a_star, b_star, Jmin = momentum_step_rk4_2solids(
                a, b, p, X1a, X2a, X1b, X2b, free_slip_box_bc, mu_s, kappa, eta_s,
                dx, dy, dt, rho_s, rho_f, phi_a, phi_b, mu_f, w_t, k_rep=k_rep, w_c=w_c)
            rho_local = rho_f * np.ones_like(a)        # equal density
            a, b, p, A, ml = pressure_projection_amg(
                a_star, b_star, dx, dy, dt, rho_local, velocity_bc=free_slip_box_bc,
                A=A, ml=ml, p_prev=p, eigenvalues=eig, bc_type='neumann')
        except FloatingPointError as e:
            print(f"  [diverged at step {step}, t={t:.3f}]: {e}")
            diverged = True
            break

        # graceful divergence check: a huge-but-finite velocity is a blow-up too
        umax = np.max(np.hypot(a, b))
        if not np.isfinite(umax) or umax > 1.0e3:
            print(f"  [diverged at step {step}, t={t:.3f}]: max|u|={umax:.2e}")
            diverged = True
            break

        cxa, cya = disc_centroid(phi_a, X, Y)
        cxb, cyb = disc_centroid(phi_b, X, Y)
        gap = cyb - cya
        t += dt
        hist.append((t, cya, cyb, gap, Jmin.min()))
        if step % 50 == 0 or t >= t_end:
            print(f"  step {step:5d} t={t:5.3f}  cya={cya:.3f} cyb={cyb:.3f} "
                  f"gap={gap:.3f}  minJ={Jmin.min():.3f}  max|u|={np.max(np.hypot(a,b)):.3f}")

    hist = np.array(hist)
    if len(hist):
        np.savetxt(os.path.join(out_dir, "centroids.csv"), hist, delimiter=",",
                   header="t,cya,cyb,gap,minJ", comments="")
        gmin = hist[:, 3].min(); imin = hist[:, 3].argmin()
        rebound = (imin < len(hist) - 1) and (hist[-1, 3] > gmin + 5e-3)
        print(f"[tg-contact] {'DIVERGED (graceful)' if diverged else 'completed'}; "
              f"min vertical gap = {gmin:.3f} (2R={2*R:.3f}); no pass-through: {gmin > 0}; "
              f"rebound: {rebound}")
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 4))
            plt.plot(hist[:, 0], hist[:, 1], label="lower disc cy")
            plt.plot(hist[:, 0], hist[:, 2], label="upper disc cy")
            plt.plot(hist[:, 0], hist[:, 3], "k--", label="vertical gap")
            plt.axhline(2 * R, color="r", ls=":", label="2R (touching)")
            plt.xlabel("t"); plt.ylabel("y"); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "tg_contact_centroids.png"), dpi=130)
            print(f"  saved {out_dir}/tg_contact_centroids.png")
        except Exception as e:
            print(f"  (plot skipped: {e})")
    return hist


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    t_end = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    U0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.12
    k_rep = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
    run(N=N, t_end=t_end, U0=U0, k_rep=k_rep)
