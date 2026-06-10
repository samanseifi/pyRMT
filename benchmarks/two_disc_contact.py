"""Two soft discs colliding — solid-solid contact (Jain 2019 Sec. 3.6/4.6).

Two neo-Hookean discs are given approaching velocities; the short-range repulsive
contact force prevents inter-penetration and they rebound. Each disc carries its
OWN reference map; the two solid stresses + fluid stress are combined with the
n=2 one-fluid mixture (Jain Eq. 29), and the contact force is added in the
momentum step (`momentum_step_rk4_2solids`).

Diagnostic: the centre-to-centre gap decreases (approach), reaches a positive
minimum (contact, no pass-through), then increases (rebound).

Usage:
    python benchmarks/two_disc_contact.py [N] [t_end] [V0] [k_rep]
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
                               check_narrow_band, disc_centroid, ensure_dir)


def run(N=128, t_end=2.0, V0=0.15, k_rep=2.0, out_root="outputs"):
    Lx = Ly = 1.0
    X, Y, dx, dy = create_grid(N, N, Lx, Ly)
    R = 0.15
    xa0, xb0, yc = 0.30, 0.70, 0.50          # left / right disc centres
    phi_init_a = lambda Xq, Yq: initialize_disc(Xq, Yq, xa0, yc, R)
    phi_init_b = lambda Xq, Yq: initialize_disc(Xq, Yq, xb0, yc, R)

    phi_a = apply_phi_BCs(phi_init_a(X, Y))
    phi_b = apply_phi_BCs(phi_init_b(X, Y))

    mu_s, kappa, rho_s, eta_s = 1.0, 0.0, 1.0, 0.0
    mu_f, rho_f = 0.01, 1.0          # low fluid viscosity -> less drag, clearer rebound
    w_t = 2.0 * dx
    w_c = 3.0 * dx                           # contact influence half-width
    nl = max(3, check_narrow_band(w_t, dx, 3))

    # reference maps (one per disc)
    ma = (phi_a <= 0).astype(float); mb = (phi_b <= 0).astype(float)
    X1a, X2a = extrapolate_reference_map(X * ma, Y * ma, phi_a, dx, dy, nl)
    X1b, X2b = extrapolate_reference_map(X * mb, Y * mb, phi_b, dx, dy, nl)

    # approaching velocity field: left disc -> +x, right disc -> -x
    Ha = smoothed_heaviside(phi_a, w_t); Hb = smoothed_heaviside(phi_b, w_t)
    a = V0 * (1 - Ha) - V0 * (1 - Hb)
    b = np.zeros((N, N))
    a, b = free_slip_box_bc(a, b)
    p = np.zeros((N, N))

    CFL, dt_min_cap = 0.2, 1e-3
    A = build_poisson_matrix(N, N, dx, dy)
    eig = _precompute_poisson_eigenvalues(N, N, dx, dy)
    ml = None

    out_dir = ensure_dir(os.path.join(out_root, f"two_disc_contact_N{N}"))
    print(f"[contact] N={N} R={R} V0={V0} k_rep={k_rep} mu_s={mu_s} t_end={t_end}")

    t = 0.0; step = 0; hist = []
    while t < t_end:
        step += 1
        dt = compute_timestep(a, b, dx, dy, CFL, dt_min_cap, mu_s, rho_s, 0.0,
                              rho_f, mu_f=mu_f, kappa=kappa)
        if t + dt > t_end:
            dt = t_end - t

        # rebuild each level set, advect each reference map, re-extrapolate
        phi_a = rebuild_phi_from_reference_map(X1a, X2a, phi_init_a)
        phi_b = rebuild_phi_from_reference_map(X1b, X2b, phi_init_b)
        ma = (phi_a <= 0).astype(float); mb = (phi_b <= 0).astype(float)
        X1a = advect_reference_map(X1a, a, b, X, Y, dt, dx, dy, phi_a, 'semilagrangian', 0.0) * ma
        X2a = advect_reference_map(X2a, a, b, X, Y, dt, dx, dy, phi_a, 'semilagrangian', 0.0) * ma
        X1b = advect_reference_map(X1b, a, b, X, Y, dt, dx, dy, phi_b, 'semilagrangian', 0.0) * mb
        X2b = advect_reference_map(X2b, a, b, X, Y, dt, dx, dy, phi_b, 'semilagrangian', 0.0) * mb
        X1a, X2a = extrapolate_reference_map(X1a, X2a, phi_a, dx, dy, nl)
        X1b, X2b = extrapolate_reference_map(X1b, X2b, phi_b, dx, dy, nl)
        phi_a = rebuild_phi_from_reference_map(X1a, X2a, phi_init_a)
        phi_b = rebuild_phi_from_reference_map(X1b, X2b, phi_init_b)

        a_star, b_star, Jmin = momentum_step_rk4_2solids(
            a, b, p, X1a, X2a, X1b, X2b, free_slip_box_bc, mu_s, kappa, eta_s,
            dx, dy, dt, rho_s, rho_f, phi_a, phi_b, mu_f, w_t, k_rep=k_rep, w_c=w_c)

        Ha = smoothed_heaviside(phi_a, w_t); Hb = smoothed_heaviside(phi_b, w_t)
        rho_local = (Ha + Hb - 1) * rho_f + (1 - Ha) * rho_s + (1 - Hb) * rho_s
        a, b, p, A, ml = pressure_projection_amg(
            a_star, b_star, dx, dy, dt, rho_local, velocity_bc=free_slip_box_bc,
            A=A, ml=ml, p_prev=p, eigenvalues=eig, bc_type='neumann')

        cxa, cya = disc_centroid(phi_a, X, Y)
        cxb, cyb = disc_centroid(phi_b, X, Y)
        gap = cxb - cxa
        t += dt
        hist.append((t, cxa, cxb, gap, Jmin.min()))
        if step % 50 == 0 or t >= t_end:
            print(f"  step {step:5d} t={t:5.3f}  cxa={cxa:.3f} cxb={cxb:.3f} "
                  f"gap={gap:.3f}  minJ={Jmin.min():.3f}  max|u|={np.max(np.hypot(a,b)):.3f}")

    hist = np.array(hist)
    np.savetxt(os.path.join(out_dir, "centroids.csv"), hist, delimiter=",",
               header="t,cxa,cxb,gap,minJ", comments="")
    gmin = hist[:, 3].min()
    approached = hist[:, 3].argmin() < len(hist) - 1
    rebounded = hist[-1, 3] > gmin + 1e-3
    print(f"[contact] min center gap = {gmin:.3f} (2R={2*R:.3f}); "
          f"{'REBOUND' if (approached and rebounded) else 'no clear rebound'}; "
          f"no pass-through: {gmin > 0}")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(hist[:, 0], hist[:, 1], label="left disc cx")
        plt.plot(hist[:, 0], hist[:, 2], label="right disc cx")
        plt.plot(hist[:, 0], hist[:, 3], "k--", label="center gap")
        plt.axhline(2 * R, color="r", ls=":", label="2R (touching)")
        plt.xlabel("t"); plt.ylabel("x"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "contact_centroids.png"), dpi=130)
        print(f"  saved {out_dir}/contact_centroids.png")
    except Exception as e:
        print(f"  (plot skipped: {e})")
    return hist


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    t_end = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    V0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
    k_rep = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    run(N=N, t_end=t_end, V0=V0, k_rep=k_rep)
