"""Soft disc in a lid-driven cavity (Jain et al. 2019, Sec. 4.5; Sugiyama 2011).

THE primary FSI validation case.  A neo-Hookean disc (R=0.2, centred at
(0.6,0.5)) is carried by the lid-driven cavity flow; its centroid trajectory is
compared with Sugiyama et al. (2011) (`data/Sugiyama_1024x1024.csv`) and
Kolahduz (2023) (`data/Kolahduz_2023.csv`).

BC/pressure pairing: no-slip walls + moving lid, Neumann pressure (consistent).

Reference parameters (Jain Sec. 4.5): mu_f=1e-2, solid viscosity eta_s=1e-2,
paper mu_s=0.05 (== code mu_s=0.1), rho_s=rho_f=1.

Usage:
    python benchmarks/soft_disc_in_lid_driven.py [N] [scheme] [t_end]
        N      : grid points per side (default 128)
        scheme : 'semilagrangian' (default) | 'central2' | 'weno5'
        t_end  : end time (default 8.0)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.functions import (
    create_grid, apply_phi_BCs, extrapolate_reference_map,
    compute_timestep, advect_reference_map, rebuild_phi_from_reference_map,
    reinitialize_phi_PDE, momentum_step_rk4, smoothed_heaviside,
    pressure_projection_amg, build_poisson_matrix, _precompute_poisson_eigenvalues,
)
from pyRMT.output import compute_kinetic_energy, compute_strain_energy
from benchmarks.common import (no_slip_lid_bc, initialize_disc, check_narrow_band,
                               disc_centroid, ensure_dir, load_xy_csv)


def run(N=128, scheme='semilagrangian', t_end=8.0, reinit=False, out_root="outputs",
        snapshot_times=None, stress_band=False, detg_clamp=3.0):
    import h5py
    Lx = Ly = 1.0
    U_lid = 1.0
    X, Y, dx, dy = create_grid(N, N, Lx, Ly)
    bc = lambda u, v: no_slip_lid_bc(u, v, U_lid)
    # field snapshots to dump (h5) when the simulation time crosses each target
    snap_targets = sorted(snapshot_times) if snapshot_times else []
    snap_idx = 0

    # disc
    x0, y0, R = 0.6, 0.5, 0.2
    phi_init = lambda Xq, Yq: initialize_disc(Xq, Yq, x0, y0, R)
    phi = apply_phi_BCs(phi_init(X, Y))
    solid_mask = (phi <= 0).astype(float)

    # physics (Jain Sec. 4.5)
    mu_s, kappa, rho_s, eta_s = 0.1, 0.0, 1.0, 0.01   # mu_s^code=0.1 == paper 0.05
    mu_f, rho_f = 0.01, 1.0
    w_t = 2.0 * dx
    gamma = 0.0
    num_layers = max(3, check_narrow_band(w_t, dx, 3))

    X1 = X * solid_mask
    X2 = Y * solid_mask
    X1, X2 = extrapolate_reference_map(X1, X2, phi, dx, dy, num_layers)

    a = np.zeros((N, N)); b = np.zeros((N, N)); p = np.zeros((N, N))

    CFL, dt_min_cap = 0.2, 1e-3
    A = build_poisson_matrix(N, N, dx, dy)
    eig = _precompute_poisson_eigenvalues(N, N, dx, dy)
    ml = None

    out_dir = ensure_dir(os.path.join(out_root, f"soft_disc_lid_N{N}_{scheme}"))
    print(f"[soft-disc-lid] N={N} scheme={scheme} mu_s={mu_s} mu_f={mu_f} "
          f"eta_s={eta_s} layers={num_layers} t_end={t_end}")

    traj = []
    t = 0.0; step = 0
    while t < t_end:
        step += 1
        dt = compute_timestep(a, b, dx, dy, CFL, dt_min_cap, mu_s, rho_s, gamma,
                              rho_f, mu_f=mu_f, eta_s=eta_s, kappa=kappa)
        if t + dt > t_end:
            dt = t_end - t

        phi = rebuild_phi_from_reference_map(X1, X2, phi_init)
        if reinit:
            phi = reinitialize_phi_PDE(phi, dx, dy, num_iters=20,
                                       apply_phi_BCs_func=None, dt_reinit_factor=0.2)
        solid_mask = (phi <= 0).astype(float)

        X1 = advect_reference_map(X1, a, b, X, Y, dt, dx, dy, phi, scheme, 0.0) * solid_mask
        X2 = advect_reference_map(X2, a, b, X, Y, dt, dx, dy, phi, scheme, 0.0) * solid_mask
        X1, X2 = extrapolate_reference_map(X1, X2, phi, dx, dy, num_layers)

        phi = rebuild_phi_from_reference_map(X1, X2, phi_init)

        a_star, b_star, sxx, sxy, syy, J = momentum_step_rk4(
            a, b, p, X1, X2, bc, mu_s, kappa, eta_s, dx, dy, dt,
            rho_s, rho_f, phi, mu_f, w_t, gamma, stress_band=stress_band, detg_clamp=detg_clamp)

        H = smoothed_heaviside(phi, w_t)
        rho_local = (1 - H) * rho_s + H * rho_f
        a, b, p, A, ml = pressure_projection_amg(
            a_star, b_star, dx, dy, dt, rho_local, velocity_bc=bc,
            A=A, ml=ml, p_prev=p, eigenvalues=eig, bc_type='neumann')

        cx, cy = disc_centroid(phi, X, Y)
        t += dt
        traj.append((t, cx, cy, J.min(), J.max()))

        # dump field snapshot when t crosses the next requested target time
        while snap_idx < len(snap_targets) and t >= snap_targets[snap_idx]:
            tt = snap_targets[snap_idx]
            with h5py.File(os.path.join(out_dir, f"snap_t{tt:05.2f}.h5"), "w") as f:
                for nm, arr in (("phi", phi), ("X1", X1), ("X2", X2), ("a", a),
                                ("b", b), ("p", p), ("J", J), ("sigma_xx", sxx),
                                ("sigma_xy", sxy), ("sigma_yy", syy)):
                    f.create_dataset(nm, data=arr)
                f.attrs["t"] = t; f.attrs["t_target"] = tt
            snap_idx += 1

        if step % 100 == 0 or t >= t_end:
            ke = compute_kinetic_energy(a, b, rho_f, rho_s, phi, w_t, dx, dy)
            print(f"  step {step:5d} t={t:6.3f} centroid=({cx:.4f},{cy:.4f}) "
                  f"KE={ke:.3e} min(J)={J.min():.3f} max(J)={J.max():.3f}")

    traj = np.array(traj)
    np.savetxt(os.path.join(out_dir, "centroid.csv"), traj, delimiter=",",
               header="t,cx,cy,minJ,maxJ", comments="")

    # ── overlay against references ────────────────────────────────────────────
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    refs = {}
    for name, fn in (("Sugiyama (2011) 1024^2", "Sugiyama_1024x1024.csv"),
                     ("Kolahduz (2023)", "Kolahduz_2023.csv")):
        path = os.path.join(data_dir, fn)
        if os.path.isfile(path):
            refs[name] = load_xy_csv(path, has_header=False)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5.5, 5.5))
        plt.plot(traj[:, 1], traj[:, 2], "-", lw=2, label=f"pyRMT (N={N}, {scheme})")
        for name, (rx, ry) in refs.items():
            plt.plot(rx, ry, "o", ms=3, label=name)
        plt.xlabel("centroid x"); plt.ylabel("centroid y")
        plt.title("Soft disc in lid-driven cavity — centroid trajectory")
        plt.legend(); plt.axis("equal"); plt.tight_layout()
        fig_path = os.path.join(out_dir, "centroid_compare.png")
        plt.savefig(fig_path, dpi=130)
        print(f"  saved {fig_path}")
    except Exception as e:
        print(f"  (plot skipped: {e})")
    return traj


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    scheme = sys.argv[2] if len(sys.argv) > 2 else 'semilagrangian'
    t_end = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
    run(N=N, scheme=scheme, t_end=t_end)
