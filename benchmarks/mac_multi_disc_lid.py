"""Several soft discs of different sizes, randomly placed in a lid-driven cavity,
stirred by the vortex and colliding via the contact STRESS (MAC grid).

Each disc carries its own reference map (X1_i, X2_i) and level set phi_i. The
blended solid stress is sum_i (1-H_i) sigma_s_i; the Rycroft contact stress is added
for every overlapping pair. Total stress divergence -> face force; lid-driven
momentum + exact projection. Honest divergence detection (folded/lost disc).

Usage: python benchmarks/mac_multi_disc_lid.py [N] [t_end] [n_discs] [seed]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.mac import (mac_grid, momentum_predictor, project, poisson_eigs_neumann,
                       divergence, contact_stress)
from pyRMT.functions import (extrapolate_reference_map, advect_reference_map,
    rebuild_phi_from_reference_map, solid_cauchy_stress, smoothed_heaviside,
    grad_central_x_2nd, grad_central_y_2nd)


def _place_discs(n, seed, Rrange=(0.07, 0.12), box=(0.18, 0.82)):
    rng = np.random.default_rng(seed)
    discs = []
    lo, hi = box
    for _ in range(2000):
        if len(discs) == n:
            break
        R = rng.uniform(*Rrange)
        cx = rng.uniform(lo + R, hi - R); cy = rng.uniform(lo + R, hi - R)
        if all((cx - d[1])**2 + (cy - d[2])**2 > (R + d[0] + 0.03)**2 for d in discs):
            discs.append((R, cx, cy))
    return discs


def run(N=128, t_end=12.0, n_discs=3, seed=3, U_lid=1.0, mu_s=0.3, mu_f=0.01,
        rho=1.0, eta=2.0, out_root="outputs"):
    dx, dy = mac_grid(N, N)
    xc = (np.arange(N) + 0.5) * dx
    Xc, Yc = np.meshgrid(xc, xc)
    Xg, Yg = np.meshgrid(np.arange(N) * dx, np.arange(N) * dy)
    w_t = 2.0 * dx; nu = mu_f / rho; eps = 3.0 * dx
    specs = _place_discs(n_discs, seed)
    print(f"[multi-disc] N={N} discs={len(specs)} eta={eta} mu_s={mu_s} t_end={t_end}")
    for k, (R, cx, cy) in enumerate(specs):
        print(f"   disc {k}: R={R:.3f} centre=({cx:.3f},{cy:.3f})")

    inits = [(lambda X, Y, cx=cx, cy=cy, R=R: np.sqrt((X-cx)**2 + (Y-cy)**2) - R)
             for (R, cx, cy) in specs]
    refs = []
    for pin in inits:
        phi = pin(Xc, Yc); m = (phi <= 0).astype(float)
        X1, X2 = extrapolate_reference_map(Xc * m, Yc * m, phi, dx, dy, 3)
        refs.append([X1, X2])

    u = np.zeros((N, N + 1)); v = np.zeros((N + 1, N))
    eig = poisson_eigs_neumann(N, N, dx, dy)
    cs = np.sqrt(mu_s / rho)
    dt = min(0.3 * dx / U_lid, 0.2 * dx * dx / nu, 0.3 * dx / (cs + 1e-9))
    out_dir = os.path.join(out_root, f"mac_multi_disc_N{N}"); os.makedirs(out_dir, exist_ok=True)

    snap_times = list(np.linspace(0, t_end, 6)); snaps = {}
    t = 0.0; step = 0
    while t < t_end:
        step += 1
        if t + dt > t_end:
            dt = t_end - t
        u_c = 0.5 * (u[:, :-1] + u[:, 1:]); v_c = 0.5 * (v[:-1, :] + v[1:, :])
        phis = []
        for k, pin in enumerate(inits):
            X1, X2 = refs[k]
            phi = rebuild_phi_from_reference_map(X1, X2, pin); m = (phi <= 0).astype(float)
            X1 = advect_reference_map(X1, u_c, v_c, Xg, Yg, dt, dx, dy, phi, 'semilagrangian', 0.0) * m
            X2 = advect_reference_map(X2, u_c, v_c, Xg, Yg, dt, dx, dy, phi, 'semilagrangian', 0.0) * m
            X1, X2 = extrapolate_reference_map(X1, X2, phi, dx, dy, 3)
            refs[k] = [X1, X2]
            phis.append(rebuild_phi_from_reference_map(X1, X2, pin))

        Sxx = np.zeros((N, N)); Sxy = np.zeros((N, N)); Syy = np.zeros((N, N))
        Jmin = 1.0; Jmax = 1.0
        for k in range(len(refs)):
            sxx, sxy, syy, J = solid_cauchy_stress(refs[k][0], refs[k][1], dx, dy, mu_s, 0.0, phis[k])
            H = smoothed_heaviside(phis[k], w_t)
            Sxx += (1 - H) * sxx; Sxy += (1 - H) * sxy; Syy += (1 - H) * syy
            Jmin = min(Jmin, J.min()); Jmax = max(Jmax, J.max())
        if eta > 0:
            for i in range(len(phis)):
                for j in range(i + 1, len(phis)):
                    txx, txy, tyy = contact_stress(phis[i], phis[j], eta, 2 * mu_s, eps, dx, dy)
                    Sxx += txx; Sxy += txy; Syy += tyy

        divx = grad_central_x_2nd(Sxx, dx) + grad_central_y_2nd(Sxy, dy)
        divy = grad_central_x_2nd(Sxy, dx) + grad_central_y_2nd(Syy, dy)
        fu = np.zeros((N, N + 1)); fu[:, 1:-1] = 0.5 * (divx[:, 1:] + divx[:, :-1])
        fv = np.zeros((N + 1, N)); fv[1:-1, :] = 0.5 * (divy[1:, :] + divy[:-1, :])
        ustar, vstar = momentum_predictor(u, v, nu, dx, dy, dt, U_lid, fu=fu, fv=fv, rho=rho)
        u, v, p = project(ustar, vstar, dx, dy, dt, rho, eig)
        t += dt

        if (not np.all(np.isfinite(u)) or Jmin < 0.0 or Jmax > 20.0
                or any(not (pp <= 0).any() for pp in phis)):
            print(f"  [DIVERGED at step {step}, t={t:.3f}: minJ={Jmin:.2f} maxJ={Jmax:.2f}]")
            break
        for ts in snap_times:
            if ts not in snaps and t >= ts:
                snaps[ts] = [pp.copy() for pp in phis]
        if step % 200 == 0 or t >= t_end:
            cents = [(Xc[pp <= 0].mean(), Yc[pp <= 0].mean()) for pp in phis]
            cs_str = " ".join(f"({c[0]:.2f},{c[1]:.2f})" for c in cents)
            print(f"  step {step:5d} t={t:6.3f} centroids={cs_str} minJ={Jmin:.2f} maxJ={Jmax:.2f} "
                  f"max|u|={np.max(np.abs(u)):.2f} div={np.max(np.abs(divergence(u,v,dx,dy))):.0e}")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ns = sorted(snaps); fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        cols = plt.cm.tab10(np.arange(len(specs)))
        for ax, ts in zip(axes.ravel(), ns):
            for k, pp in enumerate(snaps[ts]):
                ax.contourf(Xc, Yc, pp, levels=[-1e9, 0], colors=[cols[k]], alpha=0.8)
                ax.contour(Xc, Yc, pp, levels=[0], colors=['k'], linewidths=0.8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
            ax.set_title(f"t={ts:.1f}"); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{len(specs)} discs in lid-driven cavity (contact eta={eta}, N={N})")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "multi_disc_snapshots.png"), dpi=130)
        print(f"  saved {out_dir}/multi_disc_snapshots.png  (t={[f'{x:.1f}' for x in ns]})")
    except Exception as e:
        print(f"  (plot skipped: {e})")


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    t_end = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
    n_discs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    run(N=N, t_end=t_end, n_discs=n_discs, seed=seed)
