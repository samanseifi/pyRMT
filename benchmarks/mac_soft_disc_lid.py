"""Soft disc in a lid-driven cavity on the MAC grid (FSI) — vs Sugiyama (2011).

Reference map (X1,X2) lives at cell centres and is advected semi-Lagrangian with
the cell-centre velocity (faces interpolated to centres) -- the cell-centred RMT
machinery (phi rebuild, extrapolation, neo-Hookean stress) is reused from the
collocated solver. The blended solid stress divergence is added to the face
momentum. Equal density (rho_s=rho_f) -> constant-density MAC projection.

Usage: python benchmarks/mac_soft_disc_lid.py [N] [t_end]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.mac import mac_grid, momentum_predictor, project, poisson_eigs_neumann, divergence
from pyRMT.functions import (extrapolate_reference_map, advect_reference_map,
    rebuild_phi_from_reference_map, solid_cauchy_stress, smoothed_heaviside,
    grad_central_x_2nd, grad_central_y_2nd)


def run(N=128, t_end=8.0, scheme="semilagrangian", w_cut_fac=0.0, out_root="outputs"):
    dx, dy = mac_grid(N, N)
    xc = (np.arange(N) + 0.5) * dx
    Xc, Yc = np.meshgrid(xc, xc)
    # 0-based index grid for semi-Lagrangian backtrace (bilinear maps x/dx -> cell
    # index, so departure points must be measured from 0, not the 0.5dx centre).
    Xg, Yg = np.meshgrid(np.arange(N) * dx, np.arange(N) * dy)
    x0, y0, R = 0.6, 0.5, 0.2
    phi_init = lambda X, Y: np.sqrt((X - x0)**2 + (Y - y0)**2) - R
    phi = phi_init(Xc, Yc); sm = (phi <= 0).astype(float)

    mu_s, kappa, rho = 0.1, 0.0, 1.0
    mu_f = 0.01; nu = mu_f / rho; w_t = 2.0 * dx; U_lid = 1.0
    nl = 3
    X1, X2 = extrapolate_reference_map(Xc * sm, Yc * sm, phi, dx, dy, nl)

    u = np.zeros((N, N + 1)); v = np.zeros((N + 1, N))
    eig = poisson_eigs_neumann(N, N, dx, dy)
    cs = np.sqrt(mu_s / rho)
    dt = min(0.3 * dx / U_lid, 0.2 * dx * dx / nu, 0.3 * dx / (cs + 1e-9))
    out_dir = os.path.join(out_root, f"mac_soft_disc_N{N}"); os.makedirs(out_dir, exist_ok=True)
    print(f"[MAC FSI] N={N} mu_s={mu_s} mu_f={mu_f} dt={dt:.2e} t_end={t_end}")

    t = 0.0; step = 0; traj = []
    while t < t_end:
        step += 1
        # face velocity -> cell centres for advecting the reference map
        u_c = 0.5 * (u[:, :-1] + u[:, 1:])
        v_c = 0.5 * (v[:-1, :] + v[1:, :])
        phi = rebuild_phi_from_reference_map(X1, X2, phi_init); sm = (phi <= 0).astype(float)
        wc = w_cut_fac * dx
        X1 = advect_reference_map(X1, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc) * sm
        X2 = advect_reference_map(X2, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc) * sm
        X1, X2 = extrapolate_reference_map(X1, X2, phi, dx, dy, nl)
        phi = rebuild_phi_from_reference_map(X1, X2, phi_init)

        sxx, sxy, syy, J = solid_cauchy_stress(X1, X2, dx, dy, mu_s, kappa, phi)
        H = smoothed_heaviside(phi, w_t)
        Sxx = (1 - H) * sxx; Sxy = (1 - H) * sxy; Syy = (1 - H) * syy
        divx = grad_central_x_2nd(Sxx, dx) + grad_central_y_2nd(Sxy, dy)   # centres
        divy = grad_central_x_2nd(Sxy, dx) + grad_central_y_2nd(Syy, dy)
        fu = np.zeros((N, N + 1)); fu[:, 1:-1] = 0.5 * (divx[:, 1:] + divx[:, :-1])
        fv = np.zeros((N + 1, N)); fv[1:-1, :] = 0.5 * (divy[1:, :] + divy[:-1, :])

        ustar, vstar = momentum_predictor(u, v, nu, dx, dy, dt, U_lid, fu=fu, fv=fv, rho=rho)
        u, v, p = project(ustar, vstar, dx, dy, dt, rho, eig)

        msk = phi <= 0
        cx = Xc[msk].mean() if msk.any() else np.nan
        cy = Yc[msk].mean() if msk.any() else np.nan
        t += dt
        traj.append((t, cx, cy, J.min(), J.max()))
        if step % 200 == 0 or t >= t_end:
            print(f"  step {step:5d} t={t:6.3f} centroid=({cx:.3f},{cy:.3f}) "
                  f"minJ={J.min():.3f} maxJ={J.max():.3f} max|div|={np.max(np.abs(divergence(u,v,dx,dy))):.1e}")

    traj = np.array(traj)
    np.savetxt(os.path.join(out_dir, "centroid.csv"), traj, delimiter=",",
               header="t,cx,cy,minJ,maxJ", comments="")
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(5.5, 5.5))
        plt.plot(traj[:, 1], traj[:, 2], "-", lw=2, label=f"MAC (N={N})")
        for nm, fn in (("Sugiyama 1024^2", "Sugiyama_1024x1024.csv"), ("Kolahduz", "Kolahduz_2023.csv")):
            pth = os.path.join(data, fn)
            if os.path.isfile(pth):
                d = np.loadtxt(pth, delimiter=","); plt.plot(d[:, 0], d[:, 1], "o", ms=3, label=nm)
        plt.xlabel("centroid x"); plt.ylabel("centroid y"); plt.legend(); plt.axis("equal"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "centroid_compare.png"), dpi=130)
        print(f"  saved {out_dir}/centroid_compare.png")
    except Exception as e:
        print(f"  (plot skipped: {e})")
    return traj


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    t_end = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    scheme = sys.argv[3] if len(sys.argv) > 3 else "semilagrangian"
    run(N=N, t_end=t_end, scheme=scheme)
