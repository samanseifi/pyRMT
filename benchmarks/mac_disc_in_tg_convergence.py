"""Soft disc in a Taylor-Green vortex on the MAC grid — spatial convergence.

A neo-Hookean disc sits in a single decaying Taylor-Green vortex on a free-slip
box [0,1]^2 (psi = U0 sin(pi x) sin(pi y)). We run each resolution to a fixed time
and measure self-convergence: the cell-centred speed field is sampled on a common
coarse grid and the L2 difference between successive resolutions gives the spatial
order (Jain 2019 Fig. 15 style). Also tracks scalar functionals (kinetic energy,
disc centroid) with Richardson order. Saves convergence plots.

Usage: python benchmarks/mac_disc_in_tg_convergence.py [t_end]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.mac import (mac_grid, momentum_predictor_freeslip, project,
                       poisson_eigs_neumann, divergence)
from pyRMT.functions import (extrapolate_reference_map, advect_reference_map,
    rebuild_phi_from_reference_map, solid_cauchy_stress, smoothed_heaviside,
    grad_central_x_2nd, grad_central_y_2nd)


def run_one(N, t_end, R=0.15, x0=0.5, y0=0.7, mu_s=0.1, mu_f=0.01, U0=0.3, rho=1.0):
    dx, dy = mac_grid(N, N)
    xc = (np.arange(N) + 0.5) * dx
    Xc, Yc = np.meshgrid(xc, xc)
    Xg, Yg = np.meshgrid(np.arange(N) * dx, np.arange(N) * dy)
    xf = np.arange(N + 1) * dx
    Xu, Yu = np.meshgrid(xf, xc)        # u-faces (N, N+1)
    Xv, Yv = np.meshgrid(xc, xf)        # v-faces (N+1, N)

    # single Taylor-Green vortex, free-slip box (normal velocity zero at walls)
    u = U0 * np.pi * np.sin(np.pi * Xu) * np.cos(np.pi * Yu)
    v = -U0 * np.pi * np.cos(np.pi * Xv) * np.sin(np.pi * Yv)
    u[:, 0] = 0.0; u[:, -1] = 0.0; v[0, :] = 0.0; v[-1, :] = 0.0

    phi_init = lambda X, Y: np.sqrt((X - x0)**2 + (Y - y0)**2) - R
    phi = phi_init(Xc, Yc); sm = (phi <= 0).astype(float)
    X1, X2 = extrapolate_reference_map(Xc * sm, Yc * sm, phi, dx, dy, 3)
    w_t = 2.0 * dx; nu = mu_f / rho; cs = np.sqrt(mu_s / rho)
    eig = poisson_eigs_neumann(N, N, dx, dy)
    umax0 = U0 * np.pi
    dt = min(0.25 * dx / umax0, 0.2 * dx * dx / nu, 0.25 * dx / (cs + 1e-9))

    t = 0.0
    while t < t_end:
        if t + dt > t_end:
            dt = t_end - t
        u_c = 0.5 * (u[:, :-1] + u[:, 1:]); v_c = 0.5 * (v[:-1, :] + v[1:, :])
        phi = rebuild_phi_from_reference_map(X1, X2, phi_init); sm = (phi <= 0).astype(float)
        X1 = advect_reference_map(X1, u_c, v_c, Xg, Yg, dt, dx, dy, phi, 'semilagrangian', 0.0) * sm
        X2 = advect_reference_map(X2, u_c, v_c, Xg, Yg, dt, dx, dy, phi, 'semilagrangian', 0.0) * sm
        X1, X2 = extrapolate_reference_map(X1, X2, phi, dx, dy, 3)
        phi = rebuild_phi_from_reference_map(X1, X2, phi_init)
        sxx, sxy, syy, J = solid_cauchy_stress(X1, X2, dx, dy, mu_s, 0.0, phi)
        H = smoothed_heaviside(phi, w_t)
        Sxx = (1 - H) * sxx; Sxy = (1 - H) * sxy; Syy = (1 - H) * syy
        divx = grad_central_x_2nd(Sxx, dx) + grad_central_y_2nd(Sxy, dy)
        divy = grad_central_x_2nd(Sxy, dx) + grad_central_y_2nd(Syy, dy)
        fu = np.zeros((N, N + 1)); fu[:, 1:-1] = 0.5 * (divx[:, 1:] + divx[:, :-1])
        fv = np.zeros((N + 1, N)); fv[1:-1, :] = 0.5 * (divy[1:, :] + divy[:-1, :])
        us, vs = momentum_predictor_freeslip(u, v, nu, dx, dy, dt, fu=fu, fv=fv, rho=rho)
        u, v, p = project(us, vs, dx, dy, dt, rho, eig)
        t += dt

    u_c = 0.5 * (u[:, :-1] + u[:, 1:]); v_c = 0.5 * (v[:-1, :] + v[1:, :])
    speed = np.sqrt(u_c**2 + v_c**2)
    KE = 0.5 * np.sum(u_c**2 + v_c**2) * dx * dy
    msk = phi <= 0
    cx, cy = (Xc[msk].mean(), Yc[msk].mean()) if msk.any() else (np.nan, np.nan)
    return dict(N=N, dx=dx, xc=xc, speed=speed, KE=KE, cx=cx, cy=cy,
                divmax=np.max(np.abs(divergence(u, v, dx, dy))))


def _sample(field_xc, field, grid):
    """bilinear-sample a cell-centred field onto `grid` (1D coords in [0,1])."""
    from numpy import interp
    # separable bilinear via two 1-D interps
    tmp = np.array([np.interp(grid, field_xc, field[j, :]) for j in range(field.shape[0])])
    out = np.array([np.interp(grid, field_xc, tmp[:, k]) for k in range(tmp.shape[1])]).T
    return out


def run(t_end=0.6, out_root="outputs"):
    Ns = [32, 64, 128, 256]
    print(f"[MAC disc-in-TG conv] t_end={t_end}")
    res = []
    for N in Ns:
        r = run_one(N, t_end)
        res.append(r)
        print(f"  N={N:4d}  KE={r['KE']:.6f}  centroid=({r['cx']:.4f},{r['cy']:.4f})  max|div|={r['divmax']:.1e}")

    # field self-convergence: sample speed on a common 24x24 interior grid
    g = (np.arange(24) + 0.5) / 24
    sp = [_sample(r['xc'], r['speed'], g) for r in res]
    ferr = [np.sqrt(np.mean((sp[k] - sp[k + 1])**2)) for k in range(len(Ns) - 1)]
    print("  velocity-field self-convergence (||speed_N - speed_2N||_2):")
    forders = []
    for k in range(len(ferr) - 1):
        o = np.log(ferr[k] / ferr[k + 1]) / np.log(2)
        forders.append(o)
        print(f"    N {Ns[k]}->{Ns[k+1]} vs {Ns[k+1]}->{Ns[k+2]}: field order = {o:.2f}")

    # scalar (KE) Richardson order
    KE = [r['KE'] for r in res]
    print("  kinetic-energy Richardson order:")
    for k in range(len(KE) - 2):
        o = np.log(abs(KE[k] - KE[k + 1]) / abs(KE[k + 1] - KE[k + 2])) / np.log(2)
        print(f"    {Ns[k]},{Ns[k+1]},{Ns[k+2]}: order = {o:.2f}")

    out_dir = os.path.join(out_root, "mac_disc_in_tg"); os.makedirs(out_dir, exist_ok=True)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        h = np.array([1.0 / N for N in Ns[:-1]])
        plt.figure(figsize=(6, 5))
        plt.loglog(h, ferr, "o-", lw=2, label="MAC field self-conv ||u_N - u_2N||")
        c = ferr[0] / h[0]**2
        plt.loglog(h, c * h**2, "k--", label="2nd-order ref")
        plt.gca().invert_xaxis()
        plt.xlabel("h = 1/N"); plt.ylabel("L2 velocity self-difference")
        plt.title("Disc-in-Taylor-Green: MAC spatial convergence"); plt.legend(); plt.grid(True, which="both", alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "convergence_field.png"), dpi=130)
        plt.figure(figsize=(6, 4))
        plt.semilogx(Ns, KE, "s-"); plt.xlabel("N"); plt.ylabel("kinetic energy at T")
        plt.title("Disc-in-TG: KE vs resolution"); plt.grid(True, alpha=.3); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "convergence_energy.png"), dpi=130)
        print(f"  saved {out_dir}/convergence_field.png and convergence_energy.png")
    except Exception as e:
        print(f"  (plot skipped: {e})")
    return res


if __name__ == "__main__":
    t_end = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
    run(t_end=t_end)
