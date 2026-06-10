"""Soft disc in a Taylor-Green vortex on the MAC grid — convergence of ALL
quantities for ALL advection schemes (semi-Lagrangian, central2, weno5).

A neo-Hookean disc sits in a single decaying Taylor-Green vortex on a free-slip
box [0,1]^2 (psi = U0 sin(pi x) sin(pi y)). For each advection scheme and each
resolution we run to a fixed time and measure convergence vs a fixed fine
reference (N=256): velocity |u|, pressure p (L2), and kinetic / strain energy
(scalars). Saves one convergence plot per quantity with all schemes overlaid.

The advection scheme is switched with a single argument (scheme=...), shared with
the FSI drivers -- semi-Lagrangian (robust default), central2, weno5.

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
from pyRMT.output import compute_strain_energy

SCHEMES = ["semilagrangian", "central2", "weno5"]
NS = [32, 64, 128, 256]


def run_one(N, t_end, R=0.15, x0=0.5, y0=0.7, mu_s=0.1, mu_f=0.01, U0=0.3, rho=1.0,
            scheme="semilagrangian", w_cut_fac=4.0):
    dx, dy = mac_grid(N, N)
    xc = (np.arange(N) + 0.5) * dx
    Xc, Yc = np.meshgrid(xc, xc)
    Xg, Yg = np.meshgrid(np.arange(N) * dx, np.arange(N) * dy)
    xf = np.arange(N + 1) * dx
    Xu, Yu = np.meshgrid(xf, xc)
    Xv, Yv = np.meshgrid(xc, xf)

    u = U0 * np.pi * np.sin(np.pi * Xu) * np.cos(np.pi * Yu)
    v = -U0 * np.pi * np.cos(np.pi * Xv) * np.sin(np.pi * Yv)
    u[:, 0] = 0.0; u[:, -1] = 0.0; v[0, :] = 0.0; v[-1, :] = 0.0

    phi_init = lambda X, Y: np.sqrt((X - x0)**2 + (Y - y0)**2) - R
    phi = phi_init(Xc, Yc); sm = (phi <= 0).astype(float)
    X1, X2 = extrapolate_reference_map(Xc * sm, Yc * sm, phi, dx, dy, 3)
    w_t = 2.0 * dx; nu = mu_f / rho; cs = np.sqrt(mu_s / rho)
    eig = poisson_eigs_neumann(N, N, dx, dy)
    dt = min(0.25 * dx / (U0 * np.pi), 0.2 * dx * dx / nu, 0.25 * dx / (cs + 1e-9))
    p = np.zeros((N, N))

    t = 0.0
    while t < t_end:
        if t + dt > t_end:
            dt = t_end - t
        u_c = 0.5 * (u[:, :-1] + u[:, 1:]); v_c = 0.5 * (v[:-1, :] + v[1:, :])
        phi = rebuild_phi_from_reference_map(X1, X2, phi_init); sm = (phi <= 0).astype(float)
        wc = w_cut_fac * dx
        X1 = advect_reference_map(X1, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc) * sm
        X2 = advect_reference_map(X2, u_c, v_c, Xg, Yg, dt, dx, dy, phi, scheme, wc) * sm
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
    se = compute_strain_energy(X1, X2, phi, mu_s, dx, dy)
    Estrain = np.nansum(np.where(np.isfinite(se), se, 0.0)) * dx * dy
    return dict(N=N, dx=dx, xc=xc, speed=speed, p=p / dt, KE=KE, Estrain=Estrain,
                divmax=np.max(np.abs(divergence(u, v, dx, dy))))


def _sample(field_xc, field, grid):
    tmp = np.array([np.interp(grid, field_xc, field[j, :]) for j in range(field.shape[0])])
    out = np.array([np.interp(grid, field_xc, tmp[:, k]) for k in range(tmp.shape[1])]).T
    return out


def _orders(Ns, errs):
    return [np.log(errs[k] / errs[k + 1]) / np.log(Ns[k + 1] / Ns[k]) for k in range(len(errs) - 1)]


def run(t_end=0.6, out_root="outputs"):
    g = (np.arange(20) + 0.5) / 20
    data = {}
    for sch in SCHEMES:
        print(f"\n=== scheme: {sch} ===")
        res = {}
        for N in NS:
            try:
                res[N] = run_one(N, t_end, scheme=sch)
                print(f"  N={N:4d}  KE={res[N]['KE']:.6f}  Estrain={res[N]['Estrain']:.6f}  max|div|={res[N]['divmax']:.1e}")
            except Exception as e:
                print(f"  N={N:4d}  FAILED: {type(e).__name__}")
                res[N] = None
        data[sch] = res

    out_dir = os.path.join(out_root, "mac_disc_in_tg"); os.makedirs(out_dir, exist_ok=True)
    coarse = NS[:-1]; h = np.array([1.0 / N for N in coarse])

    # per-scheme errors vs that scheme's own N=256 reference
    verr = {}; perr = {}
    for sch in SCHEMES:
        r = data[sch]
        if r[256] is None or any(r[N] is None for N in coarse):
            verr[sch] = perr[sch] = None; continue
        spref = _sample(r[256]['xc'], r[256]['speed'], g)
        prref = _sample(r[256]['xc'], r[256]['p'], g)
        verr[sch] = [np.sqrt(np.mean((_sample(r[N]['xc'], r[N]['speed'], g) - spref)**2)) for N in coarse]
        perr[sch] = [np.sqrt(np.mean((_sample(r[N]['xc'], r[N]['p'], g) - prref)**2)) for N in coarse]
        print(f"\n[{sch}] velocity order:", [f"{o:.2f}" for o in _orders(coarse, verr[sch])],
              " pressure order:", [f"{o:.2f}" for o in _orders(coarse, perr[sch])])

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        col = {"semilagrangian": "C0", "central2": "C1", "weno5": "C2"}

        def conv_plot(err, title, fname):
            plt.figure(figsize=(6.2, 5))
            for sch in SCHEMES:
                if err[sch] is None:
                    continue
                plt.loglog(h, err[sch], "o-", color=col[sch], lw=2,
                           label=f"{sch}: order {_orders(coarse, err[sch])[-1]:.2f}")
            e0 = next(e for e in err.values() if e is not None)[0]
            plt.loglog(h, e0 * (h / h[0])**2, "k--", label="2nd-order ref")
            plt.gca().invert_xaxis()
            plt.xlabel("h = 1/N"); plt.ylabel("L2 error vs N=256 reference")
            plt.title(title); plt.legend(); plt.grid(True, which="both", alpha=.3); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, fname), dpi=130); plt.close()

        conv_plot(verr, "Disc-in-TG: velocity convergence (MAC)", "conv_velocity.png")
        conv_plot(perr, "Disc-in-TG: pressure convergence (MAC)", "conv_pressure.png")

        for q, fn, ttl in (("KE", "conv_kinetic_energy.png", "kinetic energy"),
                           ("Estrain", "conv_strain_energy.png", "strain energy")):
            plt.figure(figsize=(6.2, 4.2))
            for sch in SCHEMES:
                vals = [data[sch][N][q] if data[sch][N] else np.nan for N in NS]
                plt.semilogx(NS, vals, "s-", color=col[sch], label=sch)
            plt.xlabel("N"); plt.ylabel(f"{ttl} at T"); plt.title(f"Disc-in-TG: {ttl} vs resolution")
            plt.legend(); plt.grid(True, alpha=.3); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, fn), dpi=130); plt.close()
        print(f"\n  saved 4 convergence plots in {out_dir}/")
    except Exception as e:
        print(f"  (plot skipped: {e})")
    return data


if __name__ == "__main__":
    t_end = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
    run(t_end=t_end)
