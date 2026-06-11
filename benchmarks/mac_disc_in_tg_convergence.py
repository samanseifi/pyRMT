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

SCHEMES = ["semilagrangian", "semilagrangian_cubic", "central2", "weno5"]
NS = [32, 64, 128, 256]


def run_one(N, t_end, R=0.15, x0=0.5, y0=0.7, mu_s=0.1, mu_f=0.01, U0=0.3, rho=1.0,
            scheme="semilagrangian", w_cut_fac=4.0, isochoric=False):
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
        sxx, sxy, syy, J = solid_cauchy_stress(X1, X2, dx, dy, mu_s, 0.0, phi,
                                               isochoric=isochoric)
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
    # `project` returns the physical pressure (O(1)); do NOT divide by dt.
    return dict(N=N, dx=dx, xc=xc, speed=speed, p=p, KE=KE, Estrain=Estrain,
                divmax=np.max(np.abs(divergence(u, v, dx, dy))))


def _sample(field_xc, field, grid):
    tmp = np.array([np.interp(grid, field_xc, field[j, :]) for j in range(field.shape[0])])
    out = np.array([np.interp(grid, field_xc, tmp[:, k]) for k in range(tmp.shape[1])]).T
    return out


def _orders(Ns, errs):
    return [np.log(errs[k] / errs[k + 1]) / np.log(Ns[k + 1] / Ns[k]) for k in range(len(errs) - 1)]


def run(t_end=0.6, isochoric=False, out_root="outputs"):
    g = (np.arange(20) + 0.5) / 20
    data = {}
    for sch in SCHEMES:
        print(f"\n=== scheme: {sch} (isochoric={isochoric}) ===")
        res = {}
        for N in NS:
            try:
                res[N] = run_one(N, t_end, scheme=sch, isochoric=isochoric)
                print(f"  N={N:4d}  KE={res[N]['KE']:.6f}  Estrain={res[N]['Estrain']:.6f}  max|div|={res[N]['divmax']:.1e}")
            except Exception as e:
                print(f"  N={N:4d}  FAILED: {type(e).__name__}")
                res[N] = None
        data[sch] = res

    out_dir = os.path.join(out_root, "mac_disc_in_tg"); os.makedirs(out_dir, exist_ok=True)
    REF = NS[-1]; coarse = NS[:-1]; h = np.array([1.0 / N for N in coarse])

    # per-scheme L2 errors vs that scheme's own finest reference (N=REF)
    verr = {}; perr = {}
    for sch in SCHEMES:
        r = data[sch]
        if r[REF] is None or any(r[N] is None for N in coarse):
            verr[sch] = perr[sch] = None; continue
        spref = _sample(r[REF]['xc'], r[REF]['speed'], g)
        prref = _sample(r[REF]['xc'], r[REF]['p'], g)
        verr[sch] = [np.sqrt(np.mean((_sample(r[N]['xc'], r[N]['speed'], g) - spref)**2)) for N in coarse]
        perr[sch] = [np.sqrt(np.mean((_sample(r[N]['xc'], r[N]['p'], g) - prref)**2)) for N in coarse]
        print(f"\n[{sch}] velocity order:", [f"{o:.2f}" for o in _orders(coarse, verr[sch])],
              " pressure order:", [f"{o:.2f}" for o in _orders(coarse, perr[sch])])

    KE = {s: [data[s][N]['KE'] if data[s][N] else np.nan for N in NS] for s in SCHEMES}
    Es = {s: [data[s][N]['Estrain'] if data[s][N] else np.nan for N in NS] for s in SCHEMES}
    nan = [np.nan] * len(coarse)
    np.savez(os.path.join(out_dir, "conv_data.npz"),
             coarse=np.array(coarse), Ns=np.array(NS), REF=REF, h=h,
             **{f"verr_{s}": np.array(verr[s] if verr[s] else nan) for s in SCHEMES},
             **{f"perr_{s}": np.array(perr[s] if perr[s] else nan) for s in SCHEMES},
             **{f"KE_{s}": np.array(KE[s]) for s in SCHEMES},
             **{f"Es_{s}": np.array(Es[s]) for s in SCHEMES})
    plot_pub(out_dir, coarse, NS, REF, h, verr, perr, KE, Es)
    return data


def plot_pub(out_dir, coarse, Ns, REF, h, verr, perr, KE, Es):
    """Publication-quality convergence figures: per-scheme curves with 1st- and
    2nd-order reference lines (fanning from the coarsest point) plus slope
    triangles for a qualitative read."""
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (plot skipped: {e})"); return
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12.5,
                         "axes.titlesize": 12.5, "legend.fontsize": 10})
    col = {"semilagrangian": "#1f77b4", "semilagrangian_cubic": "#9467bd",
           "central2": "#ff7f0e", "weno5": "#2ca02c"}
    mk = {"semilagrangian": "o", "semilagrangian_cubic": "D", "central2": "s", "weno5": "^"}
    lbl = {"semilagrangian": "semi-Lagrangian (bilinear)",
           "semilagrangian_cubic": "semi-Lagrangian (cubic)",
           "central2": "central-2", "weno5": "WENO5"}
    h = np.asarray(h)

    def _tri(ax, slope, ybase, color):
        ha, hb = h[-2], h[-1]                       # ha>hb; hb is finer (right side)
        yb = ybase * (hb / ha) ** slope
        ax.plot([ha, hb], [ybase, ybase], color=color, lw=0.9)
        ax.plot([hb, hb], [ybase, yb], color=color, lw=0.9)
        ax.plot([ha, hb], [ybase, yb], color=color, lw=1.3)
        ax.text(hb * 0.88, (ybase * yb) ** 0.5, f"{slope}", color=color,
                fontsize=9.5, ha="right", va="center")

    def conv(err, ylabel, title, fname):
        present = [e for e in err.values() if e is not None]
        if not present:
            return
        fig, ax = plt.subplots(figsize=(6.6, 5.3))
        y0 = max(e[0] for e in present) * 1.6
        ax.loglog(h, y0 * (h / h[0]) ** 1, ls=":", color="0.55", lw=1.5, label="1st order")
        ax.loglog(h, y0 * (h / h[0]) ** 2, ls="--", color="0.3", lw=1.5, label="2nd order")
        for sch in SCHEMES:
            if err[sch] is None:
                continue
            o = _orders(coarse, err[sch])[-1]
            ax.loglog(h, err[sch], marker=mk[sch], color=col[sch], lw=2, ms=8,
                      label=f"{lbl[sch]}  (p≈{o:.2f})")
        ybase = min(e[-1] for e in present) * 0.4
        _tri(ax, 1, ybase * 2.4, "0.55")
        _tri(ax, 2, ybase, "0.3")
        ax.invert_xaxis()
        for N, hv in zip(coarse, h):
            ax.annotate(f"{N}", (hv, max(e[list(coarse).index(N)] for e in present)),
                        textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center", color="0.4")
        ax.set_xlabel("grid spacing  $h = 1/N$"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.grid(True, which="both", alpha=.25)
        ax.legend(frameon=True, framealpha=.92, loc="lower left")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, fname), dpi=200); plt.close(fig)

    conv(verr, r"$\||u|_N-|u|_{\mathrm{ref}}\|_2$",
         f"Disc-in-Taylor-Green: velocity convergence (ref $N={REF}$)", "conv_velocity.png")
    conv(perr, r"$\|p_N-p_{\mathrm{ref}}\|_2$",
         f"Disc-in-Taylor-Green: pressure convergence (ref $N={REF}$)", "conv_pressure.png")
    for vals, ttl, fn in ((KE, "kinetic energy", "conv_kinetic_energy.png"),
                          (Es, "strain energy", "conv_strain_energy.png")):
        fig, ax = plt.subplots(figsize=(6.6, 4.5))
        for sch in SCHEMES:
            ax.semilogx(Ns, vals[sch], marker=mk[sch], color=col[sch], lw=2, ms=7, label=lbl[sch])
        ax.set_xlabel("$N$"); ax.set_ylabel(f"{ttl} at $T$")
        ax.set_title(f"Disc-in-Taylor-Green: {ttl} vs resolution")
        ax.grid(True, alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fn), dpi=200); plt.close(fig)
    print(f"\n  saved 4 publication plots in {out_dir}/")


def replot(out_root="outputs"):
    """Re-make the publication figures from saved conv_data.npz (no re-run)."""
    out_dir = os.path.join(out_root, "mac_disc_in_tg")
    d = np.load(os.path.join(out_dir, "conv_data.npz"))
    coarse = list(d["coarse"]); Ns = list(d["Ns"]); REF = int(d["REF"]); h = d["h"]
    def grab(prefix):
        out = {}
        for s in SCHEMES:
            a = d[f"{prefix}_{s}"]
            out[s] = None if np.all(np.isnan(a)) else list(a)
        return out
    plot_pub(out_dir, coarse, Ns, REF, h, grab("verr"), grab("perr"), grab("KE"), grab("Es"))


if __name__ == "__main__":
    t_end = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
    isochoric = (len(sys.argv) > 2 and sys.argv[2].lower() in ("1", "true", "iso"))
    run(t_end=t_end, isochoric=isochoric)
