"""Viscoelastic money figure, CLEAN version: a soft blob at the stagnation point of
TRUE uniform planar extension u = (eps(x-1/2), -eps(y-1/2)).

This resolves the four-roll-mill caveats: the strain rate is EXACTLY eps everywhere
and for all time (no finite extensional zone, no cell-edge artifact), so
  * ELASTIC (tau->inf):  b_xx(centre) = e^{2 eps t}  -> genuinely unbounded (diverges).
  * VISCOELASTIC:        b_xx(centre) -> 1/(1 - 2 Wi) EXACTLY (Wi = eps*tau < 1/2).
The centre is a stagnation point (no advection there), so b_xx(centre) is also
resolution-independent -- we verify with a grid sweep.

Uniform extension is a steady divergence-free solution; we impose it as the initial
field and as the wall-normal velocity (the DCT-Neumann projection preserves wall-
normal velocity, and the net wall flux is zero -> compatible). Stress via b_e
log-conformation, decoupled from the reference map.

Usage:
  python benchmarks/mac_viscoelastic_uniform.py [N] [t_end] [tau]   # one run
  python benchmarks/mac_viscoelastic_uniform.py plot                # money + resolution figure
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.mac import mac_grid, momentum_predictor, project, poisson_eigs_neumann
from pyRMT.functions import (extrapolate_reference_map, advect_reference_map,
    rebuild_phi_from_reference_map, smoothed_heaviside, grad_central_x_2nd, grad_central_y_2nd)
from pyRMT.viscoelastic import logconf_local_step, sym_exp


def _lam_max(b11, b12, b22):
    tr = 0.5 * (b11 + b22)
    return tr + np.sqrt(np.maximum(0.25 * (b11 - b22)**2 + b12**2, 0.0))


def run(N=96, t_end=8.0, tau=0.5, eps_rate=0.5, G=0.3, mu_f=0.02, rho=1.0,
        R=0.13, out_root="outputs"):
    dx, dy = mac_grid(N, N)
    xc = (np.arange(N) + 0.5) * dx; xf = np.arange(N + 1) * dx
    Xc, Yc = np.meshgrid(xc, xc)
    Xu, Yu = np.meshgrid(xf, xc)          # u at x-faces (Ny,Nx+1): x=i dx, y=(j+.5)dy
    Xv, Yv = np.meshgrid(xc, xf)          # v at y-faces (Ny+1,Nx): x=(i+.5)dx, y=j dy
    Xg, Yg = np.meshgrid(np.arange(N) * dx, np.arange(N) * dy)
    w_t = 2.0 * dx; nu = mu_f / rho; e = eps_rate

    # uniform extension (initial + maintained); divergence-free, stagnation at centre
    u = e * (Xu - 0.5); v = -e * (Yv - 0.5)
    def set_walls(uu, vv):
        uu[:, 0] = -0.5 * e; uu[:, -1] = 0.5 * e         # left/right normal velocity
        vv[0, :] = 0.5 * e; vv[-1, :] = -0.5 * e          # bottom/top normal velocity
        return uu, vv
    u, v = set_walls(u, v)

    pin = lambda X, Y: np.sqrt((X - 0.5)**2 + (Y - 0.5)**2) - R
    phi = pin(Xc, Yc); m = (phi <= 0).astype(float)
    X1, X2 = extrapolate_reference_map(Xc * m, Yc * m, phi, dx, dy, 3)
    p11 = np.zeros((N, N)); p12 = np.zeros((N, N)); p22 = np.zeros((N, N))

    eig = poisson_eigs_neumann(N, N, dx, dy)
    cs = np.sqrt(G / rho)
    dt = min(0.25 * dx / max(0.5 * e, 0.1), 0.2 * dx * dx / nu, 0.3 * dx / (cs + 1e-9))
    if tau <= 0:
        tau = np.inf
    Wi = e * tau
    tag = ("elastic" if tau == np.inf else f"tau{tau:g}") + f"_N{N}"
    bxx_an = np.inf if (tau == np.inf or Wi >= 0.5) else 1.0 / (1.0 - 2.0 * Wi)
    print(f"[ve-uniform] N={N} eps={e} tau={tau} Wi={Wi:.3f} -> analytic b_xx={bxx_an:.4f} dt={dt:.2e}")

    hist = []
    t = 0.0; step = 0
    while t < t_end:
        step += 1
        if t + dt > t_end:
            dt = t_end - t
        u_c = e * (Xc - 0.5); v_c = -e * (Yc - 0.5)        # IMPOSED uniform extension
        phi = rebuild_phi_from_reference_map(X1, X2, pin); m = (phi <= 0).astype(float)
        X1 = advect_reference_map(X1, u_c, v_c, Xg, Yg, dt, dx, dy, phi, 'semilagrangian', 0.0) * m
        X2 = advect_reference_map(X2, u_c, v_c, Xg, Yg, dt, dx, dy, phi, 'semilagrangian', 0.0) * m
        X1, X2 = extrapolate_reference_map(X1, X2, phi, dx, dy, 3)
        p11 = advect_reference_map(p11, u_c, v_c, Xg, Yg, dt, dx, dy, phi, 'semilagrangian', 0.0) * m
        p12 = advect_reference_map(p12, u_c, v_c, Xg, Yg, dt, dx, dy, phi, 'semilagrangian', 0.0) * m
        p22 = advect_reference_map(p22, u_c, v_c, Xg, Yg, dt, dx, dy, phi, 'semilagrangian', 0.0) * m
        p11, p12 = extrapolate_reference_map(p11, p12, phi, dx, dy, 3)
        p22, _ = extrapolate_reference_map(p22, np.zeros((N, N)), phi, dx, dy, 3)
        L11 = grad_central_x_2nd(u_c, dx); L12 = grad_central_y_2nd(u_c, dy)
        L21 = grad_central_x_2nd(v_c, dx); L22 = grad_central_y_2nd(v_c, dy)
        p11, p12, p22 = logconf_local_step(p11, p12, p22, L11, L12, L21, L22, tau, dt)
        be11, be12, be22 = sym_exp(p11, p12, p22)
        t += dt

        sm = phi <= 0
        lam = float(_lam_max(be11, be12, be22)[sm].max()) if sm.any() else np.nan
        ci = N // 2; bxx_c = float(be11[ci, ci])
        if step % 50 == 0 or t >= t_end:
            hist.append((t, lam, bxx_c, float(np.max(np.abs(u_c)))))
        if not np.all(np.isfinite(u)) or not sm.any() or lam > 1e3:
            print(f"  [stopped step {step} t={t:.3f}: lam_max={lam:.2e}]"); break
        if step % 500 == 0:
            print(f"  step {step:5d} t={t:6.3f} lam_max={lam:8.2f} b_xx(c)={bxx_c:8.3f} "
                  f"max|u|={np.max(np.abs(u)):.3f}")

    base = os.path.join(out_root, "mac_ve_uniform"); os.makedirs(base, exist_ok=True)
    np.savez(os.path.join(base, f"hist_{tag}.npz"),
             t=np.array([h[0] for h in hist]), lam=np.array([h[1] for h in hist]),
             bxx=np.array([h[2] for h in hist]), Wi=Wi, bxx_an=bxx_an, eps=e,
             tau=(0.0 if tau == np.inf else tau), N=N)
    print(f"[ve-uniform] {tag}: t={t:.2f} final lam={hist[-1][1]:.2f} b_xx(c)={hist[-1][2]:.4f}"
          f"  (analytic {bxx_an:.4f})")
    return t, hist


def plot(out_root="outputs"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    base = os.path.join(out_root, "mac_ve_uniform")
    files = sorted(f for f in os.listdir(base) if f.startswith("hist_"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for f in files:
        d = np.load(os.path.join(base, f)); tag = f[5:-4]
        if tag.startswith("elastic") or "_N96" in tag:           # money panel: one N each
            ax[0].plot(d['t'], d['bxx'], label=tag.replace("_N96", "") +
                       (f" (Wi={float(d['Wi']):.2f})" if not tag.startswith('elastic') else ""))
            if np.isfinite(d['bxx_an']):
                ax[0].axhline(float(d['bxx_an']), ls=':', lw=1, color='gray')
    # resolution panel: tau0.5 across N
    for f in files:
        if f.startswith("hist_tau0.5_N"):
            d = np.load(os.path.join(base, f))
            ax[1].plot(d['t'], d['bxx'], label=f"N={int(d['N'])}")
    if any(f.startswith("hist_tau0.5_N") for f in files):
        d0 = np.load(os.path.join(base, [f for f in files if f.startswith("hist_tau0.5_N")][0]))
        ax[1].axhline(float(d0['bxx_an']), ls=':', color='k', label='analytic 1/(1-2Wi)')
    ax[0].set_yscale('log'); ax[0].set_xlabel('t'); ax[0].set_ylabel(r'$b_e^{xx}$ at stagnation point')
    ax[0].set_title('elastic diverges ($e^{2\\epsilon t}$); viscoelastic -> exact analytic plateau (dotted)')
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel('t'); ax[1].set_ylabel(r'$b_e^{xx}$ (tau=0.5)')
    ax[1].set_title('resolution independence (N=64,96,128)'); ax[1].legend(fontsize=8)
    fig.tight_layout(); out = os.path.join(base, "ve_uniform_money.png"); fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot()
    else:
        N = int(sys.argv[1]) if len(sys.argv) > 1 else 96
        t_end = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
        tau = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
        run(N=N, t_end=t_end, tau=tau)
