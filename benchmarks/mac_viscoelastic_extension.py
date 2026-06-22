"""THE viscoelastic money figure: a soft blob held at the extensional stagnation
point of a four-roll mill, ELASTIC vs VISCOELASTIC.

A four-roll mill (periodic body force f = nu*2k^2 * u_mill, u_mill = U0(sin kx cos ky,
-cos kx sin ky)) has a hyperbolic stagnation point at the centre: pure planar
extension at rate eps = U0*k. A blob there is stretched continuously.

  * ELASTIC (tau -> inf): the elastic Finger tensor b_e grows ~ e^{2 eps t} without
    bound -> stress diverges, the run fails. (Pure hyperelasticity under sustained
    stretching is ill-posed.)
  * VISCOELASTIC (finite tau, Wi = eps*tau < 1/2): the upper-convected Maxwell
    relaxation balances the stretching -> b_e SATURATES near the analytic planar-
    extension value b_xx -> 1/(1 - 2 Wi), and the blob reaches a steady filament.

Stress is carried by b_e via log-conformation (psi = log b_e), decoupled from the
reference map (which is used only for geometry). This is the demonstration the
lid-driven disc could NOT give (it tumbles -> b_e stays bounded for any tau).

Usage: python benchmarks/mac_viscoelastic_extension.py [N] [t_end] [tau]  (tau<=0 -> elastic)
       python benchmarks/mac_viscoelastic_extension.py plot [N]            (compare runs)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.mac import (mac_grid, momentum_predictor_periodic, project_per,
                       poisson_eigs_periodic, divergence_per)
from pyRMT.functions import (extrapolate_reference_map, advect_reference_map,
    rebuild_phi_from_reference_map, smoothed_heaviside, grad_central_x_2nd, grad_central_y_2nd)
from pyRMT.viscoelastic import logconf_local_step, sym_exp


def _lam_max(b11, b12, b22):
    tr = 0.5 * (b11 + b22)
    return tr + np.sqrt(np.maximum(0.25 * (b11 - b22)**2 + b12**2, 0.0))


def run(N=128, t_end=12.0, tau=0.5, eps_rate=0.5, G=0.3, mu_f=0.05, rho=1.0,
        R=0.13, out_root="outputs"):
    dx, dy = mac_grid(N, N)
    xc = (np.arange(N) + 0.5) * dx; xf = np.arange(N) * dx
    Xc, Yc = np.meshgrid(xc, xc)
    Xu, Yu = np.meshgrid(xf, xc)          # u at x-faces: (i dx, (j+.5)dy)
    Xv, Yv = np.meshgrid(xc, xf)          # v at y-faces: ((i+.5)dx, j dy)
    Xg, Yg = np.meshgrid(xf, xf)
    w_t = 2.0 * dx; nu = mu_f / rho
    k = 2.0 * np.pi; U0 = eps_rate / k    # extension rate at centre = U0*k = eps_rate
    u_mill = U0 * np.sin(k * Xu) * np.cos(k * Yu)
    v_mill = -U0 * np.cos(k * Xv) * np.sin(k * Yv)
    beta = 150.0                          # penalization: lock the FLUID to the mill so
                                          # the blob is kinematically stretched (can't resist)

    pin = lambda X, Y: np.sqrt((X - 0.5)**2 + (Y - 0.5)**2) - R
    phi = pin(Xc, Yc); m = (phi <= 0).astype(float)
    X1, X2 = extrapolate_reference_map(Xc * m, Yc * m, phi, dx, dy, 3)
    p11 = np.zeros((N, N)); p12 = np.zeros((N, N)); p22 = np.zeros((N, N))   # psi = log b_e = 0

    u = np.zeros((N, N)); v = np.zeros((N, N))
    eig = poisson_eigs_periodic(N, N, dx, dy)
    cs = np.sqrt(G / rho)
    dt = min(0.25 * dx / max(U0, 0.1), 0.2 * dx * dx / nu, 0.3 * dx / (cs + 1e-9))
    if tau <= 0:
        tau = np.inf
    Wi = eps_rate * tau
    tag = "elastic" if tau == np.inf else f"tau{tau:g}"
    bxx_analytic = np.inf if (tau == np.inf or Wi >= 0.5) else 1.0 / (1.0 - 2.0 * Wi)
    print(f"[ve-extension] N={N} eps={eps_rate} tau={tau} Wi={Wi:.3f} G={G} "
          f"-> analytic b_xx_steady={bxx_analytic:.3f}  dt={dt:.2e}")

    hist = []
    t = 0.0; step = 0
    while t < t_end:
        step += 1
        if t + dt > t_end:
            dt = t_end - t
        u_c = 0.5 * (u + np.roll(u, -1, 1)); v_c = 0.5 * (v + np.roll(v, -1, 0))
        phi = rebuild_phi_from_reference_map(X1, X2, pin); m = (phi <= 0).astype(float)

        # geometry
        X1 = advect_reference_map(X1, u_c, v_c, Xg, Yg, dt, dx, dy, phi, 'semilagrangian', 0.0) * m
        X2 = advect_reference_map(X2, u_c, v_c, Xg, Yg, dt, dx, dy, phi, 'semilagrangian', 0.0) * m
        X1, X2 = extrapolate_reference_map(X1, X2, phi, dx, dy, 3)
        # stress: advect psi=log(b_e), log-conformation stretch + relaxation
        p11 = advect_reference_map(p11, u_c, v_c, Xg, Yg, dt, dx, dy, phi, 'semilagrangian', 0.0) * m
        p12 = advect_reference_map(p12, u_c, v_c, Xg, Yg, dt, dx, dy, phi, 'semilagrangian', 0.0) * m
        p22 = advect_reference_map(p22, u_c, v_c, Xg, Yg, dt, dx, dy, phi, 'semilagrangian', 0.0) * m
        p11, p12 = extrapolate_reference_map(p11, p12, phi, dx, dy, 3)
        p22, _ = extrapolate_reference_map(p22, np.zeros((N, N)), phi, dx, dy, 3)
        L11 = grad_central_x_2nd(u_c, dx); L12 = grad_central_y_2nd(u_c, dy)
        L21 = grad_central_x_2nd(v_c, dx); L22 = grad_central_y_2nd(v_c, dy)
        p11, p12, p22 = logconf_local_step(p11, p12, p22, L11, L12, L21, L22, tau, dt)
        be11, be12, be22 = sym_exp(p11, p12, p22)

        H = smoothed_heaviside(phi, w_t)
        Sxx = (1 - H) * G * be11; Sxy = (1 - H) * G * be12; Syy = (1 - H) * G * be22
        divx = grad_central_x_2nd(Sxx, dx) + grad_central_y_2nd(Sxy, dy)
        divy = grad_central_x_2nd(Sxy, dx) + grad_central_y_2nd(Syy, dy)
        f_su = 0.5 * (divx + np.roll(divx, 1, 1))        # solid force -> x-faces
        f_sv = 0.5 * (divy + np.roll(divy, 1, 0))        # solid force -> y-faces
        Hu = 0.5 * (H + np.roll(H, 1, 1)); Hv = 0.5 * (H + np.roll(H, 1, 0))
        fu = f_su + Hu * beta * (u_mill - u)            # H=1 in fluid -> lock fluid to mill
        fv = f_sv + Hv * beta * (v_mill - v)

        ustar, vstar = momentum_predictor_periodic(u, v, nu, dx, dy, dt)
        ustar = ustar + dt * fu / rho; vstar = vstar + dt * fv / rho
        u, v, p = project_per(ustar, vstar, dx, dy, dt, rho, eig)
        t += dt

        sm = phi <= 0
        lam = float(_lam_max(be11, be12, be22)[sm].max()) if sm.any() else np.nan
        ci = N // 2
        bxx_c = float(be11[ci, ci])
        if step % 100 == 0 or t >= t_end:
            hist.append((t, lam, bxx_c, float(np.max(np.abs(u)))))
        if not np.all(np.isfinite(u)) or not sm.any() or lam > 1e4:
            print(f"  [stopped step {step} t={t:.3f}: lam_max={lam:.2e}]"); break
        if step % 500 == 0:
            print(f"  step {step:5d} t={t:6.3f} lam_max={lam:8.2f} b_xx(c)={bxx_c:7.3f} "
                  f"max|u|={np.max(np.abs(u)):.3f}")

    out_dir = os.path.join(out_root, "mac_ve_extension"); os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"hist_{tag}.npz"),
             t=np.array([h[0] for h in hist]), lam=np.array([h[1] for h in hist]),
             bxx=np.array([h[2] for h in hist]), umax=np.array([h[3] for h in hist]),
             Wi=Wi, bxx_analytic=bxx_analytic, eps=eps_rate, tau=(0.0 if tau == np.inf else tau))
    print(f"[ve-extension] {tag}: reached t={t:.2f}, final lam_max={hist[-1][1]:.2f}, "
          f"b_xx(c)={hist[-1][2]:.3f}")
    return t, hist


def plot(N=128, out_root="outputs"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out_dir = os.path.join(out_root, "mac_ve_extension")
    files = sorted([f for f in os.listdir(out_dir) if f.startswith("hist_")])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for f in files:
        d = np.load(os.path.join(out_dir, f)); tag = f[5:-4]
        ax[0].plot(d['t'], d['lam'], label=f"{tag} (Wi={float(d['Wi']):.2f})")
        ax[1].plot(d['t'], d['bxx'], label=tag)
        if np.isfinite(d['bxx_analytic']):
            ax[1].axhline(float(d['bxx_analytic']), ls=':', lw=1, color='gray')
    ax[0].set_yscale('log'); ax[0].set_xlabel('t'); ax[0].set_ylabel(r'max $\lambda(b_e)$ in solid')
    ax[0].set_title('elastic diverges; viscoelastic saturates'); ax[0].legend(fontsize=8)
    ax[1].set_xlabel('t'); ax[1].set_ylabel(r'$b_e^{xx}$ at centre')
    ax[1].set_title(r'centre stretch vs analytic $1/(1-2Wi)$ (dotted)'); ax[1].legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(out_dir, "ve_extension_money.png"); fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot(N=int(sys.argv[2]) if len(sys.argv) > 2 else 128)
    else:
        N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
        t_end = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
        tau = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
        run(N=N, t_end=t_end, tau=tau)
