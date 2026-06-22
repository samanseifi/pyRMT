"""The money figure: a soft disc in a lid-driven cavity, ELASTIC vs VISCOELASTIC.

A purely hyperelastic disc under sustained lid shear accumulates strain without
bound -- the elastic Finger tensor b_e grows, the stress diverges, and the solver
eventually fails (this is the ill-posed limit that reinitialization only defers).
A *viscoelastic* disc (finite relaxation time tau) relaxes the accumulated strain,
so b_e saturates and the disc reaches a steady deformed state and runs indefinitely.

DECOUPLING: the reference map xi is kept only for the geometry (phi = phi_0(xi)).
The stress comes from a SEPARATE elastic Finger tensor b_e advected with the flow
and evolved by the upper-convected Maxwell model (pyRMT/viscoelastic.py) -- so the
reference map's eventual folding no longer drives the dynamics.

Per step:  advect xi (geometry) and b_e (stress);  b_e += dt*(L b_e + b_e L^T
- (b_e-I)/tau);  sigma = (1-H) G b_e;  lid momentum + exact projection.

tau = inf reproduces the elastic case within the same code path.

Usage: python benchmarks/mac_viscoelastic_disc.py [N] [t_end] [tau]   (tau<=0 -> inf)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.mac import mac_grid, momentum_predictor, project, poisson_eigs_neumann, divergence
from pyRMT.functions import (extrapolate_reference_map, advect_reference_map,
    rebuild_phi_from_reference_map, smoothed_heaviside, grad_central_x_2nd, grad_central_y_2nd)
from pyRMT.viscoelastic import logconf_local_step, sym_exp


def _lam_max(b11, b12, b22):
    tr = 0.5 * (b11 + b22)
    dd = np.sqrt(np.maximum(0.25 * (b11 - b22)**2 + b12**2, 0.0))
    return tr + dd


def run(N=96, t_end=18.0, tau=1.0, U_lid=1.0, G=0.3, mu_f=0.01, rho=1.0,
        x0=0.6, y0=0.5, R=0.2, be_scheme='semilagrangian', out_root="outputs"):
    dx, dy = mac_grid(N, N)
    xc = (np.arange(N) + 0.5) * dx
    Xc, Yc = np.meshgrid(xc, xc)
    Xg, Yg = np.meshgrid(np.arange(N) * dx, np.arange(N) * dy)
    w_t = 2.0 * dx; nu = mu_f / rho
    pin = lambda X, Y: np.sqrt((X - x0)**2 + (Y - y0)**2) - R
    phi = pin(Xc, Yc); m = (phi <= 0).astype(float)
    X1, X2 = extrapolate_reference_map(Xc * m, Yc * m, phi, dx, dy, 3)   # geometry only
    p11 = np.zeros((N, N)); p12 = np.zeros((N, N)); p22 = np.zeros((N, N))  # psi=log(b_e)=0

    u = np.zeros((N, N + 1)); v = np.zeros((N + 1, N))
    eig = poisson_eigs_neumann(N, N, dx, dy)
    cs = np.sqrt(G / rho)
    dt = min(0.3 * dx / U_lid, 0.2 * dx * dx / nu, 0.3 * dx / (cs + 1e-9))
    if tau <= 0:
        tau = np.inf
    tag = "elastic" if tau == np.inf else f"tau{tau:g}"
    print(f"[viscoelastic-disc] N={N} G={G} tau={tau} Wi~{(tau*U_lid/(2*R)) if tau!=np.inf else np.inf:.2f} "
          f"t_end={t_end} dt={dt:.2e}")

    hist = []                     # (t, lam_max_in_solid, max|u|)
    t = 0.0; step = 0
    while t < t_end:
        step += 1
        if t + dt > t_end:
            dt = t_end - t
        u_c = 0.5 * (u[:, :-1] + u[:, 1:]); v_c = 0.5 * (v[:-1, :] + v[1:, :])
        phi = rebuild_phi_from_reference_map(X1, X2, pin); m = (phi <= 0).astype(float)

        # geometry: advect the reference map
        X1 = advect_reference_map(X1, u_c, v_c, Xg, Yg, dt, dx, dy, phi, 'semilagrangian', 0.0) * m
        X2 = advect_reference_map(X2, u_c, v_c, Xg, Yg, dt, dx, dy, phi, 'semilagrangian', 0.0) * m
        X1, X2 = extrapolate_reference_map(X1, X2, phi, dx, dy, 3)

        # stress: advect psi=log(b_e) (b_e=exp(psi) stays SPD regardless), then
        # log-conformation stretch + relaxation (operator split)
        p11 = advect_reference_map(p11, u_c, v_c, Xg, Yg, dt, dx, dy, phi, be_scheme, 0.0) * m
        p12 = advect_reference_map(p12, u_c, v_c, Xg, Yg, dt, dx, dy, phi, be_scheme, 0.0) * m
        p22 = advect_reference_map(p22, u_c, v_c, Xg, Yg, dt, dx, dy, phi, be_scheme, 0.0) * m
        p11, p12 = extrapolate_reference_map(p11, p12, phi, dx, dy, 3)
        p22, _   = extrapolate_reference_map(p22, np.zeros((N, N)), phi, dx, dy, 3)
        L11 = grad_central_x_2nd(u_c, dx); L12 = grad_central_y_2nd(u_c, dy)
        L21 = grad_central_x_2nd(v_c, dx); L22 = grad_central_y_2nd(v_c, dy)
        p11, p12, p22 = logconf_local_step(p11, p12, p22, L11, L12, L21, L22, tau, dt)
        be11, be12, be22 = sym_exp(p11, p12, p22)

        H = smoothed_heaviside(phi, w_t)
        Sxx = (1 - H) * G * be11; Sxy = (1 - H) * G * be12; Syy = (1 - H) * G * be22
        divx = grad_central_x_2nd(Sxx, dx) + grad_central_y_2nd(Sxy, dy)
        divy = grad_central_x_2nd(Sxy, dx) + grad_central_y_2nd(Syy, dy)
        fu = np.zeros((N, N + 1)); fu[:, 1:-1] = 0.5 * (divx[:, 1:] + divx[:, :-1])
        fv = np.zeros((N + 1, N)); fv[1:-1, :] = 0.5 * (divy[1:, :] + divy[:-1, :])
        ustar, vstar = momentum_predictor(u, v, nu, dx, dy, dt, U_lid, fu=fu, fv=fv, rho=rho)
        u, v, p = project(ustar, vstar, dx, dy, dt, rho, eig)
        t += dt

        sm = phi <= 0
        lam = float(_lam_max(be11, be12, be22)[sm].max()) if sm.any() else np.nan
        if not np.all(np.isfinite(u)) or not sm.any() or lam > 1e4:
            print(f"  [stopped at step {step}, t={t:.3f}: lam_max={lam:.2e}]");
            hist.append((t, lam, float(np.max(np.abs(u))))); break
        if step % 200 == 0 or t >= t_end:
            hist.append((t, lam, float(np.max(np.abs(u)))))
        if step % 1000 == 0:
            print(f"  step {step:5d} t={t:6.3f} lam_max(b_e)={lam:8.3f} max|u|={np.max(np.abs(u)):.2f}")

    out_dir = os.path.join(out_root, "mac_viscoelastic_disc"); os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"hist_{tag}.npz"),
             t=np.array([h[0] for h in hist]), lam=np.array([h[1] for h in hist]),
             umax=np.array([h[2] for h in hist]))
    print(f"[viscoelastic-disc] {tag}: reached t={t:.2f}, final lam_max={hist[-1][1]:.3f}")
    return t, hist


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    t_end = float(sys.argv[2]) if len(sys.argv) > 2 else 18.0
    tau = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    run(N=N, t_end=t_end, tau=tau)
