"""Several BIGGER soft discs in a lid-driven cavity, with reference-map
REINITIALIZATION (Rycroft 2018, F-storage) so they deform under sustained shear
indefinitely without the reference map folding -> a long animated GIF.

This is the *folding-limited* multi-body demo (contrast mac_multi_shape_reinit_gif,
which is crushing-limited by small stiff shapes colliding). Here the discs are
well resolved (R~0.14, ~18 cells across at N=128) and soft (mu_s=0.3), so the
deformation is smooth shear -- exactly the regime where reinit pays off. Same
proven-stable manual stress path as benchmarks/mac_reinit_test.py (which reached
t=18 on a single disc), extended to N discs with all-pairs + wall contact.

Run WITH and WITHOUT reinit to see the difference (reinit=0 folds; reinit=1 runs on).

Usage: python benchmarks/mac_multi_disc_reinit_gif.py [N] [t_end] [reinit 0/1]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.mac import (mac_grid, momentum_predictor, project, poisson_eigs_neumann,
                       divergence, contact_stress)
from pyRMT.functions import (extrapolate_reference_map, advect_reference_map,
    smoothed_heaviside, grad_central_x_2nd, grad_central_y_2nd, reinitialize_phi_fmm)
from pyRMT.interpolators import bilinear_interpolate


def _Fc(X1, X2, dx, dy):
    g11 = grad_central_x_2nd(X1, dx); g12 = grad_central_y_2nd(X1, dy)
    g21 = grad_central_x_2nd(X2, dx); g22 = grad_central_y_2nd(X2, dy)
    dG = g11 * g22 - g12 * g21
    dG = np.where(np.abs(dG) < 1e-9, 1e-9, dG)
    return g22 / dG, -g12 / dG, -g21 / dG, g11 / dG, 1.0 / dG


def _mm(A, B):
    return [A[0]*B[0] + A[1]*B[2], A[0]*B[1] + A[1]*B[3],
            A[2]*B[0] + A[3]*B[2], A[2]*B[1] + A[3]*B[3]]


def run(N=128, t_end=16.0, reinit=True, U_lid=1.0, mu_s=0.3, mu_f=0.01, rho=1.0,
        eta=2.0, eta_wall=2.5, reinit_J=1.7, frame_dt=0.12, out_root="outputs"):
    dx, dy = mac_grid(N, N)
    xc = (np.arange(N) + 0.5) * dx
    Xc, Yc = np.meshgrid(xc, xc)
    Xg, Yg = np.meshgrid(np.arange(N) * dx, np.arange(N) * dy)
    w_t = 2.0 * dx; nu = mu_f / rho; eps = 3.0 * dx
    phi_wall = np.minimum(np.minimum(Xc, 1.0 - Xc), np.minimum(Yc, 1.0 - Yc))
    discs = [(0.32, 0.58, 0.14), (0.63, 0.62, 0.15), (0.50, 0.31, 0.13)]
    cols = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]

    sols = []
    for (cx, cy, R) in discs:
        phi = np.sqrt((Xc - cx)**2 + (Yc - cy)**2) - R
        m = (phi <= 0).astype(float)
        X1, X2 = extrapolate_reference_map(Xc * m, Yc * m, phi, dx, dy, 3)
        sols.append(dict(X1=X1, X2=X2,
                         F=[np.ones((N, N)), np.zeros((N, N)), np.zeros((N, N)), np.ones((N, N))],
                         pref=phi.copy()))

    u = np.zeros((N, N + 1)); v = np.zeros((N + 1, N))
    eig = poisson_eigs_neumann(N, N, dx, dy)
    cs = np.sqrt(mu_s / rho)
    dt = min(0.3 * dx / U_lid, 0.2 * dx * dx / nu, 0.3 * dx / (cs + 1e-9))
    tag = "reinit" if reinit else "noreinit"
    out_dir = os.path.join(out_root, f"mac_multi_disc_{tag}_N{N}"); os.makedirs(out_dir, exist_ok=True)
    print(f"[multi-disc] N={N} reinit={reinit} mu_s={mu_s} eta={eta} reinit_J={reinit_J} "
          f"t_end={t_end} dt={dt:.2e}")

    def geom(s):
        return bilinear_interpolate(s['pref'], s['X1'] - 0.5*dx, s['X2'] - 0.5*dy, dx, dy, N, N)

    frames = []; next_frame = 0.0; nre = 0
    t = 0.0; step = 0
    while t < t_end:
        step += 1
        if t + dt > t_end:
            dt = t_end - t
        u_c = 0.5 * (u[:, :-1] + u[:, 1:]); v_c = 0.5 * (v[:-1, :] + v[1:, :])
        phis = []
        for s in sols:
            phi = geom(s); m = (phi <= 0).astype(float)
            s['X1'] = advect_reference_map(s['X1'], u_c, v_c, Xg, Yg, dt, dx, dy, phi, 'semilagrangian', 0.0) * m
            s['X2'] = advect_reference_map(s['X2'], u_c, v_c, Xg, Yg, dt, dx, dy, phi, 'semilagrangian', 0.0) * m
            for i in range(4):
                s['F'][i] = advect_reference_map(s['F'][i], u_c, v_c, Xg, Yg, dt, dx, dy, phi, 'semilagrangian', 0.0) * m
            s['X1'], s['X2'] = extrapolate_reference_map(s['X1'], s['X2'], phi, dx, dy, 3)
            s['F'][0], s['F'][1] = extrapolate_reference_map(s['F'][0], s['F'][1], phi, dx, dy, 3)
            s['F'][2], s['F'][3] = extrapolate_reference_map(s['F'][2], s['F'][3], phi, dx, dy, 3)
            phis.append(geom(s))

        Sxx = np.zeros((N, N)); Sxy = np.zeros((N, N)); Syy = np.zeros((N, N))
        Fcs = []; Jworst = 1.0
        for k, s in enumerate(sols):
            Fc = _Fc(s['X1'], s['X2'], dx, dy); Fcs.append(Fc)
            T = _mm(Fc[:4], s['F'])                       # total deformation
            b11 = T[0]**2 + T[1]**2; b12 = T[0]*T[2] + T[1]*T[3]; b22 = T[2]**2 + T[3]**2
            H = smoothed_heaviside(phis[k], w_t)
            Sxx += (1 - H) * mu_s * b11; Sxy += (1 - H) * mu_s * b12; Syy += (1 - H) * mu_s * b22
            sm = phis[k] <= 0
            if sm.any():
                jin = Fc[4][sm]
                Jworst = max(Jworst, jin.max(), 1.0 / max(jin.min(), 1e-6))
        for i in range(len(phis)):
            for j in range(i + 1, len(phis)):
                tx, ty, tz = contact_stress(phis[i], phis[j], eta, 2 * mu_s, eps, dx, dy)
                Sxx += tx; Sxy += ty; Syy += tz
            if eta_wall > 0:
                tx, ty, tz = contact_stress(phis[i], phi_wall, eta_wall, 2 * mu_s, eps, dx, dy)
                Sxx += tx; Sxy += ty; Syy += tz

        divx = grad_central_x_2nd(Sxx, dx) + grad_central_y_2nd(Sxy, dy)
        divy = grad_central_x_2nd(Sxy, dx) + grad_central_y_2nd(Syy, dy)
        fu = np.zeros((N, N + 1)); fu[:, 1:-1] = 0.5 * (divx[:, 1:] + divx[:, :-1])
        fv = np.zeros((N + 1, N)); fv[1:-1, :] = 0.5 * (divy[1:, :] + divy[:-1, :])
        ustar, vstar = momentum_predictor(u, v, nu, dx, dy, dt, U_lid, fu=fu, fv=fv, rho=rho)
        u, v, p = project(ustar, vstar, dx, dy, dt, rho, eig)
        t += dt

        if reinit:                                        # per-disc reinit on large epoch distortion
            for k, s in enumerate(sols):
                sm = phis[k] <= 0
                if not sm.any():
                    continue
                jin = Fcs[k][4][sm]
                if max(jin.max(), 1.0 / max(jin.min(), 1e-6)) > reinit_J:
                    s['F'] = _mm(Fcs[k][:4], s['F'])
                    s['F'][0], s['F'][1] = extrapolate_reference_map(s['F'][0], s['F'][1], phis[k], dx, dy, 3)
                    s['F'][2], s['F'][3] = extrapolate_reference_map(s['F'][2], s['F'][3], phis[k], dx, dy, 3)
                    s['pref'] = reinitialize_phi_fmm(phis[k].copy(), dx, dy)
                    s['X1'] = Xc.copy(); s['X2'] = Yc.copy()
                    nre += 1

        if not np.all(np.isfinite(u)) or any(not (pp <= 0).any() for pp in phis):
            print(f"  [stopped at step {step}, t={t:.3f}; reinits={nre} Jworst={Jworst:.2f}]"); break
        if t >= next_frame:
            uc = 0.5 * (u[:, :-1] + u[:, 1:]); vc = 0.5 * (v[:-1, :] + v[1:, :])
            frames.append((t, [pp.copy() for pp in phis], np.sqrt(uc**2 + vc**2)))
            next_frame += frame_dt
        if step % 300 == 0:
            print(f"  step {step:5d} t={t:5.2f} reinits={nre} Jworst={Jworst:.2f} "
                  f"max|u|={np.max(np.abs(u)):.2f} frames={len(frames)}")

    print(f"[multi-disc] reached t={t:.2f}, {nre} reinits, rendering {len(frames)} frames ...")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    imgs = []
    for (tt, pps, spd) in frames:
        fig, ax = plt.subplots(figsize=(5.6, 5.2), dpi=110)
        im = ax.imshow(spd, origin="lower", extent=[0, 1, 0, 1], cmap="viridis",
                       vmin=0, vmax=1, interpolation="bilinear")
        for k, pp in enumerate(pps):
            ax.contourf(Xc, Yc, pp, levels=[-1e9, 0.0], colors=[cols[k % len(cols)]], alpha=0.9)
            ax.contour(Xc, Yc, pp, levels=[0.0], colors=["white"], linewidths=1.2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t = {tt:4.2f}   ({tag}; colour=speed)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="|u|"); fig.tight_layout(pad=0.4)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        imgs.append(np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3].copy())
        plt.close(fig)
    gif = os.path.join(out_dir, f"multi_disc_{tag}.gif")
    imageio.mimsave(gif, imgs, duration=0.08, loop=0)
    print(f"  saved {gif}  ({len(imgs)} frames)")
    return t, nre, gif


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    t_end = float(sys.argv[2]) if len(sys.argv) > 2 else 16.0
    reinit = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True
    run(N=N, t_end=t_end, reinit=reinit)
