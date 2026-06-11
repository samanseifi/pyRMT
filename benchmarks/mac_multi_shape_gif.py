"""Three different-shaped soft solids (square, circle, smooth cross) stirred by a
lid-driven cavity, colliding via the contact STRESS -- rendered as an animated GIF
with the velocity field overlaid (MAC grid).

Each solid carries its own reference map + level set (any shape works -- phi_init is
just the analytic shape SDF). Blended solid stress + all-pairs Rycroft contact stress
-> face force; lid-driven momentum + exact projection.

Usage: python benchmarks/mac_multi_shape_gif.py [N] [t_end]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.mac import (mac_grid, momentum_predictor, project, poisson_eigs_neumann,
                       divergence, contact_stress)
from pyRMT.functions import (extrapolate_reference_map, advect_reference_map,
    rebuild_phi_from_reference_map, solid_cauchy_stress, smoothed_heaviside,
    grad_central_x_2nd, grad_central_y_2nd)


# ── shape signed-distance functions (phi < 0 inside) ─────────────────────────
def shape_circle(cx, cy, R):
    return (lambda X, Y: np.sqrt((X - cx)**2 + (Y - cy)**2) - R), R

def shape_square(cx, cy, h, r=0.02):
    # lightly-rounded square: still visually a square, but the corners aren't
    # infinitely sharp reference-map singularities (which fold fast on a 128 grid)
    return (lambda X, Y: _rbox(X, Y, cx, cy, h, h, r)), h

def _rbox(X, Y, cx, cy, bx, by, r):
    qx = np.abs(X - cx) - bx + r; qy = np.abs(Y - cy) - by + r
    return (np.sqrt(np.maximum(qx, 0)**2 + np.maximum(qy, 0)**2)
            + np.minimum(np.maximum(qx, qy), 0) - r)

def _smin(a, b, k):                      # smooth union (rounds the inner corners)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)

def shape_cross(cx, cy, arm, width, r=0.02, k=0.03):
    def f(X, Y):
        return _smin(_rbox(X, Y, cx, cy, arm, width, r),
                     _rbox(X, Y, cx, cy, width, arm, r), k)
    return f, arm


def run(N=128, t_end=6.0, U_lid=1.0, mu_s=1.2, mu_f=0.01, rho=1.0, eta=3.0,
        frame_dt=0.1, out_root="outputs"):
    dx, dy = mac_grid(N, N)
    xc = (np.arange(N) + 0.5) * dx
    Xc, Yc = np.meshgrid(xc, xc)
    Xg, Yg = np.meshgrid(np.arange(N) * dx, np.arange(N) * dy)
    w_t = 2.0 * dx; nu = mu_f / rho; eps = 3.0 * dx

    shapes = [shape_square(0.40, 0.66, 0.085),
              shape_circle(0.62, 0.62, 0.090),
              shape_cross(0.48, 0.40, 0.105, 0.042)]
    names = ["square", "circle", "smooth cross"]
    inits = [s[0] for s in shapes]
    refs = []
    for pin in inits:
        phi = pin(Xc, Yc); m = (phi <= 0).astype(float)
        X1, X2 = extrapolate_reference_map(Xc * m, Yc * m, phi, dx, dy, 3)
        refs.append([X1, X2])

    u = np.zeros((N, N + 1)); v = np.zeros((N + 1, N))
    eig = poisson_eigs_neumann(N, N, dx, dy)
    cs = np.sqrt(mu_s / rho)
    dt = min(0.3 * dx / U_lid, 0.2 * dx * dx / nu, 0.3 * dx / (cs + 1e-9))
    out_dir = os.path.join(out_root, f"mac_multi_shape_N{N}"); os.makedirs(out_dir, exist_ok=True)
    print(f"[multi-shape] N={N} solids={names} mu_s={mu_s} eta={eta} t_end={t_end} dt={dt:.2e}")

    frames = []; next_frame = 0.0
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

        if not np.all(np.isfinite(u)) or Jmin < 0.0 or Jmax > 25.0 or any(not (pp <= 0).any() for pp in phis):
            print(f"  [stopped at step {step}, t={t:.3f}: minJ={Jmin:.2f} maxJ={Jmax:.2f}]")
            break
        if t >= next_frame:
            uc = 0.5 * (u[:, :-1] + u[:, 1:]); vc = 0.5 * (v[:-1, :] + v[1:, :])
            frames.append((t, [pp.copy() for pp in phis], uc.copy(), vc.copy()))
            next_frame += frame_dt
        if step % 200 == 0:
            print(f"  step {step:5d} t={t:5.2f} minJ={Jmin:.2f} maxJ={Jmax:.2f} max|u|={np.max(np.abs(u)):.2f} frames={len(frames)}")

    # ── render GIF: filled solids (coloured) + velocity quiver ──
    print(f"[multi-shape] rendering {len(frames)} frames -> GIF ...")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    cols = ["#d62728", "#1f77b4", "#2ca02c"]
    s = max(1, N // 22); sl = (slice(None, None, s), slice(None, None, s))
    Xq, Yq = Xc[sl], Yc[sl]
    imgs = []
    for (tt, pps, uc, vc) in frames:
        fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=110)
        spd = np.sqrt(uc**2 + vc**2)
        ax.streamplot(xc, xc, uc, vc, density=1.0, color=spd, cmap="Greys",
                      linewidth=0.6, arrowsize=0.6)
        for k, pp in enumerate(pps):
            ax.contourf(Xc, Yc, pp, levels=[-1e9, 0.0], colors=[cols[k]], alpha=0.92)
            ax.contour(Xc, Yc, pp, levels=[0.0], colors=["k"], linewidths=1.0)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"t = {tt:4.2f}")
        fig.tight_layout(pad=0.4)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
        imgs.append(img.copy()); plt.close(fig)
    gif = os.path.join(out_dir, "multi_shape_lid.gif")
    imageio.mimsave(gif, imgs, duration=0.08, loop=0)
    print(f"  saved {gif}  ({len(imgs)} frames)")
    return gif


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    t_end = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
    run(N=N, t_end=t_end)
