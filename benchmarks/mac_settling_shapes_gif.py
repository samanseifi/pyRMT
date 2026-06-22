"""Six soft shapes in a closed no-slip box: an initial swirl (divergence-free
vortex) sets them moving, viscosity bleeds off the swirl, and GRAVITY settles
them into a pile on the floor -- colliding and resting on each other (and the
walls) via the Rycroft contact stress. MAC staggered grid, exact projection.

No moving lid. The drive is (1) an initial velocity field and (2) a downward
body force on the solid regions (the excess weight of the denser solids over the
displaced fluid -- a single-density "reduced gravity" that makes them fall while
the incompressible projection pushes fluid up out of the way).

Shapes: square, circle, smooth-cross, soft bar, ellipse, hexagon (all rounded,
so the reference map deforms gracefully -- see the triangle/bar resolution study).

Usage: python benchmarks/mac_settling_shapes_gif.py [N] [t_end]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.mac import (mac_grid, momentum_predictor, project, poisson_eigs_neumann,
                       divergence, contact_stress)
from pyRMT.functions import (extrapolate_reference_map, advect_reference_map,
    rebuild_phi_from_reference_map, solid_cauchy_stress, smoothed_heaviside,
    grad_central_x_2nd, grad_central_y_2nd)
from benchmarks.mac_multi_shape_gif import (shape_square, shape_circle, shape_cross,
                                            shape_bar, shape_ellipse, shape_hexagon)


def run(N=160, t_end=10.0, g=1.6, swirl=0.22, mu_s=1.5, mu_f=0.02, rho=1.0,
        eta=2.5, eta_wall=3.0, frame_dt=0.08, out_root="outputs"):
    dx, dy = mac_grid(N, N)
    xc = (np.arange(N) + 0.5) * dx
    Xc, Yc = np.meshgrid(xc, xc)
    Xg, Yg = np.meshgrid(np.arange(N) * dx, np.arange(N) * dy)
    w_t = 2.0 * dx; nu = mu_f / rho; eps = 3.0 * dx
    phi_wall = np.minimum(np.minimum(Xc, 1.0 - Xc), np.minimum(Yc, 1.0 - Yc))

    # six shapes, scattered in the upper 2/3 of the box (they fall and pile up)
    shapes = [shape_square(0.22, 0.78, 0.072, r=0.026),
              shape_circle(0.46, 0.82, 0.078),
              shape_cross(0.74, 0.78, 0.090, 0.036, r=0.022, k=0.040),
              shape_bar(0.30, 0.56, 0.100, 0.044, r=0.032),
              shape_ellipse(0.58, 0.55, 0.100, 0.060),
              shape_hexagon(0.80, 0.52, 0.078, rr=0.018)]
    names = ["square", "circle", "cross", "bar", "ellipse", "hexagon"]
    cols = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    inits = [s[0] for s in shapes]
    refs = []
    for pin in inits:
        phi = pin(Xc, Yc); m = (phi <= 0).astype(float)
        X1, X2 = extrapolate_reference_map(Xc * m, Yc * m, phi, dx, dy, 3)
        refs.append([X1, X2])

    # initial divergence-free swirl: psi = swirl * sin(pi x) sin(pi y) (vanishes on
    # walls -> no normal flow). u = d psi/dy, v = -d psi/dx.  (one big vortex)
    xf = np.arange(N + 1) * dx
    Xu, Yu = np.meshgrid(xf, xc)                       # u at x-faces
    u = swirl * np.pi * np.sin(np.pi * Xu) * np.cos(np.pi * Yu)
    yf = np.arange(N + 1) * dy
    Xv, Yv = np.meshgrid(xc, yf)                       # v at y-faces
    v = -swirl * np.pi * np.cos(np.pi * Xv) * np.sin(np.pi * Yv)
    u[:, 0] = 0.0; u[:, -1] = 0.0; v[0, :] = 0.0; v[-1, :] = 0.0

    eig = poisson_eigs_neumann(N, N, dx, dy)
    cs = np.sqrt(mu_s / rho)
    u, v, _ = project(u, v, dx, dy, 1.0, rho, eig)     # clean up the initial field
    dt = min(0.25 * dx / 1.2, 0.2 * dx * dx / nu, 0.3 * dx / (cs + 1e-9))
    out_dir = os.path.join(out_root, f"mac_settling_N{N}"); os.makedirs(out_dir, exist_ok=True)
    print(f"[settling] N={N} shapes={names} g={g} swirl={swirl} mu_s={mu_s} "
          f"t_end={t_end} dt={dt:.2e}")

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
        solid_frac = np.zeros((N, N)); Jmin = 1.0; Jmax = 1.0
        for k in range(len(refs)):
            sxx, sxy, syy, J = solid_cauchy_stress(refs[k][0], refs[k][1], dx, dy, mu_s, 0.0, phis[k])
            H = smoothed_heaviside(phis[k], w_t)
            Sxx += (1 - H) * sxx; Sxy += (1 - H) * sxy; Syy += (1 - H) * syy
            solid_frac += (1 - H)
            Jmin = min(Jmin, J.min()); Jmax = max(Jmax, J.max())
        for i in range(len(phis)):
            for j in range(i + 1, len(phis)):
                txx, txy, tyy = contact_stress(phis[i], phis[j], eta, 2 * mu_s, eps, dx, dy)
                Sxx += txx; Sxy += txy; Syy += tyy
            if eta_wall > 0:
                txx, txy, tyy = contact_stress(phis[i], phi_wall, eta_wall, 2 * mu_s, eps, dx, dy)
                Sxx += txx; Sxy += txy; Syy += tyy
        solid_frac = np.clip(solid_frac, 0.0, 1.0)

        divx = grad_central_x_2nd(Sxx, dx) + grad_central_y_2nd(Sxy, dy)
        divy = grad_central_x_2nd(Sxy, dx) + grad_central_y_2nd(Syy, dy)
        fu = np.zeros((N, N + 1)); fu[:, 1:-1] = 0.5 * (divx[:, 1:] + divx[:, :-1])
        fv = np.zeros((N + 1, N)); fv[1:-1, :] = 0.5 * (divy[1:, :] + divy[:-1, :])
        # gravity: downward body force on the solid regions (excess weight)
        sf_v = 0.5 * (solid_frac[1:, :] + solid_frac[:-1, :])
        fv[1:-1, :] += -g * sf_v
        ustar, vstar = momentum_predictor(u, v, nu, dx, dy, dt, 0.0, fu=fu, fv=fv, rho=rho)
        u, v, p = project(ustar, vstar, dx, dy, dt, rho, eig)
        t += dt

        if not np.all(np.isfinite(u)) or Jmin < 0.0 or Jmax > 25.0 or any(not (pp <= 0).any() for pp in phis):
            print(f"  [stopped at step {step}, t={t:.3f}: minJ={Jmin:.2f} maxJ={Jmax:.2f}]")
            break
        if t >= next_frame:
            uc = 0.5 * (u[:, :-1] + u[:, 1:]); vc = 0.5 * (v[:-1, :] + v[1:, :])
            frames.append((t, [pp.copy() for pp in phis], np.sqrt(uc**2 + vc**2)))
            next_frame += frame_dt
        if step % 300 == 0:
            cy = [float((Yc[pp <= 0]).mean()) if (pp <= 0).any() else np.nan for pp in phis]
            print(f"  step {step:5d} t={t:5.2f} minJ={Jmin:.2f} maxJ={Jmax:.2f} "
                  f"max|u|={np.max(np.abs(u)):.2f} <y>=[{','.join(f'{c:.2f}' for c in cy)}] "
                  f"frames={len(frames)}")

    print(f"[settling] reached t={t:.2f}, rendering {len(frames)} frames -> GIF ...")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    imgs = []
    for (tt, pps, spd) in frames:
        fig, ax = plt.subplots(figsize=(5.6, 5.2), dpi=110)
        im = ax.imshow(spd, origin="lower", extent=[0, 1, 0, 1], cmap="viridis",
                       vmin=0, vmax=0.8, interpolation="bilinear")
        for k, pp in enumerate(pps):
            ax.contourf(Xc, Yc, pp, levels=[-1e9, 0.0], colors=[cols[k % len(cols)]], alpha=0.92)
            ax.contour(Xc, Yc, pp, levels=[0.0], colors=["white"], linewidths=1.1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t = {tt:4.2f}   (swirl -> settle under gravity)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="|u|"); fig.tight_layout(pad=0.4)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        imgs.append(np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3].copy())
        plt.close(fig)
    gif = os.path.join(out_dir, "settling_shapes.gif")
    imageio.mimsave(gif, imgs, duration=0.07, loop=0)
    print(f"  saved {gif}  ({len(imgs)} frames)")
    return t, gif


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 160
    t_end = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    run(N=N, t_end=t_end)
