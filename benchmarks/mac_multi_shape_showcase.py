"""Showcase: FIVE soft shapes (square, circle, smooth-cross, soft bar, smooth
triangle) stirred by a lid-driven cavity, with two render modes.

  render_mode='speed'  -> speed field background + filled shapes (the classic view,
                          now with 5 shapes and guaranteed non-overlapping start so
                          the collisions/contact read clearly).
  render_mode='stress' -> faded velocity STREAMLINES + each shape coloured by its
                          von Mises STRESS, with its reference-map grid (dashed)
                          overlaid -- shows where the solids are loaded as they collide.

Same MAC solver, exact projection, stress-based contact + wall contact as the other
demos.  The simulation runs ONCE and both views are rendered from the stored frames
(no need to simulate twice).  Launch:
    python benchmarks/mac_multi_shape_showcase.py 256 12.0          # both views
    python benchmarks/mac_multi_shape_showcase.py 256 12.0 stress   # one view only
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
                                            shape_bar, shape_triangle)


def _check_no_overlap(inits, Xc, Yc, eps):
    """Assert the shapes' solid regions (plus a one-band guard) do not overlap at t=0."""
    masks = [(pin(Xc, Yc) <= eps) for pin in inits]          # inside + contact band
    bad = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if np.logical_and(masks[i], masks[j]).any():
                bad.append((i, j))
    if bad:
        raise SystemExit(f"[showcase] initial shapes overlap (within contact band): pairs {bad} "
                         f"-- adjust placement/sizes.")
    print("[showcase] initial placement OK: no shape overlaps (incl. contact band).")


def run(N=256, t_end=12.0, render_modes=('speed', 'stress'), U_lid=0.7, mu_s=2.5,
        mu_f=0.01, rho=1.0, eta=2.5, eta_wall=3.0, frame_dt=0.1, out_root="outputs"):
    if isinstance(render_modes, str):
        render_modes = (render_modes,)
    dx, dy = mac_grid(N, N)
    xc = (np.arange(N) + 0.5) * dx
    Xc, Yc = np.meshgrid(xc, xc)
    Xg, Yg = np.meshgrid(np.arange(N) * dx, np.arange(N) * dy)
    w_t = 2.0 * dx; nu = mu_f / rho; eps = 3.0 * dx
    phi_wall = np.minimum(np.minimum(Xc, 1.0 - Xc), np.minimum(Yc, 1.0 - Yc))

    # five shapes, spaced so none overlap (checked below). All rounded -> graceful.
    shapes = [shape_square(0.24, 0.70, 0.072, r=0.026),
              shape_circle(0.50, 0.73, 0.076),
              shape_triangle(0.77, 0.67, 0.090, r=0.030),
              shape_cross(0.30, 0.38, 0.082, 0.032, r=0.022, k=0.040),
              shape_bar(0.66, 0.39, 0.098, 0.044, r=0.032)]
    names = ["square", "circle", "triangle", "cross", "bar"]
    cols = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    inits = [s[0] for s in shapes]
    _check_no_overlap(inits, Xc, Yc, eps)

    refs = []
    for pin in inits:
        phi = pin(Xc, Yc); m = (phi <= 0).astype(float)
        X1, X2 = extrapolate_reference_map(Xc * m, Yc * m, phi, dx, dy, 3)
        refs.append([X1, X2])

    u = np.zeros((N, N + 1)); v = np.zeros((N + 1, N))
    eig = poisson_eigs_neumann(N, N, dx, dy)
    cs = np.sqrt(mu_s / rho)
    dt = min(0.3 * dx / U_lid, 0.2 * dx * dx / nu, 0.3 * dx / (cs + 1e-9))
    print(f"[showcase] N={N} modes={list(render_modes)} shapes={names} mu_s={mu_s} t_end={t_end} dt={dt:.2e}")

    def vonmises(sxx, sxy, syy):
        return np.sqrt(np.maximum(sxx*sxx - sxx*syy + syy*syy + 3.0*sxy*sxy, 0.0))

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
        vm = np.zeros((N, N)); Jmin = 1.0; Jmax = 1.0
        for k in range(len(refs)):
            sxx, sxy, syy, J = solid_cauchy_stress(refs[k][0], refs[k][1], dx, dy, mu_s, 0.0, phis[k])
            H = smoothed_heaviside(phis[k], w_t)
            Sxx += (1 - H) * sxx; Sxy += (1 - H) * sxy; Syy += (1 - H) * syy
            inside = phis[k] <= 0
            vm = np.where(inside, vonmises(sxx, sxy, syy), vm)         # per-shape, no overlap
            Jmin = min(Jmin, J.min()); Jmax = max(Jmax, J.max())
        for i in range(len(phis)):
            for j in range(i + 1, len(phis)):
                txx, txy, tyy = contact_stress(phis[i], phis[j], eta, 2 * mu_s, eps, dx, dy)
                Sxx += txx; Sxy += txy; Syy += tyy
            if eta_wall > 0:
                txx, txy, tyy = contact_stress(phis[i], phi_wall, eta_wall, 2 * mu_s, eps, dx, dy)
                Sxx += txx; Sxy += txy; Syy += tyy

        divx = grad_central_x_2nd(Sxx, dx) + grad_central_y_2nd(Sxy, dy)
        divy = grad_central_x_2nd(Sxy, dx) + grad_central_y_2nd(Syy, dy)
        fu = np.zeros((N, N + 1)); fu[:, 1:-1] = 0.5 * (divx[:, 1:] + divx[:, :-1])
        fv = np.zeros((N + 1, N)); fv[1:-1, :] = 0.5 * (divy[1:, :] + divy[:-1, :])
        ustar, vstar = momentum_predictor(u, v, nu, dx, dy, dt, U_lid, fu=fu, fv=fv, rho=rho)
        u, v, p = project(ustar, vstar, dx, dy, dt, rho, eig)
        t += dt

        if not np.all(np.isfinite(u)) or Jmin < 0.0 or Jmax > 25.0 or any(not (pp <= 0).any() for pp in phis):
            print(f"  [stopped at step {step}, t={t:.3f}: minJ={Jmin:.2f} maxJ={Jmax:.2f}]"); break
        if t >= next_frame:
            uc = 0.5 * (u[:, :-1] + u[:, 1:]); vc = 0.5 * (v[:-1, :] + v[1:, :])
            X1s = [refs[k][0].copy() for k in range(len(refs))]
            X2s = [refs[k][1].copy() for k in range(len(refs))]
            frames.append((t, [pp.copy() for pp in phis], X1s, X2s,
                           np.sqrt(uc**2 + vc**2), uc.copy(), vc.copy(), vm.copy()))
            next_frame += frame_dt
        if step % 200 == 0:
            print(f"  step {step:5d} t={t:5.2f} minJ={Jmin:.2f} maxJ={Jmax:.2f} "
                  f"vm_max={vm.max():.2f} frames={len(frames)}")

    # ── render (one simulation, both views) ──
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    glv = np.arange(0.0, 1.0001, 0.025)
    vmax_stress = max((fr[7].max() for fr in frames), default=1.0) or 1.0
    gifs = []
    for render_mode in render_modes:
        print(f"[showcase] rendering {len(frames)} frames ({render_mode}) -> GIF ...")
        out_dir = os.path.join(out_root, f"mac_showcase_{render_mode}_N{N}"); os.makedirs(out_dir, exist_ok=True)
        imgs = []
        for (tt, pps, X1s, X2s, spd, uc, vc, vmf) in frames:
            fig, ax = plt.subplots(figsize=(5.6, 5.2), dpi=110)
            if render_mode == 'speed':
                im = ax.imshow(spd, origin="lower", extent=[0, 1, 0, 1], cmap="viridis",
                               vmin=0, vmax=max(U_lid, 0.6), interpolation="bilinear")
                for k, pp in enumerate(pps):
                    ax.contourf(Xc, Yc, pp, levels=[-1e9, 0.0], colors=[cols[k]], alpha=0.92)
                    ax.contour(Xc, Yc, pp, levels=[0.0], colors=["white"], linewidths=1.1)
                cb_label = "|u|"
            else:  # stress: faded streamlines + shapes coloured by von Mises + reference-map grid
                ax.set_facecolor("#0b0b0b")
                ax.streamplot(xc, xc, uc, vc, density=1.1, color="#9ecbff",
                              linewidth=0.6, arrowsize=0.5)
                vmm = np.where(np.stack([pp <= 0 for pp in pps]).any(0), vmf, np.nan)
                im = ax.imshow(np.ma.masked_invalid(vmm), origin="lower", extent=[0, 1, 0, 1],
                               cmap="inferno", vmin=0, vmax=vmax_stress, interpolation="bilinear")
                for k, pp in enumerate(pps):
                    ax.contour(Xc, Yc, pp, levels=[0.0], colors=["white"], linewidths=1.3)
                    inside = pp <= 0
                    X1m = np.where(inside, X1s[k], np.nan); X2m = np.where(inside, X2s[k], np.nan)
                    ax.contour(Xc, Yc, X1m, levels=glv, colors=["#ffffff"], linestyles="dashed", linewidths=0.5, alpha=0.6)
                    ax.contour(Xc, Yc, X2m, levels=glv, colors=["#ffffff"], linestyles="dashed", linewidths=0.5, alpha=0.6)
                cb_label = "von Mises stress"
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"t = {tt:4.2f}   ({render_mode})")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=cb_label); fig.tight_layout(pad=0.4)
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            imgs.append(np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3].copy())
            plt.close(fig)
        gif = os.path.join(out_dir, f"showcase_{render_mode}.gif")
        imageio.mimsave(gif, imgs, duration=0.08, loop=0)
        print(f"  saved {gif}  ({len(imgs)} frames)")
        gifs.append(gif)
    return gifs


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    t_end = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
    modes = (sys.argv[3],) if len(sys.argv) > 3 else ('speed', 'stress')
    run(N=N, t_end=t_end, render_modes=modes)
