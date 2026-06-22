"""Showcase: SIX soft shapes (square, circle, smooth-triangle, smooth-cross, soft
bar, rounded trapezoid) stirred by a lid-driven cavity, with two render modes.

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
                                            shape_bar, shape_triangle, shape_trapezoid)


def _check_no_overlap(inits, Xc, Yc, eps):
    """Hard-fail if any two solids' INTERIORS overlap at t=0 (resolution-independent);
    only NOTE if they merely start within the contact band (band width ~3 dx grows on
    coarse grids, so band-proximity is not an error)."""
    phis = [pin(Xc, Yc) for pin in inits]
    interior = [(p <= 0.0) for p in phis]
    band = [(p <= eps) for p in phis]
    hard, soft = [], []
    for i in range(len(phis)):
        for j in range(i + 1, len(phis)):
            if np.logical_and(interior[i], interior[j]).any():
                hard.append((i, j))
            elif np.logical_and(band[i], band[j]).any():
                soft.append((i, j))
    if hard:
        raise SystemExit(f"[showcase] initial shapes OVERLAP (interiors): {hard} -- fix placement.")
    if soft:
        print(f"[showcase] note: pairs within contact band at t=0 (no interior overlap): {soft}")
    print("[showcase] initial placement OK: no interior overlap.")


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

    # six shapes in a non-overlapping 2x3 layout (checked below). ALL rounded
    # (no sharp corners) and gentle lid -> they deform, collide, but don't fall apart.
    shapes = [shape_square(0.22, 0.72, 0.068, r=0.024),
              shape_circle(0.50, 0.74, 0.072),
              shape_triangle(0.79, 0.71, 0.082, r=0.030),
              shape_cross(0.22, 0.40, 0.076, 0.030, r=0.022, k=0.038),
              shape_bar(0.50, 0.40, 0.090, 0.042, r=0.030),
              shape_trapezoid(0.79, 0.40, 0.082, 0.050, 0.058, r=0.022)]
    names = ["square", "circle", "triangle", "cross", "bar", "trapezoid"]
    cols = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
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

    # stream frames to disk during the sim (RAM is only ~7 GB here; holding ~120
    # float64 frames in memory OOM-kills the run -- so write each frame as it is made)
    save_dir = os.path.join(out_root, f"mac_showcase_N{N}")
    fr_dir = os.path.join(save_dir, "frames"); os.makedirs(fr_dir, exist_ok=True)
    ts = []; next_frame = 0.0; fidx = 0
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
        # solid (elastic) stress components, gated to the solids -- stored so the
        # stress measure (deviatoric von Mises) and colour range are render-time choices
        Ssxx = np.zeros((N, N)); Ssxy = np.zeros((N, N)); Ssyy = np.zeros((N, N))
        Jmin = 1.0; Jmax = 1.0
        for k in range(len(refs)):
            sxx, sxy, syy, J = solid_cauchy_stress(refs[k][0], refs[k][1], dx, dy, mu_s, 0.0, phis[k])
            H = smoothed_heaviside(phis[k], w_t)
            Sxx += (1 - H) * sxx; Sxy += (1 - H) * sxy; Syy += (1 - H) * syy
            inside = phis[k] <= 0                                       # no overlap -> where() ok
            Ssxx = np.where(inside, sxx, Ssxx); Ssxy = np.where(inside, sxy, Ssxy)
            Ssyy = np.where(inside, syy, Ssyy)
            Jmin = min(Jmin, J.min()); Jmax = max(Jmax, J.max())
        vmd = np.sqrt(3.0 * (0.25 * (Ssxx - Ssyy)**2 + Ssxy**2))       # deviatoric von Mises
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
            np.savez_compressed(
                os.path.join(fr_dir, f"f{fidx:04d}.npz"),
                phis=np.stack(phis).astype(np.float32),
                X1s=np.stack([refs[k][0] for k in range(len(refs))]).astype(np.float32),
                X2s=np.stack([refs[k][1] for k in range(len(refs))]).astype(np.float32),
                uc=uc.astype(np.float32), vc=vc.astype(np.float32),
                sxx=Ssxx.astype(np.float32), sxy=Ssxy.astype(np.float32), syy=Ssyy.astype(np.float32))
            ts.append(t); fidx += 1; next_frame += frame_dt
        if step % 200 == 0:
            print(f"  step {step:5d} t={t:5.2f} minJ={Jmin:.2f} maxJ={Jmax:.2f} "
                  f"vmDev_max={vmd.max():.2f} frames={fidx}")

    np.savez(os.path.join(save_dir, "meta.npz"), t=np.array(ts), U_lid=float(U_lid), nframes=fidx)
    print(f"[showcase] saved {fidx} frames -> {fr_dir}")
    return _render_frames(save_dir, render_modes, N, cols, out_root)


# ── rendering reads frames from disk one at a time (flat, low memory) ──

def _render_frames(save_dir, render_modes, N, cols, out_root="outputs",
                   vmax_stress=3.0, stream_color="#9ecbff"):
    if isinstance(render_modes, str):
        render_modes = (render_modes,)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    meta = np.load(os.path.join(save_dir, "meta.npz"))
    ts = meta['t']; U_lid = float(meta['U_lid']); nf = int(meta['nframes'])
    fr_dir = os.path.join(save_dir, "frames")
    dx = 1.0 / N; xc = (np.arange(N) + 0.5) * dx; Xc, Yc = np.meshgrid(xc, xc)
    glv = np.arange(0.0, 1.0001, 0.025)
    gifs = []
    for render_mode in render_modes:
        print(f"[showcase] rendering {nf} frames ({render_mode}, vmax={vmax_stress}) -> GIF ...")
        out_dir = os.path.join(out_root, f"mac_showcase_{render_mode}_N{N}"); os.makedirs(out_dir, exist_ok=True)
        imgs = []
        for i in range(nf):
            d = np.load(os.path.join(fr_dir, f"f{i:04d}.npz"))
            tt = float(ts[i]); pps = list(d['phis']); X1s = list(d['X1s']); X2s = list(d['X2s'])
            uc = d['uc']; vc = d['vc']; sxx = d['sxx']; sxy = d['sxy']; syy = d['syy']
            spd = np.sqrt(uc**2 + vc**2)
            vmf = np.sqrt(3.0 * (0.25 * (sxx - syy)**2 + sxy**2))   # deviatoric von Mises
            fig, ax = plt.subplots(figsize=(5.6, 5.2), dpi=110)
            if render_mode == 'speed':
                im = ax.imshow(spd, origin="lower", extent=[0, 1, 0, 1], cmap="viridis",
                               vmin=0, vmax=max(U_lid, 0.6), interpolation="bilinear")
                for k, pp in enumerate(pps):
                    ax.contourf(Xc, Yc, pp, levels=[-1e9, 0.0], colors=[cols[k]], alpha=0.92)
                    ax.contour(Xc, Yc, pp, levels=[0.0], colors=["white"], linewidths=1.1)
                    inside = pp <= 0                       # dashed reference-map grid (deforming)
                    X1m = np.where(inside, X1s[k], np.nan); X2m = np.where(inside, X2s[k], np.nan)
                    ax.contour(Xc, Yc, X1m, levels=glv, colors=["k"], linestyles="dashed", linewidths=0.45, alpha=0.65)
                    ax.contour(Xc, Yc, X2m, levels=glv, colors=["k"], linestyles="dashed", linewidths=0.45, alpha=0.65)
                cb_label = "|u|"
            elif render_mode == 'stress':  # faded streamlines + SOLID von Mises + reference-map grid
                ax.set_facecolor("#0b0b0b")
                ax.streamplot(xc, xc, uc, vc, density=1.1, color=stream_color,
                              linewidth=0.6, arrowsize=0.5)
                solid = np.stack([pp <= 0 for pp in pps]).any(0)
                vmm = np.where(solid, vmf, np.nan)         # solid stress only
                im = ax.imshow(np.ma.masked_invalid(vmm), origin="lower", extent=[0, 1, 0, 1],
                               cmap="inferno", vmin=0, vmax=vmax_stress, interpolation="nearest")
                for k, pp in enumerate(pps):
                    ax.contour(Xc, Yc, pp, levels=[0.0], colors=["white"], linewidths=1.3)
                    inside = pp <= 0
                    X1m = np.where(inside, X1s[k], np.nan); X2m = np.where(inside, X2s[k], np.nan)
                    ax.contour(Xc, Yc, X1m, levels=glv, colors=["#ffffff"], linestyles="dashed", linewidths=0.5, alpha=0.5)
                    ax.contour(Xc, Yc, X2m, levels=glv, colors=["#ffffff"], linestyles="dashed", linewidths=0.5, alpha=0.5)
                cb_label = "von Mises stress (solid)"
            else:  # 'contact': ONLY the contact stress -> bright exactly where parts touch
                ax.set_facecolor("#0b0b0b")
                ax.streamplot(xc, xc, uc, vc, density=1.0, color=stream_color,
                              linewidth=0.5, arrowsize=0.4)
                eta_c, gsum_c, eps_c, etaw_c = 2.5, 5.0, 3.0 * dx, 3.0   # match the run
                phiw = np.minimum(np.minimum(Xc, 1.0 - Xc), np.minimum(Yc, 1.0 - Yc))
                cxx = np.zeros((N, N)); cxy = np.zeros((N, N)); cyy = np.zeros((N, N))
                ns = len(pps)
                for a in range(ns):
                    for b in range(a + 1, ns):                          # part-part contact
                        tx, ty, tz = contact_stress(pps[a], pps[b], eta_c, gsum_c, eps_c, dx, dx)
                        cxx += tx; cxy += ty; cyy += tz
                    tx, ty, tz = contact_stress(pps[a], phiw, etaw_c, gsum_c, eps_c, dx, dx)  # part-wall
                    cxx += tx; cxy += ty; cyy += tz
                cvm = np.sqrt(3.0 * (0.25 * (cxx - cyy)**2 + cxy**2))
                im = ax.imshow(cvm, origin="lower", extent=[0, 1, 0, 1], cmap="inferno",
                               vmin=0, vmax=vmax_stress, interpolation="bilinear")
                for k, pp in enumerate(pps):
                    ax.contour(Xc, Yc, pp, levels=[0.0], colors=["white"], linewidths=1.2)
                cb_label = "contact stress"
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


def rerender(N=256, render_modes=('speed', 'stress'), out_root="outputs", vmax_stress=3.0):
    """Re-render the GIFs from saved frames on disk (instant -- no re-simulation)."""
    cols = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    save_dir = os.path.join(out_root, f"mac_showcase_N{N}")
    print(f"[showcase] re-rendering saved frames (N={N}) from {save_dir} ...")
    return _render_frames(save_dir, render_modes, N, cols, out_root, vmax_stress)


if __name__ == "__main__":
    # re-render from saved frames (no simulation):  ... rerender 256 [mode]
    if len(sys.argv) > 1 and sys.argv[1] == "rerender":
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        modes = (sys.argv[3],) if len(sys.argv) > 3 else ('speed', 'stress')
        rerender(N=N, render_modes=modes)
    else:
        N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
        t_end = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
        modes = (sys.argv[3],) if len(sys.argv) > 3 else ('speed', 'stress')
        run(N=N, t_end=t_end, render_modes=modes)
