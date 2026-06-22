"""Surface-tension relaxation on the MAC grid: a SQUARE drop should relax to a
CIRCLE (surface tension minimises interface length at fixed area).

Single-phase fluid with a level set phi tracking the drop; balanced-force CSF
(f = -gamma*kappa*grad H at faces). phi is advected semi-Lagrangian by the induced
flow and reinitialised (FMM) only every `reinit_every` steps -- per-step FMM reinit
is NOT volume-preserving and loses mass (-21% at N=64, worse at N=128); reinit_every
~50 keeps phi near signed-distance for curvature while conserving area to ~2%.
Diagnostics: area (conserved by incompressibility) and circularity (-> 1 for a circle).

Usage: python benchmarks/mac_surface_tension_relax.py [N] [t_end] [gamma]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.mac import (mac_grid, momentum_predictor_freeslip, project,
                       poisson_eigs_neumann, interfacial_force_faces, divergence)
from pyRMT.functions import (advect_reference_map, reinitialize_phi_fmm,
                             compute_curvature, smoothed_heaviside)


def _circularity(phi, dx, dy, w_t):
    """4*pi*A / P^2  (=1 for a circle). Area from H, perimeter from int |grad H|."""
    H = smoothed_heaviside(phi, w_t)
    A = np.sum(1.0 - H) * dx * dy
    gx = np.gradient(H, dx, axis=1); gy = np.gradient(H, dy, axis=0)
    P = np.sum(np.sqrt(gx**2 + gy**2)) * dx * dy
    return A, P, (4.0 * np.pi * A / max(P*P, 1e-30))


def run(N=128, t_end=4.0, gamma=0.1, half=0.20, mu=0.02, rho=1.0, reinit_every=50,
        out_root="outputs"):
    dx, dy = mac_grid(N, N)
    xc = (np.arange(N) + 0.5) * dx
    Xc, Yc = np.meshgrid(xc, xc)
    Xg, Yg = np.meshgrid(np.arange(N) * dx, np.arange(N) * dy)
    cx = cy = 0.5
    # square drop: L-infinity "distance"; reinit makes it a true signed distance
    phi = np.maximum(np.abs(Xc - cx), np.abs(Yc - cy)) - half
    phi = reinitialize_phi_fmm(phi, dx, dy)
    w_t = 2.0 * dx; nu = mu / rho
    u = np.zeros((N, N + 1)); v = np.zeros((N + 1, N))
    eig = poisson_eigs_neumann(N, N, dx, dy)
    dt = min(0.4 * np.sqrt(rho * dx**3 / (2 * np.pi * gamma)), 0.2 * dx * dx / nu)

    A0, P0, c0 = _circularity(phi, dx, dy, w_t)
    R_eq = np.sqrt(A0 / np.pi)
    out_dir = os.path.join(out_root, f"mac_st_relax_N{N}"); os.makedirs(out_dir, exist_ok=True)
    print(f"[ST relax] N={N} gamma={gamma} square half={half} -> A0={A0:.4f} "
          f"R_eq={R_eq:.4f} circ0={c0:.3f} dt={dt:.2e}")

    snaps = {}; snap_times = [0.0, 0.25, 1.0, t_end]
    t = 0.0; step = 0
    while t < t_end:
        step += 1
        if t + dt > t_end:
            dt = t_end - t
        u_c = 0.5 * (u[:, :-1] + u[:, 1:]); v_c = 0.5 * (v[:-1, :] + v[1:, :])
        phi = advect_reference_map(phi, u_c, v_c, Xg, Yg, dt, dx, dy, phi, 'semilagrangian', 0.0)
        if reinit_every > 0 and step % reinit_every == 0:
            phi = reinitialize_phi_fmm(phi, dx, dy)
        kappa = compute_curvature(phi, dx, dy)
        H = smoothed_heaviside(phi, w_t)
        fu, fv = interfacial_force_faces(kappa, H, gamma, dx, dy)
        ustar, vstar = momentum_predictor_freeslip(u, v, nu, dx, dy, dt, fu=fu, fv=fv, rho=rho)
        u, v, p = project(ustar, vstar, dx, dy, dt, rho, eig)
        t += dt
        for ts in snap_times:
            if ts not in snaps and t >= ts:
                snaps[ts] = phi.copy()
        if step % 200 == 0 or t >= t_end:
            A, P, c = _circularity(phi, dx, dy, w_t)
            print(f"  step {step:5d} t={t:5.3f}  area={A:.4f} (A0={A0:.4f}, drift={100*(A-A0)/A0:+.2f}%) "
                  f"circularity={c:.3f}  max|div|={np.max(np.abs(divergence(u,v,dx,dy))):.1e}")

    A, P, c = _circularity(phi, dx, dy, w_t)
    print(f"[ST relax] final: area drift={100*(A-A0)/A0:+.2f}%  circularity={c:.3f} "
          f"(1.0=circle); R_eq={R_eq:.4f}")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        for ts in sorted(snaps):
            ax.contour(Xc, Yc, snaps[ts], levels=[0.0], linewidths=2,
                       colors=[plt.cm.viridis(ts / max(snap_times))])
        th = np.linspace(0, 2*np.pi, 200)
        ax.plot(cx + R_eq*np.cos(th), cy + R_eq*np.sin(th), 'r--', lw=1, label=f'equal-area circle R={R_eq:.3f}')
        ax.set_aspect('equal'); ax.set_title(f'Surface tension: square -> circle (N={N})')
        ax.legend(); ax.set_xlim(0.2, 0.8); ax.set_ylim(0.2, 0.8)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "square_to_circle.png"), dpi=140)
        print(f"  saved {out_dir}/square_to_circle.png  (interface at t={sorted(snaps)})")
    except Exception as e:
        print(f"  (plot skipped: {e})")
    return c


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    t_end = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    gamma = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    run(N=N, t_end=t_end, gamma=gamma)
