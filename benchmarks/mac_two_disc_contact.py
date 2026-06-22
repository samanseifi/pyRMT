"""Two soft discs colliding on the MAC grid, with Rycroft's contact STRESS.

Two discs approach head-on; the contact stress (added to the blended solid stress,
not as a body force) should keep them from inter-penetrating and let them rebound.
THE KEY CHECK: vary eta and confirm it changes the dynamics -- the body-force contact
was projection-nullified (k_rep had zero effect); a contact stress should not be.

Usage: python benchmarks/mac_two_disc_contact.py [N] [t_end] [V0] [eta]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.mac import (mac_grid, momentum_predictor_freeslip, project,
                       poisson_eigs_neumann, divergence, contact_stress)
from pyRMT.functions import (extrapolate_reference_map, advect_reference_map,
    rebuild_phi_from_reference_map, solid_cauchy_stress, smoothed_heaviside,
    grad_central_x_2nd, grad_central_y_2nd)


def run(N=128, t_end=2.0, V0=0.3, eta=2.0, mu_s=0.2, mu_f=0.02, rho=1.0,
        R=0.12, sep=0.36, out_root="outputs"):
    dx, dy = mac_grid(N, N)
    xc = (np.arange(N) + 0.5) * dx
    Xc, Yc = np.meshgrid(xc, xc)
    Xg, Yg = np.meshgrid(np.arange(N) * dx, np.arange(N) * dy)
    xf = np.arange(N + 1) * dx
    Xu, Yu = np.meshgrid(xf, xc); Xv, Yv = np.meshgrid(xc, np.arange(N + 1) * dy)
    xa, xb, yc0 = 0.5 - sep / 2, 0.5 + sep / 2, 0.5
    pia = lambda X, Y: np.sqrt((X - xa)**2 + (Y - yc0)**2) - R
    pib = lambda X, Y: np.sqrt((X - xb)**2 + (Y - yc0)**2) - R
    pa = pia(Xc, Yc); pb = pib(Xc, Yc)
    ma = (pa <= 0).astype(float); mb = (pb <= 0).astype(float)
    X1a, X2a = extrapolate_reference_map(Xc * ma, Yc * ma, pa, dx, dy, 3)
    X1b, X2b = extrapolate_reference_map(Xc * mb, Yc * mb, pb, dx, dy, 3)
    w_t = 2.0 * dx; nu = mu_f / rho; eps = 3.0 * dx
    # approaching velocity: A -> +x, B -> -x (zeroed in fluid)
    Ha0 = smoothed_heaviside(pa, w_t); Hb0 = smoothed_heaviside(pb, w_t)
    HaU = smoothed_heaviside(pia(Xu, Yu), w_t); HbU = smoothed_heaviside(pib(Xu, Yu), w_t)
    u = V0 * (1 - HaU) - V0 * (1 - HbU)
    v = np.zeros((N + 1, N))
    u[:, 0] = 0; u[:, -1] = 0
    eig = poisson_eigs_neumann(N, N, dx, dy)
    cs = np.sqrt(mu_s / rho)
    dt = min(0.25 * dx / max(V0, 1e-9), 0.2 * dx * dx / nu, 0.25 * dx / (cs + 1e-9))
    out_dir = os.path.join(out_root, f"mac_two_disc_N{N}"); os.makedirs(out_dir, exist_ok=True)
    print(f"[2-disc] N={N} V0={V0} eta={eta} mu_s={mu_s} 2R={2*R:.3f} dt={dt:.2e}")

    t = 0.0; step = 0; hist = []
    while t < t_end:
        step += 1
        if t + dt > t_end:
            dt = t_end - t
        u_c = 0.5 * (u[:, :-1] + u[:, 1:]); v_c = 0.5 * (v[:-1, :] + v[1:, :])
        pa = rebuild_phi_from_reference_map(X1a, X2a, pia); ma = (pa <= 0).astype(float)
        pb = rebuild_phi_from_reference_map(X1b, X2b, pib); mb = (pb <= 0).astype(float)
        X1a = advect_reference_map(X1a, u_c, v_c, Xg, Yg, dt, dx, dy, pa, 'semilagrangian', 0.0) * ma
        X2a = advect_reference_map(X2a, u_c, v_c, Xg, Yg, dt, dx, dy, pa, 'semilagrangian', 0.0) * ma
        X1b = advect_reference_map(X1b, u_c, v_c, Xg, Yg, dt, dx, dy, pb, 'semilagrangian', 0.0) * mb
        X2b = advect_reference_map(X2b, u_c, v_c, Xg, Yg, dt, dx, dy, pb, 'semilagrangian', 0.0) * mb
        X1a, X2a = extrapolate_reference_map(X1a, X2a, pa, dx, dy, 3)
        X1b, X2b = extrapolate_reference_map(X1b, X2b, pb, dx, dy, 3)
        pa = rebuild_phi_from_reference_map(X1a, X2a, pia)
        pb = rebuild_phi_from_reference_map(X1b, X2b, pib)

        sAxx, sAxy, sAyy, Ja = solid_cauchy_stress(X1a, X2a, dx, dy, mu_s, 0.0, pa)
        sBxx, sBxy, sByy, Jb = solid_cauchy_stress(X1b, X2b, dx, dy, mu_s, 0.0, pb)
        Ha = smoothed_heaviside(pa, w_t); Hb = smoothed_heaviside(pb, w_t)
        Sxx = (1 - Ha) * sAxx + (1 - Hb) * sBxx
        Sxy = (1 - Ha) * sAxy + (1 - Hb) * sBxy
        Syy = (1 - Ha) * sAyy + (1 - Hb) * sByy
        if eta > 0:
            txx, txy, tyy = contact_stress(pa, pb, eta, 2 * mu_s, eps, dx, dy)
            Sxx += txx; Sxy += txy; Syy += tyy
        divx = grad_central_x_2nd(Sxx, dx) + grad_central_y_2nd(Sxy, dy)
        divy = grad_central_x_2nd(Sxy, dx) + grad_central_y_2nd(Syy, dy)
        fu = np.zeros((N, N + 1)); fu[:, 1:-1] = 0.5 * (divx[:, 1:] + divx[:, :-1])
        fv = np.zeros((N + 1, N)); fv[1:-1, :] = 0.5 * (divy[1:, :] + divy[:-1, :])
        ustar, vstar = momentum_predictor_freeslip(u, v, nu, dx, dy, dt, fu=fu, fv=fv, rho=rho)
        u, v, p = project(ustar, vstar, dx, dy, dt, rho, eig)
        t += dt
        ca = Xc[pa <= 0].mean() if (pa <= 0).any() else np.nan
        cb = Xc[pb <= 0].mean() if (pb <= 0).any() else np.nan
        hist.append((t, ca, cb, cb - ca))
        if step % 100 == 0 or t >= t_end:
            print(f"  step {step:5d} t={t:5.3f} cxA={ca:.3f} cxB={cb:.3f} gap={cb-ca:.3f} "
                  f"(2R={2*R:.3f}) max|u|={np.max(np.abs(u)):.3f} max|div|={np.max(np.abs(divergence(u,v,dx,dy))):.1e}")

    hist = np.array(hist)
    gmin = np.nanmin(hist[:, 3]); gend = hist[-1, 3]
    imin = int(np.nanargmin(hist[:, 3]))
    rebound = imin < len(hist) - 2 and gend > gmin + 5e-3
    print(f"[2-disc] eta={eta}: min gap={gmin:.3f} (2R={2*R:.3f}, overlap if <2R) at t={hist[imin,0]:.2f}, "
          f"end gap={gend:.3f}  -> {'REBOUND' if rebound else 'no clear rebound'}")
    np.savetxt(os.path.join(out_dir, f"gap_eta{eta}.csv"), hist, delimiter=",",
               header="t,cxA,cxB,gap", comments="")
    return hist


def compare(N=64, t_end=3.0, V0=0.5, etas=(0.0, 1.5), **kw):
    """Run several eta and plot gap(t) to show contact-mediated rebound."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R = kw.get("R", 0.12)
    out_dir = os.path.join(kw.get("out_root", "outputs"), f"mac_two_disc_N{N}")
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for eta in etas:
        h = run(N=N, t_end=t_end, V0=V0, eta=eta, **kw)
        ax.plot(h[:, 0], h[:, 3], lw=2, label=f"eta={eta}")
    ax.axhline(2 * R, color="r", ls="--", lw=1, label=f"2R={2*R:.2f} (touching)")
    ax.set_xlabel("t"); ax.set_ylabel("centroid gap")
    ax.set_title(f"Two-disc collision: gap vs time (V0={V0}, N={N})")
    ax.legend(); ax.grid(True, alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "collision_gap.png"), dpi=140)
    print(f"  saved {out_dir}/collision_gap.png")


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    t_end = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    V0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    eta = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    run(N=N, t_end=t_end, V0=V0, eta=eta)
