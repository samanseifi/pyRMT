"""Sharp interface next to a no-slip wall: static tests in the walled box (bc="lid", U_lid = 0).

(a) Disc with a uniform isotropic stress s I, its top g cells below the top wall. Exact solution:
    u = 0, p = s inside, 0 outside (+ const). Measures the error of the wall-adjacent cut cells.
(b) Ellipse with a uniform deviatoric stress s diag(1,-1), same placement (a physical gap of
    0.025, i.e. 3.2 cells at N = 128). No closed form: successive-grid differences.
Steady state by pseudo-time stepping of the incremental sharp step (mu = 1, dt = 0.02).

Usage: python benchmarks/iim_wall_static.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyRMT.mac import poisson_eigs_neumann
import pyRMT.iim as I

MU, S0, R = 1.0, 1.0, 0.2


def solve(N, kind, gap, nit=400, dt=0.02, xcen=0.5):
    h = 1.0 / N; xc = (np.arange(N) + 0.5) * h; X, Y = np.meshgrid(xc, xc)
    if kind == "iso":
        yc = 1.0 - R - gap; phi = np.hypot(X - xcen, Y - yc) - R
        sxx = np.full((N, N), S0); syy = sxx.copy()
    else:
        st = 1.15; yc = 1.0 - R / st - gap
        phi = np.hypot((X - xcen) / st, (Y - yc) * st) - R
        sxx = np.full((N, N), S0); syy = -sxx
    sxy = np.zeros((N, N)); eig = poisson_eigs_neumann(N, N, h, h)
    J = I.traction_jumps(phi, sxx, sxy, syy, MU); J2 = I.second_order_jumps(phi, J, 0 * sxx, 0 * sxx, mu=MU, h=h)
    u = np.zeros((N, N)); v = np.zeros((N, N)); G = (np.zeros((N, N)), np.zeros((N, N)))
    for _ in range(nit):
        u, v, p, G = I.sharp_step_per(u, v, h, h, dt, MU, 1.0, None, eig, phi, J, J2=J2, G_old=G,
                                      bc="lid", U_lid=0.0, conserve=False)
        u[:, 0] = 0.0; v[0, :] = 0.0
    return u, v, p, phi


def main():
    print("(a) isotropic disc, exact u = 0, [p] = s; gap to the lid in cells:")
    for g in (3.0, 2.0, 1.0):
        for N in (64, 128, 256):
            u, v, p, phi = solve(N, "iso", g / N)
            far = np.abs(phi) > 2.5 / N
            ex = np.where(phi <= 0, S0, 0.0); d = (p - p[far].mean()) - (ex - ex[far].mean())
            print(f"  gap {g:.0f} cells, N={N:4d}: max|u| {max(np.abs(u).max(), np.abs(v).max()):.2e}  "
                  f"max p error (>2.5 cells from the interface) {np.abs(d[far]).max():.2e}", flush=True)
    print("(b) stretched ellipse, deviatoric stress, gap 0.025: successive differences (velocity on cell centres)")
    F = {N: solve(N, "dev", 0.025) for N in (64, 128, 256)}
    cen = lambda u, v: (0.5 * (u + np.roll(u, -1, 1)), 0.5 * (v + np.roll(v, -1, 0)))
    rst = lambda f: 0.25 * (f[0::2, 0::2] + f[1::2, 0::2] + f[0::2, 1::2] + f[1::2, 1::2])
    prev = None
    for a in (64, 128):
        ua, va = cen(*F[a][:2]); ub, vb = cen(*F[2 * a][:2]); e = np.hypot(ua - rst(ub), va - rst(vb))
        top = np.zeros_like(e, bool); top[-8 * a // 64:, :] = True        # the strip between the ellipse and the lid
        r = (np.sqrt(np.mean(e ** 2)), e.max(), e[top].max())
        o = "" if prev is None else "  orders " + " ".join(f"{np.log2(x / y):.2f}" for x, y in zip(prev, r))
        print(f"  {a}/{2*a}: L2 {r[0]:.2e}  max {r[1]:.2e}  max near the lid {r[2]:.2e}{o}   (max |u| {np.hypot(ub, vb).max():.3f})")
        prev = r


if __name__ == "__main__":
    main()
