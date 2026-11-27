"""RM-IIM exact test with a NON-UNIFORM solid stress (bulk force inside the solid).

A disc carries the isotropic stress s(x) I with s varying in space. The solid force is then
chi grad s plus the interface traction -s n, which a pressure p = s inside, 0 outside balances
exactly: the exact solution is u = 0. In the sharp formulation this exercises the bulk force,
the pressure jump [p] = -s, and the jump of the normal pressure gradient, which must include
the bulk force, [p_n] = -d(G.tau)/ds - div(sigma_s).n (here = -grad s . n). Without that term
(pyRMT.iim.PN_BULK = False) the cut-face pressure gradient is wrong at first order.

Reported: max |u| of the steady state, solved directly (exact value 0), and the pressure error away
from the interface. Usage: python benchmarks/iim_nonuniform_stress.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyRMT.mac import poisson_eigs_periodic, solve_poisson_periodic, divergence_per
import pyRMT.iim as I

MU, R, C, S0 = 1.0, 0.21, (0.47, 0.52), 1.0
sfun = lambda X, Y: S0 * (1.0 + 0.6 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) + 0.3 * np.cos(4 * np.pi * Y))


def solve(N):
    """Direct steady solve of the sharp discretisation (no pseudo-time; cf. iim_eigenstress_pressure):
    mu (L u + C) + f - (grad p - g_jump) = 0, div u = -d_c."""
    h = 1.0 / N; xc = (np.arange(N) + 0.5) * h; X, Y = np.meshgrid(xc, xc)
    phi = np.hypot(X - C[0], Y - C[1]) - R; pe = poisson_eigs_periodic(N, N, h, h)
    s = sfun(X, Y); z = np.zeros((N, N))
    J = I.traction_jumps(phi, s, z, s, MU)
    fu, fv = I.bulk_solid_force_per(phi, s, z, s, h, h, mode="face")
    J2 = I.second_order_jumps(phi, J, np.gradient(s, h, axis=1), np.gradient(s, h, axis=0), mu=MU, h=h)
    pu, pv = I._to_u(phi), I._to_v(phi)
    Cu = I.laplacian_jump_correction(pu, I._to_u(J["ux"]), I._to_u(J["uy"]), h, h, gxx=I._to_u(J2["uxx"]), gyy=I._to_u(J2["uyy"]))
    Cv = I.laplacian_jump_correction(pv, I._to_v(J["vx"]), I._to_v(J["vy"]), h, h, gxx=I._to_v(J2["vxx"]), gyy=I._to_v(J2["vyy"]))
    cx, cy = I._pressure_face_jumps_per(phi, J["jp"], (J2["px"], J2["py"]), h); gxj, gyj = cx / h, cy / h
    dc = I.divergence_jump_correction_per(pu, I._to_u(J["ux"]), pv, I._to_v(J["vy"]), h, h, phi_c=phi)
    lap = lambda q: (np.roll(q, 1, 0) + np.roll(q, -1, 0) + np.roll(q, 1, 1) + np.roll(q, -1, 1) - 4 * q) / h ** 2
    p = solve_poisson_periodic(divergence_per(MU * Cu + gxj + fu, MU * Cv + gyj + fv, h, h) - MU * lap(dc), pe)
    u = solve_poisson_periodic(((p - np.roll(p, 1, 1)) / h - gxj - MU * Cu - fu) / MU, pe)
    v = solve_poisson_periodic(((p - np.roll(p, 1, 0)) / h - gyj - MU * Cv - fv) / MU, pe)
    far = np.abs(phi) > 2.5 * h
    ex = np.where(phi <= 0, s, 0.0); d = (p - p[far].mean()) - (ex - ex[far].mean())
    return max(np.abs(u - u.mean()).max(), np.abs(v - v.mean()).max()), float(np.abs(d[far]).max())


def main():
    for flag in (False, True):
        I.PN_BULK = flag; prev = None
        print(f"bulk-force term in [p_n]: {flag}")
        for N in (32, 64, 128, 256):
            r = solve(N)
            o = "" if prev is None else f"   orders {np.log2(prev[0] / r[0]):.2f} {np.log2(prev[1] / r[1]):.2f}"
            print(f"  N={N:4d}  max|u| {r[0]:.2e}   max p error (> 2.5 cells from the interface) {r[1]:.2e}{o}", flush=True)
            prev = r
    I.PN_BULK = True


if __name__ == "__main__":
    main()
