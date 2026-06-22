"""Validate reference-map reinitialization (Rycroft 2018): a single soft disc in a
lid-driven cavity, which normally folds under sustained shear, should now run
indefinitely.

Method: when the current-epoch distortion gets large, RESET the reference map to
identity (xi = x), STORE the accumulated deformation gradient F, and keep the current
shape as the new reference level set. The stress uses the TOTAL deformation
F_total = F_current . F_stored, so elasticity is preserved while the per-epoch xi
gradients stay near identity (never fold).

Usage: python benchmarks/mac_reinit_test.py [N] [t_end] [reinit 0/1]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.mac import mac_grid, momentum_predictor, project, poisson_eigs_neumann, divergence
from pyRMT.functions import (extrapolate_reference_map, advect_reference_map,
    smoothed_heaviside, grad_central_x_2nd, grad_central_y_2nd)
from pyRMT.interpolators import bilinear_interpolate


def _F_current(X1, X2, dx, dy):
    g11 = grad_central_x_2nd(X1, dx); g12 = grad_central_y_2nd(X1, dy)
    g21 = grad_central_x_2nd(X2, dx); g22 = grad_central_y_2nd(X2, dy)
    detG = g11 * g22 - g12 * g21
    detG = np.where(np.abs(detG) < 1e-9, 1e-9, detG)
    return g22 / detG, -g12 / detG, -g21 / detG, g11 / detG, 1.0 / detG


def _matmul(A, B):                       # 2x2 . 2x2 (component lists [11,12,21,22])
    return [A[0]*B[0] + A[1]*B[2], A[0]*B[1] + A[1]*B[3],
            A[2]*B[0] + A[3]*B[2], A[2]*B[1] + A[3]*B[3]]


def run(N=96, t_end=15.0, reinit=True, U_lid=1.0, mu_s=0.3, mu_f=0.01, rho=1.0,
        reinit_J=1.8):
    dx, dy = mac_grid(N, N)
    xc = (np.arange(N) + 0.5) * dx
    Xc, Yc = np.meshgrid(xc, xc)
    Xg, Yg = np.meshgrid(np.arange(N) * dx, np.arange(N) * dy)
    x0, y0, R = 0.6, 0.5, 0.2
    pin = lambda X, Y: np.sqrt((X - x0)**2 + (Y - y0)**2) - R
    phi = pin(Xc, Yc); m = (phi <= 0).astype(float)
    X1, X2 = extrapolate_reference_map(Xc * m, Yc * m, phi, dx, dy, 3)
    Fs = [np.ones((N, N)), np.zeros((N, N)), np.zeros((N, N)), np.ones((N, N))]
    phiref = phi.copy()                  # reference level set (grid)
    w_t = 2.0 * dx; nu = mu_f / rho
    u = np.zeros((N, N + 1)); v = np.zeros((N + 1, N))
    eig = poisson_eigs_neumann(N, N, dx, dy)
    cs = np.sqrt(mu_s / rho)
    dt = min(0.3 * dx / U_lid, 0.2 * dx * dx / nu, 0.3 * dx / (cs + 1e-9))
    print(f"[reinit-test] N={N} reinit={reinit} mu_s={mu_s} t_end={t_end} reinit_J={reinit_J}")

    def geom(X1, X2):                    # phi = phiref(xi)  (bilinear, cell-centred offset)
        return bilinear_interpolate(phiref, X1 - 0.5 * dx, X2 - 0.5 * dy, dx, dy, N, N)

    t = 0.0; step = 0; n_reinit = 0
    while t < t_end:
        step += 1
        if t + dt > t_end:
            dt = t_end - t
        u_c = 0.5 * (u[:, :-1] + u[:, 1:]); v_c = 0.5 * (v[:-1, :] + v[1:, :])
        phi = geom(X1, X2); m = (phi <= 0).astype(float)
        X1 = advect_reference_map(X1, u_c, v_c, Xg, Yg, dt, dx, dy, phi, 'semilagrangian', 0.0) * m
        X2 = advect_reference_map(X2, u_c, v_c, Xg, Yg, dt, dx, dy, phi, 'semilagrangian', 0.0) * m
        for i in range(4):
            Fs[i] = advect_reference_map(Fs[i], u_c, v_c, Xg, Yg, dt, dx, dy, phi, 'semilagrangian', 0.0) * m
        X1, X2 = extrapolate_reference_map(X1, X2, phi, dx, dy, 3)
        Fs[0], Fs[1] = extrapolate_reference_map(Fs[0], Fs[1], phi, dx, dy, 3)
        Fs[2], Fs[3] = extrapolate_reference_map(Fs[2], Fs[3], phi, dx, dy, 3)
        phi = geom(X1, X2)

        Fc = _F_current(X1, X2, dx, dy)               # epoch deformation
        J_ep = Fc[4]
        Ftot = _matmul(Fc[:4], Fs)                    # total deformation
        b11 = Ftot[0]**2 + Ftot[1]**2
        b12 = Ftot[0]*Ftot[2] + Ftot[1]*Ftot[3]
        b22 = Ftot[2]**2 + Ftot[3]**2
        H = smoothed_heaviside(phi, w_t)
        Sxx = (1 - H) * mu_s * b11; Sxy = (1 - H) * mu_s * b12; Syy = (1 - H) * mu_s * b22
        divx = grad_central_x_2nd(Sxx, dx) + grad_central_y_2nd(Sxy, dy)
        divy = grad_central_x_2nd(Sxy, dx) + grad_central_y_2nd(Syy, dy)
        fu = np.zeros((N, N + 1)); fu[:, 1:-1] = 0.5 * (divx[:, 1:] + divx[:, :-1])
        fv = np.zeros((N + 1, N)); fv[1:-1, :] = 0.5 * (divy[1:, :] + divy[:-1, :])
        ustar, vstar = momentum_predictor(u, v, nu, dx, dy, dt, U_lid, fu=fu, fv=fv, rho=rho)
        u, v, p = project(ustar, vstar, dx, dy, dt, rho, eig)
        t += dt

        # epoch distortion (in the solid); reinit when it gets large
        sm = phi <= 0
        if sm.any():
            jin = J_ep[sm]
            dist = max(jin.max(), 1.0 / max(jin.min(), 1e-6))
        else:
            dist = np.inf
        if reinit and dist > reinit_J and sm.any():
            Fs = _matmul(Fc[:4], Fs)                  # fold epoch into the store
            for i in range(2):
                Fs[2*i], Fs[2*i+1] = extrapolate_reference_map(Fs[2*i], Fs[2*i+1], phi, dx, dy, 3)
            phiref = phi.copy()                       # current shape -> new reference
            X1 = Xc.copy(); X2 = Yc.copy()            # reset xi = x
            n_reinit += 1

        if not np.all(np.isfinite(u)) or not sm.any():
            print(f"  DIVERGED at step {step}, t={t:.3f}"); break
        if step % 200 == 0 or t >= t_end:
            Jt = (Ftot[0]*Ftot[3] - Ftot[1]*Ftot[2])
            print(f"  step {step:5d} t={t:6.3f} epochJ[{J_ep[sm].min():.2f},{J_ep[sm].max():.2f}] "
                  f"totalJ[{Jt[sm].min():.2f},{Jt[sm].max():.2f}] reinits={n_reinit} "
                  f"max|u|={np.max(np.abs(u)):.2f} div={np.max(np.abs(divergence(u,v,dx,dy))):.0e}")
    print(f"[reinit-test] reached t={t:.2f}, {n_reinit} reinits")
    return t


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    t_end = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
    reinit = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True
    run(N=N, t_end=t_end, reinit=reinit)
