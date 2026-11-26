"""RM-IIM test with an exact answer and a traction that varies along the interface.

Steady Stokes flow in the periodic unit box driven by a uniform deviatoric stress sigma0 =
s diag(1,-1) inside an ellipse (a frozen, uniformly pre-stretched disc): force = div(chi sigma0),
i.e. a pure interface traction -sigma0.n with both normal and tangential parts varying along the
interface. Reference: the Fourier series of the solution, exact coefficient by coefficient
(chi_hat of an ellipse is 2 pi a b J1(q)/q), summed with M^2 modes by one FFT.

  smeared : stress blended by (1-H) over 2 cells + compact force (current method)
  sharp   : jumps from the stress (pyRMT.iim), first-order or with the second-order terms
"""
import os, sys
import numpy as np
from scipy.special import j1
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyRMT.mac import (poisson_eigs_periodic, lap_eigs_periodic, solve_helmholtz_periodic, project_per,
                       solid_force_faces)
from pyRMT.functions import smoothed_heaviside
from pyRMT.iim import traction_jumps, sharp_step_per, second_order_jumps

MU, S0, R, STRETCH, C = 1.0, 1.0, 0.15, 1.15, (0.5, 0.5)
A, B = R * STRETCH, R / STRETCH
M = 2048
INCREMENTAL = True


def reference():
    """u, v, p of the exact solution on the M x M grid of points (p/M, q/M)."""
    k = 2 * np.pi * np.fft.fftfreq(M, 1.0 / M); KX, KY = np.meshgrid(k, k)
    q = np.sqrt((A * KX) ** 2 + (B * KY) ** 2); q[0, 0] = 1.0
    chi = 2 * np.pi * A * B * j1(q) / q * np.exp(-1j * (KX * C[0] + KY * C[1])); chi[0, 0] = np.pi * A * B
    lam = np.sinc(KX / (2 * np.pi * M)) * np.sinc(KY / (2 * np.pi * M))        # Lanczos sigma factor
    Fx = 1j * KX * S0 * chi; Fy = -1j * KY * S0 * chi
    k2 = KX ** 2 + KY ** 2; k2[0, 0] = 1.0
    kF = (KX * Fx + KY * Fy) / k2
    ux = (Fx - KX * kF) / (MU * k2); uy = (Fy - KY * kF) / (MU * k2); ux[0, 0] = uy[0, 0] = 0
    f = lambda a: np.real(np.fft.ifft2(a * lam)) * M * M
    return f(ux), f(uy)


def level_set(X, Y):
    return np.hypot((X - C[0]) / STRETCH, (Y - C[1]) * STRETCH) - R      # phi0(xi), xi affine


def solve(N, method, second=True, nit=14, dt=1.0):
    h = 1.0 / N; xc = (np.arange(N) + 0.5) * h; X, Y = np.meshgrid(xc, xc)
    phi = level_set(X, Y); le = lap_eigs_periodic(N, N, h, h); pe = poisson_eigs_periodic(N, N, h, h)
    sxx = np.full((N, N), S0); syy = -sxx; sxy = np.zeros((N, N))
    u = np.zeros((N, N)); v = np.zeros((N, N))
    if method == "smeared":
        w = 1 - smoothed_heaviside(phi, 2 * h)
        fu, fv = solid_force_faces(w * sxx, w * sxy, w * syy, h, h, periodic=True)
        for _ in range(nit):
            us = solve_helmholtz_periodic(u + dt * fu, le, dt * MU); vs = solve_helmholtz_periodic(v + dt * fv, le, dt * MU)
            u, v, p = project_per(us, vs, h, h, dt, 1.0, pe)
    else:
        J = traction_jumps(phi, sxx, sxy, syy, MU)
        J2 = second_order_jumps(phi, J, mu=MU, h=h) if second else None
        G = (np.zeros((N, N)), np.zeros((N, N)))
        for _ in range(nit):
            if INCREMENTAL:
                u, v, p, G = sharp_step_per(u, v, h, h, dt, MU, 1.0, le, pe, phi, J, conserve=False, J2=J2, G_old=G)
            else:
                u, v, p = sharp_step_per(u, v, h, h, dt, MU, 1.0, le, pe, phi, J, conserve=False, J2=J2)
    return u - u.mean(), v - v.mean()


def main():
    ur, vr = reference(); ur -= ur.mean(); vr -= vr.mean()
    print(f"reference: max |u| {np.hypot(ur, vr).max():.4f}")
    for lab, meth, sec, dt, nit in (("smeared", "smeared", False, 1.0, 14),
                                    ("sharp incremental, 1st-order jumps, dt=0.05", "sharp", False, 0.05, 200),
                                    ("sharp incremental, 2nd-order jumps, dt=0.05", "sharp", True, 0.05, 200),
                                    ("sharp incremental, 2nd-order jumps, dt=0.5", "sharp", True, 0.5, 60)):
        prev = None; print(lab)
        for N in (32, 64, 128):
            u, v = solve(N, meth, sec, nit=nit, dt=dt); st = M // N
            eu = u - ur[st // 2::st, 0::st]; ev = v - vr[0::st, st // 2::st]      # u at (i h,(j+.5)h), v at ((i+.5)h, j h)
            e = np.concatenate([eu.ravel(), ev.ravel()]); mx, l2 = np.abs(e).max(), np.sqrt(np.mean(e ** 2))
            o = "" if prev is None else f"  orders max {np.log2(prev[0] / mx):.2f} L2 {np.log2(prev[1] / l2):.2f}"
            print(f"  N={N:4d}  max {mx:.2e}  L2 {l2:.2e}{o}", flush=True); prev = (mx, l2)


if __name__ == "__main__":
    main()
