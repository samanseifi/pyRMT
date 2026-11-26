"""RM-IIM: pressure accuracy on the exact eigenstress test (steady Stokes, stretched ellipse).

The steady state of each discretisation is solved directly (one FFT Stokes solve, no pseudo-time
stepping), so the result is free of iteration and splitting error:
    mu (L u + C) - (grad p - g_jump) = 0,   div u = -d_c
    =>  Lap p = div(mu C + g_jump) - mu L d_c,   u = L^{-1} (grad p - g_jump - mu C) / mu.
Do not compare pressures from time stepping here: the non-incremental projection scalar of the
smeared scheme is (I - dt mu L)^{-1} p, a smoothed pressure, and the incremental scheme relaxes
high-wavenumber pressure modes slowly at large dt.

Reference pressure: p_hat = s (kx^2 - ky^2)/k^2 chi_hat. p is harmonic on each side of the
interface, so a Gaussian mean of width sigma leaves it unchanged away from the interface
(mean-value property) while making the Fourier sum converge exponentially (no Gibbs error
beyond about 6 sigma = 0.012 from the interface).

Reported: velocity error (max, L2) and pressure error far from the interface (> 0.04) and near
it (> 1.5 cells and > 0.012). Usage: python benchmarks/iim_eigenstress_pressure.py
"""
import os, sys
import numpy as np
from scipy.special import j1
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import iim_eigenstress as E
import pyRMT.iim as I
from pyRMT.mac import poisson_eigs_periodic, solve_poisson_periodic, divergence_per, solid_force_faces
from pyRMT.functions import smoothed_heaviside

M, MU, SIGMA = E.M, E.MU, 0.002


def reference_p():
    k = 2 * np.pi * np.fft.fftfreq(M, 1.0 / M); KX, KY = np.meshgrid(k, k)
    q = np.sqrt((E.A * KX) ** 2 + (E.B * KY) ** 2); q[0, 0] = 1.0
    chi = 2 * np.pi * E.A * E.B * j1(q) / q * np.exp(-1j * (KX * E.C[0] + KY * E.C[1])); chi[0, 0] = np.pi * E.A * E.B
    k2 = KX ** 2 + KY ** 2; k2[0, 0] = 1.0
    ph = E.S0 * (KX ** 2 - KY ** 2) / k2 * chi; ph[0, 0] = 0
    return np.real(np.fft.ifft2(ph * np.exp(-0.5 * SIGMA ** 2 * (KX ** 2 + KY ** 2)))) * M * M


def steady(N, method):
    h = 1.0 / N; xc = (np.arange(N) + 0.5) * h; X, Y = np.meshgrid(xc, xc)
    phi = E.level_set(X, Y); pe = poisson_eigs_periodic(N, N, h, h)
    sxx = np.full((N, N), E.S0); syy = -sxx; sxy = np.zeros((N, N))
    gx = lambda p: (p - np.roll(p, 1, 1)) / h
    gy = lambda p: (p - np.roll(p, 1, 0)) / h
    if method == "smeared":
        w = 1 - smoothed_heaviside(phi, 2 * h)
        fu, fv = solid_force_faces(w * sxx, w * sxy, w * syy, h, h, periodic=True)
        p = solve_poisson_periodic(divergence_per(fu, fv, h, h), pe)
        ru, rv = gx(p) - fu, gy(p) - fv
    else:
        J = I.traction_jumps(phi, sxx, sxy, syy, MU); J2 = I.second_order_jumps(phi, J, mu=MU, h=h)
        pu, pv = I._to_u(phi), I._to_v(phi)
        Cu = I.laplacian_jump_correction(pu, I._to_u(J["ux"]), I._to_u(J["uy"]), h, h,
                                         gxx=I._to_u(J2["uxx"]), gyy=I._to_u(J2["uyy"]))
        Cv = I.laplacian_jump_correction(pv, I._to_v(J["vx"]), I._to_v(J["vy"]), h, h,
                                         gxx=I._to_v(J2["vxx"]), gyy=I._to_v(J2["vyy"]))
        cx, cy = I._pressure_face_jumps_per(phi, J["jp"], (J2["px"], J2["py"]), h)
        gxj, gyj = cx / h, cy / h
        dc = I.divergence_jump_correction_per(pu, I._to_u(J["ux"]), pv, I._to_v(J["vy"]), h, h, phi_c=phi)
        lap = lambda f: (np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f) / h ** 2
        p = solve_poisson_periodic(divergence_per(MU * Cu + gxj, MU * Cv + gyj, h, h) - MU * lap(dc), pe)
        ru, rv = gx(p) - gxj - MU * Cu, gy(p) - gyj - MU * Cv
    u = solve_poisson_periodic(ru / MU, pe); v = solve_poisson_periodic(rv / MU, pe)
    return u - u.mean(), v - v.mean(), p, phi


def main():
    ur, vr = E.reference(); ur -= ur.mean(); vr -= vr.mean(); pr = reference_p()
    for meth in ("smeared", "sharp"):
        print(meth); prev = None
        for N in (32, 64, 128, 256):
            u, v, p, phi = steady(N, meth); st = M // N
            e = np.concatenate([(u - ur[st // 2::st, 0::st]).ravel(), (v - vr[0::st, st // 2::st]).ravel()])
            pex = pr[st // 2::st, st // 2::st]; s = phi <= 0
            dist = np.where(s, ndi.distance_transform_edt(s), ndi.distance_transform_edt(~s)) / N
            far = dist > 0.04; near = (dist > 1.5 / N) & (dist > 6 * SIGMA)
            d = (p - p[far].mean()) - (pex - pex[far].mean())
            r = [np.abs(e).max(), np.sqrt(np.mean(e ** 2)), np.sqrt(np.mean(d[far] ** 2)), np.abs(d[far]).max(),
                 np.sqrt(np.mean(d[near] ** 2)), np.abs(d[near]).max()]
            o = "" if prev is None else "  orders " + " ".join(f"{np.log2(a / b):.2f}" for a, b in zip(prev, r))
            print(f"  N={N:4d} u max {r[0]:.2e} L2 {r[1]:.2e} | p far L2 {r[2]:.2e} max {r[3]:.2e}"
                  f" | p near L2 {r[4]:.2e} max {r[5]:.2e}{o}", flush=True)
            prev = r


if __name__ == "__main__":
    main()
