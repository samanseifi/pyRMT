"""Taylor-Green decaying-vortex convergence on the periodic MAC solver.

Exact NS solution on [0,2pi]^2:
    u = -cos(x) sin(y) e^{-2 nu t},  v = sin(x) cos(y) e^{-2 nu t},
    p = -1/4 (cos 2x + cos 2y) e^{-4 nu t}.
We integrate to T with a small fixed dt (so temporal error is a low floor) and
measure the spatial L2 error of u vs N -> the solver's spatial order. On the
staggered grid with consistent operators this should be ~2 (the order the
collocated solver could not reach for the interface-coupled fields, #18).

Usage: python benchmarks/mac_taylor_green_convergence.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyRMT.mac import (momentum_predictor_periodic, project_per,
                       poisson_eigs_periodic, divergence_per)

L = 2 * np.pi


def _faces(N):
    dx = L / N
    i = np.arange(N)
    xf = i * dx                      # x-face / y-face node lines
    xc = (i + 0.5) * dx              # cell centres
    Xu, Yu = np.meshgrid(xf, xc)     # u at (i dx, (j+.5)dy)
    Xv, Yv = np.meshgrid(xc, xf)     # v at ((i+.5)dx, j dy)
    return dx, Xu, Yu, Xv, Yv


def tg_exact(Xu, Yu, Xv, Yv, nu, t):
    e = np.exp(-2 * nu * t)
    return (-np.cos(Xu) * np.sin(Yu) * e, np.sin(Xv) * np.cos(Yv) * e)


def run_one(N, nu=0.05, T=0.2, dt=2e-4):
    dx = L / N
    _, Xu, Yu, Xv, Yv = _faces(N)
    u, v = tg_exact(Xu, Yu, Xv, Yv, nu, 0.0)
    eig = poisson_eigs_periodic(N, N, dx, dx)
    nsteps = int(round(T / dt))
    for _ in range(nsteps):
        us, vs = momentum_predictor_periodic(u, v, nu, dx, dx, dt)
        u, v, _ = project_per(us, vs, dx, dx, dt, 1.0, eig)
    ue, ve = tg_exact(Xu, Yu, Xv, Yv, nu, nsteps * dt)
    erru = np.sqrt(np.mean((u - ue) ** 2))
    divmax = np.max(np.abs(divergence_per(u, v, dx, dx)))
    return erru, divmax


def run():
    nu, T, dt = 0.05, 0.2, 2e-4
    Ns = [32, 64, 128]
    print(f"[MAC TG conv] nu={nu} T={T} dt={dt} (forward-Euler, central)")
    errs = []
    for N in Ns:
        e, dmax = run_one(N, nu, T, dt)
        errs.append(e)
        print(f"  N={N:4d}  L2 err(u)={e:.3e}  max|div|={dmax:.1e}")
    print("  spatial order (Richardson):")
    for k in range(1, len(Ns)):
        p = np.log(errs[k - 1] / errs[k]) / np.log(Ns[k] / Ns[k - 1])
        print(f"    N {Ns[k-1]}->{Ns[k]}:  order = {p:.2f}")
    return Ns, errs


if __name__ == "__main__":
    run()
